#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import xarray as xr
import zarr

from manual_inference.prediction.dataset import build_predictions_dataset
from manual_inference.prediction.predict import (
    _compute_x_interp_for_export,
    _get_parallel_info,
    _init_model_comm_group,
    _load_objects,
    _parse_json,
    _predict_from_dataloader,
    _resolve_device,
)

LOG = logging.getLogger(__name__)

CANONICAL_HDATES = [
    "2023-08-26",
    "2023-08-27",
    "2023-08-28",
    "2023-08-29",
    "2023-08-30",
]
CANONICAL_STEPS = [24, 48, 72, 96, 120]
DEFAULT_HDATES = CANONICAL_HDATES
DEFAULT_STEPS = CANONICAL_STEPS
DEFAULT_SAMPLER_SEEDS = [1234 + idx for idx in range(10)]
DEFAULT_MAPPING_ZARR = (
    "/home/ecm5702/scratch/codex/active/20260629-regional-tc-testbed/"
    "scratch/tc_atlantic_mdr_west_tc_window_fullvars/in_lres_o320.zarr"
)
DEFAULT_EXTRA_ARGS_JSON = json.dumps({
    "schedule_type": "experimental_piecewise",
    "num_steps": 30,
    "sigma_max": 10000.0,
    "sigma_transition": 10.0,
    "sigma_min": 0.03,
    "high_schedule_type": "exponential",
    "low_schedule_type": "karras",
    "num_steps_high": 10,
    "num_steps_low": 20,
    "rho": 7.0,
    "sampler": "heun",
    "S_churn": 2.5,
    "S_min": 0.75,
    "S_max": 10000.0,
    "S_noise": 1.05,
})


@dataclass(frozen=True)
class TargetSample:
    fake_date: str
    idx: int
    refdate: str
    hdate: str
    step: int


def _parse_csv_dates(value: str) -> list[str]:
    out = [part.strip() for part in value.split(",") if part.strip()]
    if not out:
        raise ValueError("Expected at least one hdate")
    return out


def _parse_csv_steps(value: str) -> list[int]:
    out = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not out:
        raise ValueError("Expected at least one step")
    return out


def _parse_csv_ints(value: str, *, label: str) -> list[int]:
    out = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not out:
        raise ValueError(f"Expected at least one {label}")
    return out


def _norm_dt(value) -> str:
    arr = np.asarray(value, dtype="datetime64[s]")
    return np.datetime_as_string(arr, unit="s")


def _set_all_rng_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_request_shape(
    *,
    hdates: Sequence[str],
    steps: Sequence[int],
    sampler_seeds: Sequence[int],
    allow_noncanonical_shape: bool,
) -> None:
    if len(set(int(seed) for seed in sampler_seeds)) != len(sampler_seeds):
        raise ValueError(f"Sampler seeds must be unique, got {list(sampler_seeds)}")
    if not allow_noncanonical_shape:
        if list(hdates) != CANONICAL_HDATES:
            raise ValueError(
                "Canonical tc_o320_o1280 regional base requires Franklin/Idalia "
                f"hdates {CANONICAL_HDATES}, got {list(hdates)}. "
                "Pass --allow-noncanonical-shape only for an explicit off-base experiment."
            )
        if list(steps) != CANONICAL_STEPS:
            raise ValueError(
                "Canonical tc_o320_o1280 regional base requires lead steps "
                f"{CANONICAL_STEPS}, got {list(steps)}. "
                "Pass --allow-noncanonical-shape only for an explicit off-base experiment."
            )
        if len(sampler_seeds) != len(DEFAULT_SAMPLER_SEEDS):
            raise ValueError(
                "Canonical tc_o320_o1280 regional base requires 10 sampler seeds "
                f"(got {len(sampler_seeds)}). "
                "Pass --allow-noncanonical-shape only for an explicit off-base experiment."
            )


def _discover_targets(mapping_zarr: Path, *, hdates: Sequence[str], steps: Sequence[int]) -> dict[str, list[TargetSample]]:
    z = zarr.open(str(mapping_zarr), mode="r")
    fake_hindcasts = dict(z.attrs["fake_hindcasts"])
    fake_dates_axis = [_norm_dt(v) for v in np.asarray(z["dates"])]
    idx_by_fake_date = {fake_date: idx for idx, fake_date in enumerate(fake_dates_axis)}

    grouped: dict[str, list[TargetSample]] = defaultdict(list)
    step_set = {int(step) for step in steps}
    target_hdates = {str(hdate) for hdate in hdates}

    for fake_date, triple in fake_hindcasts.items():
        refdate, hdate, step = triple
        hdate_s = str(hdate)[:10]
        step_i = int(step)
        if hdate_s not in target_hdates or step_i not in step_set:
            continue
        fake_date_s = _norm_dt(fake_date)
        if fake_date_s not in idx_by_fake_date:
            raise KeyError(f"Fake date {fake_date_s} missing from zarr dates axis")
        grouped[hdate_s].append(
            TargetSample(
                fake_date=fake_date_s,
                idx=int(idx_by_fake_date[fake_date_s]),
                refdate=str(refdate)[:10],
                hdate=hdate_s,
                step=step_i,
            )
        )

    missing_hdates = [hdate for hdate in hdates if hdate not in grouped]
    if missing_hdates:
        raise ValueError(f"No fake-hindcast rows found for hdates: {missing_hdates}")

    for hdate, rows in grouped.items():
        rows.sort(key=lambda row: row.step)
        present_steps = [row.step for row in rows]
        if present_steps != list(steps):
            raise ValueError(f"hdate {hdate}: expected steps {list(steps)}, got {present_steps}")
        idxs = [row.idx for row in rows]
        if idxs != list(range(idxs[0], idxs[0] + len(idxs))):
            raise ValueError(f"hdate {hdate}: indices are not contiguous: {idxs}")
    return dict(grouped)


def _add_eval_metadata(
    ds: xr.Dataset,
    *,
    hdate: str,
    refdate: str,
    step: int,
    checkpoint: str,
    extra_args_json: str,
    validation_frequency: str,
    output_weather_state_mode: str,
    output_weather_states: Sequence[str],
    sampler_seeds: Sequence[int],
) -> xr.Dataset:
    hdate64 = np.asarray([np.datetime64(hdate)], dtype="datetime64[ns]")
    valid64 = np.asarray([np.datetime64(hdate) + np.timedelta64(int(step), "h")], dtype="datetime64[ns]")
    lead64 = np.asarray([np.timedelta64(int(step), "h")], dtype="timedelta64[ns]")
    ds = ds.copy()
    ds = ds.assign_coords(sampler_seed=("ensemble_member", np.asarray(sampler_seeds, dtype=np.int64)))
    ds["date"] = xr.DataArray(hdate64, dims=["sample"])
    ds["init_date"] = xr.DataArray(hdate64, dims=["sample"])
    ds["lead_step_hours"] = xr.DataArray(lead64, dims=["sample"])
    ds["valid_time"] = xr.DataArray(valid64, dims=["sample"])
    ds.attrs["init_date"] = hdate
    ds.attrs["lead_step_hours"] = int(step)
    ds.attrs["valid_time"] = str(valid64[0])
    ds.attrs["checkpoint_path"] = checkpoint
    ds.attrs["sampling_config_json"] = extra_args_json
    ds.attrs["validation_frequency"] = validation_frequency
    ds.attrs["x_interp_exported"] = 1
    ds.attrs["output_weather_state_mode"] = output_weather_state_mode
    ds.attrs["output_weather_states"] = ",".join(output_weather_states)
    ds.attrs["native_regional_evalcli_adapter"] = 1
    ds.attrs["hdate"] = hdate
    ds.attrs["refdate"] = refdate
    ds.attrs["lead_step_hours_int"] = int(step)
    ds.attrs["sampler_seed_members"] = ",".join(str(seed) for seed in sampler_seeds)
    return ds


def build_predictions_dir(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped = _discover_targets(
        Path(args.mapping_zarr).expanduser().resolve(),
        hdates=args.hdates,
        steps=args.steps,
    )

    checkpoint_path = str(Path(args.checkpoint).expanduser().resolve())
    checkpoint_name = Path(checkpoint_path).name
    if checkpoint_name.startswith("inference-"):
        candidate = Path(checkpoint_path).with_name(checkpoint_name[len("inference-"):])
        if candidate.exists():
            checkpoint_path = str(candidate)

    global_rank, local_rank, world_size = _get_parallel_info()
    device = _resolve_device(args.device, local_rank)
    model_comm_group = _init_model_comm_group(device, global_rank, world_size)
    inference_model, datamodule, _dir_exp, _name_exp = _load_objects(
        ckpt_path=checkpoint_path,
        device=device,
        validation_frequency=args.validation_frequency,
        precision=args.precision,
    )
    extra_args_base = _parse_json(args.extra_args_json)
    if "seed" in extra_args_base:
        LOG.warning("Ignoring seed=%s from --extra-args-json; sampler seeds are controlled by --sampler-seeds.", extra_args_base["seed"])
        extra_args_base = dict(extra_args_base)
        extra_args_base.pop("seed", None)

    manifest_rows: list[dict[str, str | int]] = []
    weather_states_written: list[str] | None = None
    member_ids = list(range(1, len(args.sampler_seeds) + 1))
    sampling_config_json = json.dumps(
        {
            **extra_args_base,
            "seed_members": list(args.sampler_seeds),
        },
        sort_keys=True,
    )

    for hdate in args.hdates:
        rows = grouped[hdate]
        start_idx = rows[0].idx
        n_samples = len(rows)
        LOG.info(
            "Predicting hdate=%s idx=%s n_samples=%s steps=%s sampler_members=%s",
            hdate,
            start_idx,
            n_samples,
            [row.step for row in rows],
            len(args.sampler_seeds),
        )
        x_members: list[np.ndarray] = []
        y_members: list[np.ndarray] = []
        y_pred_members: list[np.ndarray] = []
        x_interp_members: list[np.ndarray] = []
        lon_lres = lat_lres = lon_hres = lat_hres = dates = None

        for member_id, sampler_seed in zip(member_ids, args.sampler_seeds, strict=True):
            LOG.info(
                "Predicting hdate=%s member=%s/%s sampler_seed=%s",
                hdate,
                member_id,
                len(member_ids),
                sampler_seed,
            )
            extra_args = dict(extra_args_base)
            extra_args["seed"] = int(sampler_seed)
            _set_all_rng_seeds(int(sampler_seed))
            (
                x_seed,
                y_seed,
                y_pred_seed,
                lon_lres_seed,
                lat_lres_seed,
                lon_hres_seed,
                lat_hres_seed,
                weather_states,
                dates_seed,
            ) = _predict_from_dataloader(
                inference_model=inference_model,
                datamodule=datamodule,
                device=device,
                idx=start_idx,
                n_samples=n_samples,
                members=[0],
                extra_args=extra_args,
                precision=args.precision,
                model_comm_group=model_comm_group,
                output_weather_state_mode=args.output_weather_state_mode,
                output_weather_states=None,
                split=args.split,
            )
            x_interp_seed = _compute_x_interp_for_export(
                inference_model=inference_model,
                x=x_seed,
                device=device,
                model_comm_group=model_comm_group,
            )

            if weather_states_written is None:
                weather_states_written = list(weather_states)
                lon_lres = lon_lres_seed
                lat_lres = lat_lres_seed
                lon_hres = lon_hres_seed
                lat_hres = lat_hres_seed
                dates = dates_seed
            elif list(weather_states) != weather_states_written:
                raise ValueError(
                    f"Weather states changed across sampler seeds for hdate {hdate}: "
                    f"{weather_states_written} vs {list(weather_states)}"
                )

            x_members.append(x_seed)
            y_members.append(y_seed)
            y_pred_members.append(y_pred_seed)
            x_interp_members.append(x_interp_seed)

        if lon_lres is None or lat_lres is None or lon_hres is None or lat_hres is None or dates is None:
            raise RuntimeError(f"No predictions were produced for hdate {hdate}")

        x = np.concatenate(x_members, axis=1)
        y = np.concatenate(y_members, axis=1)
        y_pred = np.concatenate(y_pred_members, axis=1)
        x_interp = np.concatenate(x_interp_members, axis=1)
        block_ds = build_predictions_dataset(
            x=x,
            y=y,
            y_pred=y_pred,
            lon_lres=lon_lres,
            lat_lres=lat_lres,
            lon_hres=lon_hres,
            lat_hres=lat_hres,
            weather_states=weather_states,
            dates=dates,
            member_ids=member_ids,
            x_interp=x_interp,
            include_member_views=True,
        )
        for sample_idx, row in enumerate(rows):
            sample_ds = block_ds.isel(sample=slice(sample_idx, sample_idx + 1))
            sample_ds = _add_eval_metadata(
                sample_ds,
                hdate=row.hdate,
                refdate=row.refdate,
                step=row.step,
                checkpoint=checkpoint_path,
                extra_args_json=sampling_config_json,
                validation_frequency=args.validation_frequency,
                output_weather_state_mode=args.output_weather_state_mode,
                output_weather_states=weather_states,
                sampler_seeds=args.sampler_seeds,
            )
            out_name = f"predictions_{row.hdate.replace('-', '')}_step{row.step:03d}.nc"
            out_path = out_dir / out_name
            sample_ds.to_netcdf(out_path)
            manifest_rows.append({
                "filename": out_name,
                "hdate": row.hdate,
                "refdate": row.refdate,
                "step_hours": row.step,
                "fake_date": row.fake_date,
                "sample_index": row.idx,
            })
            LOG.info("Wrote %s", out_path)

    manifest_path = out_dir / "predictions_manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "hdate", "refdate", "step_hours", "fake_date", "sample_index"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "checkpoint": checkpoint_path,
        "mapping_zarr": args.mapping_zarr,
        "split": args.split,
        "hdates": args.hdates,
        "steps": args.steps,
        "sampler_seeds": args.sampler_seeds,
        "n_members": len(args.sampler_seeds),
        "n_files": len(manifest_rows),
        "n_predictions_total": len(manifest_rows) * len(args.sampler_seeds),
        "weather_states": weather_states_written,
        "out_dir": str(out_dir),
    }
    (out_dir / "native_adapter_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    LOG.info("Wrote %d prediction files to %s", len(manifest_rows), out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build eval.cli-style predictions_*.nc files from the native regional dataloader path.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mapping-zarr", default=DEFAULT_MAPPING_ZARR)
    parser.add_argument("--hdates", default=",".join(DEFAULT_HDATES))
    parser.add_argument("--steps", default=",".join(str(step) for step in DEFAULT_STEPS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--validation-frequency", default="50h")
    parser.add_argument("--extra-args-json", default=DEFAULT_EXTRA_ARGS_JSON)
    parser.add_argument("--sampler-seeds", default=",".join(str(seed) for seed in DEFAULT_SAMPLER_SEEDS))
    parser.add_argument("--split", choices=["valid", "train"], default="valid")
    parser.add_argument("--output-weather-state-mode", default="all")
    parser.add_argument("--allow-noncanonical-shape", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    args.hdates = _parse_csv_dates(args.hdates)
    args.steps = _parse_csv_steps(args.steps)
    args.sampler_seeds = _parse_csv_ints(args.sampler_seeds, label="sampler seed")
    _validate_request_shape(
        hdates=args.hdates,
        steps=args.steps,
        sampler_seeds=args.sampler_seeds,
        allow_noncanonical_shape=args.allow_noncanonical_shape,
    )
    build_predictions_dir(args)


if __name__ == "__main__":
    main()
