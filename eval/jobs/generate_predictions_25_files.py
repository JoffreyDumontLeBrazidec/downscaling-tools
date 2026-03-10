#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from manual_inference.prediction.dataset import build_predictions_dataset
from manual_inference.prediction.predict import (
    DEFAULT_EXTRA_ARGS_JSON,
    _get_parallel_info,
    _init_model_comm_group,
    _load_objects,
    _predict_from_bundle,
    _resolve_ckpt_path,
    _resolve_device,
)

BUNDLE_RE = re.compile(
    r"date(?P<date>\d{8})_time\d{4}_mem(?P<member>\d+)_step(?P<step>\d{3})h_input_bundle\.nc$"
)


@dataclass(frozen=True)
class BundleKey:
    date: str
    step: int
    member: int


def parse_bundle_key(path: Path) -> BundleKey | None:
    m = BUNDLE_RE.search(path.name)
    if not m:
        return None
    return BundleKey(date=m.group("date"), step=int(m.group("step")), member=int(m.group("member")))


def discover_bundles(input_root: Path) -> dict[BundleKey, Path]:
    out: dict[BundleKey, Path] = {}
    for p in sorted(input_root.rglob("*_input_bundle.nc")):
        key = parse_bundle_key(p)
        if key is None:
            continue
        if key not in out or p.stat().st_mtime >= out[key].stat().st_mtime:
            out[key] = p
    return out


def parse_int_list(raw: str) -> list[int]:
    return sorted({int(x.strip()) for x in raw.split(",") if x.strip()})


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate 25 predictions_YYYYMMDD_stepXXX.nc files from bundle inputs.")
    ap.add_argument("--input-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ckpt-id", default="")
    ap.add_argument(
        "--name-ckpt",
        default="",
        help="Explicit checkpoint path under --ckpt-root or an absolute .ckpt path. Overrides --ckpt-id.",
    )
    ap.add_argument("--ckpt-root", default="/home/ecm5702/scratch/aifs/checkpoint")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-gpus-per-model", type=int, default=1)
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    ap.add_argument("--validation-frequency", default="50h")
    ap.add_argument("--batch-index", type=int, default=0)
    ap.add_argument("--members", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--steps", default="24,48,72,96,120")
    ap.add_argument("--dates", default="20230826,20230827,20230828,20230829,20230830")
    ap.add_argument(
        "--extra-args-json",
        default=DEFAULT_EXTRA_ARGS_JSON,
    )
    ap.add_argument("--manifest", default="")
    ap.add_argument(
        "--allow-missing-target",
        action="store_true",
        help="Deprecated in new stack: y truth is required for predictions output.",
    )
    ap.add_argument(
        "--allow-existing-out-dir",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    ap.add_argument(
        "--allow-overwrite-existing-files",
        action="store_true",
        help="Allow overwriting existing predictions_*.nc files.",
    )
    args = ap.parse_args()

    if args.allow_missing_target:
        raise SystemExit(
            "--allow-missing-target is no longer supported in the new stack. "
            "Bundles must include target_hres_* so y is always present."
        )

    input_root = Path(args.input_root)
    out_dir = Path(args.out_dir)

    global_rank, local_rank, world_size = _get_parallel_info()
    dir_check_error = None
    if global_rank == 0:
        try:
            if out_dir.exists():
                if not args.allow_existing_out_dir and any(out_dir.iterdir()):
                    raise SystemExit(
                        f"Output directory already exists and is not empty: {out_dir}. "
                        "Use a fresh run folder or pass --allow-existing-out-dir explicitly."
                    )
            else:
                out_dir.mkdir(parents=True, exist_ok=False)
        except BaseException as exc:  # propagate rank-0 startup failure cleanly
            dir_check_error = exc

    if world_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
        sync_device_ids = [local_rank] if args.device == "cuda" and torch.cuda.is_available() else None
        torch.distributed.barrier(device_ids=sync_device_ids)

    if dir_check_error is not None:
        raise dir_check_error

    members = parse_int_list(args.members)
    steps = parse_int_list(args.steps)
    dates = [d.strip() for d in args.dates.split(",") if d.strip()]

    bundle_map = discover_bundles(input_root)
    if not bundle_map:
        raise SystemExit(f"No bundle files found in {input_root}")

    required = [BundleKey(d, s, m) for d in dates for s in steps for m in members]
    missing = [k for k in required if k not in bundle_map]
    if missing:
        head = ", ".join(f"{m.date}/step{m.step}/mem{m.member}" for m in missing[:15])
        raise SystemExit(f"Missing {len(missing)} required bundle(s). First missing: {head}")

    if args.name_ckpt:
        ckpt_path = _resolve_ckpt_path(args.name_ckpt, args.ckpt_root)
    elif args.ckpt_id:
        ckpt_path = os.path.join(args.ckpt_root, args.ckpt_id, "last.ckpt")
    else:
        raise SystemExit("Pass either --name-ckpt or --ckpt-id.")
    extra_args = json.loads(args.extra_args_json) if args.extra_args_json else {}

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    device = _resolve_device(args.device, local_rank)
    if str(device).startswith("cuda"):
        torch.cuda.set_device(int(str(device).split(":")[1]))
    if args.num_gpus_per_model > 1 and world_size != args.num_gpus_per_model:
        raise SystemExit(
            f"Expected world_size={args.num_gpus_per_model} for model-parallel inference, got {world_size}. "
            "Launch with matching srun --ntasks."
        )
    model_comm_group = _init_model_comm_group(device, global_rank, world_size)

    inference_model, datamodule, _, _ = _load_objects(
        ckpt_path=ckpt_path,
        device=device,
        validation_frequency=args.validation_frequency,
        precision=args.precision,
        num_gpus_per_model_override=args.num_gpus_per_model,
    )

    manifest_lines = ["date,step,member,bundle_path,predictions_path"]
    total_files = len(dates) * len(steps)
    done_files = 0

    for date in dates:
        for step in steps:
            x_members: list[np.ndarray] = []
            y_members: list[np.ndarray | None] = []
            yp_members: list[np.ndarray] = []
            source_paths: list[str] = []
            members_missing_target: list[int] = []

            lon_lres = lat_lres = lon_hres = lat_hres = weather_states = None
            for m in members:
                key = BundleKey(date, step, m)
                bundle_path = bundle_map[key]
                x, y, y_pred, lon_lres, lat_lres, lon_hres, lat_hres, weather_states, _dates = _predict_from_bundle(
                    inference_model=inference_model,
                    datamodule=datamodule,
                    device=device,
                    bundle_nc=str(bundle_path),
                    batch_index=args.batch_index,
                    member_index=0,
                    extra_args=extra_args,
                    precision=args.precision,
                    model_comm_group=model_comm_group,
                )
                x_members.append(x[0])
                y_members.append(None if y is None else y[0, 0])
                if y is None:
                    members_missing_target.append(m)
                yp_members.append(y_pred[0, 0])
                source_paths.append(str(bundle_path))

            x_stack = np.stack(x_members, axis=0)[None, ...]
            y_stack = None
            if any(y is not None for y in y_members):
                template = next(y for y in y_members if y is not None)
                y_filled = [
                    (y if y is not None else np.full_like(template, np.nan, dtype=np.float32))
                    for y in y_members
                ]
                y_stack = np.stack(y_filled, axis=0)[None, ...]
                if members_missing_target:
                    print(
                        f"WARNING date={date} step={step}: missing target y for members {members_missing_target}; "
                        "filled with NaN to keep y present in predictions file."
                    )
            else:
                raise SystemExit(
                    f"No target truth (y) extracted for date={date} step={step}. "
                    "Rebuild bundles with target_hres_* fields."
                )
            yp_stack = np.stack(yp_members, axis=0)[None, ...]

            ds = build_predictions_dataset(
                x=x_stack,
                y=y_stack,
                y_pred=yp_stack,
                lon_lres=lon_lres,
                lat_lres=lat_lres,
                lon_hres=lon_hres,
                lat_hres=lat_hres,
                weather_states=weather_states,
                dates=None,
                member_ids=members,
            )
            ds = ds.assign_coords(sample=[0])
            ds["init_date"] = xr.DataArray([date], dims=("sample",))
            ds["lead_step_hours"] = xr.DataArray([step], dims=("sample",))
            ds["source_bundle"] = xr.DataArray(np.array([source_paths], dtype=object), dims=("sample", "ensemble_member"))

            ds.attrs["checkpoint_id"] = args.ckpt_id
            ds.attrs["checkpoint_path"] = ckpt_path
            ds.attrs["sampling_config_json"] = args.extra_args_json
            ds.attrs["validation_frequency"] = args.validation_frequency
            ds.attrs["target_members_with_data"] = int(sum(y is not None for y in y_members))
            ds.attrs["target_members_missing_data"] = int(len(members_missing_target))
            ds.attrs["target_missing_member_ids"] = ",".join(str(m) for m in members_missing_target)
            ds.attrs["target_weather_state_count"] = int(len(weather_states))

            out_path = out_dir / f"predictions_{date}_step{step:03d}.nc"
            if global_rank == 0:
                if out_path.exists():
                    if args.allow_overwrite_existing_files:
                        out_path.unlink()
                    else:
                        raise SystemExit(
                            f"Refusing to overwrite existing prediction file: {out_path}. "
                            "Use a fresh output directory or pass --allow-overwrite-existing-files explicitly."
                        )
                ds.to_netcdf(out_path)

                for m, src in zip(members, source_paths):
                    manifest_lines.append(f"{date},{step},{m},{src},{out_path}")

                done_files += 1
                print(f"[{done_files}/{total_files}] wrote {out_path}")

    manifest_path = Path(args.manifest) if args.manifest else (out_dir / "predictions_manifest.csv")
    if global_rank == 0:
        manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
        print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
