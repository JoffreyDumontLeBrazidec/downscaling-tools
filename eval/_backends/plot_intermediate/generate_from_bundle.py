"""Bundle-based intermediate-state NC generator.

Loads a checkpoint, runs sampling against an input bundle (a pre-built input/target
pair on the lane's grid), captures intermediate sampling states, and writes a
minimal NC compatible with `plot_intermediate_region_grid`.

Why a separate script rather than `plot_intermediate.py checkpoint`:
  - The `checkpoint` mode loads input via the model's default DATA_DIR plumbing,
    which works for o96_o320 (78-channel input matches the ckpt) but fails for
    o48_o96 (67-channel input vs 78-channel ckpt expectation).
  - The legacy archived `eval/archive/jobs/generate_intermediate_from_bundle.py`
    uses bundles correctly but then dies at `build_predictions_dataset` because
    that function forces input `x` and output `y_pred` onto the same
    `weather_state` dim — they have different channel counts (e.g. 78 vs 67).
  - This script writes only what the plot needs: `inter_state`, `y_pred`, `y`,
    `lat_hres`, `lon_hres`, `sampling_step` — no input field, so no dim conflict.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from eval._backends.plot_intermediate.plot_intermediate import _predict_with_intermediates_single_member
from manual_inference.config import DEFAULT_EXTRA_ARGS_JSON
from manual_inference.input_data_construction.bundle import (
    extract_target_from_bundle,
    load_inputs_from_bundle_numpy,
)
from manual_inference.prediction.predict import (
    _get_parallel_info,
    _init_model_comm_group,
    _load_objects,
    _resolve_ckpt_path,
    _resolve_device,
)


def _parse_steps(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _resolve_checkpoint(ckpt_ref: str, ckpt_root: str) -> tuple[Path, str]:
    candidate = Path(ckpt_ref).expanduser()
    if candidate.exists():
        return candidate.resolve(), str(candidate.resolve())
    resolved = _resolve_ckpt_path(ckpt_ref, ckpt_root)
    return Path(resolved).resolve(), ckpt_ref


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Generate intermediate-state NC from one input bundle (no input field saved).",
    )
    ap.add_argument("--bundle-nc", required=True)
    ap.add_argument("--out-nc", required=True)
    ap.add_argument("--ckpt-ref", required=True)
    ap.add_argument("--ckpt-root", default="/home/ecm5702/scratch/aifs/checkpoint")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-gpus-per-model", type=int, default=0)
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    ap.add_argument("--validation-frequency", default="50h")
    ap.add_argument("--extra-args-json", default=DEFAULT_EXTRA_ARGS_JSON)
    ap.add_argument(
        "--capture-steps",
        default="0,4,8,12,16,20,24,29",
        help="Comma-separated diffusion step ids to retain.",
    )
    ap.add_argument("--include-init-state", action="store_true")
    return ap


def main() -> None:
    args = _build_parser().parse_args()

    bundle_nc = Path(args.bundle_nc).expanduser().resolve()
    if not bundle_nc.exists():
        raise SystemExit(f"Bundle file not found: {bundle_nc}")
    out_nc = Path(args.out_nc).expanduser().resolve()
    out_nc.parent.mkdir(parents=True, exist_ok=True)

    global_rank, local_rank, world_size = _get_parallel_info()
    num_gpus_per_model = int(args.num_gpus_per_model) if int(args.num_gpus_per_model) > 0 else int(world_size)
    requested_device = args.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        requested_device = "cpu"
    device = _resolve_device(requested_device, local_rank)
    if str(device).startswith("cuda"):
        torch.cuda.set_device(int(str(device).split(":")[1]))

    if num_gpus_per_model > 1 and world_size != num_gpus_per_model:
        raise SystemExit(
            f"Expected world_size={num_gpus_per_model} for model-parallel inference, got {world_size}."
        )

    model_comm_group = _init_model_comm_group(device, global_rank, world_size)
    ckpt_path, ckpt_label = _resolve_checkpoint(args.ckpt_ref, args.ckpt_root)

    interface, datamodule, _, _ = _load_objects(
        ckpt_path=ckpt_path,
        device=device,
        validation_frequency=args.validation_frequency,
        precision=args.precision,
        num_gpus_per_model_override=num_gpus_per_model,
    )

    output_weather_states = list(datamodule.data_indices.model.output.name_to_index.keys())
    x_lres_np, x_hres_np, lon_lres, lat_lres, lon_hres, lat_hres = load_inputs_from_bundle_numpy(
        bundle_nc,
        datamodule.data_indices.data.input[0].name_to_index,
        datamodule.data_indices.data.input[1].name_to_index,
    )

    x_lres = torch.from_numpy(x_lres_np).to(device)[None, None, None, ...]
    x_hres = torch.from_numpy(x_hres_np).to(device)[None, None, None, ...]
    extra_args = json.loads(args.extra_args_json) if args.extra_args_json else {}

    with torch.inference_mode():
        final_pred, inter_steps, sampling_step_ids, _x_interp_state = _predict_with_intermediates_single_member(
            interface=interface,
            x_in_lres=x_lres,
            x_in_hres=x_hres,
            extra_args=extra_args,
            model_comm_group=model_comm_group,
            capture_steps=_parse_steps(args.capture_steps),
            include_init_state=bool(args.include_init_state),
        )

    target_np, _found_target_channels = extract_target_from_bundle(bundle_nc, output_weather_states)
    if target_np is None:
        raise SystemExit(f"Could not extract target from bundle {bundle_nc}.")

    if global_rank != 0:
        return

    # Build a minimal dataset on the OUTPUT weather_state axis only.
    # No input field — that's the bug we're avoiding.
    ds = xr.Dataset(
        data_vars={
            "y_pred": (
                ["sample", "ensemble_member", "grid_point_hres", "weather_state"],
                final_pred[None, None, :, :].astype("float32"),
            ),
            "y": (
                ["sample", "ensemble_member", "grid_point_hres", "weather_state"],
                target_np[None, None, :, :].astype("float32"),
            ),
            "inter_state": (
                ["sample", "ensemble_member", "sampling_step", "grid_point_hres", "weather_state"],
                inter_steps[None, None, ...].astype("float32"),
            ),
            "lat_hres": (["grid_point_hres"], np.asarray(lat_hres, dtype="float64")),
            "lon_hres": (["grid_point_hres"], np.asarray(lon_hres, dtype="float64")),
        },
        coords={
            "sample": np.array([0], dtype="int64"),
            "ensemble_member": np.array([0], dtype="int64"),
            "sampling_step": np.asarray(sampling_step_ids, dtype="int64"),
            "weather_state": np.asarray(output_weather_states),
        },
        attrs={
            "intermediate_source": "bundle",
            "bundle_nc": str(bundle_nc),
            "checkpoint_path": str(ckpt_path),
            "checkpoint_label": ckpt_label,
            "sampling_config_json": args.extra_args_json,
        },
    )

    if out_nc.exists():
        out_nc.unlink()
    ds.to_netcdf(out_nc)
    print(f"Saved intermediate dataset (bundle path): {out_nc}")


if __name__ == "__main__":
    main()
