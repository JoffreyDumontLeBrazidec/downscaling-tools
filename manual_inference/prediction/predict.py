from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from manual_inference.checkpoints import (
    adapt_config_hpc,
    get_checkpoint,
    get_datamodule,
    instantiate_config,
)
from manual_inference.input_data_construction.bundle import fill_inputs_from_bundle
from manual_inference.prediction.dataset import build_predictions_dataset
from manual_inference.prediction.utils import extract_filtered_input_from_output


def _split_ckpt_path(path: str) -> tuple[str, str, str, str]:
    ckpt_path = os.path.abspath(os.path.expanduser(path))
    name_ckpt = os.path.basename(ckpt_path)
    name_exp = os.path.basename(os.path.dirname(ckpt_path))
    dir_exp = os.path.dirname(os.path.dirname(ckpt_path))
    return ckpt_path, dir_exp, name_exp, name_ckpt


def _resolve_ckpt_path(name_ckpt: str, ckpt_root: str) -> str:
    raw = os.path.expanduser(name_ckpt)
    if os.path.isabs(raw):
        return raw
    root = os.path.expanduser(ckpt_root)
    if raw.endswith(".ckpt"):
        return os.path.join(root, raw)
    return os.path.join(root, raw, "last.ckpt")


def _get_template_batch(datamodule, batch_index: int, device: str):
    val_loader = datamodule.val_dataloader()
    for idx, sample in enumerate(val_loader):
        if idx == batch_index:
            batch = [x.to(device) for x in sample]
            return batch[0], batch[1], batch[2]
    raise RuntimeError(f"Could not find validation batch index {batch_index}")




def _load_objects(
    *,
    ckpt_path: str,
    device: str,
    validation_frequency: str,
):
    ckpt_path, dir_exp, name_exp, name_ckpt = _split_ckpt_path(ckpt_path)
    checkpoint, config_checkpoint = get_checkpoint(dir_exp, name_exp, name_ckpt)
    config_for_datamodule = instantiate_config()
    config_checkpoint = adapt_config_hpc(config_checkpoint, config_for_datamodule)
    config_for_datamodule.dataloader.validation.frequency = validation_frequency
    if hasattr(config_for_datamodule.dataloader.validation, "num_workers"):
        config_for_datamodule.dataloader.validation.num_workers = 0

    inference_model = torch.load(
        os.path.join(dir_exp, name_exp, "inference-" + name_ckpt),
        map_location=torch.device(device),
        weights_only=False,
    ).to(device)
    graph_data = inference_model.graph_data
    datamodule = get_datamodule(config_for_datamodule, graph_data)
    return inference_model, datamodule, dir_exp, name_exp


def _parse_members(value: str, max_members: int) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(max_members))
    members = [int(v.strip()) for v in value.split(",") if v.strip()]
    invalid = [m for m in members if m < 0 or m >= max_members]
    if invalid:
        raise ValueError(
            f"Requested member(s) {invalid} out of range for available members [0, {max_members - 1}]."
        )
    return members


def _predict_from_dataloader(
    *,
    inference_model,
    datamodule,
    device: str,
    idx: int,
    n_samples: int,
    members: Sequence[int],
    extra_args: dict,
):
    data = datamodule.ds_valid.data
    x_in = np.asarray(data[idx : idx + n_samples][0])  # [dates, vars, ens, grid]
    x_in_hres = np.asarray(data[idx : idx + n_samples][1])
    y = np.asarray(data[idx : idx + n_samples][2])

    x_in = np.transpose(x_in, (0, 2, 3, 1))  # [sample, ens, grid, vars]
    x_in_hres = np.transpose(x_in_hres, (0, 2, 3, 1))
    y = np.transpose(y, (0, 2, 3, 1))

    name_to_idx_in = datamodule.data_indices.data.input[0].name_to_index
    name_to_idx_out = datamodule.data_indices.model.output.name_to_index

    x_in, _ = extract_filtered_input_from_output(
        x_in, name_to_idx_in, name_to_idx_out
    )

    lon_lres = np.asarray(data.longitudes[0])
    lat_lres = np.asarray(data.latitudes[0])
    lon_hres = np.asarray(data.longitudes[2])
    lat_hres = np.asarray(data.latitudes[2])
    dates = np.asarray(data.dates[idx : idx + n_samples])
    weather_states = list(name_to_idx_out.keys())

    y_pred = np.zeros(
        (n_samples, len(members), lon_hres.shape[0], len(weather_states)),
        dtype=np.float32,
    )

    for i_sample in range(n_samples):
        for j, m in enumerate(members):
            x_l = torch.from_numpy(x_in[i_sample, m]).to(device)
            x_h = torch.from_numpy(x_in_hres[i_sample, m]).to(device)
            x_l = x_l[None, None, None, ...]
            x_h = x_h[None, None, None, ...]
            with torch.inference_mode():
                pred = inference_model.predict_step(
                    x_l,
                    x_h,
                    extra_args=extra_args,
                )
            y_pred[i_sample, j] = (
                pred[0, 0, 0].detach().cpu().numpy().astype(np.float32)
            )

    if not members:
        raise ValueError("No members selected. Pass at least one member id.")
    x_out = x_in[:, members, :, :]
    y_out = y[:, members, :, :]
    return (
        x_out,
        y_out,
        y_pred,
        lon_lres,
        lat_lres,
        lon_hres,
        lat_hres,
        weather_states,
        dates,
    )


def _predict_from_bundle(
    *,
    inference_model,
    datamodule,
    device: str,
    bundle_nc: str,
    batch_index: int,
    extra_args: dict,
):
    x_template, x_hres_template, _ = _get_template_batch(
        datamodule, batch_index, device
    )
    x_in = x_template.clone()
    x_in_hres = x_hres_template.clone()
    name_to_idx_lres = datamodule.data_indices.data.input[0].name_to_index
    name_to_idx_hres = datamodule.data_indices.data.input[1].name_to_index
    fill_inputs_from_bundle(
        bundle_nc,
        x_in,
        x_in_hres,
        name_to_idx_lres,
        name_to_idx_hres,
        device,
    )

    with torch.inference_mode():
        pred = inference_model.predict_step(
            x_in[0:1],
            x_in_hres[0:1],
            extra_args=extra_args,
        )

    x_np = x_in[0, 0, 0].detach().cpu().numpy().astype(np.float32)
    pred_np = pred[0, 0, 0].detach().cpu().numpy().astype(np.float32)

    lon_lres = np.asarray(datamodule.ds_valid.data.longitudes[0])
    lat_lres = np.asarray(datamodule.ds_valid.data.latitudes[0])
    lon_hres = np.asarray(datamodule.ds_valid.data.longitudes[2])
    lat_hres = np.asarray(datamodule.ds_valid.data.latitudes[2])
    weather_states = list(datamodule.data_indices.model.output.name_to_index.keys())
    dates = None

    x_np, _ = extract_filtered_input_from_output(
        x_np, datamodule.data_indices.data.input[0].name_to_index, datamodule.data_indices.model.output.name_to_index
    )

    return (
        x_np[None, ...],
        None,
        pred_np[None, None, ...],
        lon_lres,
        lat_lres,
        lon_hres,
        lat_hres,
        weather_states,
        dates,
    )




def _parse_json(value: str) -> dict:
    if not value:
        return {}
    return json.loads(value)


def main() -> None:
    ckpt_root_default = os.environ.get(
        "AIFS_CKPT_ROOT", "/home/ecm5702/scratch/aifs/checkpoint"
    )
    parser = argparse.ArgumentParser(description="Generate predictions.nc from a checkpoint.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser("from-dataloader", help="Use dataloader inputs.")
    p_dl.add_argument("--name-ckpt", required=True)
    p_dl.add_argument("--ckpt-root", default=ckpt_root_default)
    p_dl.add_argument("--device", default="cuda")
    p_dl.add_argument("--idx", type=int, default=0)
    p_dl.add_argument("--n-samples", type=int, default=1)
    p_dl.add_argument("--members", default="0")
    p_dl.add_argument("--validation-frequency", default="50h")
    p_dl.add_argument("--extra-args-json", default="")
    p_dl.add_argument("--out", default="")

    p_bundle = sub.add_parser("from-bundle", help="Use a prebuilt input bundle NetCDF.")
    p_bundle.add_argument("--name-ckpt", required=True)
    p_bundle.add_argument("--ckpt-root", default=ckpt_root_default)
    p_bundle.add_argument("--device", default="cuda")
    p_bundle.add_argument("--bundle-nc", required=True)
    p_bundle.add_argument("--batch-index", type=int, default=0)
    p_bundle.add_argument("--validation-frequency", default="50h")
    p_bundle.add_argument("--extra-args-json", default="")
    p_bundle.add_argument("--out", default="")

    p_bundle_build = sub.add_parser("build-bundle", help="Create input bundle from GRIB.")
    p_bundle_build.add_argument("--lres-sfc-grib", required=True)
    p_bundle_build.add_argument("--lres-pl-grib", required=True)
    p_bundle_build.add_argument("--hres-grib", required=True)
    p_bundle_build.add_argument("--out", required=True)
    p_bundle_build.add_argument("--step-hours", type=int, default=None)
    p_bundle_build.add_argument("--member", type=int, default=None)

    args = parser.parse_args()

    if args.cmd == "build-bundle":
        from manual_inference.input_data_construction.bundle import (
            build_input_bundle_from_grib,
        )

        out = build_input_bundle_from_grib(
            lres_sfc_grib=args.lres_sfc_grib,
            lres_pl_grib=args.lres_pl_grib,
            hres_grib=args.hres_grib,
            out_nc=args.out,
            step_hours=args.step_hours,
            member=args.member,
        )
        print(f"Saved bundle: {out}")
        return

    resolved_ckpt = _resolve_ckpt_path(args.name_ckpt, args.ckpt_root)
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    inference_model, datamodule, dir_exp, name_exp = _load_objects(
        ckpt_path=resolved_ckpt,
        device=args.device,
        validation_frequency=args.validation_frequency,
    )

    extra_args = _parse_json(args.extra_args_json)
    if args.cmd == "from-dataloader":
        data = datamodule.ds_valid.data
        max_members = int(np.asarray(data[args.idx : args.idx + 1][0]).shape[2])
        members = _parse_members(args.members, max_members)
        (
            x,
            y,
            y_pred,
            lon_lres,
            lat_lres,
            lon_hres,
            lat_hres,
            weather_states,
            dates,
        ) = _predict_from_dataloader(
            inference_model=inference_model,
            datamodule=datamodule,
            device=args.device,
            idx=args.idx,
            n_samples=args.n_samples,
            members=members,
            extra_args=extra_args,
        )
        member_ids = members
    elif args.cmd == "from-bundle":
        (
            x,
            y,
            y_pred,
            lon_lres,
            lat_lres,
            lon_hres,
            lat_hres,
            weather_states,
            dates,
        ) = _predict_from_bundle(
            inference_model=inference_model,
            datamodule=datamodule,
            device=args.device,
            bundle_nc=args.bundle_nc,
            batch_index=args.batch_index,
            extra_args=extra_args,
        )
        member_ids = [0]
    else:
        raise SystemExit("Unknown command")

    ds = build_predictions_dataset(
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
    )

    out_path = args.out
    if not out_path:
        out_path = os.path.join(
            "/home/ecm5702/hpcperm/experiments", name_exp, "predictions.nc"
        )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    ds.to_netcdf(out_path)
    print(f"Saved predictions: {out_path}")


if __name__ == "__main__":
    main()
