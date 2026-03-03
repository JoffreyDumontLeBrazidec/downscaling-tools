from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import xarray as xr

from eval.region_plotting.local_plotting import get_region_ds
from manual_inference.prediction.dataset import build_predictions_dataset
from manual_inference.prediction.predict import (
    _get_parallel_info,
    _init_model_comm_group,
    _load_objects,
    _parse_json,
    _resolve_ckpt_path,
    _resolve_device,
)
from manual_inference.prediction.utils import extract_filtered_input_from_output

try:
    from anemoi.models.samplers import diffusion_samplers
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Could not import anemoi diffusion samplers. "
        "Ensure anemoi-core/anemoi-models is available in PYTHONPATH."
    ) from exc

try:
    from anemoi.training.diagnostics.maps import Coastlines
except Exception:  # pragma: no cover
    Coastlines = None


DEFAULT_EXTRA_ARGS_JSON = (
    '{"num_steps":40,"sigma_max":1000.0,"sigma_min":0.03,'
    '"rho":7.0,"sampler":"heun","S_max":1000.0}'
)

NOISE_KEYS = {"schedule_type", "sigma_max", "sigma_min", "rho", "num_steps"}
SAMPLER_KEYS = {"sampler", "S_churn", "S_min", "S_max", "S_noise"}


def select_sampling_steps(total_steps: int, max_panels: int) -> list[int]:
    if total_steps <= 0:
        return []
    n = min(total_steps, max_panels)
    return sorted({int(i) for i in np.linspace(0, total_steps - 1, n)})


def resolve_capture_steps(
    total_steps: int,
    explicit_steps: Sequence[int] | None = None,
    capture_max_steps: int = 0,
) -> list[int]:
    if total_steps <= 0:
        return []
    if explicit_steps:
        valid = sorted({int(s) for s in explicit_steps if 0 <= int(s) < total_steps})
        if not valid:
            raise ValueError("No valid capture steps within [0, total_steps).")
        if (total_steps - 1) not in valid:
            valid.append(total_steps - 1)
        return sorted(valid)
    if capture_max_steps and capture_max_steps > 0:
        return select_sampling_steps(total_steps, capture_max_steps)
    return list(range(total_steps))


def _parse_steps_csv(steps: str) -> list[int]:
    if not steps:
        return []
    return [int(x.strip()) for x in steps.split(",") if x.strip()]


def _split_sampling_args(extra_args: dict) -> tuple[dict, dict]:
    noise = {k: v for k, v in extra_args.items() if k in NOISE_KEYS}
    sampler = {k: v for k, v in extra_args.items() if k in SAMPLER_KEYS}
    return noise, sampler


def _build_sigmas(model, x_in_interp_to_hres: torch.Tensor, noise_scheduler_params: dict) -> torch.Tensor:
    noise_scheduler_config = dict(model.inference_defaults.noise_scheduler)
    noise_scheduler_config.update(noise_scheduler_params)
    schedule_type = noise_scheduler_config.pop("schedule_type")
    if schedule_type not in diffusion_samplers.NOISE_SCHEDULERS:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
    scheduler = diffusion_samplers.NOISE_SCHEDULERS[schedule_type](**noise_scheduler_config)
    return scheduler.get_schedule(x_in_interp_to_hres.device, torch.float64)


def _sample_heun_with_intermediates(
    *,
    model,
    x_in_interp: torch.Tensor,
    x_in_hres: torch.Tensor,
    y: torch.Tensor,
    sigmas: torch.Tensor,
    model_comm_group,
    grid_shard_shapes,
    sampler_params: dict,
    capture_steps: set[int] | None = None,
) -> list[torch.Tensor]:
    cfg = dict(model.inference_defaults.diffusion_sampler)
    cfg.update(sampler_params)
    cfg.pop("sampler", None)

    s_churn = cfg.get("S_churn", 0.0)
    s_min = cfg.get("S_min", 0.0)
    s_max = cfg.get("S_max", float("inf"))
    s_noise = cfg.get("S_noise", 1.0)
    dtype = sigmas.dtype
    eps_prec = cfg.get("eps_prec", 1e-10)

    batch_size = x_in_interp.shape[0]
    ensemble_size = x_in_interp.shape[2]
    num_steps = len(sigmas) - 1

    out_steps: list[torch.Tensor] = []
    for i in range(num_steps):
        sigma_i = sigmas[i]
        sigma_next = sigmas[i + 1]

        apply_churn = s_min <= sigma_i <= s_max and s_churn > 0.0
        if apply_churn:
            gamma = min(
                s_churn / num_steps,
                torch.sqrt(torch.tensor(2.0, dtype=sigma_i.dtype)) - 1,
            )
            sigma_effective = sigma_i + gamma * sigma_i
            epsilon = torch.randn_like(y) * s_noise
            y = y + torch.sqrt(sigma_effective**2 - sigma_i**2) * epsilon
        else:
            sigma_effective = sigma_i

        sigma_eff_exp = sigma_effective.view(1, 1, 1, 1).expand(batch_size, ensemble_size, 1, 1)
        d1 = model.fwd_with_preconditioning(
            x_in_interp,
            x_in_hres,
            y.to(dtype=x_in_interp.dtype),
            sigma_eff_exp.to(x_in_interp.dtype),
            model_comm_group,
            grid_shard_shapes,
        ).to(dtype)

        d = (y - d1) / (sigma_effective + eps_prec)
        y_next = y + (sigma_next - sigma_effective) * d

        if sigma_next > eps_prec:
            sigma_next_exp = sigma_next.view(1, 1, 1, 1).expand(batch_size, ensemble_size, 1, 1)
            d2 = model.fwd_with_preconditioning(
                x_in_interp,
                x_in_hres,
                y_next.to(dtype=x_in_interp.dtype),
                sigma_next_exp.to(dtype=x_in_interp.dtype),
                model_comm_group,
                grid_shard_shapes,
            ).to(dtype)
            d_prime = (y_next - d2) / (sigma_next + eps_prec)
            y = y + (sigma_next - sigma_effective) * (d + d_prime) / 2
        else:
            y = y_next

        if (capture_steps is None) or (i in capture_steps):
            out_steps.append(y.clone())

    return out_steps


def _sample_dpmpp_2m_with_intermediates(
    *,
    model,
    x_in_interp: torch.Tensor,
    x_in_hres: torch.Tensor,
    y: torch.Tensor,
    sigmas: torch.Tensor,
    model_comm_group,
    grid_shard_shapes,
    capture_steps: set[int] | None = None,
) -> list[torch.Tensor]:
    y = y.to(x_in_interp.dtype)
    sigmas = sigmas.to(x_in_interp.dtype)

    batch_size = x_in_interp.shape[0]
    ensemble_size = x_in_interp.shape[2]
    num_steps = len(sigmas) - 1
    old_denoised = None

    out_steps: list[torch.Tensor] = []
    for i in range(num_steps):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        sigma_exp = sigma.view(1, 1, 1, 1).expand(batch_size, ensemble_size, 1, 1)
        denoised = model.fwd_with_preconditioning(
            x_in_interp,
            x_in_hres,
            y,
            sigma_exp,
            model_comm_group,
            grid_shard_shapes,
        )

        if sigma_next == 0:
            y = denoised
            if (capture_steps is None) or (i in capture_steps):
                out_steps.append(y.clone())
            break

        t = -torch.log(sigma + 1e-10)
        t_next = -torch.log(sigma_next + 1e-10) if sigma_next > 0 else float("inf")
        h = t_next - t

        if old_denoised is None:
            x0 = denoised
            y = (sigma_next / sigma) * y - (torch.exp(-h) - 1) * x0
        else:
            h_last = -torch.log(sigmas[i - 1] + 1e-10) - t if i > 0 else h
            r = h_last / h
            x0 = denoised
            x0_last = old_denoised
            coeff1 = 1 + 1 / (2 * r)
            coeff2 = -1 / (2 * r)
            d = coeff1 * x0 + coeff2 * x0_last
            y = (sigma_next / sigma) * y - (torch.exp(-h) - 1) * d

        old_denoised = denoised
        if (capture_steps is None) or (i in capture_steps):
            out_steps.append(y.clone())

    return out_steps


def _predict_with_intermediates_single_member(
    *,
    interface,
    x_in_lres: torch.Tensor,
    x_in_hres: torch.Tensor,
    extra_args: dict,
    model_comm_group,
    capture_steps: Sequence[int] | None = None,
    include_init_state: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model = interface.model

    predict_kwargs = {
        "x_in": x_in_lres,
        "x_in_hres": x_in_hres,
        "pre_processors": interface.pre_processors,
        "post_processors": interface.post_processors,
        "multi_step": interface.multi_step,
        "model_comm_group": model_comm_group,
    }
    if hasattr(interface, "pre_processors_tendencies"):
        predict_kwargs["pre_processors_tendencies"] = interface.pre_processors_tendencies
    if hasattr(interface, "post_processors_tendencies"):
        predict_kwargs["post_processors_tendencies"] = interface.post_processors_tendencies

    before_sampling_data, grid_shard_shapes = model._before_sampling(**predict_kwargs)
    x_interp = before_sampling_data[0]
    x_hres_proc = before_sampling_data[1]
    x_interp_state = x_interp[0, 0, 0].detach().cpu().numpy().astype(np.float32)

    noise_args, sampler_args = _split_sampling_args(extra_args)
    sigmas = _build_sigmas(model, x_interp, noise_args)
    total_steps = len(sigmas) - 1
    capture_steps_resolved = resolve_capture_steps(
        total_steps=total_steps,
        explicit_steps=capture_steps,
        capture_max_steps=0,
    )
    capture_steps_set = set(capture_steps_resolved)

    shape = (x_interp.shape[0], 1, x_interp.shape[2], x_interp.shape[-2], model.num_output_channels)
    y = torch.randn(shape, device=x_interp.device) * sigmas[0]
    y_init = y.clone()

    sampler_name = sampler_args.get("sampler", model.inference_defaults.diffusion_sampler.get("sampler", "heun"))
    if sampler_name == "heun":
        latent_steps = _sample_heun_with_intermediates(
            model=model,
            x_in_interp=x_interp,
            x_in_hres=x_hres_proc,
            y=y,
            sigmas=sigmas,
            model_comm_group=model_comm_group,
            grid_shard_shapes=grid_shard_shapes,
            sampler_params=sampler_args,
            capture_steps=capture_steps_set,
        )
    elif sampler_name == "dpmpp_2m":
        latent_steps = _sample_dpmpp_2m_with_intermediates(
            model=model,
            x_in_interp=x_interp,
            x_in_hres=x_hres_proc,
            y=y,
            sigmas=sigmas,
            model_comm_group=model_comm_group,
            grid_shard_shapes=grid_shard_shapes,
            capture_steps=capture_steps_set,
        )
    else:
        raise ValueError(f"Unsupported sampler for intermediate plotting: {sampler_name}")

    if not latent_steps:
        raise RuntimeError("No intermediate states were produced by the sampler.")

    post_tend = getattr(interface, "post_processors_tendencies", None)
    if include_init_state:
        latent_steps = [y_init] + latent_steps
        capture_steps_resolved = [-1] + capture_steps_resolved

    state_steps: list[np.ndarray] = []
    for latent in latent_steps:
        state = model._after_sampling(
            latent.to(x_interp.dtype),
            interface.post_processors,
            before_sampling_data,
            model_comm_group=model_comm_group,
            grid_shard_shapes=grid_shard_shapes,
            gather_out=True,
            post_processors_tendencies=post_tend,
        )
        state_steps.append(state[0, 0, 0].detach().cpu().numpy().astype(np.float32))

    final_pred = state_steps[-1]
    inter = np.stack(state_steps, axis=0)
    return final_pred, inter, np.asarray(capture_steps_resolved, dtype=np.int32), x_interp_state


def _build_intermediate_dataset_from_checkpoint(args: argparse.Namespace) -> xr.Dataset:
    global_rank, local_rank, world_size = _get_parallel_info()
    requested_device = args.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        requested_device = "cpu"
    device = _resolve_device(requested_device, local_rank)
    if str(device).startswith("cuda"):
        torch.cuda.set_device(int(str(device).split(":")[1]))

    model_comm_group = _init_model_comm_group(device, global_rank, world_size)

    ckpt_path = _resolve_ckpt_path(args.name_ckpt, args.ckpt_root)
    interface, datamodule, _, _ = _load_objects(
        ckpt_path=ckpt_path,
        device=device,
        validation_frequency=args.validation_frequency,
        precision=args.precision,
    )

    data = datamodule.ds_valid.data
    member_id = int(args.member)
    idx = int(args.idx)

    x = np.asarray(data[idx : idx + 1][0])
    x_h = np.asarray(data[idx : idx + 1][1])
    y = np.asarray(data[idx : idx + 1][2])

    max_members = int(x.shape[2])
    if member_id < 0 or member_id >= max_members:
        raise ValueError(f"member={member_id} out of range [0, {max_members - 1}]")

    x = np.transpose(x, (0, 2, 3, 1))
    x_h = np.transpose(x_h, (0, 2, 3, 1))
    y = np.transpose(y, (0, 2, 3, 1))

    name_to_idx_in = datamodule.data_indices.data.input[0].name_to_index
    name_to_idx_out = datamodule.data_indices.model.output.name_to_index
    x, _ = extract_filtered_input_from_output(x, name_to_idx_in, name_to_idx_out)

    x_l = torch.from_numpy(x[0, member_id]).to(device)[None, None, None, ...]
    x_hres = torch.from_numpy(x_h[0, member_id]).to(device)[None, None, None, ...]

    extra_args = _parse_json(args.extra_args_json)
    capture_steps = _parse_steps_csv(args.capture_steps)
    if not capture_steps:
        total_steps = int(extra_args.get("num_steps", 40))
        capture_steps = resolve_capture_steps(
            total_steps=total_steps,
            explicit_steps=None,
            capture_max_steps=int(args.capture_max_steps),
        )
    with torch.inference_mode():
        final_pred, inter_steps, sampling_step_ids, x_interp_state = _predict_with_intermediates_single_member(
            interface=interface,
            x_in_lres=x_l,
            x_in_hres=x_hres,
            extra_args=extra_args,
            model_comm_group=model_comm_group,
            capture_steps=capture_steps,
            include_init_state=bool(args.include_init_state),
        )

    x_out = x[:, [member_id], :, :]
    y_out = y[:, [member_id], :, :]
    y_pred = final_pred[None, None, :, :]

    ds = build_predictions_dataset(
        x=x_out,
        y=y_out,
        y_pred=y_pred,
        lon_lres=np.asarray(data.longitudes[0]),
        lat_lres=np.asarray(data.latitudes[0]),
        lon_hres=np.asarray(data.longitudes[2]),
        lat_hres=np.asarray(data.latitudes[2]),
        weather_states=list(name_to_idx_out.keys()),
        dates=np.asarray(data.dates[idx : idx + 1]),
        member_ids=[member_id],
    )
    ds["inter_state"] = (
        ["sample", "ensemble_member", "sampling_step", "grid_point_hres", "weather_state"],
        inter_steps[None, None, ...],
    )
    if (
        x_interp_state.shape[-1] == len(ds.weather_state)
        and x_interp_state.shape[0] == int(ds.sizes.get("grid_point_hres", -1))
    ):
        ds["x_interp"] = (
            ["sample", "ensemble_member", "grid_point_hres", "weather_state"],
            x_interp_state[None, None, ...],
        )
    ds = ds.assign_coords(sampling_step=sampling_step_ids)
    ds.attrs["intermediate_source"] = "checkpoint"
    ds.attrs["name_ckpt"] = args.name_ckpt
    ds.attrs["idx"] = idx
    ds.attrs["member"] = member_id
    ds.attrs["sampling_config_json"] = args.extra_args_json
    return ds


def _draw_field(
    ax,
    lon: np.ndarray,
    lat: np.ndarray,
    field: np.ndarray,
    title: str,
    vmin: float,
    vmax: float,
):
    scatter = ax.scatter(
        lon,
        lat,
        c=field,
        s=max(1.0, 75_000 / max(1, len(lon))),
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
        rasterized=True,
    )
    if Coastlines is not None:
        Coastlines().plot_continents(ax)
    ax.set_title(title)
    ax.set_xlim(float(np.nanmin(lon)), float(np.nanmax(lon)))
    ax.set_ylim(float(np.nanmin(lat)), float(np.nanmax(lat)))
    ax.set_aspect("auto", adjustable=None)
    ax.set_box_aspect(1)
    return scatter


def _nearest_interp_lres_to_hres(
    *,
    lon_lres: np.ndarray,
    lat_lres: np.ndarray,
    values_lres: np.ndarray,
    lon_hres: np.ndarray,
    lat_hres: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbor interpolation from lres points to hres points."""
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(np.column_stack([lon_lres, lat_lres]))
        _, idx = tree.query(np.column_stack([lon_hres, lat_hres]), k=1)
        return values_lres[idx]
    except Exception:
        out = np.empty_like(lon_hres, dtype=values_lres.dtype)
        for i, (x, y) in enumerate(zip(lon_hres, lat_hres)):
            j = np.argmin((lon_lres - x) ** 2 + (lat_lres - y) ** 2)
            out[i] = values_lres[j]
        return out


def plot_intermediate_trajectory(
    *,
    ds: xr.Dataset,
    weather_state: str,
    sample: int,
    member: int,
    out_path: str,
    region: str = "default",
    max_panels: int = 8,
    sampling_steps: Sequence[int] | None = None,
    independent_color_scales: bool = False,
    show_residuals: bool = True,
) -> Path:
    if "inter_state" not in ds.variables:
        raise ValueError("Dataset does not contain 'inter_state'.")
    if weather_state not in ds.weather_state.values:
        raise ValueError(f"Unknown weather_state={weather_state}")

    ds_region = get_region_ds(ds, region)
    ds_sel = ds_region.sel(sample=sample, ensemble_member=member, weather_state=weather_state)

    available_steps = [int(v) for v in ds_sel.sampling_step.values]
    if sampling_steps:
        steps = [s for s in [int(v) for v in sampling_steps] if s in set(available_steps)]
    else:
        pos = select_sampling_steps(len(available_steps), max_panels)
        steps = [available_steps[i] for i in pos]
    if not steps:
        raise ValueError("No sampling steps selected.")

    inter_data = ds_sel["inter_state"].sel(sampling_step=steps).values
    pred_data = ds_sel["y_pred"].values
    truth_data = ds_sel["y"].values if "y" in ds_sel else None
    base_data = None
    if "x" in ds_sel:
        x_vals = ds_sel["x"].values
        if x_vals.shape == pred_data.shape:
            base_data = x_vals
        elif (
            "grid_point_lres" in ds_sel["x"].dims
            and "lon_lres" in ds_sel
            and "lat_lres" in ds_sel
            and "lon_hres" in ds_sel
            and "lat_hres" in ds_sel
        ):
            base_data = _nearest_interp_lres_to_hres(
                lon_lres=ds_sel["lon_lres"].values,
                lat_lres=ds_sel["lat_lres"].values,
                values_lres=x_vals,
                lon_hres=ds_sel["lon_hres"].values,
                lat_hres=ds_sel["lat_hres"].values,
            )

    panel_titles: list[str] = [f"step={int(s)}" for s in steps]
    panel_fields: list[np.ndarray] = [ds_sel["inter_state"].sel(sampling_step=s).values for s in steps]
    panel_titles.append("y_pred")
    panel_fields.append(pred_data)
    if truth_data is not None:
        panel_titles.append("y")
        panel_fields.append(truth_data)

    all_fields = [f.reshape(-1) for f in panel_fields]
    global_vmin = float(np.nanmin(np.concatenate(all_fields)))
    global_vmax = float(np.nanmax(np.concatenate(all_fields)))

    residual_enabled = bool(show_residuals and (base_data is not None))
    residual_fields = [field - base_data for field in panel_fields] if residual_enabled else []
    if residual_enabled:
        residual_concat = np.concatenate([rf.reshape(-1) for rf in residual_fields])

    ncols = len(panel_fields)
    nrows = 2 if residual_enabled else 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(5.1 * ncols, 5.0 * nrows), squeeze=False)

    lon = ds_sel["lon_hres"].values
    lat = ds_sel["lat_hres"].values

    if independent_color_scales:
        row0_vmin, row0_vmax = global_vmin, global_vmax
    else:
        row0_vmin = float(np.nanpercentile(np.concatenate([f.reshape(-1) for f in panel_fields]), 1))
        row0_vmax = float(np.nanpercentile(np.concatenate([f.reshape(-1) for f in panel_fields]), 99))
        if not np.isfinite(row0_vmin) or not np.isfinite(row0_vmax) or row0_vmin == row0_vmax:
            row0_vmin, row0_vmax = global_vmin, global_vmax

    for i, field in enumerate(panel_fields):
        if independent_color_scales:
            vmin = float(np.nanpercentile(field, 1))
            vmax = float(np.nanpercentile(field, 99))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = row0_vmin, row0_vmax
        else:
            vmin, vmax = row0_vmin, row0_vmax
        mappable = _draw_field(
            axs[0, i],
            lon,
            lat,
            field,
            title=panel_titles[i],
            vmin=vmin,
            vmax=vmax,
        )
        cbar = fig.colorbar(mappable, ax=axs[0, i], orientation="vertical", pad=0.05)
        cbar.outline.set_edgecolor("black")
        cbar.outline.set_linewidth(1.0)
        cbar.ax.tick_params(labelsize=10)

    if residual_enabled:
        if independent_color_scales:
            row1_vmin, row1_vmax = float(np.nanmin(residual_concat)), float(np.nanmax(residual_concat))
        else:
            row1_vmin = float(np.nanpercentile(residual_concat, 1))
            row1_vmax = float(np.nanpercentile(residual_concat, 99))
            if not np.isfinite(row1_vmin) or not np.isfinite(row1_vmax) or row1_vmin == row1_vmax:
                row1_vmin, row1_vmax = float(np.nanmin(residual_concat)), float(np.nanmax(residual_concat))

        for i, rfield in enumerate(residual_fields):
            if independent_color_scales:
                rvmin = float(np.nanpercentile(rfield, 1))
                rvmax = float(np.nanpercentile(rfield, 99))
                if not np.isfinite(rvmin) or not np.isfinite(rvmax) or rvmin == rvmax:
                    rvmin, rvmax = row1_vmin, row1_vmax
            else:
                rvmin, rvmax = row1_vmin, row1_vmax
            rmappable = _draw_field(
                axs[1, i],
                lon,
                lat,
                rfield,
                title=f"{panel_titles[i]} - x_interp",
                vmin=rvmin,
                vmax=rvmax,
            )
            rmappable.set_cmap("viridis")
            cbar = fig.colorbar(rmappable, ax=axs[1, i], orientation="vertical", pad=0.05)
            cbar.outline.set_edgecolor("black")
            cbar.outline.set_linewidth(1.0)
            cbar.ax.tick_params(labelsize=10)

    for r in range(nrows):
        for c in range(ncols):
            ax = axs[r, c]
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d°"))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d°"))
            ax.tick_params(axis="both", which="major", labelsize=10)
            ax.grid(False)
            ax.patch.set_edgecolor("black")
            ax.patch.set_linewidth(2)

    for r in range(nrows):
        axs[r, 0].set_ylabel("Latitude (°)", fontsize=12)
    for c in range(ncols):
        axs[-1, c].set_xlabel("Longitude (°)", fontsize=12)
    fig.suptitle(
        f"Intermediate diffusion states | region={region} | sample={sample} | member={member} | state={weather_state}",
        y=1.02,
    )
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def _load_dataset(args: argparse.Namespace) -> xr.Dataset:
    if args.cmd == "checkpoint":
        ds = _build_intermediate_dataset_from_checkpoint(args)
        if args.save_intermediate_nc:
            out_nc = Path(args.save_intermediate_nc)
            out_nc.parent.mkdir(parents=True, exist_ok=True)
            if out_nc.exists():
                out_nc.unlink()
            ds.to_netcdf(out_nc)
            print(f"Saved intermediate dataset: {out_nc}")
        return ds

    ds = xr.open_dataset(args.predictions_nc)
    return ds


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot intermediate diffusion sampling states from checkpoint or dataset.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ckpt = sub.add_parser("checkpoint", help="Generate and plot intermediates from a checkpoint.")
    p_ckpt.add_argument("--name-ckpt", required=True)
    p_ckpt.add_argument("--ckpt-root", default="/home/ecm5702/scratch/aifs/checkpoint")
    p_ckpt.add_argument("--device", default="cuda")
    p_ckpt.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    p_ckpt.add_argument("--validation-frequency", default="50h")
    p_ckpt.add_argument("--idx", type=int, default=0)
    p_ckpt.add_argument("--member", type=int, default=0)
    p_ckpt.add_argument("--extra-args-json", default=DEFAULT_EXTRA_ARGS_JSON)
    p_ckpt.add_argument("--save-intermediate-nc", default="")
    p_ckpt.add_argument(
        "--capture-steps",
        default="",
        help="Comma-separated diffusion step ids to retain in inter_state.",
    )
    p_ckpt.add_argument(
        "--capture-max-steps",
        type=int,
        default=8,
        help="If --capture-steps is empty, keep at most this many steps (0 keeps all).",
    )
    p_ckpt.add_argument(
        "--include-init-state",
        action="store_true",
        help="Include pre-denoising initialization state as sampling_step=-1.",
    )

    p_ds = sub.add_parser("dataset", help="Plot intermediates from an existing dataset with inter_state.")
    p_ds.add_argument("--predictions-nc", required=True)

    for p in (p_ckpt, p_ds):
        p.add_argument("--sample", type=int, default=0)
        p.add_argument("--weather-state", required=True)
        p.add_argument("--region", default="default")
        p.add_argument("--max-panels", type=int, default=8)
        p.add_argument("--steps", default="", help="Comma-separated sampling_step indices.")
        p.add_argument(
            "--independent-color-scales",
            action="store_true",
            help="Use independent color scales per panel instead of one shared scale.",
        )
        p.add_argument(
            "--no-residuals",
            action="store_true",
            help="Disable residual row (field - x_interp).",
        )
        p.add_argument("--out", required=True, help="Output image path (png/pdf).")

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    ds = _load_dataset(args)
    steps = _parse_steps_csv(args.steps)

    out = plot_intermediate_trajectory(
        ds=ds,
        weather_state=args.weather_state,
        sample=int(args.sample),
        member=int(getattr(args, "member", 0)),
        out_path=args.out,
        region=args.region,
        max_panels=int(args.max_panels),
        sampling_steps=steps,
        independent_color_scales=bool(args.independent_color_scales),
        show_residuals=not bool(args.no_residuals),
    )
    print(f"Saved intermediate trajectory plot: {out}")


if __name__ == "__main__":
    main()
