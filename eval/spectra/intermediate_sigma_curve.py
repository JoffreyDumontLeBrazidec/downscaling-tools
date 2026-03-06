from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from eval.spectra.calibrate_fast_spectra_proxy import cl_from_unstructured


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure spectra discrepancy vs target for each intermediate step."
    )
    parser.add_argument(
        "--intermediate-nc",
        default="/home/ecm5702/perm/eval/j2hh_old_intermediate_idalia_20260304/j2hh_idalia_mem06_step048h__intermediate_cached.nc",
        help="Cached intermediate NetCDF from the TARGET run.",
    )
    parser.add_argument("--param", default="2t", help="Weather state (e.g., 2t, 10u, msl).")
    parser.add_argument(
        "--target-var",
        default="y",
        choices=["y", "y_pred"],
        help="Field used as the spectral target.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Sample index in the NetCDF to analyze.",
    )
    parser.add_argument(
        "--ensemble-index",
        type=int,
        default=0,
        help="Ensemble member in the NetCDF to analyze.",
    )
    parser.add_argument(
        "--nside",
        type=int,
        default=128,
        help="Healpix nside for the spherical transform.",
    )
    parser.add_argument(
        "--lmax",
        type=int,
        default=319,
        help="Maximum spherical harmonic degree.",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/ecm5702/perm/eval/j2hh_old_intermediate_idalia_20260304/spectra_sigma_curve_20260305",
        help="Directory where the curve/table/README will be written.",
    )
    return parser.parse_args()


def build_sigma_schedule(num_steps: int, sigma_max: float, sigma_min: float, rho: float) -> Sequence[float]:
    if num_steps <= 1:
        return [sigma_min]
    sigma_max_r = sigma_max ** (1.0 / rho)
    sigma_min_r = sigma_min ** (1.0 / rho)
    schedule = []
    for step in range(num_steps):
        frac = step / (num_steps - 1)
        sigma_r = sigma_max_r + frac * (sigma_min_r - sigma_max_r)
        schedule.append(float(sigma_r ** rho))
    return schedule


def compute_cl_for_field(
    lat: np.ndarray,
    lon: np.ndarray,
    values: np.ndarray,
    nside: int,
    lmax: int,
) -> np.ndarray:
    clean_values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return cl_from_unstructured(lat, lon, clean_values, nside=nside, lmax=lmax)


def format_fixed_width_table(records: Sequence[dict]) -> str:
    header = "step  sigma    mean_abs_log10  rmse    relative_l2"
    lines = [header]
    for rec in records:
        lines.append(
            f"{rec['step']:>4d}  "
            f"{rec['sigma']:>7.2f}  "
            f"{rec['mean_abs_log10']:>14.6f}  "
            f"{rec['rmse']:>6.4f}  "
            f"{rec['relative_l2']:>11.6f}"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    ds = xr.open_dataset(args.intermediate_nc)

    lat = np.asarray(ds["lat_hres"], dtype=np.float64)
    lon = np.asarray(ds["lon_hres"], dtype=np.float64)
    weather_states = ds["weather_state"].values.astype(str)
    if args.param not in weather_states:
        raise ValueError(f"{args.param} not found in weather_state coordinates.")
    weather_index = int(np.where(weather_states == args.param)[0][0])

    target_da = ds[args.target_var]
    target_values = target_da.isel(
        sample=args.sample_index,
        ensemble_member=args.ensemble_index,
        weather_state=weather_index,
    ).values
    target_cl = compute_cl_for_field(lat, lon, target_values.ravel(), args.nside, args.lmax)

    inter = ds["inter_state"]
    num_steps = int(inter.sizes["sampling_step"])
    sampling_config = json.loads(ds.attrs.get("sampling_config_json", "{}") or "{}")
    schedule = build_sigma_schedule(
        num_steps,
        float(sampling_config.get("sigma_max", 1000.0)),
        float(sampling_config.get("sigma_min", 0.03)),
        float(sampling_config.get("rho", 7.0)),
    )

    records = []
    eps = 1e-30
    for step in range(num_steps):
        values = inter.isel(
            sample=args.sample_index,
            ensemble_member=args.ensemble_index,
            sampling_step=step,
            weather_state=weather_index,
        ).values
        cl = compute_cl_for_field(lat, lon, values.ravel(), args.nside, args.lmax)
        n = min(len(cl), len(target_cl))
        inter_slice = cl[:n]
        target_slice = target_cl[:n]
        ratio = np.maximum(inter_slice, eps) / np.maximum(target_slice, eps)
        mean_abs_log10 = float(np.mean(np.abs(np.log10(ratio))))
        rmse = float(np.sqrt(np.mean((inter_slice - target_slice) ** 2)))
        relative_l2 = float(np.linalg.norm(inter_slice - target_slice) / max(np.linalg.norm(target_slice), eps))

        records.append(
            {
                "step": step,
                "sigma": schedule[step] if step < len(schedule) else 0.0,
                "mean_abs_log10": mean_abs_log10,
                "rmse": rmse,
                "relative_l2": relative_l2,
            }
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table_path = out_dir / "spectra_sigma_curve.txt"
    table_text = format_fixed_width_table(records)
    table_path.write_text(table_text + "\n")

    plot_path = out_dir / "spectra_sigma_curve.pdf"
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    sigmas = [rec["sigma"] for rec in records]
    mean_abs = [rec["mean_abs_log10"] for rec in records]
    ax.plot(sigmas, mean_abs, label="Mean |log10 ratio|", color="#1f77b4", marker="o", linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Sigma")
    ax.set_ylabel("Mean |log10(inter / target)|")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    ax2 = ax.twinx()
    rmse = [rec["rmse"] for rec in records]
    ax2.plot(sigmas, rmse, label="RMSE", color="#d62728", marker="s", linewidth=1.2)
    ax2.set_ylabel("RMSE (power spectrum)")
    ax2.set_yscale("linear")
    ax2.tick_params(axis="y", colors="#d62728")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=240, bbox_inches="tight")
    png_path = out_dir / "spectra_sigma_curve.png"
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "param": args.param,
        "target_var": args.target_var,
        "sigma_schedule": schedule,
        "table": str(table_path),
        "plot": str(plot_path),
        "plot_png": str(png_path),
    }
    metrics_path = out_dir / "spectra_sigma_curve.metadata.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"Wrote table -> {table_path}")
    print(f"Wrote plot -> {plot_path}")
    print(f"Wrote PNG -> {png_path}")
    print(f"Wrote metadata -> {metrics_path}")


if __name__ == "__main__":
    main()
