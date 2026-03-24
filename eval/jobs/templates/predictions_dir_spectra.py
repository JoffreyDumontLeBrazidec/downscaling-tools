#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

try:
    from eval.spectra.calibrate_fast_spectra_proxy import (
        cl_from_unstructured as _healpy_cl_from_unstructured,
    )
except ImportError:
    _healpy_cl_from_unstructured = None


SPECTRA_SCORE_WAVENUMBER_MIN_EXCLUSIVE_DEFAULT = 100.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute simple prediction-vs-truth spectra diagnostics from predictions_*.nc files."
    )
    p.add_argument("--predictions-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-label", required=True)
    p.add_argument("--weather-states", default="10u,10v,2t,msl,t_850,z_500")
    p.add_argument("--nside", type=int, default=64)
    p.add_argument("--lmax", type=int, default=319)
    p.add_argument(
        "--spectra-method",
        choices=["auto", "healpy", "fft_proxy"],
        default="auto",
        help=(
            "How to convert unstructured lat/lon fields into spectra. "
            "'auto' prefers healpy when available and otherwise falls back to fft_proxy."
        ),
    )
    p.add_argument(
        "--member-aggregation",
        choices=["per-file-mean", "raw-members"],
        default="per-file-mean",
        help=(
            "How to treat ensemble members. "
            "'per-file-mean' averages member spectra within each prediction file first, "
            "so one curve contributes per case/file. "
            "'raw-members' keeps one curve per member."
        ),
    )
    p.add_argument(
        "--show-individual-curves",
        action="store_true",
        help="Overlay every aggregated case curve in the background instead of plotting only mean curves.",
    )
    p.add_argument(
        "--score-wavenumber-min-exclusive",
        type=float,
        default=SPECTRA_SCORE_WAVENUMBER_MIN_EXCLUSIVE_DEFAULT,
        help=(
            "Only wavenumbers strictly above this threshold contribute to relative_l2_mean_curve. "
            f"Default: {SPECTRA_SCORE_WAVENUMBER_MIN_EXCLUSIVE_DEFAULT:g}."
        ),
    )
    return p.parse_args()


def parse_weather_states(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def valid_prediction_files(pred_dir: Path) -> list[Path]:
    files = sorted(pred_dir.glob("predictions_*.nc"))
    if not files:
        raise FileNotFoundError(f"No predictions_*.nc files found in {pred_dir}")
    return files


def _fft_proxy_cl_from_unstructured(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    values: np.ndarray,
    *,
    lmax: int,
) -> np.ndarray:
    lat = np.asarray(lat_deg, dtype=np.float64).reshape(-1)
    lon = np.mod(np.asarray(lon_deg, dtype=np.float64).reshape(-1), 360.0)
    val = np.asarray(values, dtype=np.float64).reshape(-1)

    nlat = int(min(max(lmax + 1, 256), 2048))
    nlon = int(2 * nlat)

    lat_idx = np.clip(((lat + 90.0) / 180.0 * nlat).astype(np.int64), 0, nlat - 1)
    lon_idx = np.clip((lon / 360.0 * nlon).astype(np.int64), 0, nlon - 1)

    sums = np.zeros((nlat, nlon), dtype=np.float64)
    counts = np.zeros((nlat, nlon), dtype=np.int64)
    np.add.at(sums, (lat_idx, lon_idx), val)
    np.add.at(counts, (lat_idx, lon_idx), 1)

    grid = np.zeros((nlat, nlon), dtype=np.float64)
    valid = counts > 0
    if not np.any(valid):
        raise RuntimeError("No valid grid coverage for FFT spectra proxy.")

    grid[valid] = sums[valid] / counts[valid]
    grid_mean = float(np.mean(grid[valid]))
    grid[valid] = grid[valid] - grid_mean

    lat_centers = -90.0 + (np.arange(nlat, dtype=np.float64) + 0.5) * (180.0 / nlat)
    weights = np.sqrt(np.clip(np.cos(np.deg2rad(lat_centers)), 1e-6, None))[:, None]
    weighted = grid * weights

    spec2 = np.abs(np.fft.rfft2(weighted)) ** 2
    ky = np.fft.fftfreq(nlat)[:, None] * nlat
    kx = np.fft.rfftfreq(nlon)[None, :] * nlon
    kr = np.sqrt(kx**2 + ky**2)

    shell = np.floor(kr).astype(np.int64)
    keep = shell <= lmax
    power_sum = np.bincount(shell[keep].ravel(), weights=spec2[keep].ravel(), minlength=lmax + 1)
    power_count = np.bincount(shell[keep].ravel(), minlength=lmax + 1)
    cl = power_sum / np.maximum(power_count, 1)

    coverage = float(np.mean(valid))
    return cl / max(coverage, 1e-6)


def resolve_spectra_method(method: str):
    if method == "healpy":
        if _healpy_cl_from_unstructured is None:
            raise RuntimeError("Requested spectra-method=healpy but healpy path is unavailable.")
        return _healpy_cl_from_unstructured, "healpy"
    if method == "fft_proxy":
        return lambda lat, lon, val, nside, lmax: _fft_proxy_cl_from_unstructured(lat, lon, val, lmax=lmax), "fft_proxy"
    if _healpy_cl_from_unstructured is not None:
        return _healpy_cl_from_unstructured, "healpy"
    return lambda lat, lon, val, nside, lmax: _fft_proxy_cl_from_unstructured(lat, lon, val, lmax=lmax), "fft_proxy"


def finite_positive_mask(
    arr: np.ndarray,
    *,
    wavenumbers: np.ndarray | None = None,
    score_wavenumber_min_exclusive: float = SPECTRA_SCORE_WAVENUMBER_MIN_EXCLUSIVE_DEFAULT,
) -> np.ndarray:
    ell = np.arange(arr.shape[0], dtype=np.float64) if wavenumbers is None else np.asarray(wavenumbers, dtype=np.float64)
    if ell.shape != arr.shape:
        raise ValueError(f"wavenumber length mismatch: curve={arr.shape[0]} wavenumbers={ell.shape[0]}")
    return np.isfinite(arr) & (arr > 0.0) & np.isfinite(ell) & (ell > score_wavenumber_min_exclusive)


def member_indices(ds: xr.Dataset) -> Iterable[int]:
    if "ensemble_member" not in ds.dims:
        return [0]
    return range(int(ds.sizes["ensemble_member"]))


def mean_curve(curves: list[np.ndarray]) -> np.ndarray:
    return np.nanmean(np.stack(curves, axis=0), axis=0)


def plot_one_state(
    *,
    state: str,
    pred_cls: list[np.ndarray],
    truth_cls: list[np.ndarray],
    out_pdf: Path,
    run_label: str,
    show_individual_curves: bool,
    scope_label: str,
    score_wavenumber_min_exclusive: float,
) -> dict[str, float]:
    pred_mean = mean_curve(pred_cls)
    truth_mean = mean_curve(truth_cls)

    ell = np.arange(pred_mean.shape[0], dtype=np.float64)
    kp = finite_positive_mask(pred_mean, wavenumbers=ell, score_wavenumber_min_exclusive=1.0)
    kt = finite_positive_mask(truth_mean, wavenumbers=ell, score_wavenumber_min_exclusive=1.0)
    score_keep = finite_positive_mask(
        pred_mean,
        wavenumbers=ell,
        score_wavenumber_min_exclusive=score_wavenumber_min_exclusive,
    ) & finite_positive_mask(
        truth_mean,
        wavenumbers=ell,
        score_wavenumber_min_exclusive=score_wavenumber_min_exclusive,
    )

    fig, ax = plt.subplots(figsize=(8.2, 5.2))

    if show_individual_curves:
        for arr in truth_cls:
            keep = finite_positive_mask(arr)
            ax.plot(ell[keep], arr[keep], color="#888888", alpha=0.18, linewidth=0.8)
        for arr in pred_cls:
            keep = finite_positive_mask(arr)
            ax.plot(ell[keep], arr[keep], color="#1f77b4", alpha=0.18, linewidth=0.8)

    ax.plot(ell[kt], truth_mean[kt], color="#333333", linewidth=2.2, label="truth mean")
    ax.plot(ell[kp], pred_mean[kp], color="#1f77b4", linewidth=2.2, label="prediction mean")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Wavenumber l")
    ax.set_ylabel("Power")
    ax.set_title(f"{run_label} | {state} | spectra ({scope_label})")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)

    if not np.any(score_keep):
        rel_l2 = float("nan")
    else:
        rel_l2 = float(np.linalg.norm((pred_mean - truth_mean)[score_keep]) / max(np.linalg.norm(truth_mean[score_keep]), 1e-12))
    return {"relative_l2_mean_curve": rel_l2}


def main() -> None:
    args = parse_args()
    pred_dir = Path(args.predictions_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cl_from_unstructured, spectra_method_used = resolve_spectra_method(args.spectra_method)

    files = valid_prediction_files(pred_dir)
    states = parse_weather_states(args.weather_states)

    # Collect per-state spectra across all prediction files.
    all_pred: dict[str, list[np.ndarray]] = {s: [] for s in states}
    all_truth: dict[str, list[np.ndarray]] = {s: [] for s in states}
    raw_member_curve_counts: dict[str, int] = {s: 0 for s in states}

    for f in files:
        with xr.open_dataset(f) as ds:
            weather_states = set(map(str, ds["weather_state"].values.tolist()))
            lat = ds["lat_hres"].values
            lon = ds["lon_hres"].values
            for state in states:
                if state not in weather_states:
                    continue
                pred_member_curves: list[np.ndarray] = []
                truth_member_curves: list[np.ndarray] = []
                for member_idx in member_indices(ds):
                    pred = (
                        ds["y_pred"]
                        .isel(sample=0, ensemble_member=member_idx)
                        .sel(weather_state=state)
                        .values
                        .astype(np.float64)
                    )
                    truth = (
                        ds["y"]
                        .isel(sample=0, ensemble_member=member_idx)
                        .sel(weather_state=state)
                        .values
                        .astype(np.float64)
                    )
                    pred_member_curves.append(
                        cl_from_unstructured(lat, lon, pred, nside=args.nside, lmax=args.lmax)
                    )
                    truth_member_curves.append(
                        cl_from_unstructured(lat, lon, truth, nside=args.nside, lmax=args.lmax)
                    )
                raw_member_curve_counts[state] += len(pred_member_curves)
                if args.member_aggregation == "raw-members":
                    all_pred[state].extend(pred_member_curves)
                    all_truth[state].extend(truth_member_curves)
                else:
                    all_pred[state].append(mean_curve(pred_member_curves))
                    all_truth[state].append(mean_curve(truth_member_curves))

    summary: dict[str, object] = {
        "run_label": args.run_label,
        "predictions_dir": str(pred_dir),
        "num_files": len(files),
        "nside": args.nside,
        "lmax": args.lmax,
        "score_wavenumber_min_exclusive": args.score_wavenumber_min_exclusive,
        "spectra_method_requested": args.spectra_method,
        "spectra_method_used": spectra_method_used,
        "member_aggregation": args.member_aggregation,
        "weather_states": {},
    }
    scope_label = f"{len(files)}-file aggregate"

    for state in states:
        if not all_pred[state] or not all_truth[state]:
            summary["weather_states"][state] = {"status": "missing"}
            continue
        out_pdf = out_dir / f"spectra_{state}.pdf"
        metrics = plot_one_state(
            state=state,
            pred_cls=all_pred[state],
            truth_cls=all_truth[state],
            out_pdf=out_pdf,
            run_label=args.run_label,
            show_individual_curves=args.show_individual_curves,
            scope_label=scope_label,
            score_wavenumber_min_exclusive=args.score_wavenumber_min_exclusive,
        )
        summary["weather_states"][state] = {
            "status": "ok",
            "n_curves": len(all_pred[state]),
            "n_member_curves": raw_member_curve_counts[state],
            "pdf": str(out_pdf),
            **metrics,
        }

    out_json = out_dir / "spectra_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote spectra summary: {out_json}")


if __name__ == "__main__":
    main()
