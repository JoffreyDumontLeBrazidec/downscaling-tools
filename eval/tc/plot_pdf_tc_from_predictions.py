from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages

from eval.tc.tools.loading_data import DataRetriever

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class EventDomain:
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


IDALIA_DOMAIN = EventDomain(
    name="idalia",
    lat_min=10.0,
    lat_max=40.0,
    lon_min=-100.0,
    lon_max=-70.0,
)

MSLP_BINS = np.arange(980, 1021, 1.0)
WIND_BINS = np.arange(0, 35.01, 1.0)
IDALIA_EXTREME_MSLP_RANGE = (980.0, 990.0)
IDALIA_EXTREME_WIND_MIN = 25.0


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full_like(num, np.nan, dtype=np.float64)
    m = den > 0
    out[m] = num[m] / den[m]
    return out


def _finite_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    return x[np.isfinite(x)]


def _summary_stats(x: np.ndarray) -> dict:
    vals = _finite_1d(x)
    if vals.size == 0:
        return {"n": 0}
    return {
        "n": int(vals.size),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "q01": float(np.quantile(vals, 0.01)),
        "q05": float(np.quantile(vals, 0.05)),
        "q50": float(np.quantile(vals, 0.50)),
        "q95": float(np.quantile(vals, 0.95)),
        "q99": float(np.quantile(vals, 0.99)),
    }


def _distribution_metrics(hist_ref: np.ndarray, hist_other: np.ndarray, bin_width: float) -> dict:
    pref = np.asarray(hist_ref, dtype=np.float64) * bin_width
    poth = np.asarray(hist_other, dtype=np.float64) * bin_width
    sref = float(np.sum(pref))
    soth = float(np.sum(poth))
    if sref <= 0.0 or soth <= 0.0:
        return {
            "l1_mass": math.nan,
            "total_variation": math.nan,
            "rmse_density": math.nan,
            "max_abs_density_diff": math.nan,
            "ks_hist": math.nan,
            "kl_ref_to_other": math.nan,
            "kl_other_to_ref": math.nan,
            "js_divergence": math.nan,
        }
    pref /= sref
    poth /= soth
    diff = pref - poth
    l1 = float(np.sum(np.abs(diff)))
    tv = 0.5 * l1
    max_abs_density_diff = float(np.max(np.abs(hist_ref - hist_other)))
    rmse_density = float(np.sqrt(np.mean((hist_ref - hist_other) ** 2)))
    cdf_ref = np.cumsum(pref)
    cdf_oth = np.cumsum(poth)
    ks_hist = float(np.max(np.abs(cdf_ref - cdf_oth)))
    eps = 1e-12
    kl_ref_to_other = float(np.sum(pref * np.log((pref + eps) / (poth + eps))))
    kl_other_to_ref = float(np.sum(poth * np.log((poth + eps) / (pref + eps))))
    m = 0.5 * (pref + poth)
    js = 0.5 * (
        np.sum(pref * np.log((pref + eps) / (m + eps)))
        + np.sum(poth * np.log((poth + eps) / (m + eps)))
    )
    return {
        "l1_mass": l1,
        "total_variation": tv,
        "rmse_density": rmse_density,
        "max_abs_density_diff": max_abs_density_diff,
        "ks_hist": ks_hist,
        "kl_ref_to_other": kl_ref_to_other,
        "kl_other_to_ref": kl_other_to_ref,
        "js_divergence": float(js),
    }


def _ratio_metrics(hist_ref: np.ndarray, hist_other: np.ndarray) -> dict:
    ratio = _safe_ratio(hist_other, hist_ref)
    valid = np.isfinite(ratio)
    if not np.any(valid):
        return {
            "valid_bins": 0,
            "ratio_mean": math.nan,
            "ratio_std": math.nan,
            "ratio_min": math.nan,
            "ratio_max": math.nan,
            "ratio_mae_to_1": math.nan,
            "ratio_max_abs_dev_from_1": math.nan,
        }
    r = ratio[valid]
    return {
        "valid_bins": int(r.size),
        "ratio_mean": float(np.mean(r)),
        "ratio_std": float(np.std(r)),
        "ratio_min": float(np.min(r)),
        "ratio_max": float(np.max(r)),
        "ratio_mae_to_1": float(np.mean(np.abs(r - 1.0))),
        "ratio_max_abs_dev_from_1": float(np.max(np.abs(r - 1.0))),
    }


def _extreme_tail_table(
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    mslp_range: tuple[float, float],
    wind_gt: float,
) -> dict:
    rows: list[dict[str, object]] = []
    pmin, pmax = mslp_range
    for exp, (msl_arr, wind_arr) in series.items():
        msl = _finite_1d(msl_arr)
        wind = _finite_1d(wind_arr)
        msl_hit = (msl >= pmin) & (msl <= pmax)
        wind_hit = wind > wind_gt
        rows.append(
            {
                "exp": exp,
                "mslp_980_990_count": int(np.sum(msl_hit)),
                "mslp_980_990_fraction": float(np.mean(msl_hit)) if msl.size else math.nan,
                "wind_gt_25_count": int(np.sum(wind_hit)),
                "wind_gt_25_fraction": float(np.mean(wind_hit)) if wind.size else math.nan,
                "n_msl": int(msl.size),
                "n_wind": int(wind.size),
            }
        )

    m_vals = [float(r["mslp_980_990_fraction"]) for r in rows if np.isfinite(r["mslp_980_990_fraction"])]
    w_vals = [float(r["wind_gt_25_fraction"]) for r in rows if np.isfinite(r["wind_gt_25_fraction"])]
    m_min, m_max = (min(m_vals), max(m_vals)) if m_vals else (math.nan, math.nan)
    w_min, w_max = (min(w_vals), max(w_vals)) if w_vals else (math.nan, math.nan)

    for r in rows:
        mf = float(r["mslp_980_990_fraction"])
        wf = float(r["wind_gt_25_fraction"])
        m_norm = (mf - m_min) / (m_max - m_min) if np.isfinite(mf) and m_max > m_min else 0.0
        w_norm = (wf - w_min) / (w_max - w_min) if np.isfinite(wf) and w_max > w_min else 0.0
        r["extreme_score"] = 0.5 * m_norm + 0.5 * w_norm

    rows.sort(key=lambda x: float(x["extreme_score"]), reverse=True)
    return {
        "thresholds": {
            "mslp_hpa_range": [pmin, pmax],
            "wind_ms_gt": wind_gt,
        },
        "rows": rows,
    }


def _step_to_index(step: int) -> int:
    # In DataRetriever output, OPER is trimmed to +24h..+120h => indices 0..4.
    idx = (int(step) // 24) - 1
    if idx < 0 or idx > 4:
        raise ValueError(f"Unsupported step={step}; expected one of 24,48,72,96,120")
    return idx


def _normalize_lon(lon: np.ndarray) -> np.ndarray:
    # Convert [0, 360) to [-180, 180) if needed.
    return ((lon + 180.0) % 360.0) - 180.0


def _discover_prediction_files(pred_dir: Path) -> list[tuple[Path, int, int]]:
    files = sorted(pred_dir.glob("predictions_*.nc"))
    out: list[tuple[Path, int, int]] = []
    rx = re.compile(r"predictions_(\d{8})_step(\d{3})\.nc$")
    for f in files:
        m = rx.match(f.name)
        if not m:
            continue
        ymd = int(m.group(1))
        step = int(m.group(2))
        out.append((f, ymd, step))
    return out


def _extract_pred_vectors(
    pred_files: list[tuple[Path, int, int]],
    domain: EventDomain,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int], str, str]:
    msl_vals: list[np.ndarray] = []
    wind_vals: list[np.ndarray] = []
    days: list[int] = []
    steps: list[int] = []
    year: str | None = None
    month: str | None = None

    for path, ymd, step in pred_files:
        ds = xr.open_dataset(path)
        try:
            ws = ds["weather_state"].values.tolist()
            i_msl = ws.index("msl")
            i_u10 = ws.index("10u")
            i_v10 = ws.index("10v")

            lon = _normalize_lon(ds["lon_hres"].values)
            lat = ds["lat_hres"].values
            mask = (
                (lat >= domain.lat_min)
                & (lat <= domain.lat_max)
                & (lon >= domain.lon_min)
                & (lon <= domain.lon_max)
            )
            if not np.any(mask):
                LOG.warning("No points in domain for %s", path)
                continue

            y_pred = ds["y_pred"].values  # [sample, member, point, weather_state]
            # sample dim is 1 for this workflow.
            arr = y_pred[0]  # [member, point, weather_state]

            msl = arr[:, mask, i_msl] / 100.0
            u10 = arr[:, mask, i_u10]
            v10 = arr[:, mask, i_v10]
            wind = np.sqrt(u10 * u10 + v10 * v10)

            msl_vals.append(msl.reshape(-1))
            wind_vals.append(wind.reshape(-1))
        finally:
            ds.close()

        ymd_s = f"{ymd:08d}"
        year = ymd_s[:4] if year is None else year
        month = ymd_s[4:6] if month is None else month
        days.append(int(ymd_s[6:8]))
        steps.append(step)

    if not msl_vals or not wind_vals:
        raise RuntimeError("No prediction values extracted from input files.")
    if year is None or month is None:
        raise RuntimeError("Could not infer year/month from prediction files.")

    return (
        np.concatenate(msl_vals),
        np.concatenate(wind_vals),
        sorted(set(days)),
        sorted(set(steps)),
        year,
        month,
    )


def _extract_reference_vectors(
    *,
    base_dir: str,
    domain: EventDomain,
    year: str,
    month: str,
    days: list[int],
    steps: list[int],
    extra_reference_expids: list[str] | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    step_indices = [_step_to_index(s) for s in steps]
    msl_oper_all: list[np.ndarray] = []
    msl_enfo_all: list[np.ndarray] = []
    msl_eefo_all: list[np.ndarray] = []
    msl_ip6y_all: list[np.ndarray] = []
    msl_extra_all: dict[str, list[np.ndarray]] = {}
    wind_oper_all: list[np.ndarray] = []
    wind_enfo_all: list[np.ndarray] = []
    wind_eefo_all: list[np.ndarray] = []
    wind_ip6y_all: list[np.ndarray] = []
    wind_extra_all: dict[str, list[np.ndarray]] = {}
    extra_reference_expids = extra_reference_expids or []

    for day in days:
        retriever = DataRetriever(
            f"{base_dir}/{domain.name}",
            [day],
            year,
            month,
            0.25,
            domain.lat_min,
            domain.lat_max,
            domain.lon_min,
            domain.lon_max,
        )
        ml_refs = ["ENFO_O320_ip6y", *extra_reference_expids]
        msl, wind10m = retriever.retrieve_all_data(
            "OPER_O320_0001",
            "ENFO_O320_0001",
            "EEFO_O96_0001",
            ml_refs,
        )

        # Keep first 10 members to match prediction ensemble size.
        msl_oper_all.append(msl["OPER_O320_0001"][:, step_indices, :, :].reshape(-1))
        msl_enfo_all.append(msl["ENFO_O320_0001"][:, :10, step_indices, :, :].reshape(-1))
        msl_eefo_all.append(msl["EEFO_O96_0001"][:, :10, step_indices, :, :].reshape(-1))
        msl_ip6y_all.append(msl["ENFO_O320_ip6y"][:, :10, step_indices, :, :].reshape(-1))
        wind_oper_all.append(wind10m["OPER_O320_0001"][:, step_indices, :, :].reshape(-1))
        wind_enfo_all.append(wind10m["ENFO_O320_0001"][:, :10, step_indices, :, :].reshape(-1))
        wind_eefo_all.append(wind10m["EEFO_O96_0001"][:, :10, step_indices, :, :].reshape(-1))
        wind_ip6y_all.append(wind10m["ENFO_O320_ip6y"][:, :10, step_indices, :, :].reshape(-1))
        for expid in extra_reference_expids:
            msl_extra_all.setdefault(expid, []).append(msl[expid][:, :10, step_indices, :, :].reshape(-1))
            wind_extra_all.setdefault(expid, []).append(wind10m[expid][:, :10, step_indices, :, :].reshape(-1))

    msl_extra = {k: np.concatenate(v) for k, v in msl_extra_all.items()}
    wind_extra = {k: np.concatenate(v) for k, v in wind_extra_all.items()}

    return (
        np.concatenate(msl_oper_all),
        np.concatenate(msl_enfo_all),
        np.concatenate(msl_eefo_all),
        np.concatenate(msl_ip6y_all),
        np.concatenate(wind_oper_all),
        np.concatenate(wind_enfo_all),
        np.concatenate(wind_eefo_all),
        np.concatenate(wind_ip6y_all),
        msl_extra,
        wind_extra,
    )


def run_tc_pdf_from_predictions(
    *,
    predictions_dir: str,
    outdir: str,
    run_label: str,
    out_name: str = "",
    base_tc_dir: str = "/home/ecm5702/hpcperm/data/tc",
    extra_reference_expids: list[str] | None = None,
) -> str:
    pred_dir = Path(predictions_dir).expanduser().resolve()
    out_dir = Path(outdir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_files = _discover_prediction_files(pred_dir)
    if not pred_files:
        raise FileNotFoundError(f"No predictions_*.nc files found in {pred_dir}")

    # Manual_o320r2 predictions are Idalia-only.
    pred_msl, pred_wind, days, steps, year, month = _extract_pred_vectors(
        pred_files,
        IDALIA_DOMAIN,
    )
    (
        msl_oper,
        msl_enfo,
        msl_eefo,
        msl_ip6y,
        wind_oper,
        wind_enfo,
        wind_eefo,
        wind_ip6y,
        msl_extra,
        wind_extra,
    ) = _extract_reference_vectors(
        base_dir=base_tc_dir,
        domain=IDALIA_DOMAIN,
        year=year,
        month=month,
        days=days,
        steps=steps,
        extra_reference_expids=extra_reference_expids or [],
    )

    if not out_name:
        out_name = f"tc_normed_pdfs_{IDALIA_DOMAIN.name}_{run_label}_from_predictions.pdf"
    out_pdf = out_dir / out_name
    out_stats = out_dir / f"{out_pdf.stem}.stats.json"

    msl_mid = (MSLP_BINS[:-1] + MSLP_BINS[1:]) / 2.0
    wind_mid = (WIND_BINS[:-1] + WIND_BINS[1:]) / 2.0

    hist_msl_oper, _ = np.histogram(msl_oper, bins=MSLP_BINS, density=True)
    hist_msl_pred, _ = np.histogram(pred_msl, bins=MSLP_BINS, density=True)
    hist_msl_enfo, _ = np.histogram(msl_enfo, bins=MSLP_BINS, density=True)
    hist_msl_eefo, _ = np.histogram(msl_eefo, bins=MSLP_BINS, density=True)
    hist_msl_ip6y, _ = np.histogram(msl_ip6y, bins=MSLP_BINS, density=True)
    hist_msl_extra: dict[str, np.ndarray] = {}
    for expid, arr in msl_extra.items():
        hist_msl_extra[expid], _ = np.histogram(arr, bins=MSLP_BINS, density=True)

    hist_wind_oper, _ = np.histogram(wind_oper, bins=WIND_BINS, density=True)
    hist_wind_pred, _ = np.histogram(pred_wind, bins=WIND_BINS, density=True)
    hist_wind_enfo, _ = np.histogram(wind_enfo, bins=WIND_BINS, density=True)
    hist_wind_eefo, _ = np.histogram(wind_eefo, bins=WIND_BINS, density=True)
    hist_wind_ip6y, _ = np.histogram(wind_ip6y, bins=WIND_BINS, density=True)
    hist_wind_extra: dict[str, np.ndarray] = {}
    for expid, arr in wind_extra.items():
        hist_wind_extra[expid], _ = np.histogram(arr, bins=WIND_BINS, density=True)

    stats_payload: dict[str, object] = {
        "run_label": run_label,
        "predictions_dir": str(pred_dir),
        "pdf_file": str(out_pdf),
        "event": IDALIA_DOMAIN.name,
        "year": year,
        "month": month,
        "days": days,
        "steps_hours": steps,
        "variables": {
            "mslp_hpa": {"curves": {}},
            "wind10m_ms": {"curves": {}},
        },
    }

    stats_payload["variables"]["mslp_hpa"]["oper"] = {"summary": _summary_stats(msl_oper)}
    stats_payload["variables"]["mslp_hpa"]["curves"][run_label] = {
        "label": run_label,
        "summary": _summary_stats(pred_msl),
        "vs_oper": {
            **_distribution_metrics(hist_msl_oper, hist_msl_pred, 1.0),
            **_ratio_metrics(hist_msl_oper, hist_msl_pred),
        },
    }
    stats_payload["variables"]["mslp_hpa"]["curves"]["ENFO_O320_0001"] = {
        "label": "enfo_o320",
        "summary": _summary_stats(msl_enfo),
        "vs_oper": {
            **_distribution_metrics(hist_msl_oper, hist_msl_enfo, 1.0),
            **_ratio_metrics(hist_msl_oper, hist_msl_enfo),
        },
    }
    stats_payload["variables"]["mslp_hpa"]["curves"]["EEFO_O96_0001"] = {
        "label": "eefo_o96",
        "summary": _summary_stats(msl_eefo),
        "vs_oper": {
            **_distribution_metrics(hist_msl_oper, hist_msl_eefo, 1.0),
            **_ratio_metrics(hist_msl_oper, hist_msl_eefo),
        },
    }
    stats_payload["variables"]["mslp_hpa"]["curves"]["ENFO_O320_ip6y"] = {
        "label": "ip6y",
        "summary": _summary_stats(msl_ip6y),
        "vs_oper": {
            **_distribution_metrics(hist_msl_oper, hist_msl_ip6y, 1.0),
            **_ratio_metrics(hist_msl_oper, hist_msl_ip6y),
        },
    }
    for expid, arr in msl_extra.items():
        stats_payload["variables"]["mslp_hpa"]["curves"][expid] = {
            "label": expid.replace("ENFO_O320_", ""),
            "summary": _summary_stats(arr),
            "vs_oper": {
                **_distribution_metrics(hist_msl_oper, hist_msl_extra[expid], 1.0),
                **_ratio_metrics(hist_msl_oper, hist_msl_extra[expid]),
            },
        }

    stats_payload["variables"]["wind10m_ms"]["oper"] = {"summary": _summary_stats(wind_oper)}
    stats_payload["variables"]["wind10m_ms"]["curves"][run_label] = {
        "label": run_label,
        "summary": _summary_stats(pred_wind),
        "vs_oper": {
            **_distribution_metrics(hist_wind_oper, hist_wind_pred, 1.0),
            **_ratio_metrics(hist_wind_oper, hist_wind_pred),
        },
    }
    stats_payload["variables"]["wind10m_ms"]["curves"]["ENFO_O320_0001"] = {
        "label": "enfo_o320",
        "summary": _summary_stats(wind_enfo),
        "vs_oper": {
            **_distribution_metrics(hist_wind_oper, hist_wind_enfo, 1.0),
            **_ratio_metrics(hist_wind_oper, hist_wind_enfo),
        },
    }
    stats_payload["variables"]["wind10m_ms"]["curves"]["EEFO_O96_0001"] = {
        "label": "eefo_o96",
        "summary": _summary_stats(wind_eefo),
        "vs_oper": {
            **_distribution_metrics(hist_wind_oper, hist_wind_eefo, 1.0),
            **_ratio_metrics(hist_wind_oper, hist_wind_eefo),
        },
    }
    stats_payload["variables"]["wind10m_ms"]["curves"]["ENFO_O320_ip6y"] = {
        "label": "ip6y",
        "summary": _summary_stats(wind_ip6y),
        "vs_oper": {
            **_distribution_metrics(hist_wind_oper, hist_wind_ip6y, 1.0),
            **_ratio_metrics(hist_wind_oper, hist_wind_ip6y),
        },
    }
    for expid, arr in wind_extra.items():
        stats_payload["variables"]["wind10m_ms"]["curves"][expid] = {
            "label": expid.replace("ENFO_O320_", ""),
            "summary": _summary_stats(arr),
            "vs_oper": {
                **_distribution_metrics(hist_wind_oper, hist_wind_extra[expid], 1.0),
                **_ratio_metrics(hist_wind_oper, hist_wind_extra[expid]),
            },
        }

    extreme_series: dict[str, tuple[np.ndarray, np.ndarray]] = {
        run_label: (pred_msl, pred_wind),
        "ENFO_O320_0001": (msl_enfo, wind_enfo),
        "EEFO_O96_0001": (msl_eefo, wind_eefo),
        "ENFO_O320_ip6y": (msl_ip6y, wind_ip6y),
        "OPER_O320_0001": (msl_oper, wind_oper),
    }
    for expid in sorted(msl_extra.keys()):
        extreme_series[expid] = (msl_extra[expid], wind_extra[expid])
    stats_payload["extreme_tail"] = _extreme_tail_table(
        extreme_series,
        mslp_range=IDALIA_EXTREME_MSLP_RANGE,
        wind_gt=IDALIA_EXTREME_WIND_MIN,
    )

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(msl_mid, _safe_ratio(hist_msl_pred, hist_msl_oper), color="royalblue", lw=3, label=run_label)
    extra_colors = ["magenta", "deepskyblue", "limegreen", "goldenrod"]
    for i, expid in enumerate(sorted(hist_msl_extra.keys())):
        axs[0].plot(
            msl_mid,
            _safe_ratio(hist_msl_extra[expid], hist_msl_oper),
            color=extra_colors[i % len(extra_colors)],
            lw=2,
            ls="-",
            label=expid.replace("ENFO_O320_", ""),
        )
    axs[0].plot(msl_mid, _safe_ratio(hist_msl_enfo, hist_msl_oper), color="black", lw=2, ls="-.", label="enfo_o320")
    axs[0].plot(msl_mid, _safe_ratio(hist_msl_eefo, hist_msl_oper), color="red", lw=2, ls="--", label="eefo_o96")
    axs[0].plot(msl_mid, _safe_ratio(hist_msl_ip6y, hist_msl_oper), color="orange", lw=2, ls=":", label="ip6y")
    axs[0].set_xlabel("MSLP [hPa]")
    axs[0].set_ylabel("PDF ratio to OPER")
    axs[0].set_title("Idalia - MSLP")
    axs[0].set_ylim(0, 4)
    axs[0].legend()
    axs[0].grid(alpha=0.3)

    axs[1].plot(wind_mid, _safe_ratio(hist_wind_pred, hist_wind_oper), color="royalblue", lw=3, label=run_label)
    for i, expid in enumerate(sorted(hist_wind_extra.keys())):
        axs[1].plot(
            wind_mid,
            _safe_ratio(hist_wind_extra[expid], hist_wind_oper),
            color=extra_colors[i % len(extra_colors)],
            lw=2,
            ls="-",
            label=expid.replace("ENFO_O320_", ""),
        )
    axs[1].plot(wind_mid, _safe_ratio(hist_wind_enfo, hist_wind_oper), color="black", lw=2, ls="-.", label="enfo_o320")
    axs[1].plot(wind_mid, _safe_ratio(hist_wind_eefo, hist_wind_oper), color="red", lw=2, ls="--", label="eefo_o96")
    axs[1].plot(wind_mid, _safe_ratio(hist_wind_ip6y, hist_wind_oper), color="orange", lw=2, ls=":", label="ip6y")
    axs[1].set_xlabel("10m wind [m/s]")
    axs[1].set_ylabel("PDF ratio to OPER")
    axs[1].set_title("Idalia - 10m Wind")
    axs[1].set_ylim(0, 2)
    axs[1].legend()
    axs[1].grid(alpha=0.3)

    fig.suptitle(f"TC PDFs from predictions ({run_label}) - dates={days}, steps={steps}")
    fig.tight_layout()

    with PdfPages(out_pdf) as pdf:
        pdf.savefig(fig, dpi=300)
    plt.close(fig)
    with open(out_stats, "w", encoding="utf-8") as f:
        json.dump(stats_payload, f, indent=2, sort_keys=True)

    LOG.info("Saved TC PDF from predictions: %s", out_pdf)
    LOG.info("Saved TC stats JSON: %s", out_stats)
    return str(out_pdf)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot TC PDF ratios directly from predictions_*.nc files.")
    ap.add_argument("--predictions-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--run-label", required=True, help="Legend label for prediction curve, e.g. manual_o320r2")
    ap.add_argument("--out-name", default="")
    ap.add_argument("--base-tc-dir", default="/home/ecm5702/hpcperm/data/tc")
    ap.add_argument(
        "--extra-reference-expids",
        default="",
        help="Comma-separated extra reference expids, e.g. ENFO_O320_j138,ENFO_O320_j24v",
    )
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    extras = [x.strip() for x in args.extra_reference_expids.split(",") if x.strip()]

    out = run_tc_pdf_from_predictions(
        predictions_dir=args.predictions_dir,
        outdir=args.outdir,
        run_label=args.run_label,
        out_name=args.out_name,
        base_tc_dir=args.base_tc_dir,
        extra_reference_expids=extras,
    )
    print(out)


if __name__ == "__main__":
    main()
