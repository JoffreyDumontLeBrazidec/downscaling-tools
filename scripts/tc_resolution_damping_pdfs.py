#!/usr/bin/env python3
"""TC PDF comparison: resolution damping effect.

Produces two PDF figures comparing TC wind/MSLP distributions at different
resolutions to visualize the extreme-value damping at coarse resolution:

  1. Humberto IEKM: O96 vs O2560 (global fields → bbox crop)
  2. Franklin ENFO: O320 vs O1280 (subarea extractions)

Uses the existing TC evaluation infrastructure in eval._backends.tc.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import earthkit.data as ekd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# Ensure eval package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval._backends.tc.data_types import BoundingBox, CurveVectors
from eval._backends.tc.grid import normalize_lon, point_mask
import cmcrameri.cm as cm
import seaborn as sns

from eval._backends.tc.plot_config import TCPlotConfig
from eval._backends.tc.stats import _finite_1d, safe_ratio

sns.set_theme(style="ticks", rc={"font.family": "DejaVu Sans"})

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger(__name__)

# ── Data paths ──────────────────────────────────────────────────────────────

HUMBERTO_DATES = ["20250926", "20250927", "20250928", "20250929", "20250930"]
HUMBERTO_BBOX = BoundingBox(north=45.0, south=15.0, east=-50.0, west=-90.0)

O96_DIR = "/home/ecm5702/hpcperm/data/input_data/o48_o96/humberto_20250926_20250930"
O2560_DIR = "/home/ecm5702/hpcperm/data/input_data/destine_iekm_o2560_targets_humberto_20250926_20250930"

FRANKLIN_DATES = ["20230825", "20230826", "20230827"]
FRANKLIN_BBOX = BoundingBox(north=38.0, south=15.0, east=-58.0, west=-78.0)
FRANKLIN_DIR = "/home/ecm5702/hpcperm/data/tc/franklin"

OUTDIR = Path("/home/ecm5702/scratch/eval/resolution_damping_analysis")

# Steps to use (24h intervals)
TARGET_STEP_HOURS = [24, 48, 72, 96, 120]


# ── Loading helpers ─────────────────────────────────────────────────────────


def _load_native_bbox_curve(
    files: list[str],
    bbox: BoundingBox,
    step_hours: list[int] | None = None,
) -> CurveVectors:
    """Load GRIB files field-by-field, apply bbox mask, return CurveVectors.

    Uses earthkit's field-level API to avoid loading entire datasets into
    memory (critical for O2560 with 26M grid points per field).
    step_hours: if set, select only these forecast steps (in hours).
    """
    target_steps = set(step_hours) if step_hours else None
    msl_chunks: list[np.ndarray] = []
    wind_chunks: list[np.ndarray] = []

    for path in files:
        LOG.info("  Loading %s", Path(path).name)
        source = ekd.from_source("file", path)

        # Build bbox mask from first field's lat/lon
        ll = source[0].to_latlon()
        lat = np.asarray(ll["lat"], dtype=np.float64)
        lon = normalize_lon(np.asarray(ll["lon"], dtype=np.float64))
        mask = point_mask(lon, lat, bbox)
        LOG.info("    bbox filter: %d / %d points inside box", int(mask.sum()), len(mask))

        # Group fields by (step, member): accumulate masked values as lists
        # Key: (step_h, member) → {"msl": arr, "10u": arr, "10v": arr}
        cell_data: dict[tuple[int, int], dict[str, np.ndarray]] = {}
        for field in source:
            md = field.metadata()
            param = md.get("shortName", "")
            step_h = int(md.get("step", 0))
            member = int(md.get("number", 0))
            if param not in ("msl", "10u", "10v"):
                continue
            if target_steps and step_h not in target_steps:
                continue
            vals = np.asarray(field.values, dtype=np.float64)
            cell_data.setdefault((step_h, member), {})[param] = vals[mask]

        for key in sorted(cell_data):
            sd = cell_data[key]
            if not all(k in sd for k in ("msl", "10u", "10v")):
                continue
            msl_chunks.append(sd["msl"] / 100.0)  # Pa → hPa
            wind_chunks.append(np.sqrt(sd["10u"] ** 2 + sd["10v"] ** 2))

    return CurveVectors(
        msl=np.concatenate(msl_chunks),
        wind=np.concatenate(wind_chunks),
    )


# ── Plotting ────────────────────────────────────────────────────────────────


def render_comparison_figure(
    curves: dict[str, CurveVectors],
    *,
    highres_key: str,
    lowres_key: str,
    highres_label: str,
    lowres_label: str,
    plot_config: TCPlotConfig,
    title: str,
    out_path: Path,
    bin_width_msl: float = 4.0,
    bin_width_wind: float = 3.0,
    ratio_bin_width_msl: float = 4.0,
    ratio_bin_width_wind: float = 3.0,
) -> None:
    """Render a 2×2 figure: top = log-scale raw PDFs, bottom = smoothed ratio.

    Both rows use wider-than-default bins so the low-res curve (which may have
    650× fewer grid points) isn't dominated by sampling noise.
    """
    hi_msl = _finite_1d(curves[highres_key].msl)
    hi_wind = _finite_1d(curves[highres_key].wind)
    lo_msl = _finite_1d(curves[lowres_key].msl)
    lo_wind = _finite_1d(curves[lowres_key].wind)

    # Bins for raw PDF
    xbins_msl = np.arange(plot_config.mslp_bin_range[0], plot_config.mslp_bin_range[1], bin_width_msl)
    xbins_wind = np.arange(plot_config.wind_bin_range[0], plot_config.wind_bin_range[1], bin_width_wind)
    mids_msl = (xbins_msl[:-1] + xbins_msl[1:]) / 2.0
    mids_wind = (xbins_wind[:-1] + xbins_wind[1:]) / 2.0

    hist_hi_msl, _ = np.histogram(hi_msl, bins=xbins_msl, density=True)
    hist_lo_msl, _ = np.histogram(lo_msl, bins=xbins_msl, density=True)
    hist_hi_wind, _ = np.histogram(hi_wind, bins=xbins_wind, density=True)
    hist_lo_wind, _ = np.histogram(lo_wind, bins=xbins_wind, density=True)

    # Wider bins for ratio
    rbins_msl = np.arange(plot_config.mslp_bin_range[0], plot_config.mslp_bin_range[1], ratio_bin_width_msl)
    rbins_wind = np.arange(plot_config.wind_bin_range[0], plot_config.wind_bin_range[1], ratio_bin_width_wind)
    rmids_msl = (rbins_msl[:-1] + rbins_msl[1:]) / 2.0
    rmids_wind = (rbins_wind[:-1] + rbins_wind[1:]) / 2.0

    rhist_hi_msl, _ = np.histogram(hi_msl, bins=rbins_msl, density=True)
    rhist_lo_msl, _ = np.histogram(lo_msl, bins=rbins_msl, density=True)
    rhist_hi_wind, _ = np.histogram(hi_wind, bins=rbins_wind, density=True)
    rhist_lo_wind, _ = np.histogram(lo_wind, bins=rbins_wind, density=True)

    COLOR_HI = "#1b7837"   # green for high-res
    COLOR_LO = "#c51b7d"   # magenta for low-res

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))

    # ── Top row: raw PDFs (log y-scale to reveal tails) ──
    for ax, mids, hist_hi, hist_lo, xlabel, var_title in [
        (axs[0, 0], mids_msl, hist_hi_msl, hist_lo_msl,
         "Mean Sea Level Pressure (hPa)",
         "MSLP Distribution (log scale, %g hPa bins)" % bin_width_msl),
        (axs[0, 1], mids_wind, hist_hi_wind, hist_lo_wind,
         "10m wind speed (m/s)",
         "10m Wind Speed Distribution (log scale, %g m/s bins)" % bin_width_wind),
    ]:
        ax.plot(mids, hist_hi, linewidth=2, color=COLOR_HI, label=highres_label)
        ax.plot(mids, hist_lo, linewidth=2, color=COLOR_LO, label=lowres_label, linestyle="--")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Probability Density", fontsize=12)
        ax.set_title(var_title, fontsize=13)
        ax.legend(fontsize=11)

    # Auto-crop x-axis
    all_msl = np.concatenate([hi_msl, lo_msl])
    all_wind = np.concatenate([hi_wind, lo_wind])
    axs[0, 0].set_xlim(max(xbins_msl[0], all_msl.min() - 3), min(xbins_msl[-1], all_msl.max() + 3))
    axs[0, 1].set_xlim(0, min(xbins_wind[-1], all_wind.max() + 2))

    # ── Bottom row: ratio with wider bins ──
    for ax, rmids, rhist_hi, rhist_lo, xlabel, var_title in [
        (axs[1, 0], rmids_msl, rhist_hi_msl, rhist_lo_msl,
         "Mean Sea Level Pressure (hPa)",
         "PDF Ratio: %s / %s  (MSLP, %g hPa bins)" % (lowres_label, highres_label, ratio_bin_width_msl)),
        (axs[1, 1], rmids_wind, rhist_hi_wind, rhist_lo_wind,
         "10m wind speed (m/s)",
         "PDF Ratio: %s / %s  (Wind, %g m/s bins)" % (lowres_label, highres_label, ratio_bin_width_wind)),
    ]:
        ratio = safe_ratio(rhist_lo, rhist_hi)
        ax.plot(rmids, ratio, linewidth=2.5, color=COLOR_LO, marker="o", markersize=4, label=lowres_label)
        ax.axhline(1.0, linestyle="--", linewidth=1.5, color=COLOR_HI, label=highres_label + " (=1)")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("PDF Ratio", fontsize=12)
        ax.set_title(var_title, fontsize=13)
        ax.legend(fontsize=11)
        ax.set_ylim(0, max(plot_config.mslp_ylim[1], plot_config.wind_ylim[1]))

    # Match x-limits between top and bottom
    axs[1, 0].set_xlim(axs[0, 0].get_xlim())
    axs[1, 1].set_xlim(axs[0, 1].get_xlim())

    # Summary annotation
    summary = (
        f"{highres_label}: MSLP [{hi_msl.min():.0f}\u2013{hi_msl.max():.0f}] hPa, "
        f"Wind max {hi_wind.max():.1f} m/s (n={hi_msl.size:,})  |  "
        f"{lowres_label}: MSLP [{lo_msl.min():.0f}\u2013{lo_msl.max():.0f}] hPa, "
        f"Wind max {lo_wind.max():.1f} m/s (n={lo_msl.size:,})"
    )
    fig.text(0.5, 0.01, summary, ha="center", fontsize=10, style="italic", color="gray")

    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])

    with PdfPages(out_path) as pdf:
        pdf.savefig(fig, dpi=300)
    plt.close(fig)
    LOG.info("Saved: %s", out_path)


# ── Main ────────────────────────────────────────────────────────────────────


def run_humberto_iekm() -> None:
    """Humberto IEKM: O96 vs O2560."""
    LOG.info("=== Humberto IEKM: O96 vs O2560 ===")

    o96_files = [
        f"{O96_DIR}/iekm_o96_iekm_date{d}_time0000_step24to120_sfc_y.grib"
        for d in HUMBERTO_DATES
    ]
    o2560_files = [
        f"{O2560_DIR}/iekm_o2560_iekm_date{d}_time0000_step24to120_sfc_y.grib"
        for d in HUMBERTO_DATES
    ]

    LOG.info("Loading IEKM O2560 (high-res reference)...")
    curve_o2560 = _load_native_bbox_curve(o2560_files, HUMBERTO_BBOX)
    LOG.info("  O2560: %d MSL values, %d wind values", curve_o2560.msl.size, curve_o2560.wind.size)

    LOG.info("Loading IEKM O96 (low-res)...")
    curve_o96 = _load_native_bbox_curve(o96_files, HUMBERTO_BBOX)
    LOG.info("  O96: %d MSL values, %d wind values", curve_o96.msl.size, curve_o96.wind.size)

    curves = {"IEKM_O2560": curve_o2560, "IEKM_O96": curve_o96}

    plot_cfg = TCPlotConfig(
        mslp_bin_range=(910, 1025, 1),
        wind_bin_range=(0, 60.01, 1),
        mslp_ylim=(0, 4),
        wind_ylim=(0, 4),
    )

    render_comparison_figure(
        curves,
        highres_key="IEKM_O2560",
        lowres_key="IEKM_O96",
        highres_label="IEKM O2560",
        lowres_label="IEKM O96",
        plot_config=plot_cfg,
        title="Humberto 2025-09 | Resolution Damping: IEKM O96 vs O2560",
        out_path=OUTDIR / "humberto_iekm_resolution_damping.pdf",
    )


def run_franklin_enfo() -> None:
    """Franklin ENFO: O320 vs O1280."""
    LOG.info("=== Franklin ENFO: O320 vs O1280 ===")

    o320_files = [
        f"{FRANKLIN_DIR}/surface_pf_ENFO_O320_0001_{d}.grib"
        for d in FRANKLIN_DATES
    ]
    o1280_files = [
        f"{FRANKLIN_DIR}/surface_pf_ENFO_O1280_0001_{d}.grib"
        for d in FRANKLIN_DATES
    ]

    LOG.info("Loading ENFO O1280 (high-res reference)...")
    curve_o1280 = _load_native_bbox_curve(
        o1280_files, FRANKLIN_BBOX, step_hours=TARGET_STEP_HOURS,
    )
    LOG.info("  O1280: %d MSL values, %d wind values", curve_o1280.msl.size, curve_o1280.wind.size)

    LOG.info("Loading ENFO O320 (low-res)...")
    curve_o320 = _load_native_bbox_curve(o320_files, FRANKLIN_BBOX)
    LOG.info("  O320: %d MSL values, %d wind values", curve_o320.msl.size, curve_o320.wind.size)

    curves = {"ENFO_O1280": curve_o1280, "ENFO_O320": curve_o320}

    plot_cfg = TCPlotConfig(
        mslp_bin_range=(980, 1021, 1),
        wind_bin_range=(0, 35.01, 1),
        mslp_ylim=(0, 4),
        wind_ylim=(0, 4),
    )

    render_comparison_figure(
        curves,
        highres_key="ENFO_O1280",
        lowres_key="ENFO_O320",
        highres_label="ENFO O1280 (10 mbr)",
        lowres_label="ENFO O320 (50 mbr)",
        plot_config=plot_cfg,
        title="Franklin 2023-08 | Resolution Damping: ENFO O320 vs O1280",
        out_path=OUTDIR / "franklin_enfo_resolution_damping.pdf",
    )


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    run_humberto_iekm()
    run_franklin_enfo()
    LOG.info("Done. All outputs in %s", OUTDIR)


if __name__ == "__main__":
    main()
