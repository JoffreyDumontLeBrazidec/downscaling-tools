# tc_pdf_plot.py
"""
Plotting logic for TC PDF comparisons.
PLOT-IDENTICAL to original script.
"""

import os
import warnings
import logging
from typing import Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cmcrameri.cm as cm
import earthkit.data as ekd

from tc_events import TCEvent

logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message=".*decode_timedelta will default to False.*",
    category=FutureWarning,
    module="cfgrib.xarray_plugin",
)

sns.set_theme(style="ticks", rc={"font.family": "DejaVu Sans"})
np.seterr(divide="ignore", invalid="ignore")


def to_xarray_from_files(files):
    logger.debug("Loading %d GRIB files", len(files))
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        logger.error("Missing %d files", len(missing))
        raise FileNotFoundError("Missing GRIB files:\n" + "\n".join(missing))
    return ekd.from_source("file", files).to_xarray(engine="cfgrib")


def plot_event(
    cfg: TCEvent,
    *,
    dir_data_base: str,
    out_path: str,
    include_ml: Optional[Iterable[str]] = None,
    exclude_ml: Optional[Iterable[str]] = None,
) -> plt.Figure:
    """
    Exact refactor of original plotting script.
    """

    logger.info("Starting event: %s", cfg.name)

    dir_data = os.path.join(dir_data_base, cfg.name)
    logger.info("Data directory: %s", dir_data)

    # --- select ML experiments (no reordering)
    if include_ml is not None:
        ml_exps = list(include_ml)
    else:
        ml_exps = list(cfg.list_expid_ml)

    if exclude_ml is not None:
        ml_exps = [e for e in ml_exps if e not in exclude_ml]

    logger.info("ML experiments (%d): %s", len(ml_exps), ml_exps)

    # --- load ML datasets
    all_datasets = {}
    for exp in ml_exps:
        logger.info("Loading ML experiment: %s", exp)
        files = [
            os.path.join(dir_data, f"surface_pf_{exp}_{cfg.year}{cfg.month}{d}.grib")
            for d in cfg.dates
        ]
        ds = to_xarray_from_files(files)
        all_datasets[exp] = ds
        logger.info("Loaded %s, msl shape=%s", exp, ds["msl"].shape)

    # --- load references
    logger.info("Loading ENFO O320: %s", cfg.expid_enfo_o320)
    dataset_enfo_o320 = to_xarray_from_files(
        [
            os.path.join(
                dir_data,
                f"surface_pf_{cfg.expid_enfo_o320}_{cfg.year}{cfg.month}{d}.grib",
            )
            for d in cfg.dates
        ]
    )

    logger.info("Loading EEFO O96: %s", cfg.expid_eefo_o96)
    dataset_eefo_o96 = to_xarray_from_files(
        [
            os.path.join(
                dir_data,
                f"surface_pf_{cfg.expid_eefo_o96}_{cfg.year}{cfg.month}{d}.grib",
            )
            for d in cfg.dates
        ]
    )

    logger.info("Loading OPER analysis: %s", cfg.analysis)
    dataset_oper = to_xarray_from_files(
        [
            os.path.join(dir_data, f"surface_an_{cfg.analysis}_{d}.grib")
            for d in cfg.analysis_dates
        ]
    )

    # --- plotting (UNCHANGED)
    logger.info("Creating plots")

    cmap = cm.batlow
    colors = cmap(np.linspace(0, 1, len(all_datasets)))

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # =====================
    # MSLP
    # =====================
    logger.info("Processing MSLP")

    xbins_msl = np.arange(*cfg.mslp_bin_range)
    mids_msl = (xbins_msl[:-1] + xbins_msl[1:]) / 2

    oper_msl = (dataset_oper["msl"].values.ravel() / 100.0).astype(np.float64)
    hist_oper_msl, _ = np.histogram(oper_msl, bins=xbins_msl, density=True)
    logger.info(
        "OPER MSLP range: %.2f – %.2f hPa",
        oper_msl.min(),
        oper_msl.max(),
    )

    for i, exp in enumerate(all_datasets.keys()):
        vals = (all_datasets[exp]["msl"].values.ravel() / 100.0).astype(np.float64)
        hist_exp, _ = np.histogram(vals, bins=xbins_msl, density=True)
        axs[0].plot(
            mids_msl,
            hist_exp / hist_oper_msl,
            linestyle="-",
            linewidth=3,
            label=exp,
            color=colors[i],
        )
        logger.debug(
            "%s MSLP range: %.2f – %.2f hPa",
            exp,
            vals.min(),
            vals.max(),
        )

    vals = (dataset_enfo_o320["msl"].values.ravel() / 100.0).astype(np.float64)
    hist, _ = np.histogram(vals, bins=xbins_msl, density=True)
    axs[0].plot(
        mids_msl,
        hist / hist_oper_msl,
        "-.",
        linewidth=2,
        color="black",
        label="enfo_o320",
    )

    vals = (dataset_eefo_o96["msl"].values.ravel() / 100.0).astype(np.float64)
    hist, _ = np.histogram(vals, bins=xbins_msl, density=True)
    axs[0].plot(
        mids_msl, hist / hist_oper_msl, "--", linewidth=2, color="red", label="eefo_o96"
    )

    axs[0].plot(
        mids_msl,
        np.ones_like(mids_msl),
        "--",
        linewidth=2,
        color="green",
        label="OPER AN",
    )
    axs[0].set_ylim(*cfg.mslp_ylim)
    axs[0].set_xlabel("Mean Sea Level Pressure (hPa)", fontsize=14)
    axs[0].set_ylabel("Normalized Probability Density", fontsize=14)
    axs[0].set_title("Normalized (by AN O320) Distribution MSLP", fontsize=14)
    axs[0].legend()

    # =====================
    # WIND
    # =====================
    logger.info("Processing 10m wind speed")

    xbins_wind = np.arange(*cfg.wind_bin_range)
    mids_wind = (xbins_wind[:-1] + xbins_wind[1:]) / 2

    oper = (
        np.sqrt(dataset_oper["u10"] ** 2 + dataset_oper["v10"] ** 2)
        .values.ravel()
        .astype(np.float64)
    )
    hist_oper, _ = np.histogram(oper, bins=xbins_wind, density=True)
    logger.info(
        "OPER wind range: %.2f – %.2f m/s",
        oper.min(),
        oper.max(),
    )

    for i, exp in enumerate(all_datasets.keys()):
        vals = (
            np.sqrt(all_datasets[exp]["u10"] ** 2 + all_datasets[exp]["v10"] ** 2)
            .values.ravel()
            .astype(np.float64)
        )
        hist, _ = np.histogram(vals, bins=xbins_wind, density=True)
        axs[1].plot(
            mids_wind, hist / hist_oper, "-", linewidth=3, color=colors[i], label=exp
        )
        logger.debug(
            "%s wind range: %.2f – %.2f m/s",
            exp,
            vals.min(),
            vals.max(),
        )

    vals = (
        np.sqrt(dataset_enfo_o320["u10"] ** 2 + dataset_enfo_o320["v10"] ** 2)
        .values.ravel()
        .astype(np.float64)
    )
    hist, _ = np.histogram(vals, bins=xbins_wind, density=True)
    axs[1].plot(
        mids_wind, hist / hist_oper, "-.", linewidth=2, color="black", label="enfo_o320"
    )

    vals = (
        np.sqrt(dataset_eefo_o96["u10"] ** 2 + dataset_eefo_o96["v10"] ** 2)
        .values.ravel()
        .astype(np.float64)
    )
    hist, _ = np.histogram(vals, bins=xbins_wind, density=True)
    axs[1].plot(
        mids_wind, hist / hist_oper, "--", linewidth=2, color="red", label="eefo_o96"
    )

    axs[1].plot(
        mids_wind,
        np.ones_like(mids_wind),
        "--",
        linewidth=2,
        color="green",
        label="OPER AN",
    )
    axs[1].set_ylim(*cfg.wind_ylim)
    axs[1].set_xlabel("10m wind speed (m/s)", fontsize=14)
    axs[1].set_ylabel("Normalized Probability Density", fontsize=14)
    axs[1].set_title("Normalized (by AN O320) Distribution 10m Wind Speed", fontsize=14)
    axs[1].legend()

    fig.suptitle(cfg.plot_title)
    plt.tight_layout()

    logger.info("Event %s complete", cfg.name)
    return fig
