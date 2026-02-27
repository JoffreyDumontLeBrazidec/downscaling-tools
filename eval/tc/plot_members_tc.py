from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from cartopy import crs

LOG = logging.getLogger(__name__)


CASES = [
    {
        "name": "dora",
        "lat": (5.5, 23.0),
        "lon": (-158.75, -141.0),
        "dates": [6],
        "time": 1,
        "msl_levels": np.linspace(985, 1015, 31),
        "wind_levels": np.linspace(0, 30, 31),
    },
    {
        "name": "fernanda",
        "lat": (1.0, 29.0),
        "lon": (-134.0, -105.0),
        "dates": [13],
        "time": 2,
        "msl_levels": np.linspace(985, 1015, 31),
        "wind_levels": np.linspace(0, 30, 31),
    },
    {
        "name": "hilary",
        "lat": (11.0, 29.0),
        "lon": (-120.0, -100.0),
        "dates": [17],
        "time": 1,
        "msl_levels": np.linspace(965, 1015, 31),
        "wind_levels": np.linspace(0, 35, 31),
    },
    {
        "name": "idalia",
        "lat": (11.0, 39.01),
        "lon": (-99.0, -71.01),
        "dates": [28],
        "time": 1,
        "msl_levels": np.linspace(985, 1015, 31),
        "wind_levels": np.linspace(0, 30, 31),
    },
    {
        "name": "franklin",
        "lat": (11.0, 39.0),
        "lon": (-79.0, -51.0),
        "dates": [28],
        "time": 1,
        "msl_levels": np.linspace(985, 1015, 31),
        "wind_levels": np.linspace(0, 30, 31),
    },
]


def _parse_members(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _candidate_expids(expver: str) -> list[str]:
    out = [expver]
    if not expver.startswith("ENFO_O320_"):
        out.insert(0, f"ENFO_O320_{expver}")
    # de-duplicate while preserving order
    seen = set()
    uniq: list[str] = []
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        uniq.append(x)
    return uniq


def _resolve_expid_for_case(*, base_dir: str, case_name: str, year: str, month: str, day: int, expver: str) -> str:
    for expid in _candidate_expids(expver):
        p = Path(base_dir) / case_name / f"surface_pf_{expid}_{year}{month}{int(day):02d}.grib"
        if p.exists():
            return expid
    return expver


def _plot_member_data_legacy(
    *,
    data_dict: dict[str, np.ndarray],
    lon2: np.ndarray,
    lat2: np.ndarray,
    levels: np.ndarray,
    title: str,
    filename: str,
    members: list[int],
    time_index: int,
    day: int,
    out_dir: Path,
) -> None:
    datasets = [(data, key) for key, data in data_dict.items()]
    ncols, nrows = len(datasets), len(members)
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(8 * ncols, 6 * nrows),
        subplot_kw={"projection": crs.PlateCarree()},
    )
    fig.subplots_adjust(left=0.07, right=0.93, bottom=0.08, top=0.92, wspace=-0.1, hspace=0.15)
    axs = np.array(axs).reshape(nrows, ncols)

    images = []
    for row, member in enumerate(members):
        for col, (data, label) in enumerate(datasets):
            ax = axs[row, col]
            arr = data[day, member, time_index][10:100, 10:100]
            im = ax.pcolormesh(
                lon2[10:100, 10:100],
                lat2[10:100, 10:100],
                arr,
                transform=crs.PlateCarree(),
                vmin=float(levels[0]),
                vmax=float(levels[-1]),
                shading="gouraud",
                cmap="viridis",
            )
            ax.contour(
                lon2[10:100, 10:100],
                lat2[10:100, 10:100],
                arr,
                transform=crs.PlateCarree(),
                levels=levels,
                colors="black",
                linewidths=0.5,
            )
            ax.set_title(f"mbr{member} {label}", fontsize=12)
            ax.coastlines()
            ax.grid(color="white", linestyle="--", linewidth=0.5)
            gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
            gl.top_labels = False
            gl.right_labels = False
            images.append(im)

    for row in range(nrows):
        cbar = fig.colorbar(
            images[row * ncols],
            ax=axs[row],
            orientation="vertical",
            shrink=0.8,
            pad=0.02,
        )
        cbar.ax.set_ylabel("")

    fig.suptitle(title, fontsize=14)
    out_path = out_dir / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _build_retriever(case: dict, base_dir: str, year: str, month: str, resol: float) -> Any:
    # Import lazily so the module can be imported in test environments
    # where the metview executable is not available.
    from .tools.loading_data import DataRetriever

    lat = np.arange(case["lat"][0], case["lat"][1] + resol / 4, resol)
    lon = np.arange(case["lon"][0], case["lon"][1] + resol / 4, resol)
    return DataRetriever(
        f"{base_dir}/{case['name']}",
        case["dates"],
        year,
        month,
        resol,
        float(lat.min()),
        float(lat.max()),
        float(lon.min()),
        float(lon.max()),
    )


def run_tc_member_plots(
    *,
    expver: str,
    outdir: str,
    base_dir: str = "/home/ecm5702/hpcperm/data/tc",
    members: list[int] | None = None,
    year: str = "2023",
    month: str = "08",
    resol: float = 0.25,
) -> list[str]:
    members = members or [1, 2, 5, 7, 9]
    out_root = Path(outdir).expanduser().resolve() / "tc_members"
    out_root.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    analysis = "OPER_O320_0001"
    expid_enfo_o320 = "ENFO_O320_0001"
    expid_eefo_o96 = "EEFO_O96_0001"

    for case in CASES:
        retriever = _build_retriever(case, base_dir, year, month, resol)
        ml_expid = _resolve_expid_for_case(
            base_dir=base_dir,
            case_name=case["name"],
            year=year,
            month=month,
            day=case["dates"][0],
            expver=expver,
        )
        try:
            msl, wind10m = retriever.retrieve_all_data(
                analysis,
                expid_enfo_o320,
                expid_eefo_o96,
                [ml_expid],
            )
        except Exception as exc:
            LOG.warning("Skipping case %s: %s", case["name"], exc)
            continue

        lat = np.arange(case["lat"][0], case["lat"][1] + resol / 4, resol)
        lon = np.arange(case["lon"][0], case["lon"][1] + resol / 4, resol)
        lon2, lat2 = np.meshgrid(lon, lat)

        pf_keys = [k for k in msl.keys() if "OPER" not in k]
        if not pf_keys:
            LOG.warning("Skipping case %s: no ensemble datasets found", case["name"])
            continue
        max_member = min(int(msl[k].shape[1]) - 1 for k in pf_keys)
        safe_members = [m for m in members if 0 <= m <= max_member]
        if not safe_members:
            LOG.warning("Skipping case %s: no valid members in %s", case["name"], members)
            continue

        t_idx = int(case["time"])
        max_time = min(int(msl[k].shape[2]) for k in pf_keys) - 1
        if t_idx > max_time:
            LOG.warning("Skipping case %s: time index %s out of bounds", case["name"], t_idx)
            continue

        msl_data = {k: v for k, v in msl.items() if "OPER" not in k}
        wind_data = {k: v for k, v in wind10m.items() if "OPER" not in k}
        if not msl_data or not wind_data:
            LOG.warning("Skipping case %s: no non-OPER datasets available", case["name"])
            continue

        msl_name = f"{case['name']}_msl_fields_{case['dates'][0]}_{month}_step{t_idx}.png"
        wind_name = f"{case['name']}_wind10m_fields_{case['dates'][0]}_{month}_step{t_idx}.png"
        _plot_member_data_legacy(
            data_dict=msl_data,
            lon2=lon2,
            lat2=lat2,
            levels=case["msl_levels"],
            title="Mean Sea Level Pressure",
            filename=msl_name,
            members=safe_members,
            time_index=t_idx,
            day=0,
            out_dir=out_root,
        )
        _plot_member_data_legacy(
            data_dict=wind_data,
            lon2=lon2,
            lat2=lat2,
            levels=case["wind_levels"],
            title=f"Wind Speed 10m - {month} Lead Time {t_idx}",
            filename=wind_name,
            members=safe_members,
            time_index=t_idx,
            day=0,
            out_dir=out_root,
        )
        generated.extend([str(out_root / msl_name), str(out_root / wind_name)])
        LOG.info("Generated TC member plots for %s", case["name"])

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot TC member maps for configured August events.")
    parser.add_argument("--expver", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--base-dir", default="/home/ecm5702/hpcperm/data/tc")
    parser.add_argument("--members", default="1,2,5,7,9")
    parser.add_argument("--year", default="2023")
    parser.add_argument("--month", default="08")
    parser.add_argument("--resol", type=float, default=0.25)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    generated = run_tc_member_plots(
        expver=args.expver,
        outdir=args.outdir,
        base_dir=args.base_dir,
        members=_parse_members(args.members),
        year=args.year,
        month=args.month,
        resol=args.resol,
    )
    print(f"Generated {len(generated)} files in {Path(args.outdir).expanduser().resolve() / 'tc_members'}")


if __name__ == "__main__":
    main()
