"""Figure suite for track-based TC comparison — two page-oriented reports.

Report 1 (``tc_tracks_report.pdf``) — one page per basin, in the standard
``eval.cli tc`` PDF style (two log-density panels, MSLP inverted + wind),
pooling ALL track points of all TCs of the basin.

Report 2 (``tc_tracks_diagnostics.pdf``) — the remaining diagnostics:

  P1  overview     — headline table (all basins/roles) + focus-basin lifetime
                     min-MSLP log-PDF with a ratio-vs-target inset
  P2  focus basin  — consolidated grid: density diffs vs target, intensity vs
                     lead time, lifetime max-wind PDF, counts, classification
  P3  other basins — the same grid, one compact row per remaining basin
  P4+ case pages   — one page per selected storm (deepest reference tracks):
                     single map with every role's associated tracks, MSLP
                     spaghetti, per-member deepest-MSLP strip plot, stats box

Every page carries a provenance footer (support contract + member sets +
completeness). Role colours come from the house style (``eval.plotting``) so the
same role reads the same across every figure of the framework: target (the
truth) black, input blue, model red, ctrl (a control run) grey dashed like any
other reference; extra roles take the colour-blind-safe sequence. Diagnostic panel only: pages show distributions and
references side by side and never verdict language.

Multi-month runs default to pooled ("all") pages; per-month focus-basin pages
only render behind ``per_month=True``.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eval.plotting import (
    AXIS,
    FigureBook,
    INPUT_COLOR,
    MODEL_COLOR,
    SEQUENCE,
    TRUTH_COLOR,
    add_geography,
    axis_label,
    extend_for,
    pdf_label,
    reference_style,
    role_style,
    save_figure,
    sequence_style,
    styled,
    symmetric_norm,
)

from .scorer import DENSITY_BIN_DEG, TS_WIND_MS, density_grid, scope_mask, select_cases

LOG = logging.getLogger(__name__)

ROLE_COLORS = {
    "target": TRUTH_COLOR,
    "input": INPUT_COLOR,
    "ctrl": reference_style(0)["color"],
    "model": MODEL_COLOR,
}
_EXTRA_CYCLE = list(SEQUENCE)

# Words used on the figures for the tccompare roles.
ROLE_NAMES = {"target": "truth", "model": "model", "input": "input", "ctrl": "control"}

BASIN_NAMES = {
    "atl": "North Atlantic",
    "enp": "Eastern North Pacific",
    "cnp": "Central North Pacific",
    "wnp": "Western North Pacific",
    "nin": "North Indian Ocean",
    "sin": "South Indian Ocean",
    "aus": "Australian region",
    "spc": "South Pacific",
}

_MSLP_LABEL = axis_label("msl")          # "Mean sea level pressure (hPa)"
_WIND_LABEL = axis_label("10ff")         # "10 m wind speed (m s⁻¹)"
_WIND_UNIT = "m s⁻¹"

# Map extents per basin: (lon_w, lon_e, lat_s, lat_n), degrees east 0..360.
BASIN_EXTENTS = {
    "atl": (250, 360, 0, 60),
    "enp": (180, 280, 0, 45),
    "cnp": (180, 230, 0, 40),
    "wnp": (100, 200, 0, 50),
    "nin": (40, 110, 0, 35),
    "sin": (30, 115, -40, 0),
    "aus": (90, 160, -40, 0),
    "spc": (135, 240, -40, 0),
}

MSLP_BINS = np.arange(890.0, 1021.0, 5.0)
WIND_BINS = np.arange(0.0, 82.5, 2.5)
PAGE_SIZE = (11.69, 8.27)  # A4 landscape
_VALID_TIME_FMT = "%Y/%m/%d/%H"


def role_color(role: str, index: int = 0) -> str:
    return ROLE_COLORS.get(role, _EXTRA_CYCLE[index % len(_EXTRA_CYCLE)])


def role_line(role: str, index: int = 0, **overrides) -> dict:
    """Line keyword arguments for a tccompare role (house role styles)."""
    if role == "target":
        return role_style("truth", **overrides)
    if role in ("model", "input"):
        return role_style(role, **overrides)
    if role == "ctrl":
        return reference_style(0, **overrides)
    return sequence_style(index, **overrides)


def role_name(role: str) -> str:
    """Readable name of a role ("truth" for target, "control" for ctrl)."""
    return ROLE_NAMES.get(role, role.replace("_", " "))


_SOURCE_ID = re.compile(r"^(?:(?P<cls>od|ai|rd)_)?(?P<stream>enfo|eefo|oper|iekm)_(?P<expver>\w+)$", re.I)


def source_name(src: dict[str, Any], role: str) -> str:
    """Readable source name: "ENFO" for od_enfo_0001, "AI ENFO" for ai_enfo_0001, the expver
    (for example "ja6g") for a model run."""
    sid = str((src.get("provenance") or {}).get("source_id") or role)
    m = _SOURCE_ID.match(sid)
    if not m:
        return sid
    prefix = "AI " if (m.group("cls") or "").lower() == "ai" else ""
    return f"{prefix}{m.group('stream').upper()}"


def role_label(role: str, src: dict[str, Any] | None = None) -> str:
    """"truth (ENFO)", "model (ja6g)" ..."""
    name = role_name(role)
    if src is None:
        return name
    sname = source_name(src, role)
    return name if sname in (role, name) else f"{name} ({sname})"


def basin_title(basin: str) -> str:
    name = BASIN_NAMES.get(basin)
    return f"{name} ({basin.upper()})" if name else basin.upper()


def month_text(month: str) -> str:
    try:
        return pd.to_datetime(str(month), format="%Y%m").strftime("%B %Y")
    except (ValueError, TypeError):
        return str(month)


def _footer_text(sources: dict[str, dict[str, Any]], support: dict[str, Any]) -> str:
    contract = support.get("contract", {})
    def _txt(value, default):
        return default if value in (None, "", []) else str(value)

    parts = [
        f"tracker support {_txt(contract.get('grid_support'), 'n/a').upper()}, "
        f"steps {_txt(contract.get('steps'), 'all')}, "
        f"vorticity {_txt(contract.get('vorticity'), 'default')}",
    ]
    for role, src in sources.items():
        prov = src.get("provenance") or {}
        compl = prov.get("completeness")
        compl_txt = f"{100 * float(compl):.0f}%" if compl is not None else "n/a"
        parts.append(
            f"{role}={prov.get('source_id')} m{len(prov.get('members') or [])} "
            f"compl {compl_txt}"
        )
    pinned = next(
        (p.get("dates_pinned") for p in
         ((s.get("provenance") or {}) for s in sources.values()) if p.get("dates_pinned")),
        None,
    )
    if pinned:
        parts.append(f"PINNED {len(pinned)} dates {pinned[0]}..{pinned[-1]}")
    if not support.get("consistent", True):
        parts.append("!! SUPPORT CONTRACT VIOLATIONS — see metrics json")
    return " | ".join(parts)


def _scoped_records(src: dict[str, Any], months, basin) -> pd.DataFrame:
    rec = src["records"]
    if rec.empty:
        return rec
    return rec[scope_mask(rec, months, [basin])]


def _fmt_latlon(lat: float, lon_e: float) -> str:
    lon = lon_e % 360.0
    lon_txt = f"{360 - lon:.0f}W" if lon > 180 else f"{lon:.0f}E"
    lat_txt = f"{abs(lat):.0f}{'N' if lat >= 0 else 'S'}"
    return f"{lat_txt} {lon_txt}"


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians((lon2 - lon1 + 180.0) % 360.0 - 180.0)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


# ---------------------------------------------------------------------------
# Axes-level building blocks (each draws into a provided axes)
# ---------------------------------------------------------------------------

def _add_map_ax(fig, spec, extent, *, label_size: float = 6.5):
    """One map axes on a gridspec slot; cartopy when available."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        proj = ccrs.PlateCarree(central_longitude=(extent[0] + extent[1]) / 2 % 360)
        ax = fig.add_subplot(spec, projection=proj)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="0.93", zorder=0)
        add_geography(ax, label_size=label_size)
        return ax, ccrs.PlateCarree()
    except Exception:  # cartopy unavailable — degrade to plain axes
        ax = fig.add_subplot(spec)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("auto")
        return ax, None


def _density_per_forecast(sources, months, basin):
    """{role: track points / forecast on the 2° grid} + bin centres."""
    grids: dict[str, np.ndarray] = {}
    n_fc: dict[str, int] = {}
    edges = None
    for role, src in sources.items():
        rec = _scoped_records(src, months, basin)
        grid = density_grid(rec)
        fc = src["forecasts"]
        present = fc[fc["present"]] if ("present" in fc and not fc.empty) else fc
        if not present.empty and months:
            present = present[present["init_date"].astype(str).str[:6].isin(months)]
        n = max(1, len(present))
        grids[role] = grid["hist"] / n
        n_fc[role] = n
        edges = (grid["lat_edges"], grid["lon_edges"])
    lat_c = (edges[0][:-1] + edges[0][1:]) / 2
    lon_c = (edges[1][:-1] + edges[1][1:]) / 2
    return grids, n_fc, lat_c, lon_c


def _draw_density_diffs(fig, specs, sources, months, basin, *, title_size=9, cbar_size=9,
                        cbar_shrink=0.85):
    """Density difference maps vs target (or absolute maps without a target).

    Returns the list of map axes; one spec is consumed per drawn panel. ``cbar_shrink`` is the
    height of the shared colour bar relative to the row of maps.
    """
    def _colorbar(pcm, axes, **kw):
        return fig.colorbar(pcm, ax=axes, shrink=cbar_shrink, pad=0.015, aspect=25, **kw)

    extent = BASIN_EXTENTS.get(basin, (0, 360, -60, 60))
    grids, n_fc, lat_c, lon_c = _density_per_forecast(sources, months, basin)
    axes = []
    if "target" in grids:
        roles = [r for r in grids if r != "target"][: len(specs)]
        diffs = {r: grids[r] - grids["target"] for r in roles}
        # one zero-centred scale shared by the row; arrows mark the rare cells beyond it
        norm, _lim = symmetric_norm(*[np.where(d != 0, d, np.nan) for d in diffs.values()], q=99.0)
        pcm = None
        for spec, role in zip(specs, roles):
            ax, transform = _add_map_ax(fig, spec, extent)
            kwargs = {"transform": transform} if transform is not None else {}
            pcm = ax.pcolormesh(lon_c, lat_c, diffs[role], norm=norm, cmap="RdBu_r",
                                rasterized=True, **kwargs)
            ax.set_title(f"{role_name(role).capitalize()} minus truth: track density",
                         fontsize=title_size)
            axes.append(ax)
        if axes:
            cb = _colorbar(pcm, axes, extend=extend_for(norm, *diffs.values()))
            cb.set_label("Track density difference\n(points per forecast per 2° box)",
                         fontsize=cbar_size)
            cb.ax.tick_params(labelsize=cbar_size - 1)
    else:  # no target — absolute densities
        roles = list(grids)[: len(specs)]
        vmax = max((grids[r].max() for r in roles), default=0) or 1.0
        pcm = None
        for spec, role in zip(specs, roles):
            ax, transform = _add_map_ax(fig, spec, extent)
            kwargs = {"transform": transform} if transform is not None else {}
            pcm = ax.pcolormesh(lon_c, lat_c,
                                np.where(grids[role] > 0, grids[role], np.nan),
                                vmin=0, vmax=vmax, cmap="viridis", rasterized=True, **kwargs)
            ax.set_title(f"{role_name(role).capitalize()}: track density "
                         f"(n = {n_fc[role]} forecasts)", fontsize=title_size)
            axes.append(ax)
        if axes:
            cb = _colorbar(pcm, axes)
            cb.set_label("Track density\n(points per forecast per 2° box)", fontsize=cbar_size)
            cb.ax.tick_params(labelsize=cbar_size - 1)
    return axes


def _draw_step_intensity(ax, sources, months, basin, *, label_size=8):
    drew = False
    for idx, (role, src) in enumerate(sources.items()):
        rec = _scoped_records(src, months, basin)
        if rec.empty:
            continue
        grp = rec.groupby("step_h")["mslp_hpa"]
        steps = sorted(rec["step_h"].unique())
        med = grp.median().reindex(steps)
        q10 = grp.quantile(0.10).reindex(steps)
        q90 = grp.quantile(0.90).reindex(steps)
        color = role_color(role, idx)
        ax.plot(steps, med, label=role_name(role), **role_line(role, idx, linewidth=1.8))
        ax.fill_between(steps, q10, q90, color=color, alpha=0.12, linewidth=0)
        drew = True
    ax.set_xlabel(AXIS["lead"], fontsize=label_size)
    ax.set_ylabel("Track MSLP (hPa)\nmedian and 10–90 % range", fontsize=label_size)
    ax.invert_yaxis()
    ax.tick_params(labelsize=label_size - 1)
    if drew:
        ax.legend(fontsize=label_size - 1)
    ax.grid(alpha=0.25)
    return drew


def _lifetime_values(sources, months, basin, stat):
    out: dict[str, np.ndarray] = {}
    for role, src in sources.items():
        summ = src["summary"]
        vals = (summ[scope_mask(summ, months, [basin])][stat].dropna().to_numpy(dtype=float)
                if not summ.empty else np.array([]))
        out[role] = vals
    return out


def _draw_intensity_pdf(ax, sources, months, basin, stat, bins, xlabel,
                        *, ratio_ax=None, label_size=8):
    """Log-PDF of a lifetime statistic per role.

    When ``ratio_ax`` is given (an axes placed under ``ax`` that shares its x axis) the ratio of
    each role's PDF to the truth's is drawn there, so it never covers the curves above.
    """
    values = _lifetime_values(sources, months, basin, stat)
    centers = (bins[:-1] + bins[1:]) / 2
    hists = {}
    for idx, (role, vals) in enumerate(values.items()):
        hist, _ = np.histogram(vals, bins=bins, density=True)
        hists[role] = hist
        ax.semilogy(centers, np.where(hist > 0, hist, np.nan),
                    label=f"{role_name(role)} (n = {len(vals)} tracks)",
                    **role_line(role, idx, linewidth=1.8))
    unit = "hPa" if "hPa" in xlabel else _WIND_UNIT
    ax.set_ylabel(pdf_label(unit), fontsize=label_size)
    ax.tick_params(labelsize=label_size - 1)
    # bulk of both distributions sits at the benign end: MSLP right, wind left, so the legend
    # goes to the empty upper corner on the other side
    ax.legend(fontsize=label_size - 1,
              loc=("upper right" if stat == "wind_max_ms" else "upper left"))
    ax.grid(alpha=0.25)
    has_ratio = ratio_ax is not None and len(values.get("target", ())) > 0
    if has_ratio:
        tgt = hists["target"]
        for idx, (role, hist) in enumerate(hists.items()):
            if role == "target":
                continue
            ratio = np.where(tgt > 0, hist / np.where(tgt > 0, tgt, np.nan), np.nan)
            ratio_ax.plot(centers, ratio, **role_line(role, idx, linewidth=1.2))
        ratio_ax.axhline(1.0, **role_line("target", linewidth=1.0))
        ratio_ax.set_ylim(0, 3.5)
        ratio_ax.set_ylabel("Ratio to\nthe truth", fontsize=label_size - 1)
        ratio_ax.set_xlabel(xlabel, fontsize=label_size)
        ratio_ax.tick_params(labelsize=label_size - 1)
        ratio_ax.grid(alpha=0.25)
        ax.tick_params(labelbottom=False)
    else:
        ax.set_xlabel(xlabel, fontsize=label_size)
        if ratio_ax is not None:
            ratio_ax.axis("off")


def _reserve_footer(fig):
    """Keep constrained-layout axes clear of the provenance footer line."""
    try:
        fig.get_layout_engine().set(rect=(0.0, 0.03, 1.0, 0.97))
    except Exception:
        pass


def _basin_metric_rows(metrics, basin, scope):
    return {row["role"]: row for row in metrics.get("metrics", [])
            if row.get("basin") == basin and row.get("scope") == scope}


def _headline_scope(metrics) -> str:
    scopes = metrics.get("scopes") or []
    return "all" if "all" in scopes else (scopes[0] if scopes else "all")


def _draw_counts(ax, metrics, basin, scope, *, label_size=8):
    """Grouped per-forecast count bars (tracks + TC-days) per role."""
    rows = _basin_metric_rows(metrics, basin, scope)
    roles = list(rows)
    if not roles:
        ax.axis("off")
        return
    width = 0.8 / len(roles)
    x = np.arange(2)
    for idx, role in enumerate(roles):
        row = rows[role]
        vals = [row.get("tracks_per_forecast") or 0.0,
                row.get("tc_days_per_forecast") or 0.0]
        ci = row.get("tracks_per_forecast_ci")
        yerr = None
        if isinstance(ci, (list, tuple)) and vals[0]:
            yerr = np.array([[vals[0] - ci[0], 0.0], [ci[1] - vals[0], 0.0]])
        ax.bar(x + idx * width, vals, width, yerr=yerr, capsize=2,
               color=role_color(role, idx), label=role_name(role))
    ax.set_xticks(x + 0.4 - width / 2)
    ax.set_xticklabels(["tracks", "TC days"], fontsize=label_size)
    ax.set_ylabel("Number per forecast", fontsize=label_size)
    ax.tick_params(labelsize=label_size - 1)
    ax.grid(alpha=0.25, axis="y")
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.25)  # headroom so the legend clears the bars
    ax.legend(fontsize=label_size - 2)


# Intensity classes form an ordered scale: one sequential colour map, strongest darkest
# (never a qualitative red/green set); non-tropical classes in greys.
_CLASS_ORDER = ["HR5", "HR4", "HR3", "HR2", "HR1", "TS", "TD"]
_CLASS_GREYS = {"ET": "0.55", "SSD": "0.72", "unknown": "0.86"}


def _class_color(cat: str):
    if cat in _CLASS_ORDER:
        frac = _CLASS_ORDER.index(cat) / (len(_CLASS_ORDER) - 1)
        return plt.get_cmap("magma")(0.08 + 0.80 * frac)
    return _CLASS_GREYS.get(cat, "0.8")


def _draw_classification(ax, metrics, basin, scope, *, label_size=8):
    rows = _basin_metric_rows(metrics, basin, scope)
    cls_rows = []
    for role, row in rows.items():
        counts = row.get("classification_counts")
        if not isinstance(counts, dict):
            counts = {}
        total = sum(counts.values()) or 1
        cls_rows.append({"role": role_name(role), **{k: v / total for k, v in counts.items()}})
    if not cls_rows:
        ax.axis("off")
        return
    cls_df = pd.DataFrame(cls_rows).set_index("role").fillna(0.0)
    order = [c for c in ["HR5", "HR4", "HR3", "HR2", "HR1", "TS", "TD", "ET", "SSD", "unknown"]
             if c in cls_df]
    bottom = np.zeros(len(cls_df))
    for cat in order:
        ax.bar(cls_df.index, cls_df[cat], bottom=bottom, label=cat, color=_class_color(cat),
               edgecolor="white", linewidth=0.4)
        bottom += cls_df[cat].to_numpy()
    ax.set_ylabel("Fraction of track points", fontsize=label_size)
    ax.grid(False)
    ax.tick_params(labelsize=label_size - 1)
    ax.tick_params(axis="x", rotation=45)
    if order:
        ax.legend(fontsize=label_size - 1.5, loc="center left", bbox_to_anchor=(1.0, 0.5))


# ---------------------------------------------------------------------------
# Report 1: per-basin TC distributions in the eval.cli `tc` PDF style
# ---------------------------------------------------------------------------

# Styling mirrors eval/evaluators/tc/core/pdf_plot.py (plot_pdf_log): two log-density
# panels (MSLP left with inverted x so intensity increases rightward, wind
# right), log-floor instead of gaps, truth drawn black/solid/thick. Line styles
# are the house role styles (eval.plotting.role_style).
ROLE_LINESTYLES = {r: role_line(r)["linestyle"] for r in ROLE_COLORS}
ROLE_LINEWIDTHS = {r: role_line(r)["linewidth"] for r in ROLE_COLORS}

MSLP_POINT_BINS = np.arange(890.0, 1032.5, 2.5)
WIND_POINT_BINS = np.arange(0.0, 81.0, 1.0)


def _log_floor(series) -> float:
    positive = [s[np.isfinite(s) & (s > 0)] for s in series]
    positive = [s for s in positive if s.size]
    if not positive:
        return 1e-12
    return max(float(np.min(np.concatenate(positive))) * 0.1, 1e-12)


@styled
def page_basin_distributions(sources, months, basin, scope_label, dist_stat="records") -> plt.Figure:
    """One tc-style page: pooled MSLP + wind PDFs over all TCs of one basin.

    ``dist_stat='records'`` pools every 24-hourly track point of every TC in
    scope (the all-TC analogue of the box `tc` evaluator's field PDFs).
    """
    data: dict[str, dict[str, np.ndarray]] = {}
    for role, src in sources.items():
        rec = _scoped_records(src, months, basin)
        data[role] = {
            "mslp": rec["mslp_hpa"].dropna().to_numpy(dtype=float) if not rec.empty else np.array([]),
            "wind": rec["wind_ms"].dropna().to_numpy(dtype=float) if not rec.empty else np.array([]),
        }

    fig, axs = plt.subplots(1, 2, figsize=(13.8, 5.6))
    panels = [
        ("mslp", MSLP_POINT_BINS, _MSLP_LABEL, "Mean sea level pressure at the track points"),
        ("wind", WIND_POINT_BINS, _WIND_LABEL, "Maximum 10 m wind speed at the track points"),
    ]
    for ax, (var, bins, xlabel, title) in zip(axs, panels):
        mids = (bins[:-1] + bins[1:]) / 2
        hists = {}
        for role in data:
            hist, _ = np.histogram(data[role][var], bins=bins, density=True)
            hists[role] = hist
        floor = _log_floor(list(hists.values()))
        # truth (target) first, black and thick, like the analysis in `tc`
        order = (["target"] if "target" in data else []) + [r for r in data if r != "target"]
        for idx, role in enumerate(order):
            n = len(data[role][var])
            ax.plot(mids, np.where(np.isfinite(hists[role]) & (hists[role] > 0),
                                   hists[role], floor),
                    label=f"{role_label(role, sources[role])} (n = {n} track points)",
                    **role_line(role, idx, alpha=0.96))
        ax.set_yscale("log")
        ax.set_ylim(bottom=floor)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(pdf_label("hPa" if var == "mslp" else _WIND_UNIT))
        ax.set_title(title)
        ax.legend(loc="upper left" if var == "mslp" else "upper right")
        # data-range crop; MSLP inverted so TC intensity increases rightward
        pooled = np.concatenate([v[var] for v in data.values() if v[var].size]) \
            if any(v[var].size for v in data.values()) else np.array([0.0, 1.0])
        if var == "mslp":
            ax.set_xlim(min(bins[-1], pooled.max() + 5.0),
                        max(bins[0], pooled.min() - 5.0))
        else:
            ax.set_xlim(0, min(bins[-1], pooled.max() + 2.0))
    fig.suptitle(
        f"{basin_title(basin)}: distributions over all track points of all tropical "
        f"cyclones, {scope_label}",
    )
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.16, top=0.87, wspace=0.22)
    return fig


# ---------------------------------------------------------------------------
# Diagnostics report: overview page
# ---------------------------------------------------------------------------

def _fmt_ci(value, ci, fmt="{:.1f}") -> str:
    if value is None:
        return "—"
    txt = fmt.format(value)
    if isinstance(ci, (list, tuple)) and len(ci) == 2:
        txt += f" [{fmt.format(ci[0])}, {fmt.format(ci[1])}]"
    return txt


@styled
def page_overview(sources, metrics, months, basins, focus_basin) -> plt.Figure:
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.text(0.04, 0.94, "Tropical cyclone track comparison", fontsize=17, weight="bold")
    scope = _headline_scope(metrics)
    roles_txt = ", ".join(role_label(r, sources[r]) for r in sources)
    subtitle = (f"{', '.join(month_text(m) for m in months)}; basins "
                f"{', '.join(b.upper() for b in basins)}; statistics over "
                f"{_scope_display(scope, months)}; sources: {roles_txt}")
    fig.text(0.04, 0.905, subtitle, fontsize=9, color="0.25")
    footer = _footer_text(sources, metrics.get("support", {}))
    fig.text(0.04, 0.885, footer.replace(" | ", "\n"), fontsize=6.5, color="0.35", va="top")

    # headline table, one row per (basin, role)
    col_labels = ["basin", "source", "fore-\ncasts", "tracks per\nforecast\n(95 % CI)",
                  "5th pct of\nmin MSLP\n(hPa, 95 % CI)", "deepest\nMSLP\n(hPa)",
                  "95th pct\nwind\n(m s⁻¹)", "TC days\nper\nforecast"]
    cells, row_colors, row_roles = [], [], []
    for basin in basins:
        rows = _basin_metric_rows(metrics, basin, scope)
        for role, row in rows.items():
            cells.append([
                basin.upper(), role_name(role),
                str(row.get("n_forecasts", "—")),
                _fmt_ci(row.get("tracks_per_forecast"), row.get("tracks_per_forecast_ci"), "{:.2f}"),
                _fmt_ci(row.get("mslp_p5"), row.get("mslp_p5_ci"), "{:.1f}"),
                f"{row['mslp_min']:.0f}" if row.get("mslp_min") is not None else "—",
                f"{row['wind_p95']:.1f}" if row.get("wind_p95") is not None else "—",
                f"{row['tc_days_per_forecast']:.2f}" if row.get("tc_days_per_forecast") is not None else "—",
            ])
            row_colors.append("#f3f3f3" if (basins.index(basin) % 2) else "white")
            row_roles.append(role)
    ax_tab = fig.add_axes([0.03, 0.06, 0.52, 0.72])
    ax_tab.axis("off")
    if cells:
        col_widths = [0.07, 0.09, 0.07, 0.19, 0.21, 0.11, 0.11, 0.11]
        table = ax_tab.table(cellText=cells, colLabels=col_labels,
                             colWidths=col_widths, cellLoc="center", loc="upper center")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.55)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("0.8")
            if r == 0:
                cell.set_text_props(weight="bold", fontsize=7)
                cell.set_height(cell.get_height() * 1.6)
            else:
                cell.set_facecolor(row_colors[r - 1])
                if c == 1:
                    cell.set_text_props(color=role_color(row_roles[r - 1]), weight="bold")
    ax_tab.set_title(f"Headline statistics, {_scope_display(scope, months)}", fontsize=10)

    # focus-basin deep-tail PDF with ratio inset
    # (the ratio to the truth has its own axes under the PDF, so it hides no curve)
    ax_pdf = fig.add_axes([0.65, 0.36, 0.32, 0.40])
    ax_ratio = fig.add_axes([0.65, 0.12, 0.32, 0.18], sharex=ax_pdf)
    _draw_intensity_pdf(ax_pdf, sources, months, focus_basin,
                        "mslp_min_hpa", MSLP_BINS, "Lifetime minimum MSLP (hPa)",
                        ratio_ax=ax_ratio, label_size=9)
    ax_pdf.set_title(f"{basin_title(focus_basin)}: lifetime minimum MSLP per track",
                     fontsize=10)
    return fig


# ---------------------------------------------------------------------------
# P2 focus-basin consolidated grid
# ---------------------------------------------------------------------------

def _scope_display(scope_name, months) -> str:
    if scope_name != "all":
        return month_text(scope_name)
    return (month_text(months[0]) if len(months) == 1
            else f"all months ({', '.join(month_text(m) for m in months)})")


@styled
def page_basin_grid(sources, metrics, months, basin, scope_name) -> plt.Figure:
    fig = plt.figure(figsize=PAGE_SIZE, layout="constrained")
    _reserve_footer(fig)
    gs = fig.add_gridspec(2, 6, height_ratios=[1.25, 1.0])
    # Top row: one density map per source other than the truth, sharing the whole row, with
    # one colour bar as tall as the maps (no empty map slot, no oversized colour bar).
    n_diffs = min(3, max(1, len([r for r in sources if r != "target"]) or len(sources)))
    top = gs[0, :].subgridspec(1, n_diffs)
    specs = [top[0, i] for i in range(n_diffs)]
    _draw_density_diffs(fig, specs, sources, months, basin, cbar_shrink=0.6)

    ax_step = fig.add_subplot(gs[1, 0:2])
    _draw_step_intensity(ax_step, sources, months, basin)
    ax_step.set_title("Track MSLP against lead time", fontsize=9)

    # Middle of the bottom row: the lifetime maximum wind PDF with the ratio to the truth in
    # its own axes underneath.
    wind = gs[1, 2:4].subgridspec(2, 1, height_ratios=[3.0, 1.15], hspace=0.06)
    ax_wind = fig.add_subplot(wind[0])
    ax_wind_ratio = fig.add_subplot(wind[1], sharex=ax_wind)
    _draw_intensity_pdf(ax_wind, sources, months, basin,
                        "wind_max_ms", WIND_BINS, f"Lifetime maximum 10 m wind speed ({_WIND_UNIT})",
                        ratio_ax=ax_wind_ratio)
    ax_wind.set_title("Lifetime maximum wind per track", fontsize=9)

    scope = scope_name if scope_name in (metrics.get("scopes") or []) else _headline_scope(metrics)
    ax_counts = fig.add_subplot(gs[1, 4])
    _draw_counts(ax_counts, metrics, basin, scope)
    ax_counts.set_title("Counts per forecast", fontsize=9)
    ax_cls = fig.add_subplot(gs[1, 5])
    _draw_classification(ax_cls, metrics, basin, scope)
    ax_cls.set_title("Intensity classes", fontsize=9)

    fig.suptitle(f"{basin_title(basin)}: all tropical cyclones, "
                 f"{_scope_display(scope_name, months)}")
    return fig


# ---------------------------------------------------------------------------
# P3 other basins, one compact row each
# ---------------------------------------------------------------------------

@styled
def page_other_basins(sources, metrics, months, other_basins, scope_name) -> plt.Figure:
    nrows = len(other_basins)
    fig = plt.figure(figsize=PAGE_SIZE, layout="constrained")
    _reserve_footer(fig)
    gs = fig.add_gridspec(nrows, 4)
    for i, basin in enumerate(other_basins):
        ax_pdf = fig.add_subplot(gs[i, 0])
        _draw_intensity_pdf(ax_pdf, sources, months, basin,
                            "mslp_min_hpa", MSLP_BINS, "Lifetime minimum MSLP (hPa)",
                            label_size=7)
        ax_pdf.set_title(f"{basin.upper()}: lifetime minimum MSLP", fontsize=9)
        _draw_density_diffs(fig, [gs[i, 1], gs[i, 2]], sources, months, basin,
                            title_size=8, cbar_size=6.5)
        ax_step = fig.add_subplot(gs[i, 3])
        _draw_step_intensity(ax_step, sources, months, basin, label_size=7)
        ax_step.set_title(f"{basin.upper()}: MSLP by lead time", fontsize=9)
    fig.suptitle(
        f"Other basins for context ({', '.join(b.upper() for b in other_basins)}), "
        f"{_scope_display(scope_name, months)}",
    )
    return fig


# ---------------------------------------------------------------------------
# P4+ case pages (one storm per page)
# ---------------------------------------------------------------------------

def _case_member_tracks(sources, case):
    """{role: list of (member_meta, records_df)} for a case's associations."""
    out: dict[str, list[tuple[dict, pd.DataFrame]]] = {}
    for role, src in sources.items():
        rec, summ = src["records"], src["summary"]
        entries = []
        for m in case["members"].get(role) or []:
            key = ((summ["init_date"] == str(m["init_date"]))
                   & (summ["member"] == m["member"])
                   & (summ["track_id"] == m["track_id"])) if not summ.empty else None
            meta = dict(m)
            if key is not None and key.any():  # enrich from summary when fields absent
                srow = summ[key].iloc[0]
                for col in ("mslp_min_hpa", "mslp_min_lat", "mslp_min_lon_e",
                            "mslp_min_valid_time", "wind_max_ms"):
                    meta.setdefault(col, srow.get(col))
            tr = rec[(rec["init_date"] == str(m["init_date"]))
                     & (rec["member"] == m["member"])
                     & (rec["track_id"] == m["track_id"])] if not rec.empty else pd.DataFrame()
            entries.append((meta, tr))
        out[role] = entries
    return out


def case_label(case) -> str:
    at = case["at"]
    try:
        date_txt = pd.to_datetime(str(at["valid_time"]), format=_VALID_TIME_FMT).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        date_txt = str(at["valid_time"])
    basin = str(case.get("basin") or case["case_id"].split("_")[0]).upper()
    return (f"{basin}, deepest {date_txt} {case['mslp_min_hpa']:.0f} hPa "
            f"near {_fmt_latlon(at['lat'], at['lon_e'])}")


def _case_map_extent(members_by_role, at):
    ref_lon = float(at["lon_e"]) % 360.0
    lons, lats = [ref_lon], [float(at["lat"])]
    for entries in members_by_role.values():
        for _, tr in entries:
            if tr.empty:
                continue
            lon = tr["lon_e"].to_numpy(dtype=float) % 360.0
            lon = np.where(lon < ref_lon - 180, lon + 360,
                           np.where(lon > ref_lon + 180, lon - 360, lon))
            lons.extend(lon.tolist())
            lats.extend(tr["lat"].to_numpy(dtype=float).tolist())
    pad = 4.0
    return (min(lons) - pad, max(lons) + pad, min(lats) - pad, max(lats) + pad), ref_lon


@styled
def page_case(sources, case, months) -> plt.Figure:
    fig = plt.figure(figsize=PAGE_SIZE, layout="constrained")
    _reserve_footer(fig)
    # top row: MSLP spaghetti + deepest-MSLP strip; bottom row: wide map + stats
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.9],
                          height_ratios=[1.0, 0.95])
    members_by_role = _case_member_tracks(sources, case)
    at = case["at"]
    try:
        ref_time = pd.to_datetime(str(at["valid_time"]), format=_VALID_TIME_FMT)
    except (ValueError, TypeError):
        ref_time = None

    # --- single map, all roles overlaid, color = role ---
    extent, ref_lon = _case_map_extent(members_by_role, at)
    ax_map, transform = _add_map_ax(fig, gs[1, 0:2], extent)
    kwargs = {"transform": transform} if transform is not None else {}
    for idx, role in enumerate(sources):
        color = role_color(role, idx)
        labelled = False
        for _, tr in members_by_role.get(role, []):
            if tr.empty:
                continue
            lon = tr["lon_e"].to_numpy(dtype=float) % 360.0
            lon = np.where(lon < ref_lon - 180, lon + 360,
                           np.where(lon > ref_lon + 180, lon - 360, lon))
            ax_map.plot(lon, tr["lat"], lw=0.9, alpha=0.55, color=color,
                        label=(None if labelled else role_label(role, sources[role])), **kwargs)
            ax_map.plot(lon[:1], tr["lat"].to_numpy()[:1], ".", ms=3,
                        color=color, alpha=0.8, **kwargs)
            labelled = True
    ax_map.plot([ref_lon], [float(at["lat"])], marker="*", ms=14,
                color=role_color(case.get("reference_role", "target")),
                mec="white", mew=0.6, zorder=5, **kwargs)
    leg = ax_map.legend(fontsize=8, loc="upper left", frameon=True, framealpha=0.85)
    for line in leg.get_lines():
        line.set_linewidth(2.0)
        line.set_alpha(1.0)
    ax_map.set_title("Associated tracks (★ = deepest point of the truth track)", fontsize=9)

    # --- MSLP spaghetti vs time ---
    ax_ts = fig.add_subplot(gs[0, 0:2])
    for idx, role in enumerate(sources):
        color = role_color(role, idx)
        entries = members_by_role.get(role, [])
        labelled = False
        for _, tr in entries:
            if tr.empty:
                continue
            t = pd.to_datetime(tr["valid_time"], format=_VALID_TIME_FMT)
            ax_ts.plot(t, tr["mslp_hpa"], lw=0.9, alpha=0.6, color=color,
                       label=(None if labelled else f"{role_label(role, sources[role])} "
                                                    f"(n = {len(entries)} tracks)"))
            labelled = True
    if ref_time is not None:
        ax_ts.axvline(ref_time, color="0.5", lw=0.8, ls="--")
    ax_ts.invert_yaxis()
    ax_ts.set_ylabel(_MSLP_LABEL)
    ax_ts.set_xlabel("Valid date (month-day)")
    leg = ax_ts.legend(fontsize=8)
    for line in leg.get_lines():
        line.set_linewidth(2.0)
        line.set_alpha(1.0)
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax_ts.tick_params(axis="x", rotation=30, labelsize=8)
    ax_ts.set_title("MSLP along the associated tracks", fontsize=9)

    # --- per-member deepest MSLP strip plot ---
    ax_strip = fig.add_subplot(gs[0, 2])
    roles = list(sources)
    for idx, role in enumerate(roles):
        vals = np.array([float(meta["mslp_min_hpa"])
                         for meta, _ in members_by_role.get(role, [])
                         if meta.get("mslp_min_hpa") is not None and
                         meta["mslp_min_hpa"] == meta["mslp_min_hpa"]])
        if not len(vals):
            continue
        jitter = np.linspace(-0.18, 0.18, len(vals)) if len(vals) > 1 else np.array([0.0])
        ax_strip.plot(idx + jitter, vals, "o", ms=4, alpha=0.65,
                      color=role_color(role, idx))
        ax_strip.hlines(np.median(vals), idx - 0.28, idx + 0.28,
                        color=role_color(role, idx), lw=1.8)
    ax_strip.axhline(case["mslp_min_hpa"], color="0.5", lw=0.8, ls="--")
    ax_strip.set_xticks(range(len(roles)))
    ax_strip.set_xticklabels([role_name(r) for r in roles], fontsize=8)
    ax_strip.invert_yaxis()
    ax_strip.set_ylabel("Deepest MSLP of the track (hPa)", fontsize=8)
    ax_strip.grid(False, axis="x")
    ax_strip.set_title("Deepest MSLP per associated track\n(bar = median; dashed = deepest truth)",
                       fontsize=8)

    # --- stats box ---
    ax_stats = fig.add_subplot(gs[1, 2])
    ax_stats.axis("off")
    lines = ["Association statistics per source", ""]
    for role in roles:
        entries = members_by_role.get(role, [])
        vals, dts, dists = [], [], []
        for meta, _ in entries:
            v = meta.get("mslp_min_hpa")
            if v is not None and v == v:
                vals.append(float(v))
            try:
                t = pd.to_datetime(str(meta.get("mslp_min_valid_time")), format=_VALID_TIME_FMT)
                if ref_time is not None:
                    dts.append((t - ref_time).total_seconds() / 3600.0)
            except (ValueError, TypeError):
                pass
            la, lo = meta.get("mslp_min_lat"), meta.get("mslp_min_lon_e")
            if la is not None and lo is not None and la == la and lo == lo:
                dists.append(_haversine_km(float(la), float(lo),
                                           float(at["lat"]), float(at["lon_e"])))
        if not entries:
            lines.append(f"{role_name(role)}: no associated track")
            continue
        txt = f"{role_name(role)}: n = {len(entries)}"
        if vals:
            txt += f", deepest {min(vals):.0f}, median {np.median(vals):.0f} hPa"
        lines.append(txt)
        off = "    offset of deepest point from truth:"
        if dts:
            off += f" Δt med {np.median(dts):+.0f} h"
        if dists:
            off += f", Δx med {np.median(dists):.0f} km"
        if dts or dists:
            lines.append(off)
    ax_stats.text(0.0, 0.98, "\n".join(lines), fontsize=8, family="monospace",
                  va="top", ha="left", transform=ax_stats.transAxes)

    months_txt = ", ".join(month_text(m) for m in months) if months else "all months"
    fig.suptitle(f"Case {case['case_id']}: {case_label(case)} ({months_txt})")
    return fig


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def render_all(
    sources,
    metrics,
    months,
    basins,
    out_dir: Path,
    *,
    per_month: bool = False,
    case_basins: list[str] | None = None,
    top_k_cases: int = 3,
) -> list[Path]:
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    footer = _footer_text(sources, metrics.get("support", {}))
    focus = "atl" if "atl" in basins else basins[0]
    others = [b for b in basins if b != focus]
    if case_basins is None:
        case_basins = [focus]
    case_basins = [b for b in case_basins if b in basins] or [focus]
    scope_label = _scope_display("all", months)

    # Report 1 — the headline product: one tc-style distribution page per basin.
    dist_pages: list[tuple[str, plt.Figure]] = [
        (f"dist_{basin}", page_basin_distributions(sources, months, basin, scope_label))
        for basin in basins
    ]

    # Report 2 — everything else (overview table, consolidated grids, cases).
    diag_pages: list[tuple[str, plt.Figure]] = []
    diag_pages.append(("page1_overview", page_overview(sources, metrics, months, basins, focus)))
    diag_pages.append((f"page2_{focus}_all_tcs", page_basin_grid(sources, metrics, months, focus, "all")))
    if others:
        diag_pages.append(("page3_other_basins",
                           page_other_basins(sources, metrics, months, others, "all")))
    if per_month and len(months) > 1:
        for month in months:
            diag_pages.append((f"month_{focus}_{month}",
                               page_basin_grid(sources, metrics, [month], focus, month)))
    for basin in case_basins:
        for case in select_cases(sources, months, basin, top_k=top_k_cases):
            diag_pages.append((f"case_{case['case_id']}", page_case(sources, case, months)))

    paths: list[Path] = []
    reports = [
        (out_dir / "tc_tracks_report.pdf", dist_pages),
        (out_dir / "tc_tracks_diagnostics.pdf", diag_pages),
    ]
    for pdf_path, pages in reports:
        with FigureBook(pdf_path) as book:
            for stem, fig in pages:
                fig.text(0.01, 0.005, footer, fontsize=6, color="0.35", ha="left", va="bottom")
                # figures/<stem>.png (150 dpi) plus its PDF sibling; the page keeps its A4 size
                png, _pdf = save_figure(fig, figures_dir / stem, tight=False)
                book.add(fig, name=stem, tight=False)
                paths.append(png)
        LOG.info("%s: %d pages", pdf_path, len(pages))
    return paths + [p for p, _ in reports]
