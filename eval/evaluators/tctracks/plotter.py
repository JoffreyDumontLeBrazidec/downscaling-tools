"""Figure suite for track-based TC comparison — page-oriented report.

The report is a small set of dense landscape pages instead of one figure per
statistic:

  P1  overview     — headline table (all basins/roles) + focus-basin lifetime
                     min-MSLP log-PDF with a ratio-vs-target inset
  P2  focus basin  — consolidated grid: density diffs vs target, intensity vs
                     lead time, lifetime max-wind PDF, counts, classification
  P3  other basins — the same grid, one compact row per remaining basin
  P4+ case pages   — one page per selected storm (deepest reference tracks):
                     single map with every role's associated tracks, MSLP
                     spaghetti, per-member deepest-MSLP strip plot, stats box

Every page carries a provenance footer (support contract + member sets +
completeness). Role colors are fixed so the same role reads the same across
every figure: target black, input blue, ctrl grey, model red; extra roles
cycle a fallback palette. Diagnostic panel only: pages show distributions and
references side by side and never verdict language.

Multi-month runs default to pooled ("all") pages; per-month focus-basin pages
only render behind ``per_month=True``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .scorer import DENSITY_BIN_DEG, TS_WIND_MS, density_grid, scope_mask, select_cases

LOG = logging.getLogger(__name__)

ROLE_COLORS = {
    "target": "#000000",
    "input": "#1f77b4",
    "ctrl": "#7f7f7f",
    "model": "#d62728",
}
_EXTRA_CYCLE = ["#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#bcbd22", "#17becf"]

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


def _footer_text(sources: dict[str, dict[str, Any]], support: dict[str, Any]) -> str:
    contract = support.get("contract", {})
    parts = [
        f"support {contract.get('grid_support')} steps {contract.get('steps')} "
        f"vort {contract.get('vorticity')}",
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

def _add_map_ax(fig, spec, extent):
    """One map axes on a gridspec slot; cartopy when available."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        proj = ccrs.PlateCarree(central_longitude=(extent[0] + extent[1]) / 2 % 360)
        ax = fig.add_subplot(spec, projection=proj)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.4, color="0.4")
        ax.add_feature(cfeature.LAND, facecolor="0.93", zorder=0)
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


def _draw_density_diffs(fig, specs, sources, months, basin, *, title_size=9):
    """Density difference maps vs target (or absolute maps without a target).

    Returns the list of map axes; one spec is consumed per drawn panel.
    """
    extent = BASIN_EXTENTS.get(basin, (0, 360, -60, 60))
    grids, n_fc, lat_c, lon_c = _density_per_forecast(sources, months, basin)
    axes = []
    if "target" in grids:
        roles = [r for r in grids if r != "target"][: len(specs)]
        dmax = max((abs(grids[r] - grids["target"]).max() for r in roles), default=0) or 1.0
        pcm = None
        for spec, role in zip(specs, roles):
            ax, transform = _add_map_ax(fig, spec, extent)
            kwargs = {"transform": transform} if transform is not None else {}
            pcm = ax.pcolormesh(lon_c, lat_c, grids[role] - grids["target"],
                                vmin=-dmax, vmax=dmax, cmap="RdBu_r", **kwargs)
            ax.set_title(f"{role} − target track density", fontsize=title_size)
            axes.append(ax)
        if axes:
            fig.colorbar(pcm, ax=axes, shrink=0.85, pad=0.015,
                         label="Δ track points / forecast / 2°")
    else:  # no target — absolute densities
        roles = list(grids)[: len(specs)]
        vmax = max((grids[r].max() for r in roles), default=0) or 1.0
        pcm = None
        for spec, role in zip(specs, roles):
            ax, transform = _add_map_ax(fig, spec, extent)
            kwargs = {"transform": transform} if transform is not None else {}
            pcm = ax.pcolormesh(lon_c, lat_c,
                                np.where(grids[role] > 0, grids[role], np.nan),
                                vmin=0, vmax=vmax, cmap="viridis", **kwargs)
            ax.set_title(f"{role} track density (n_fc={n_fc[role]})", fontsize=title_size)
            axes.append(ax)
        if axes:
            fig.colorbar(pcm, ax=axes, shrink=0.85, pad=0.015,
                         label="track points / forecast / 2°")
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
        ax.plot(steps, med, lw=1.6, color=color, label=role)
        ax.fill_between(steps, q10, q90, color=color, alpha=0.12)
        drew = True
    ax.set_xlabel("forecast step [h]", fontsize=label_size)
    ax.set_ylabel("track MSLP [hPa]\n(median, q10–q90)", fontsize=label_size)
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
                        *, ratio_inset=True, label_size=8):
    """Log-PDF of a lifetime statistic per role, ratio-vs-target inset."""
    values = _lifetime_values(sources, months, basin, stat)
    centers = (bins[:-1] + bins[1:]) / 2
    hists = {}
    for idx, (role, vals) in enumerate(values.items()):
        hist, _ = np.histogram(vals, bins=bins, density=True)
        hists[role] = hist
        ax.semilogy(centers, np.where(hist > 0, hist, np.nan), lw=1.6,
                    color=role_color(role, idx), label=f"{role} (n={len(vals)})")
    ax.set_xlabel(xlabel, fontsize=label_size)
    ax.set_ylabel("PDF (log)", fontsize=label_size)
    ax.tick_params(labelsize=label_size - 1)
    # bulk of both distributions sits at the benign end: MSLP right, wind left
    ax.legend(fontsize=label_size - 1,
              loc=("lower left" if stat == "wind_max_ms" else "upper left"))
    ax.grid(alpha=0.25)
    if ratio_inset and len(values.get("target", ())) > 0:
        loc = [0.58, 0.64, 0.39, 0.32] if stat == "wind_max_ms" else [0.55, 0.66, 0.42, 0.30]
        axr = ax.inset_axes(loc)
        tgt = hists["target"]
        for idx, (role, hist) in enumerate(hists.items()):
            if role == "target":
                continue
            ratio = np.where(tgt > 0, hist / np.where(tgt > 0, tgt, np.nan), np.nan)
            axr.plot(centers, ratio, lw=1.2, color=role_color(role, idx))
        axr.axhline(1.0, color="k", lw=0.7)
        axr.set_ylim(0, 3.5)
        axr.tick_params(labelsize=label_size - 3)
        axr.set_title("ratio vs target", fontsize=label_size - 2)
        axr.grid(alpha=0.2)


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
               color=role_color(role, idx), label=role)
    ax.set_xticks(x + 0.4 - width / 2)
    ax.set_xticklabels(["tracks/fc", "TC-days/fc"], fontsize=label_size)
    ax.tick_params(labelsize=label_size - 1)
    ax.grid(alpha=0.25, axis="y")
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.25)  # headroom so the legend clears the bars
    ax.legend(fontsize=label_size - 2)


def _draw_classification(ax, metrics, basin, scope, *, label_size=8):
    rows = _basin_metric_rows(metrics, basin, scope)
    cls_rows = []
    for role, row in rows.items():
        counts = row.get("classification_counts")
        if not isinstance(counts, dict):
            counts = {}
        total = sum(counts.values()) or 1
        cls_rows.append({"role": role, **{k: v / total for k, v in counts.items()}})
    if not cls_rows:
        ax.axis("off")
        return
    cls_df = pd.DataFrame(cls_rows).set_index("role").fillna(0.0)
    order = [c for c in ["HR5", "HR4", "HR3", "HR2", "HR1", "TS", "TD", "ET", "SSD", "unknown"]
             if c in cls_df]
    bottom = np.zeros(len(cls_df))
    for cat in order:
        ax.bar(cls_df.index, cls_df[cat], bottom=bottom, label=cat)
        bottom += cls_df[cat].to_numpy()
    ax.set_ylabel("classification fraction", fontsize=label_size)
    ax.tick_params(labelsize=label_size - 1)
    ax.tick_params(axis="x", rotation=45)
    if order:
        ax.legend(fontsize=label_size - 3, loc="center left", bbox_to_anchor=(1.0, 0.5))


# ---------------------------------------------------------------------------
# P1 overview
# ---------------------------------------------------------------------------

def _fmt_ci(value, ci, fmt="{:.1f}") -> str:
    if value is None:
        return "—"
    txt = fmt.format(value)
    if isinstance(ci, (list, tuple)) and len(ci) == 2:
        txt += f" [{fmt.format(ci[0])}, {fmt.format(ci[1])}]"
    return txt


def page_overview(sources, metrics, months, basins, focus_basin) -> plt.Figure:
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.text(0.04, 0.94, "TC track comparison report", fontsize=17, weight="bold")
    scope = _headline_scope(metrics)
    subtitle = (f"months {', '.join(months)} — basins {', '.join(b.upper() for b in basins)}"
                f" — scope {scope} — diagnostic panel (distributions vs references)")
    fig.text(0.04, 0.905, subtitle, fontsize=9, color="0.25")
    footer = _footer_text(sources, metrics.get("support", {}))
    fig.text(0.04, 0.885, footer.replace(" | ", "\n"), fontsize=6.5, color="0.35", va="top")

    # headline table, one row per (basin, role)
    col_labels = ["basin", "role", "fc", "tracks/fc [95% CI]",
                  "min MSLP p5 [95% CI]\n[hPa]", "deepest\n[hPa]",
                  "wind p95\n[m/s]", "TC-days\n/fc"]
    cells, row_colors = [], []
    for basin in basins:
        rows = _basin_metric_rows(metrics, basin, scope)
        for role, row in rows.items():
            cells.append([
                basin.upper(), role,
                str(row.get("n_forecasts", "—")),
                _fmt_ci(row.get("tracks_per_forecast"), row.get("tracks_per_forecast_ci"), "{:.2f}"),
                _fmt_ci(row.get("mslp_p5"), row.get("mslp_p5_ci"), "{:.1f}"),
                f"{row['mslp_min']:.0f}" if row.get("mslp_min") is not None else "—",
                f"{row['wind_p95']:.1f}" if row.get("wind_p95") is not None else "—",
                f"{row['tc_days_per_forecast']:.2f}" if row.get("tc_days_per_forecast") is not None else "—",
            ])
            row_colors.append("#f3f3f3" if (basins.index(basin) % 2) else "white")
    ax_tab = fig.add_axes([0.03, 0.06, 0.52, 0.72])
    ax_tab.axis("off")
    if cells:
        col_widths = [0.08, 0.09, 0.05, 0.21, 0.23, 0.10, 0.10, 0.09]
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
                    cell.set_text_props(color=role_color(cells[r - 1][1]), weight="bold")
    ax_tab.set_title(f"headline statistics — scope {scope}", fontsize=10)

    # focus-basin deep-tail PDF with ratio inset
    ax_pdf = fig.add_axes([0.62, 0.12, 0.35, 0.64])
    _draw_intensity_pdf(ax_pdf, sources, months, focus_basin,
                        "mslp_min_hpa", MSLP_BINS, "lifetime min MSLP [hPa]",
                        label_size=9)
    ax_pdf.set_title(f"{focus_basin.upper()} lifetime min MSLP — the deep tail",
                     fontsize=10)
    return fig


# ---------------------------------------------------------------------------
# P2 focus-basin consolidated grid
# ---------------------------------------------------------------------------

def _scope_display(scope_name, months) -> str:
    if scope_name != "all":
        return scope_name
    return months[0] if len(months) == 1 else f"all months ({', '.join(months)})"


def page_basin_grid(sources, metrics, months, basin, scope_name) -> plt.Figure:
    fig = plt.figure(figsize=PAGE_SIZE, layout="constrained")
    _reserve_footer(fig)
    gs = fig.add_gridspec(2, 6, height_ratios=[1.25, 1.0])
    n_diffs = min(3, max(1, len([r for r in sources if r != "target"]) or len(sources)))
    specs = [gs[0, 2 * i:2 * i + 2] for i in range(n_diffs)]
    _draw_density_diffs(fig, specs, sources, months, basin)

    ax_step = fig.add_subplot(gs[1, 0:2])
    _draw_step_intensity(ax_step, sources, months, basin)
    ax_step.set_title("intensity vs lead time", fontsize=9)

    ax_wind = fig.add_subplot(gs[1, 2:4])
    _draw_intensity_pdf(ax_wind, sources, months, basin,
                        "wind_max_ms", WIND_BINS, "lifetime max wind [m/s]")
    ax_wind.set_title("lifetime max wind", fontsize=9)

    scope = scope_name if scope_name in (metrics.get("scopes") or []) else _headline_scope(metrics)
    ax_counts = fig.add_subplot(gs[1, 4])
    _draw_counts(ax_counts, metrics, basin, scope)
    ax_counts.set_title("counts per forecast", fontsize=9)
    ax_cls = fig.add_subplot(gs[1, 5])
    _draw_classification(ax_cls, metrics, basin, scope)
    ax_cls.set_title("classification mix", fontsize=9)

    fig.suptitle(f"{basin.upper()} — all TCs — {_scope_display(scope_name, months)}",
                 fontsize=13)
    return fig


# ---------------------------------------------------------------------------
# P3 other basins, one compact row each
# ---------------------------------------------------------------------------

def page_other_basins(sources, metrics, months, other_basins, scope_name) -> plt.Figure:
    nrows = len(other_basins)
    fig = plt.figure(figsize=PAGE_SIZE, layout="constrained")
    _reserve_footer(fig)
    gs = fig.add_gridspec(nrows, 4)
    for i, basin in enumerate(other_basins):
        ax_pdf = fig.add_subplot(gs[i, 0])
        _draw_intensity_pdf(ax_pdf, sources, months, basin,
                            "mslp_min_hpa", MSLP_BINS, "lifetime min MSLP [hPa]",
                            ratio_inset=False, label_size=7)
        ax_pdf.set_title(f"{basin.upper()} — min MSLP", fontsize=9)
        _draw_density_diffs(fig, [gs[i, 1], gs[i, 2]], sources, months, basin,
                            title_size=8)
        ax_step = fig.add_subplot(gs[i, 3])
        _draw_step_intensity(ax_step, sources, months, basin, label_size=7)
        ax_step.set_title(f"{basin.upper()} — vs lead time", fontsize=9)
    fig.suptitle(
        f"Other basins (context) — {', '.join(b.upper() for b in other_basins)}"
        f" — {_scope_display(scope_name, months)}",
        fontsize=13,
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
                        label=(None if labelled else role), **kwargs)
            ax_map.plot(lon[:1], tr["lat"].to_numpy()[:1], ".", ms=3,
                        color=color, alpha=0.8, **kwargs)
            labelled = True
    ax_map.plot([ref_lon], [float(at["lat"])], marker="*", ms=14,
                color=role_color(case.get("reference_role", "target")),
                mec="white", mew=0.6, zorder=5, **kwargs)
    ax_map.legend(fontsize=8, loc="upper left")
    ax_map.set_title("associated tracks (★ = reference deepest point)", fontsize=9)

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
                       label=(None if labelled else f"{role} ({len(entries)} tracks)"))
            labelled = True
    if ref_time is not None:
        ax_ts.axvline(ref_time, color="0.5", lw=0.8, ls="--")
    ax_ts.invert_yaxis()
    ax_ts.set_ylabel("MSLP [hPa]")
    ax_ts.grid(alpha=0.25)
    ax_ts.legend(fontsize=8)
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax_ts.tick_params(axis="x", rotation=30, labelsize=8)
    ax_ts.set_title("associated-track MSLP vs valid time", fontsize=9)

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
    ax_strip.set_xticklabels(roles, fontsize=8)
    ax_strip.invert_yaxis()
    ax_strip.set_ylabel("track deepest MSLP [hPa]", fontsize=8)
    ax_strip.grid(alpha=0.25, axis="y")
    ax_strip.set_title("deepest MSLP per associated track\n(median bar; -- = reference deepest)",
                       fontsize=8)

    # --- stats box ---
    ax_stats = fig.add_subplot(gs[1, 2])
    ax_stats.axis("off")
    lines = ["per-role association stats", ""]
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
            lines.append(f"{role}: no associated track")
            continue
        txt = f"{role}: n={len(entries)}"
        if vals:
            txt += f", deepest {min(vals):.0f}, median {np.median(vals):.0f} hPa"
        lines.append(txt)
        off = "    deepest-point offset vs target:"
        if dts:
            off += f" Δt med {np.median(dts):+.0f} h"
        if dists:
            off += f", Δx med {np.median(dists):.0f} km"
        if dts or dists:
            lines.append(off)
    ax_stats.text(0.0, 0.98, "\n".join(lines), fontsize=8, family="monospace",
                  va="top", ha="left", transform=ax_stats.transAxes)

    months_txt = ", ".join(months) if months else "all"
    fig.suptitle(f"Case {case['case_id']} — {case_label(case)} — scope {months_txt}",
                 fontsize=12)
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
    from matplotlib.backends.backend_pdf import PdfPages

    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    footer = _footer_text(sources, metrics.get("support", {}))
    focus = "atl" if "atl" in basins else basins[0]
    others = [b for b in basins if b != focus]
    if case_basins is None:
        case_basins = [focus]
    case_basins = [b for b in case_basins if b in basins] or [focus]

    pages: list[tuple[str, plt.Figure]] = []
    pages.append(("page1_overview", page_overview(sources, metrics, months, basins, focus)))
    pages.append((f"page2_{focus}_all_tcs", page_basin_grid(sources, metrics, months, focus, "all")))
    if others:
        pages.append(("page3_other_basins",
                      page_other_basins(sources, metrics, months, others, "all")))
    if per_month and len(months) > 1:
        for month in months:
            pages.append((f"month_{focus}_{month}",
                          page_basin_grid(sources, metrics, [month], focus, month)))
    for basin in case_basins:
        for case in select_cases(sources, months, basin, top_k=top_k_cases):
            pages.append((f"case_{case['case_id']}", page_case(sources, case, months)))

    paths: list[Path] = []
    pdf_path = out_dir / "tc_tracks_report.pdf"
    with PdfPages(pdf_path) as pdf:
        for stem, fig in pages:
            fig.text(0.01, 0.005, footer, fontsize=6, color="0.35", ha="left", va="bottom")
            png = figures_dir / f"{stem}.png"
            fig.savefig(png, dpi=150)
            pdf.savefig(fig)
            plt.close(fig)
            paths.append(png)
    LOG.info("report: %d pages -> %s", len(pages), pdf_path)
    return paths + [pdf_path]
