"""Figure suite for track-based TC comparison (F1..F6).

Every figure carries a provenance footer (support contract + member sets +
completeness) and an explicit completeness annotation when any source is
below 100%. Role colors are fixed so the same role reads the same across
every figure: target black, input blue, ctrl grey, model red; extra roles
cycle tab10.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
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


def _finish(fig, out_path: Path, footer: str) -> Path:
    fig.text(0.01, 0.005, footer, fontsize=6, color="0.35", ha="left", va="bottom")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _map_axes(fig, nrows: int, ncols: int, extent: tuple[float, float, float, float]):
    """Cartopy axes when available, plain lat/lon axes otherwise."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        proj = ccrs.PlateCarree(central_longitude=(extent[0] + extent[1]) / 2 % 360)
        axes = []
        for i in range(nrows * ncols):
            ax = fig.add_subplot(nrows, ncols, i + 1, projection=proj)
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.coastlines(linewidth=0.4, color="0.4")
            ax.add_feature(cfeature.LAND, facecolor="0.93", zorder=0)
            axes.append(ax)
        return axes, ccrs.PlateCarree()
    except Exception:  # cartopy unavailable — degrade to plain axes
        axes = []
        for i in range(nrows * ncols):
            ax = fig.add_subplot(nrows, ncols, i + 1)
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_aspect("auto")
            axes.append(ax)
        return axes, None


def _scoped_records(src: dict[str, Any], months, basin) -> pd.DataFrame:
    rec = src["records"]
    if rec.empty:
        return rec
    return rec[scope_mask(rec, months, [basin])]


# ---------------------------------------------------------------------------
# F1 track maps
# ---------------------------------------------------------------------------

def plot_track_maps(sources, months, basin, scope_name, out_dir, footer, max_tracks=300):
    extent = BASIN_EXTENTS.get(basin, (0, 360, -60, 60))
    roles = list(sources)
    ncols = min(2, len(roles))
    nrows = int(np.ceil(len(roles) / ncols))
    fig = plt.figure(figsize=(7.5 * ncols, 4.0 * nrows + 0.6))
    axes, transform = _map_axes(fig, nrows, ncols, extent)
    for idx, role in enumerate(roles):
        ax = axes[idx]
        rec = _scoped_records(sources[role], months, basin)
        n_total = rec.groupby(["init_date", "member", "track_id"]).ngroups if not rec.empty else 0
        subtitle = f"{role} — {n_total} tracks"
        if not rec.empty:
            groups = list(rec.groupby(["init_date", "member", "track_id"]))
            if len(groups) > max_tracks:
                stride = int(np.ceil(len(groups) / max_tracks))
                groups = groups[::stride]
                subtitle += f" (drawn 1/{stride})"
            kwargs = {"transform": transform} if transform is not None else {}
            for _, g in groups:
                lon = g["lon_e"].to_numpy() % 360.0
                lon = np.where(lon < extent[0] - 10, lon + 360, lon)
                ax.plot(lon, g["lat"], lw=0.5, alpha=0.35, color=role_color(role, idx), **kwargs)
                ax.plot(lon[:1], g["lat"].to_numpy()[:1], ".", ms=2.5,
                        color=role_color(role, idx), alpha=0.7, **kwargs)
        ax.set_title(subtitle, fontsize=10)
    for ax in axes[len(roles):]:
        ax.set_visible(False)
    fig.suptitle(f"TC tracks — {basin.upper()} — {scope_name}", fontsize=12)
    return _finish(fig, out_dir / f"track_maps_{basin}_{scope_name}.png", footer)


# ---------------------------------------------------------------------------
# F2 density maps (+ diff vs target)
# ---------------------------------------------------------------------------

def plot_density_maps(sources, months, basin, scope_name, out_dir, footer):
    extent = BASIN_EXTENTS.get(basin, (0, 360, -60, 60))
    grids: dict[str, np.ndarray] = {}
    n_fc: dict[str, int] = {}
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
    roles = list(sources)
    has_target = "target" in grids
    diff_roles = [r for r in roles if r != "target"] if has_target else []
    ncols = max(len(roles), len(diff_roles)) or 1
    nrows = 1 + (1 if diff_roles else 0)
    fig = plt.figure(figsize=(5.5 * ncols, 3.2 * nrows + 0.8))
    axes, transform = _map_axes(fig, nrows, ncols, extent)
    lat_c = (edges[0][:-1] + edges[0][1:]) / 2
    lon_c = (edges[1][:-1] + edges[1][1:]) / 2
    vmax = max(g.max() for g in grids.values()) or 1.0
    kwargs = {"transform": transform} if transform is not None else {}
    for idx, role in enumerate(roles):
        ax = axes[idx]
        pcm = ax.pcolormesh(lon_c, lat_c, np.where(grids[role] > 0, grids[role], np.nan),
                            vmin=0, vmax=vmax, cmap="viridis", **kwargs)
        ax.set_title(f"{role} (n_fc={n_fc[role]})", fontsize=9)
    fig.colorbar(pcm, ax=axes[:len(roles)], shrink=0.8, label="track points / forecast / 2°")
    if diff_roles:
        dmax = max(abs(grids[r] - grids["target"]).max() for r in diff_roles) or 1.0
        for j, role in enumerate(diff_roles):
            ax = axes[ncols + j]
            pcm = ax.pcolormesh(lon_c, lat_c, grids[role] - grids["target"],
                                vmin=-dmax, vmax=dmax, cmap="RdBu_r", **kwargs)
            ax.set_title(f"{role} − target", fontsize=9)
        fig.colorbar(pcm, ax=axes[ncols:ncols + len(diff_roles)], shrink=0.8, label="Δ density")
    for ax in axes[len(roles):ncols]:
        ax.set_visible(False)
    if diff_roles:
        for ax in axes[ncols + len(diff_roles):]:
            ax.set_visible(False)
    fig.suptitle(f"Track density — {basin.upper()} — {scope_name}", fontsize=12)
    return _finish(fig, out_dir / f"density_{basin}_{scope_name}.png", footer)


# ---------------------------------------------------------------------------
# F3 intensity distributions (log-PDF + ratio vs target)
# ---------------------------------------------------------------------------

def plot_intensity_pdfs(sources, months, basin, scope_name, out_dir, footer):
    paths = []
    for stat, bins, label in (
        ("mslp_min_hpa", MSLP_BINS, "lifetime min MSLP [hPa]"),
        ("wind_max_ms", WIND_BINS, "lifetime max wind [m/s]"),
    ):
        hists: dict[str, np.ndarray] = {}
        counts: dict[str, int] = {}
        for role, src in sources.items():
            summ = src["summary"]
            vals = (summ[scope_mask(summ, months, [basin])][stat].dropna().to_numpy(dtype=float)
                    if not summ.empty else np.array([]))
            hist, _ = np.histogram(vals, bins=bins, density=True)
            hists[role] = hist
            counts[role] = len(vals)
        centers = (bins[:-1] + bins[1:]) / 2
        has_target = counts.get("target", 0) > 0
        ncols = 2 if has_target else 1
        fig, axs = plt.subplots(1, ncols, figsize=(6.2 * ncols, 4.2), squeeze=False)
        ax = axs[0][0]
        for idx, (role, hist) in enumerate(hists.items()):
            ax.semilogy(centers, np.where(hist > 0, hist, np.nan), lw=1.6,
                        color=role_color(role, idx), label=f"{role} (n={counts[role]})")
        ax.set_xlabel(label)
        ax.set_ylabel("PDF (log)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        if has_target:
            axr = axs[0][1]
            tgt = hists["target"]
            for idx, (role, hist) in enumerate(hists.items()):
                if role == "target":
                    continue
                ratio = np.where(tgt > 0, hist / np.where(tgt > 0, tgt, np.nan), np.nan)
                axr.plot(centers, ratio, lw=1.6, color=role_color(role, idx), label=role)
            axr.axhline(1.0, color="k", lw=0.8)
            axr.set_xlabel(label)
            axr.set_ylabel("PDF ratio vs target")
            axr.set_ylim(0, 3.5)
            axr.legend(fontsize=8)
            axr.grid(alpha=0.25)
        fig.suptitle(f"{label} — {basin.upper()} — {scope_name}", fontsize=12)
        stem = "mslp" if "MSLP" in label else "wind"
        paths.append(_finish(fig, out_dir / f"intensity_pdf_{stem}_{basin}_{scope_name}.png", footer))
    return paths


# ---------------------------------------------------------------------------
# F4 counts
# ---------------------------------------------------------------------------

def plot_counts(metrics_rows, months, basins, out_dir, footer):
    df = pd.DataFrame(metrics_rows)
    df = df[df["scope"].isin(months)]
    if df.empty:
        return []
    paths = []
    for basin in basins:
        sub = df[df["basin"] == basin]
        if sub.empty:
            continue
        roles = list(dict.fromkeys(sub["role"]))
        fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.2))
        width = 0.8 / max(1, len(roles))
        x = np.arange(len(months))
        for idx, role in enumerate(roles):
            rsub = sub[sub["role"] == role].set_index("scope").reindex(months)
            vals = rsub["tracks_per_forecast"].astype(float).to_numpy()
            cis = rsub["tracks_per_forecast_ci"]
            err = np.array([
                [v - ci[0], ci[1] - v] if isinstance(ci, (list, tuple)) and v == v else [0, 0]
                for v, ci in zip(vals, cis)
            ]).T
            axs[0].bar(x + idx * width, np.nan_to_num(vals), width,
                       yerr=err, capsize=2, color=role_color(role, idx), label=role)
        axs[0].set_xticks(x + 0.4 - width / 2)
        axs[0].set_xticklabels(months)
        axs[0].set_ylabel("tracks / forecast")
        axs[0].legend(fontsize=8)
        axs[0].grid(alpha=0.25, axis="y")
        # classification mix pooled over scope months
        cls_rows = []
        for role in roles:
            pooled: dict[str, int] = {}
            for _, row in sub[sub["role"] == role].iterrows():
                for k, v in (row.get("classification_counts") or {}).items():
                    pooled[k] = pooled.get(k, 0) + int(v)
            total = sum(pooled.values()) or 1
            cls_rows.append({"role": role, **{k: v / total for k, v in pooled.items()}})
        cls_df = pd.DataFrame(cls_rows).set_index("role").fillna(0.0)
        order = [c for c in ["HR3", "HR2", "HR1", "TS", "TD", "ET", "SSD", "unknown"] if c in cls_df]
        bottom = np.zeros(len(cls_df))
        for cat in order:
            axs[1].bar(cls_df.index, cls_df[cat], bottom=bottom, label=cat)
            bottom += cls_df[cat].to_numpy()
        axs[1].set_ylabel("classification fraction")
        axs[1].legend(fontsize=7, ncol=2)
        fig.suptitle(f"Track counts & classification — {basin.upper()}", fontsize=12)
        paths.append(_finish(fig, out_dir / f"counts_{basin}.png", footer))
    return paths


# ---------------------------------------------------------------------------
# F5 step-conditional intensity
# ---------------------------------------------------------------------------

def plot_step_intensity(sources, months, basin, scope_name, out_dir, footer):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
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
        ax.plot(steps, med, lw=1.8, color=color, label=role)
        ax.fill_between(steps, q10, q90, color=color, alpha=0.12)
        drew = True
    if not drew:
        plt.close(fig)
        return None
    ax.set_xlabel("forecast step [h]")
    ax.set_ylabel("track-point MSLP [hPa] (median, q10–q90)")
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.suptitle(f"Intensity vs lead time — {basin.upper()} — {scope_name}", fontsize=12)
    return _finish(fig, out_dir / f"step_intensity_{basin}_{scope_name}.png", footer)


# ---------------------------------------------------------------------------
# F6 case panels
# ---------------------------------------------------------------------------

def plot_cases(sources, months, basin, scope_name, out_dir, footer, top_k=3):
    cases = select_cases(sources, months, basin, top_k=top_k)
    paths = []
    for case in cases:
        fig, ax = plt.subplots(figsize=(8.0, 4.5))
        for idx, (role, src) in enumerate(sources.items()):
            rec = src["records"]
            if rec.empty:
                continue
            members = case["members"].get(role) or []
            color = role_color(role, idx)
            labelled = False
            for m in members:
                tr = rec[(rec["init_date"] == str(m["init_date"]))
                         & (rec["member"] == m["member"])
                         & (rec["track_id"] == m["track_id"])]
                if tr.empty:
                    continue
                t = pd.to_datetime(tr["valid_time"], format="%Y/%m/%d/%H")
                ax.plot(t, tr["mslp_hpa"], lw=0.9, alpha=0.6, color=color,
                        label=(None if labelled else f"{role} ({len(members)} tracks)"))
                labelled = True
        ax.invert_yaxis()
        ax.set_ylabel("MSLP [hPa]")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        at = case["at"]
        fig.suptitle(
            f"Case {case['case_id']} — deepest {case['reference_role']} "
            f"{case['mslp_min_hpa']:.0f} hPa @ ({at['lat']:.1f}, {at['lon_e']:.1f}) {at['valid_time']}",
            fontsize=11,
        )
        fig.autofmt_xdate()
        paths.append(_finish(fig, out_dir / f"case_{case['case_id']}_{scope_name}.png", footer))
    return paths


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def render_all(sources, metrics, months, basins, out_dir: Path) -> list[Path]:
    from matplotlib.backends.backend_pdf import PdfPages

    figures_dir = out_dir / "figures"
    footer = _footer_text(sources, metrics["support"])
    paths: list[Path] = []
    scopes: list[tuple[str, list[str]]] = [(m, [m]) for m in months]
    if len(months) != 1:
        scopes.append(("all", months))
    for basin in basins:
        for scope_name, scope_months in scopes:
            paths.append(plot_track_maps(sources, scope_months, basin, scope_name, figures_dir, footer))
            paths.append(plot_density_maps(sources, scope_months, basin, scope_name, figures_dir, footer))
            paths.extend(plot_intensity_pdfs(sources, scope_months, basin, scope_name, figures_dir, footer))
            p = plot_step_intensity(sources, scope_months, basin, scope_name, figures_dir, footer)
            if p:
                paths.append(p)
        paths.extend(plot_cases(sources, months, basin, "all", figures_dir, footer))
    paths.extend(plot_counts(metrics["metrics"], months, basins, figures_dir, footer))
    paths = [p for p in paths if p]

    pdf_path = out_dir / "tc_tracks_report.pdf"
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.06, 0.9, "TC track comparison report", fontsize=18, weight="bold")
        fig.text(0.06, 0.84, footer.replace(" | ", "\n"), fontsize=9, va="top")
        pdf.savefig(fig)
        plt.close(fig)
        for path in paths:
            image = plt.imread(path)
            fig = plt.figure(figsize=(11, 11 * image.shape[0] / image.shape[1]))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(image)
            ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)
    return paths + [pdf_path]
