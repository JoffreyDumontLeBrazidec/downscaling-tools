"""Plots for the spread proxy: curves, ratio maps, and spread spectra.

Every figure is drawn in the house style of ``eval.plotting``. The model (ML) ensemble uses
the model role, the ENFO ensemble it is compared with the truth role, and the EEFO driving
ensemble the input role. Spreads are shown in the display unit of the variable (hPa for
pressure). Each multi-page PDF keeps its name and gets PNG copies of its pages in
``<name>_pages/``.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

DOMAIN_TEXT = {
    "global": "Globe",
    "n.hem": "Northern Hemisphere extratropics",
    "s.hem": "Southern Hemisphere extratropics",
    "tropics": "Tropics",
    "europe": "Europe",
}
CURVES = (  # (csv metric, role, legend label)
    ("spread_ml", "model", "model ensemble"),
    ("spread_enfo", "truth", "ENFO ensemble (truth)"),
    ("spread_input", "input", "EEFO input ensemble"),
)
_MARKER = {"model": "o", "truth": "s", "input": "^"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _unit_factor(field: str) -> tuple[str, float]:
    """Display unit of a spread (a difference) and the factor from the native unit."""
    from eval.plotting import convert_difference, variable_spec

    spec = variable_spec(field)
    return spec.unit, float(np.asarray(convert_difference(field, 1.0)))


def _page_name(field: str) -> str:
    return field


def plot_spread_curves(summary_csv: str | Path, output_pdf: str | Path,
                       *, title_prefix: str = "Spread proxy") -> Path:
    """One page per field: area-mean spread vs lead, ML vs ENFO, per domain."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from eval.plotting import AXIS, FigureBook, eval_style, role_style, variable_spec

    rows = _read_csv(Path(summary_csv))
    if not rows:
        raise ValueError(f"No rows found in {summary_csv}")

    by_field: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_field[row["weather_state"]].append(row)

    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with eval_style(), FigureBook(output_pdf, png=True) as book:
        for field in sorted(by_field):
            unit, factor = _unit_factor(field)
            name = variable_spec(field).name
            domains = sorted({r["domain"] for r in by_field[field]})
            n = len(domains)
            ncols = min(n, 3)
            nrows = int(np.ceil(n / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.9 * nrows),
                                     squeeze=False, layout="constrained")
            for ax in axes.flat[n:]:
                ax.axis("off")
            n_dates = set()
            for ax, domain in zip(axes.flat, domains):
                sub = [r for r in by_field[field] if r["domain"] == domain]
                n_dates.update(int(r["n_dates"]) for r in sub if r.get("n_dates"))
                for metric, role, label in CURVES:
                    pts = sorted(
                        (int(r["step"]), float(r["mean"]), float(r["stderr"]))
                        for r in sub if r["metric"] == metric
                    )
                    if not pts:
                        continue
                    x, y, err = zip(*pts)
                    ax.errorbar(x, np.asarray(y) * factor, yerr=np.asarray(err) * factor,
                                marker=_MARKER[role], ms=4, capsize=2, label=label,
                                **role_style(role))
                ratios = [float(r["mean"]) for r in sub if r["metric"] == "spread_ratio"]
                title = DOMAIN_TEXT.get(domain, domain)
                if ratios:
                    title += f"\nmean ratio model / ENFO {np.mean(ratios):.3f}"
                ax.set_title(title, fontsize=10)
                ax.set_xlabel(AXIS["lead"])
                ax.set_ylabel(f"Ensemble spread ({unit})" if unit else "Ensemble spread")
            handles, labels = axes.flat[0].get_legend_handles_labels()
            dates = f"n = {max(n_dates)} start dates per lead time" if n_dates else ""
            fig.legend(handles, [f"{lab} ({dates})" if dates else lab for lab in labels],
                       loc="outside lower center", ncol=len(labels))
            fig.suptitle(f"{title_prefix}: {name}, area-mean ensemble spread by lead time\n"
                         "error bars: standard error over start dates")
            book.add(fig, name=_page_name(field))
    return output_pdf


def plot_spread_maps(maps_npz: str | Path, output_pdf: str | Path,
                     *, title_prefix: str = "Spread proxy") -> Path:
    """One page per field: log2 of the ML/ENFO RMS-spread ratio on the coarse grid.

    All pages share one colour scale, symmetric about zero (equal spread), so that the
    fields can be compared with each other.
    """
    import matplotlib

    matplotlib.use("Agg")
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    from eval.plotting import FigureBook, add_geography, eval_style, extend_for, symmetric_norm, variable_spec

    data = np.load(Path(maps_npz))
    lat = data["lat_centers"]
    lon = data["lon_centers"]
    fields = sorted({k.split("__")[0] for k in data.files if k.endswith("__ml_var")})

    log2 = {}
    for field in fields:
        ml_var = data[f"{field}__ml_var"]
        enfo_var = data[f"{field}__enfo_var"]
        count = data[f"{field}__count"]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.sqrt(ml_var / enfo_var)
            log2[field] = np.where((count > 0) & (enfo_var > 0), np.log2(ratio), np.nan)
    # one scale for every page: the 99th percentile of |log2 ratio| over all fields, at least
    # a factor 2^0.5, rounded up to a quarter so the colour-bar ticks stay readable
    norm, lim = symmetric_norm(*log2.values(), q=99.0)
    lim = max(0.5, float(np.ceil(lim * 4.0) / 4.0))
    norm, lim = symmetric_norm(limit=lim)

    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with eval_style(), FigureBook(output_pdf, png=True) as book:
        for field in fields:
            log2r = log2[field]
            fig = plt.figure(figsize=(11.5, 6.4))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
            ax.set_global()
            mesh = ax.pcolormesh(lon, lat, log2r, cmap="RdBu_r", norm=norm, shading="nearest",
                                 transform=ccrs.PlateCarree(), rasterized=True)
            add_geography(ax, resolution="110m", borders=True)
            cbar = fig.colorbar(mesh, ax=ax, shrink=0.8, pad=0.03, extend=extend_for(norm, log2r))
            cbar.set_label("log₂(model spread / ENFO spread)\n+1 = model spread twice ENFO's, "
                           "0 = equal, −1 = half")
            title = f"{title_prefix}: {variable_spec(field).name}, ratio of model to ENFO ensemble spread"
            finite = np.isfinite(log2r)
            if np.any(finite):
                w = np.cos(np.deg2rad(lat))[:, None] * finite
                gmean = float(np.nansum(np.where(finite, log2r, 0.0) * w) / np.sum(w))
                title += (f"\nspread = root-mean-square over all start dates and lead times; "
                          f"area-weighted mean log₂ ratio {gmean:+.3f}")
            ax.set_title(title, fontsize=11)
            book.add(fig, name=_page_name(field))
    return output_pdf


def plot_spread_spectra(spectra_npz: str | Path, output_pdf: str | Path,
                        *, title_prefix: str = "Spread proxy") -> Path:
    """One page per field: deviation power spectra ML vs ENFO, plus their ratio.

    Colour marks the lead time (neutral sequence colours); the line style marks the source
    (solid = model, dashed = ENFO, dotted = EEFO input).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from eval.plotting import AXIS, FigureBook, eval_style, sequence_colors, variable_spec

    data = np.load(Path(spectra_npz))
    ell = data["ell"]
    keys = [k for k in data.files if k.endswith("__ml")]
    by_field: dict[str, list[int]] = defaultdict(list)
    for key in keys:
        field, step_part, _ = key.split("__")
        by_field[field].append(int(step_part.replace("step", "")))

    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with eval_style(), FigureBook(output_pdf, png=True) as book:
        for field in sorted(by_field):
            unit, factor = _unit_factor(field)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), layout="constrained")
            steps = sorted(by_field[field])
            colours = sequence_colors(len(steps))
            has_input = False
            n_dates = set()
            for colour, step in zip(colours, steps):
                cl_ml = data[f"{field}__step{step:03d}__ml"] * factor ** 2
                cl_enfo = data[f"{field}__step{step:03d}__enfo"] * factor ** 2
                nd_key = f"{field}__step{step:03d}__n_dates"
                if nd_key in data.files:
                    n_dates.add(int(np.asarray(data[nd_key]).ravel()[0]))
                sel = ell >= 1
                ax1.loglog(ell[sel], cl_ml[sel], "-", lw=1.8, color=colour)
                ax1.loglog(ell[sel], cl_enfo[sel], "--", lw=1.5, color=colour)
                input_key = f"{field}__step{step:03d}__input"
                if input_key in data.files:
                    has_input = True
                    ax1.loglog(ell[sel], data[input_key][sel] * factor ** 2, ":", lw=1.5,
                               color=colour)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ax2.semilogx(ell[sel], cl_ml[sel] / cl_enfo[sel], "-", lw=1.8,
                                 color=colour, label=f"lead time {step} h")
            ax1.set_xlabel(AXIS["wavenumber"])
            ax1.set_ylabel(f"Member-deviation power C_ℓ ({unit}²)" if unit else "Member-deviation power C_ℓ")
            ax1.set_title("Spectra of the member deviations from the ensemble mean")
            src = [Line2D([], [], color="0.2", ls="-", lw=1.8, label="model ensemble"),
                   Line2D([], [], color="0.2", ls="--", lw=1.5, label="ENFO ensemble")]
            if has_input:
                src.append(Line2D([], [], color="0.2", ls=":", lw=1.5, label="EEFO input ensemble"))
            lead = [Line2D([], [], color=c, lw=2.2, label=f"lead time {s} h")
                    for c, s in zip(colours, steps)]
            ax1.legend(handles=src + lead, fontsize=8, loc="lower left", ncol=2)
            ax1.grid(alpha=0.3, which="both")
            ax2.axhline(1.0, color="0.2", lw=0.9, ls=":")
            ax2.set_xlabel(AXIS["wavenumber"])
            ax2.set_ylabel("Spread power ratio (model / ENFO)")
            ax2.set_ylim(0, 3)
            ax2.set_title("Ratio of the spread power, scale by scale")
            ax2.legend(fontsize=8)
            ax2.grid(alpha=0.3, which="both")
            dates = f" (n = {max(n_dates)} start dates)" if n_dates else ""
            fig.suptitle(f"{title_prefix}: {variable_spec(field).name}, ensemble spread by scale{dates}")
            book.add(fig, name=_page_name(field))
    return output_pdf


def plot_all(results_dir: Path, plots_dir: Path, *, title_prefix: str = "Spread proxy") -> list[Path]:
    """Render every readout whose inputs exist; return the PDFs written."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    summary_csv = results_dir / "summary_by_lead.csv"
    if summary_csv.exists():
        written.append(plot_spread_curves(
            summary_csv, plots_dir / "spread_curves.pdf", title_prefix=title_prefix))
    maps_npz = results_dir / "spread_maps.npz"
    if maps_npz.exists():
        written.append(plot_spread_maps(
            maps_npz, plots_dir / "spread_ratio_maps.pdf", title_prefix=title_prefix))
    spectra_npz = results_dir / "spread_spectra.npz"
    if spectra_npz.exists():
        written.append(plot_spread_spectra(
            spectra_npz, plots_dir / "spread_spectra.pdf", title_prefix=title_prefix))
    return written
