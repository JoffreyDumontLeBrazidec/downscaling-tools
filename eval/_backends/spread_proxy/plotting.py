"""Plots for the spread proxy: curves, ratio maps, and spread spectra."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def plot_spread_curves(summary_csv: str | Path, output_pdf: str | Path,
                       *, title_prefix: str = "Spread proxy") -> Path:
    """One page per field: area-mean spread vs lead, ML vs ENFO, per domain."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    rows = _read_csv(Path(summary_csv))
    if not rows:
        raise ValueError(f"No rows found in {summary_csv}")

    by_field: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_field[row["weather_state"]].append(row)

    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        for field in sorted(by_field):
            domains = sorted({r["domain"] for r in by_field[field]})
            n = len(domains)
            ncols = min(n, 3)
            nrows = int(np.ceil(n / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows),
                                     squeeze=False)
            for ax in axes.flat[n:]:
                ax.axis("off")
            for ax, domain in zip(axes.flat, domains):
                sub = [r for r in by_field[field] if r["domain"] == domain]
                for metric, style, label in (("spread_ml", "-o", "ML"),
                                             ("spread_enfo", "--s", "ENFO"),
                                             ("spread_input", ":^", "EEFO input")):
                    pts = sorted(
                        (int(r["step"]), float(r["mean"]), float(r["stderr"]))
                        for r in sub if r["metric"] == metric
                    )
                    if not pts:
                        continue
                    x, y, err = zip(*pts)
                    ax.errorbar(x, y, yerr=err, fmt=style, ms=3, capsize=2, label=label)
                ratios = [float(r["mean"]) for r in sub if r["metric"] == "spread_ratio"]
                if ratios:
                    ax.set_title(f"{domain}  (mean ratio {np.mean(ratios):.3f})", fontsize=9)
                ax.set_xlabel("lead (h)", fontsize=8)
                ax.set_ylabel("spread", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(alpha=0.3)
                ax.legend(fontsize=7)
            fig.suptitle(f"{title_prefix} — {field}: area-mean ensemble spread vs lead")
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            pdf.savefig(fig)
            plt.close(fig)
    return output_pdf


def plot_spread_maps(maps_npz: str | Path, output_pdf: str | Path,
                     *, title_prefix: str = "Spread proxy") -> Path:
    """One page per field: log2 of the ML/ENFO RMS-spread ratio on the coarse grid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    data = np.load(Path(maps_npz))
    lat = data["lat_centers"]
    lon = data["lon_centers"]
    fields = sorted({k.split("__")[0] for k in data.files if k.endswith("__ml_var")})

    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        for field in fields:
            ml_var = data[f"{field}__ml_var"]
            enfo_var = data[f"{field}__enfo_var"]
            count = data[f"{field}__count"]
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.sqrt(ml_var / enfo_var)
                log2r = np.where((count > 0) & (enfo_var > 0), np.log2(ratio), np.nan)
            fig, ax = plt.subplots(figsize=(11, 5.5))
            mesh = ax.pcolormesh(lon, lat, log2r, cmap="RdBu_r", vmin=-1.0, vmax=1.0,
                                 shading="nearest")
            cbar = fig.colorbar(mesh, ax=ax, shrink=0.85)
            cbar.set_label("log2(spread_ML / spread_ENFO)   [+1 = 2x, 0 = equal]")
            finite = np.isfinite(log2r)
            if np.any(finite):
                w = np.cos(np.deg2rad(lat))[:, None] * finite
                gmean = float(np.nansum(np.where(finite, log2r, 0.0) * w) / np.sum(w))
                ax.set_title(
                    f"{title_prefix} — {field}: spread ratio map "
                    f"(all dates+steps; area-mean log2 ratio {gmean:+.3f})"
                )
            ax.set_xlabel("lon")
            ax.set_ylabel("lat")
            pdf.savefig(fig)
            plt.close(fig)
    return output_pdf


def plot_spread_spectra(spectra_npz: str | Path, output_pdf: str | Path,
                        *, title_prefix: str = "Spread proxy") -> Path:
    """One page per field: deviation power spectra ML vs ENFO, plus their ratio."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    data = np.load(Path(spectra_npz))
    ell = data["ell"]
    keys = [k for k in data.files if k.endswith("__ml")]
    by_field: dict[str, list[int]] = defaultdict(list)
    for key in keys:
        field, step_part, _ = key.split("__")
        by_field[field].append(int(step_part.replace("step", "")))

    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        for field in sorted(by_field):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
            for step in sorted(by_field[field]):
                cl_ml = data[f"{field}__step{step:03d}__ml"]
                cl_enfo = data[f"{field}__step{step:03d}__enfo"]
                sel = ell >= 1
                line, = ax1.loglog(ell[sel], cl_ml[sel], "-", lw=1.2, label=f"ML +{step}h")
                ax1.loglog(ell[sel], cl_enfo[sel], "--", lw=1.2,
                           color=line.get_color(), label=f"ENFO +{step}h")
                input_key = f"{field}__step{step:03d}__input"
                if input_key in data.files:
                    ax1.loglog(ell[sel], data[input_key][sel], ":", lw=1.0,
                               color=line.get_color(), label=f"input +{step}h")
                with np.errstate(divide="ignore", invalid="ignore"):
                    ax2.semilogx(ell[sel], cl_ml[sel] / cl_enfo[sel], "-", lw=1.2,
                                 color=line.get_color(), label=f"+{step}h")
            ax1.set_xlabel("spherical wavenumber")
            ax1.set_ylabel("deviation power C_l")
            ax1.set_title("member-deviation spectra")
            ax1.legend(fontsize=7)
            ax1.grid(alpha=0.3, which="both")
            ax2.axhline(1.0, color="k", lw=0.8)
            ax2.set_xlabel("spherical wavenumber")
            ax2.set_ylabel("ML / ENFO")
            ax2.set_ylim(0, 3)
            ax2.set_title("spread-power ratio by scale")
            ax2.legend(fontsize=7)
            ax2.grid(alpha=0.3, which="both")
            fig.suptitle(f"{title_prefix} — {field}: band-resolved ensemble spread")
            fig.tight_layout(rect=(0, 0, 1, 0.93))
            pdf.savefig(fig)
            plt.close(fig)
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
