"""Plot local probabilistic scores in a quaver-like lead-time style."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def plot_probabilistic_summary(
    summary_csv: str | Path,
    output_pdf: str | Path,
    *,
    title_prefix: str = "Probabilistic scores",
    reference_curves: str | Path | None = None,
) -> Path:
    """Create a multi-page lead-time PDF from ``summary_by_lead.csv``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    summary_csv = Path(summary_csv)
    output_pdf = Path(output_pdf)
    rows = _read_csv(summary_csv)
    if not rows:
        raise ValueError(f"No rows found in {summary_csv}")

    refs: list[dict[str, str]] = []
    if reference_curves:
        ref_path = Path(reference_curves).expanduser()
        if ref_path.exists():
            refs = _read_csv(ref_path)

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["weather_state"], row["domain"])].append(row)

    ref_grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in refs:
        ref_grouped[(row.get("weather_state", ""), row.get("domain", ""), row.get("metric", ""))].append(row)

    palette = {
        "crps": "#377eb8",
        "fcrps": "#4daf4a",
        "spread": "#984ea3",
        "rmse_ens_mean": "#e41a1c",
    }
    metrics = ["fcrps", "crps", "spread", "rmse_ens_mean"]
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_pdf) as pdf:
        for (weather_state, domain), group_rows in sorted(grouped.items()):
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
            axes_flat = axes.ravel()
            for ax, metric in zip(axes_flat, metrics):
                metric_rows = [r for r in group_rows if r["metric"] == metric]
                metric_rows.sort(key=lambda r: _float(r, "step"))
                if metric_rows:
                    steps = [_float(r, "step") for r in metric_rows]
                    means = [_float(r, "mean") for r in metric_rows]
                    stderrs = [_float(r, "stderr") for r in metric_rows]
                    color = palette.get(metric, "black")
                    ax.plot(steps, means, marker="o", color=color, label="local")
                    if any(v > 0.0 for v in stderrs):
                        lower = [m - 1.96 * s for m, s in zip(means, stderrs)]
                        upper = [m + 1.96 * s for m, s in zip(means, stderrs)]
                        ax.fill_between(steps, lower, upper, color=color, alpha=0.18, linewidth=0)
                for ref in ref_grouped.get((weather_state, domain, metric), []):
                    label = ref.get("label") or "reference"
                    ax.scatter([_float(ref, "step")], [_float(ref, "value", _float(ref, "mean"))], label=label, marker="x")
                ax.set_title(metric)
                ax.set_xlabel("Lead time (h)")
                ax.set_ylabel(metric)
                ax.grid(True, alpha=0.25)
                ax.legend(loc="best", fontsize="small")
            fig.suptitle(f"{title_prefix}: {weather_state} / {domain}")
            pdf.savefig(fig)
            plt.close(fig)
    return output_pdf
