"""Per-rung TC proxy plot: the eye distribution behind the two headline numbers."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

LOG = logging.getLogger(__name__)


def plot(
    results_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    *,
    output_dir: str | Path | None = None,
    **kwargs,
) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results_dir = Path(results_dir)
    plots_dir = Path(output_dir or results_dir) / "plots"
    path = results_dir / "per_instance.json"
    if not path.exists():
        LOG.warning("tc_proxy: nothing to plot, %s missing", path)
        return None
    report = json.loads(path.read_text())
    events = report.get("events") or {}
    if not events:
        return None
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(events), figsize=(6.2 * len(events), 4.6), squeeze=False)
    for ax, (event, payload) in zip(axes[0], events.items()):
        for curve, colour in (("model", "#1f77b4"), ("enfo", "black"), ("eefo", "#777777")):
            table = (payload.get("tables") or {}).get(curve)
            if not table:
                continue
            arr = np.asarray(table["eye"], dtype=np.float64)
            ax.plot(np.sort(arr.ravel()), np.linspace(0, 1, arr.size), lw=2,
                    color=colour, label=f"{curve}  (deepest {arr.min():.1f})")
        ax.set_xlabel("eye MSLP [hPa]")
        ax.set_ylabel("cumulative fraction of instances")
        ax.set_title(event)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    out = plots_dir / "tc_proxy_eye_distribution.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
