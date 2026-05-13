"""Generate a self-contained HTML report for an evaluation run.

Produces report.html + report_assets/ (copied PDFs) that can be
rsynced together and viewed in any browser.
"""
from __future__ import annotations

import csv
import html
import json
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PDF discovery
# ---------------------------------------------------------------------------

# Tab name assignment by filename pattern (order matters — first match wins)
_TAB_PATTERNS: list[tuple[str, str]] = [
    ("spectra", "Spectra"),
    ("region", "Region Plots"),
    ("tc_pdfs", "TC Distributions"),
]
# tc_members_<event>_... gets special handling below


def _tab_name(filename: str) -> str:
    """Derive a human-readable tab name from a PDF filename."""
    stem = Path(filename).stem
    for pattern, label in _TAB_PATTERNS:
        if pattern in stem.lower():
            return label
    # TC member maps: tc_members_<event>_...
    m = re.match(r"tc_members_([a-zA-Z]+)", stem)
    if m:
        event = m.group(1).capitalize()
        return f"TC {event} Members"
    # Fallback: humanise the stem
    return stem.replace("_", " ").title()


def _discover_pdfs(run_dir: Path) -> list[Path]:
    """Find PDFs, preferring plots/ consolidated copies over evaluator subdirs."""
    plots_dir = run_dir / "plots"
    data_dir = run_dir / "data"

    # Consolidated plots (preferred) — deduplicate prefixed copies
    # e.g. region_plot__all_regions_plots.pdf is a dup of all_regions_plots.pdf
    if plots_dir.exists():
        consolidated = sorted(plots_dir.glob("*.pdf"))
        if consolidated:
            base_names = {p.name for p in consolidated}
            deduped: list[Path] = []
            for p in consolidated:
                # Skip evaluator-prefixed copies if the bare name exists
                if "__" in p.name:
                    bare = p.name.split("__", 1)[1]
                    if bare in base_names:
                        continue
                deduped.append(p)
            return deduped

    # Fallback: gather from evaluator subdirectories
    seen_names: set[str] = set()
    pdfs: list[Path] = []
    search_dirs = [
        data_dir / "evaluators",
    ]
    for base in search_dirs:
        if not base.exists():
            continue
        for pdf in sorted(base.rglob("*.pdf")):
            if pdf.name not in seen_names:
                seen_names.add(pdf.name)
                pdfs.append(pdf)
    return pdfs


# ---------------------------------------------------------------------------
# Metrics loading
# ---------------------------------------------------------------------------

def _load_metrics_csv(run_dir: Path) -> list[tuple[str, str, float, str]]:
    """Load (evaluator, metric, value, unit) from scoreboard CSV."""
    csv_path = run_dir / "data" / "scoreboard" / "scores.csv"
    if not csv_path.exists():
        return []
    rows: list[tuple[str, str, float, str]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                val = float(row["value"])
            except (ValueError, KeyError):
                continue
            rows.append((
                row.get("evaluator", ""),
                row.get("metric", ""),
                val,
                row.get("unit", ""),
            ))
    return rows


def _load_metrics_json(run_dir: Path) -> list[tuple[str, str, float, str]]:
    """Fallback: load metrics from evaluator metrics.json files."""
    evaluators_dir = run_dir / "data" / "evaluators"
    if not evaluators_dir.exists():
        return []
    rows: list[tuple[str, str, float, str]] = []
    for mj in sorted(evaluators_dir.glob("*/metrics.json")):
        evaluator = mj.parent.name
        try:
            data = json.loads(mj.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict) and "metric" in entry and "value" in entry:
                    try:
                        val = float(entry["value"])
                    except (ValueError, TypeError):
                        continue
                    rows.append((evaluator, entry["metric"], val, entry.get("unit", "")))
    return rows


def _load_metrics(run_dir: Path) -> list[tuple[str, str, float, str]]:
    """Load metrics, preferring scoreboard CSV."""
    metrics = _load_metrics_csv(run_dir)
    if metrics:
        return metrics
    return _load_metrics_json(run_dir)


# ---------------------------------------------------------------------------
# Run identity
# ---------------------------------------------------------------------------

def _load_identity(run_dir: Path) -> dict[str, str]:
    """Load run identity from effective_config.json or dir name."""
    info: dict[str, str] = {"run_dir": str(run_dir), "label": run_dir.name}
    cfg_path = run_dir / "data" / "effective_config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
            info["lane"] = cfg.get("lane", "")
            info["checkpoint"] = str(cfg.get("checkpoint", "") or "")
            info["timestamp"] = cfg.get("timestamp_utc", "")
            info["git_commit"] = cfg.get("git_commit", "")
            info["host"] = cfg.get("host", "")
        except (json.JSONDecodeError, OSError):
            pass
    return info


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

_CSS = """\
:root { --bg:#0f172a; --fg:#e2e8f0; --card:#1e293b; --muted:#94a3b8;
        --accent:#38bdf8; --border: rgba(255,255,255,.06); }
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background:var(--bg); color:var(--fg);
       font: 14.5px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       padding: 0 1rem 4rem; }
.wrap { max-width: 1400px; margin: 0 auto; }
header { margin: 2rem 0 1.25rem; padding: 1.35rem 1.65rem; border-radius: 14px;
         background: linear-gradient(135deg,#1e3a8a 0%,#312e81 60%,#1e293b 100%);
         box-shadow: 0 10px 30px -10px rgba(0,0,0,.5); }
header h1 { font-size: 1.65rem; font-weight: 700; letter-spacing:-.01em;
            background: linear-gradient(90deg,#38bdf8,#a78bfa); -webkit-background-clip:text;
            background-clip:text; color: transparent; }
header .meta { margin: .4rem 0 0; color:#cbd5e1; font-size: .85rem; }
header .meta code { background: rgba(255,255,255,.06); padding: 1px 5px;
                    border-radius: 4px; font-size: .85em;
                    font-family: ui-monospace, "SF Mono", Menlo, monospace; }
.cards { display: flex; flex-wrap: wrap; gap: .65rem; margin: 1rem 0; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 10px;
        padding: .7rem 1rem; min-width: 160px; flex: 1 1 160px; max-width: 220px; }
.card .card-label { font-size: .7rem; text-transform: uppercase; letter-spacing: .06em;
                    color: var(--muted); }
.card .card-value { font-size: 1.15rem; font-weight: 700; color: var(--accent);
                    font-family: ui-monospace, "SF Mono", Menlo, monospace; margin-top: .15rem; }
.card .card-unit { font-size: .7rem; color: var(--muted); }
section.metrics-table { background: var(--card); border-radius: 12px; overflow-x: auto;
                        border:1px solid var(--border); margin-top: .75rem; }
section.metrics-table h2 { padding: .65rem 1rem 0; font-size: .8rem; text-transform: uppercase;
                           letter-spacing: .08em; color: var(--accent); font-weight: 600; }
table { width: 100%; border-collapse: collapse; font-size: .8rem;
        font-variant-numeric: tabular-nums; }
thead th { position: sticky; top: 0; background: #0f172a; color: var(--muted);
           font-weight: 500; font-size: .7rem; text-transform: uppercase;
           letter-spacing: .04em; padding: .5rem .6rem; text-align: left;
           border-bottom: 1px solid var(--border); white-space: nowrap; }
tbody td { padding: .3rem .6rem; border-bottom: 1px solid var(--border); white-space: nowrap; }
tbody tr:hover { background: rgba(56,189,248,.05); }
td.num { text-align: right; font-family: ui-monospace, "SF Mono", Menlo, monospace; }
.tabs { margin: 1.25rem 0; }
.tabs h2 { font-size: .8rem; text-transform: uppercase; letter-spacing: .08em;
           color: var(--accent); font-weight: 600; margin-bottom: .5rem; }
.tab-bar { display: flex; flex-wrap: wrap; gap: .35rem; margin-bottom: .6rem; }
.tab-btn { background: var(--card); border: 1px solid var(--border); border-radius: 8px;
           padding: .35rem .85rem; color: var(--muted); cursor: pointer; font-size: .78rem;
           transition: background .15s, color .15s; }
.tab-btn:hover { background: rgba(56,189,248,.1); color: var(--fg); }
.tab-btn.active { background: rgba(56,189,248,.15); color: var(--accent);
                  border-color: var(--accent); }
.tab-panel { display: none; background: var(--card); border: 1px solid var(--border);
             border-radius: 12px; overflow: hidden; }
.tab-panel.active { display: block; }
.tab-panel iframe { width: 100%; height: 85vh; border: none; }
footer { color: var(--muted); font-size: .77rem; margin: 1.25rem 0 0; padding: 0 .5rem; }
"""

_TAB_JS = """\
document.addEventListener('DOMContentLoaded', function() {
  var btns = document.querySelectorAll('.tab-btn');
  var panels = document.querySelectorAll('.tab-panel');
  btns.forEach(function(btn) {
    btn.addEventListener('click', function() {
      btns.forEach(function(b) { b.classList.remove('active'); });
      panels.forEach(function(p) { p.classList.remove('active'); });
      btn.classList.add('active');
      var target = document.getElementById(btn.getAttribute('data-target'));
      if (target) target.classList.add('active');
    });
  });
});
"""


def _key_metrics(metrics: list[tuple[str, str, float, str]]) -> list[tuple[str, str, float, str]]:
    """Select a few key metrics for the card overview."""
    # Prefer: spectra_mean_score, surface_weighted_nmse, plus any tc_ scores
    key_names = {"spectra_mean_score", "surface_weighted_nmse"}
    result = [m for m in metrics if m[1] in key_names]
    # Add TC scores if present
    result.extend(m for m in metrics if m[1].startswith("tc_") and m[1].endswith("_score"))
    # If we got nothing, take first 4
    if not result:
        result = metrics[:4]
    return result


def _format_value(v: float) -> str:
    """Format a metric value for display."""
    if abs(v) >= 100:
        return f"{v:.1f}"
    if abs(v) >= 1:
        return f"{v:.4f}"
    return f"{v:.6f}"


def generate_report(
    run_dir: Path,
    output_path: Path | None = None,
) -> Path:
    """Generate an HTML report for an evaluation run.

    Parameters
    ----------
    run_dir : Path
        Root directory of the evaluation run.
    output_path : Path | None
        Where to write report.html. Defaults to run_dir / "report.html".

    Returns
    -------
    Path
        Path to the written report.html.
    """
    run_dir = Path(run_dir).resolve()
    if output_path is None:
        output_path = run_dir / "report.html"
    else:
        output_path = Path(output_path).resolve()

    assets_dir = output_path.parent / "report_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # --- Gather data ---
    pdfs = _discover_pdfs(run_dir)
    metrics = _load_metrics(run_dir)
    identity = _load_identity(run_dir)

    # --- Copy PDFs to assets ---
    pdf_entries: list[tuple[str, str, str]] = []  # (tab_name, filename, rel_path)
    seen_names: set[str] = set()
    for pdf in pdfs:
        if pdf.name in seen_names:
            continue
        seen_names.add(pdf.name)
        dest = assets_dir / pdf.name
        shutil.copy2(pdf, dest)
        rel = f"report_assets/{pdf.name}"
        pdf_entries.append((_tab_name(pdf.name), pdf.name, rel))

    LOG.info("Discovered %d PDFs, %d metrics", len(pdf_entries), len(metrics))

    # --- Build HTML ---
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    title = f"Eval Report: {identity.get('label', run_dir.name)}"

    parts: list[str] = []
    parts.append("<!doctype html>")
    parts.append('<html lang="en"><head><meta charset="utf-8"/>')
    parts.append(f"<title>{html.escape(title)}</title>")
    parts.append('<meta name="viewport" content="width=device-width,initial-scale=1"/>')
    parts.append(f"<style>{_CSS}</style>")
    parts.append(f"<script>{_TAB_JS}</script>")
    parts.append('</head><body><div class="wrap">')

    # Header
    parts.append("<header>")
    parts.append(f"<h1>{html.escape(title)}</h1>")
    meta_items = [f"Generated {now}"]
    if identity.get("lane"):
        meta_items.append(f"Lane: <code>{html.escape(identity['lane'])}</code>")
    if identity.get("checkpoint"):
        meta_items.append(f"Checkpoint: <code>{html.escape(identity['checkpoint'])}</code>")
    if identity.get("git_commit"):
        meta_items.append(f"Commit: <code>{html.escape(identity['git_commit'])}</code>")
    if identity.get("host"):
        meta_items.append(f"Host: <code>{html.escape(identity['host'])}</code>")
    parts.append(f'<p class="meta">{" · ".join(meta_items)}</p>')
    parts.append("</header>")

    # Key metric cards
    key = _key_metrics(metrics)
    if key:
        parts.append('<div class="cards">')
        for evaluator, metric, value, unit in key:
            display_name = metric.replace("_", " ").title()
            parts.append('<div class="card">')
            parts.append(f'<div class="card-label">{html.escape(display_name)}</div>')
            parts.append(f'<div class="card-value">{_format_value(value)}</div>')
            parts.append(f'<div class="card-unit">{html.escape(unit)}</div>')
            parts.append("</div>")
        parts.append("</div>")

    # Full metrics table
    if metrics:
        parts.append('<section class="metrics-table">')
        parts.append("<h2>All Metrics</h2>")
        parts.append("<table><thead><tr>")
        parts.append("<th>Evaluator</th><th>Metric</th><th>Value</th><th>Unit</th>")
        parts.append("</tr></thead><tbody>")
        for evaluator, metric, value, unit in metrics:
            parts.append("<tr>")
            parts.append(f"<td>{html.escape(evaluator)}</td>")
            parts.append(f"<td>{html.escape(metric)}</td>")
            parts.append(f'<td class="num">{_format_value(value)}</td>')
            parts.append(f"<td>{html.escape(unit)}</td>")
            parts.append("</tr>")
        parts.append("</tbody></table></section>")

    # Tabbed PDF viewer
    if pdf_entries:
        parts.append('<div class="tabs">')
        parts.append("<h2>Plots</h2>")
        parts.append('<div class="tab-bar">')
        for i, (tab_name, _fname, _rel) in enumerate(pdf_entries):
            active = " active" if i == 0 else ""
            parts.append(
                f'<button class="tab-btn{active}" data-target="pdf-{i}">'
                f"{html.escape(tab_name)}</button>"
            )
        parts.append("</div>")
        for i, (tab_name, fname, rel) in enumerate(pdf_entries):
            active = " active" if i == 0 else ""
            parts.append(f'<div id="pdf-{i}" class="tab-panel{active}">')
            parts.append(f'<iframe src="{html.escape(rel)}" title="{html.escape(tab_name)}"></iframe>')
            parts.append("</div>")
        parts.append("</div>")

    # Footer
    parts.append(
        f'<footer>{len(metrics)} metrics · {len(pdf_entries)} plots · '
        f'run directory: <code>{html.escape(str(run_dir))}</code></footer>'
    )

    parts.append("</div></body></html>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts))
    LOG.info("Report written to %s", output_path)
    return output_path
