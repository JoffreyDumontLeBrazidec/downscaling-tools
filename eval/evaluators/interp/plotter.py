"""Interp evaluator plotter — calls interp.viz and copies PDFs into the eval run.

Generates the canonical PDFs at ~/perm/interp/<ckpt_id>/plots/ AND mirrors
them into the eval framework's per-evaluator plots/ dir so that
_consolidate_plots in eval.cli picks them up for the run-level plots/ folder.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

from eval.evaluators.interp.common import resolve_ckpt_id, perm_run_dirs

LOG = logging.getLogger(__name__)


def plot(
    results_dir,
    lane_config: dict,
    eval_config: dict,
    *,
    output_dir=None,
    checkpoint: str | None = None,
    **_kwargs,
) -> Path:
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else results_dir
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Eval CLI does not pass `checkpoint` to plot(); read it from results metadata
    # if available (from runner) or fall back to lane_config.
    ckpt = checkpoint or eval_config.get("checkpoint") or lane_config.get("checkpoint")
    if not ckpt:
        # try to read from runner output
        meta_path = results_dir / "interp_run_meta.json"
        if meta_path.exists():
            import json
            ckpt = json.loads(meta_path.read_text()).get("checkpoint")
    if not ckpt:
        LOG.warning("interp.plot: no checkpoint available — skipping")
        return plots_dir

    ckpt_id = resolve_ckpt_id(ckpt, eval_config)
    run_dirs = perm_run_dirs(ckpt_id)
    LOG.info("interp.plot: ckpt_id=%s run_dirs=%s", ckpt_id, [d.name for d in run_dirs])

    if not run_dirs:
        LOG.warning("interp.plot: no perm dirs for %s — run the interp sbatch jobs first",
                    ckpt_id)
        return plots_dir

    # (Re)generate report.pdf for the base run dir AND every case-study dir
    # (<ckpt_id>_humberto, ...), then mirror everything into the
    # eval-framework plots dir so _consolidate_plots picks them up.
    n_mirrored = 0
    for rd in run_dirs:
        cmd = [sys.executable, "-m", "interp.viz", "--run-dir", str(rd)]
        LOG.info("interp.plot: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            LOG.error("interp.viz failed for %s (exit %d) — skipping", rd.name,
                      exc.returncode)
            continue
        for pdf in sorted((rd / "plots").glob("*.pdf")):
            shutil.copy2(pdf, plots_dir / f"interp_{rd.name}_{pdf.name}")
            n_mirrored += 1
    LOG.info("interp.plot: mirrored %d PDFs to %s", n_mirrored, plots_dir)
    return plots_dir
