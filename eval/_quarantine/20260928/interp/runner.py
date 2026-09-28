"""Interp evaluator runner — no-op stub.

The interp computations run out-of-band on AG via sbatch (they need a GPU
and the inference checkpoint, and they touch the model forward path). This
evaluator's `run` only verifies that the perm-stored JSONs exist; the
heavy work is `plot`.
"""
from __future__ import annotations

import logging
from pathlib import Path

from eval.evaluators.interp.common import resolve_ckpt_id, perm_run_dirs

LOG = logging.getLogger(__name__)


def run(
    predictions_dir,
    lane_config: dict,
    eval_config: dict,
    *,
    output_dir,
    overwrite: bool = False,
    checkpoint: str | None = None,
    run_label: str | None = None,
    **_kwargs,
) -> dict:
    """Verify that interp JSONs exist for this checkpoint (base run dir plus
    any case-study dirs like <ckpt_id>_humberto)."""
    if not checkpoint:
        LOG.warning("interp: no checkpoint provided — nothing to do")
        return {"available_tools": []}

    # find_json knows the tool -> subdir/json layout (old and new schemas).
    from interp.viz.report import TOOL_REGISTRY, find_json

    ckpt_id = resolve_ckpt_id(checkpoint, eval_config)
    run_dirs = perm_run_dirs(ckpt_id)
    LOG.info("interp.run: ckpt_id=%s run_dirs=%s", ckpt_id,
             [d.name for d in run_dirs])

    found: dict[str, list[str]] = {}
    for rd in run_dirs:
        tools = [tool for tool in TOOL_REGISTRY if find_json(rd, tool)]
        if tools:
            found[rd.name] = tools
            LOG.info("interp: %s -> %s", rd.name, tools)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Persist the checkpoint path so plot() (which doesn't receive --checkpoint)
    # can recover the same ckpt_id we resolved here.
    import json
    (out / "interp_run_meta.json").write_text(json.dumps({
        "checkpoint": str(checkpoint),
        "ckpt_id": ckpt_id,
        "run_dirs": {name: tools for name, tools in found.items()},
    }, indent=2) + "\n")
    return {"ckpt_id": ckpt_id, "run_dirs": list(found), "available_tools": found}
