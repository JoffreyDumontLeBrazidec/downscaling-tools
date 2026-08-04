"""Lean run-root layout projection.

The eval harness writes each evaluator's outputs into a self-contained
``evaluators/<name>/`` tree (run/score/plot artifacts plus a ``.complete``
marker). That tree is the source of truth and drives skip/overwrite/re-run.

This module projects that tree into the lean run-root bundle that operators and
downstream tooling consume::

    <run_root>/
    ├── metrics.json                 # assembled from evaluators/*/metrics.json
    ├── <deliverable>.pdf / .png     # consolidated per-evaluator outputs, promoted
    ├── plots/<name>/                # per-variable plot trees
    ├── data/<name>/                 # full raw evaluator outputs
    └── evaluators/<name>/           # source of truth (untouched)

The projection is a non-destructive, idempotent set of relative symlinks plus
one assembled ``metrics.json``. Nothing is moved, so re-running a single
evaluator and re-projecting always yields a correct snapshot — and the old
standalone ``finalize_lean_eval_layout.sbatch`` reorg step is unnecessary
because the harness now lays the bundle down itself.

What lands at the top level is registry-driven: each evaluator may declare a
``deliverables`` block in its ``EVALUATOR_SPEC``::

    "deliverables": {
        # explicit promotions (path relative to the evaluator dir) -> run root,
        # with an optional clean name. Needed for deep paths or canonical renames.
        "top_level": [{"src": "plots/all_tc_distributions.pdf",
                       "as": "tc_pdf_distributions.pdf"}],
        # plot subdirs projected under plots/<name>/ (default: ["plots"])
        "plots": ["plots", "member_maps"],
        # auto-promote root-level *.pdf / *.png not already declared (default True)
        "auto_promote": True,
    }

Convention covers the common case with no declaration: any ``*.pdf`` / ``*.png``
sitting at the evaluator dir root is promoted to the run root under its own name,
and a ``plots/`` subdir becomes ``plots/<name>/``.

Every step is best-effort: a projection hiccup must never fail a run whose
metrics and completion marker are already written.
"""
from __future__ import annotations

import contextlib
import importlib
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)

# Raw predict-stage artifacts (siblings of evaluators/) surfaced under data/.
_RAW_DATA_ARTIFACTS = (
    "predictions",
    "bundles_with_y",
    "bundles",
    "logs",
    "predictions_manifest.csv",
)

# Sibling dirs that mark a directory as a run root (used to disambiguate a
# nested analysis dir such as eval_plots/ from the real run root).
_RUN_ROOT_SIBLINGS = ("manual", "prepml", "predictions")


def resolve_run_root(output_dir: Path) -> Path:
    """Resolve the true run root from an evaluator ``output_dir``.

    Accepts the layouts the harness produces:
      * the run root itself (contains ``evaluators/`` or ``predictions/``),
      * ``<run_root>/data`` (canonical when evaluate runs with
        ``--predictions-dir``),
      * a nested analysis dir such as ``<run_root>/eval_plots`` whose parent is
        the real run root (carries ``manual/``/``prepml/``/``predictions/``
        siblings). This nested case previously resolved to the analysis dir
        itself, burying ``plots/`` and ``data/`` one level too deep.
    """
    output_dir = Path(output_dir)
    if output_dir.name == "data":
        return output_dir.parent

    has_evaluators = (output_dir / "evaluators").exists()
    has_predictions = (output_dir / "predictions").exists()

    # Nested analysis dir: has the evaluator tree but not its own predictions,
    # and its parent looks like the real run root.
    if has_evaluators and not has_predictions:
        parent = output_dir.parent
        if parent != output_dir and any(
            (parent / sib).exists() for sib in _RUN_ROOT_SIBLINGS
        ):
            return parent

    if has_evaluators or has_predictions:
        return output_dir

    raise ValueError(
        f"Cannot resolve run root from output_dir={output_dir!r}. Expected the "
        "run root, <run_root>/data, or a nested analysis dir with run-root "
        "siblings (manual/, prepml/, predictions/)."
    )


def _deliverables(name: str) -> dict[str, Any]:
    """Return the ``deliverables`` block from an evaluator's spec (or {})."""
    try:
        mod = importlib.import_module(f"eval.evaluators.{name}")
    except ImportError:
        return {}
    return dict(getattr(mod, "EVALUATOR_SPEC", {}).get("deliverables", {}))


def _relink(link: Path, target: Path) -> None:
    """Point a *relative* symlink at ``link`` to ``target`` (best-effort).

    Refuses to clobber a real (non-symlink) file/dir already at ``link`` so the
    projection can never destroy a hand-placed artifact.
    """
    try:
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            LOG.debug("Not overwriting real path during projection: %s", link)
            return
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(target, link.parent))
    except OSError:
        LOG.warning("Could not link %s -> %s (non-fatal)", link, target)


def _reset_managed_dir(path: Path) -> None:
    """Rebuild a harness-managed view dir from scratch so stale links vanish."""
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def project_lean_layout(output_dir: Path) -> Path | None:
    """Project the ``evaluators/`` tree into the lean run-root bundle.

    Returns the run root on success, ``None`` on a (logged, non-fatal) failure.
    """
    try:
        output_dir = Path(output_dir)
        run_root = resolve_run_root(output_dir)

        evaluators_dir = output_dir / "evaluators"
        if not evaluators_dir.exists():
            evaluators_dir = run_root / "evaluators"
        if not evaluators_dir.exists():
            LOG.info("No evaluators/ tree to project (output_dir=%s)", output_dir)
            return run_root

        plots_root = run_root / "plots"
        data_root = run_root / "data"
        _reset_managed_dir(plots_root)
        _reset_managed_dir(data_root)

        assembled: dict[str, Any] = {
            "schema_version": "1.0",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "run_root": str(run_root),
            "evaluators": {},
        }
        promoted: set[str] = set()

        for eval_dir in sorted(p for p in evaluators_dir.iterdir() if p.is_dir()):
            name = eval_dir.name
            if name == "__pycache__":
                continue
            spec = _deliverables(name)

            # 1. data/<name> -> the whole evaluator output (source of truth).
            _relink(data_root / name, eval_dir)

            # 2. plots/<name>[/_<sub>] -> declared plot subdirs (default plots/).
            for sub in spec.get("plots", ["plots"]):
                src = eval_dir / sub
                if src.is_dir() and any(src.iterdir()):
                    link_name = name if sub == "plots" else f"{name}_{sub}"
                    _relink(plots_root / link_name, src)

            # 3. top-level deliverables: explicit declarations first, then the
            #    root-level *.pdf/*.png convention for anything not declared.
            declared: set[str] = set()
            for item in spec.get("top_level", []):
                src_rel = item["src"]
                declared.add(src_rel)
                src = eval_dir / src_rel
                if src.exists():
                    dst = item.get("as", Path(src_rel).name)
                    _relink(run_root / dst, src)
                    promoted.add(dst)
            if spec.get("auto_promote", True):
                for child in sorted(eval_dir.iterdir()):
                    if (
                        child.is_file()
                        and child.suffix.lower() in (".pdf", ".png")
                        and child.name not in declared
                    ):
                        _relink(run_root / child.name, child)
                        promoted.add(child.name)

            # 4. metrics.json -> assembled, keyed by evaluator.
            metrics_path = eval_dir / "metrics.json"
            if metrics_path.exists():
                with contextlib.suppress(Exception):
                    assembled["evaluators"][name] = json.loads(
                        metrics_path.read_text()
                    )

        # 5. Surface raw predict-stage artifacts under data/ (non-destructive).
        for artifact in _RAW_DATA_ARTIFACTS:
            src = run_root / artifact
            if src.exists() and not src.is_symlink():
                _relink(data_root / artifact, src)

        assembled["top_level_deliverables"] = sorted(promoted)
        (run_root / "metrics.json").write_text(
            json.dumps(assembled, indent=2, default=str) + "\n"
        )
        LOG.info(
            "Projected lean layout at %s (%d deliverables, %d evaluators with metrics)",
            run_root, len(promoted), len(assembled["evaluators"]),
        )
        return run_root
    except Exception:
        # Last-resort guard: projection is cosmetic and must not fail the run.
        LOG.warning("Lean layout projection failed (non-fatal)", exc_info=True)
        return None
