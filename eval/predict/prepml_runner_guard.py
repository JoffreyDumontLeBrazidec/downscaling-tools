"""Refuse a prepml launch whose runner venv carries an unfixed downscaling_unified runner.

Two defects of season-scale-tools/downscaling_unified_runner.py were fixed on 2026-11-10:

1. Sampler keys silently dropped. The runner routed development_hacks.extra_args by the
   signature of AnemoiModelInterface.predict_step, which takes only **kwargs, so the seven
   piecewise scheduler keys (num_steps_high, num_steps_low, sigma_transition,
   high/low_schedule_type, rho_high, rho_low) went as loose kwargs that the model ignores:
   a 10+20 lane ran 15+15.
2. Seeding. The runner seeded once per process, so every (date, member) prepml task started
   from the same noise, against standing rule 31 (one seed per date, lead and member, one
   global noise field).

The fixed runner carries the module constant ``RUNNER_FIX_LEVEL``. Before eval.cli pushes a
prepml suite, ``enforce_prepml_runner`` sources the wrap's ``bin/activate`` in a clean bash,
reads the PYTHONPATH it sets, finds the downscaling_unified_runner.py that Python would
import, and refuses unless that file declares an accepted fix level (read as text, never
imported). A wrap whose own ``site/`` directory carries ``gl512_runner_patches.py`` (the GL512
wrap, which applies the same two fixes as a monkeypatch over a frozen runner copy) counts as
fixed. ``--allow-unfixed-runner`` overrides the refusal. Lanes whose base runner is not
``downscaling_unified`` do not load this file and are not checked.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)

UNIFIED_BASE_RUNNER = "downscaling_unified"
RUNNER_MODULE_FILE = "downscaling_unified_runner.py"
ACCEPTED_RUNNER_FIX_LEVELS = frozenset({"20261110-sampler-dicts-seed31"})
# Monkeypatch modules applying the same fixes from the wrap's own site/ directory.
EQUIVALENT_PATCH_MODULES = ("gl512_runner_patches.py",)
_FIX_LEVEL_RE = re.compile(r"""^RUNNER_FIX_LEVEL\s*=\s*["']([^"']+)["']""", re.MULTILINE)


@dataclass
class RunnerCheck:
    ok: bool
    message: str
    venv: str = ""
    runner_file: str | None = None
    fix_level: str | None = None


def base_runner_of(prepml_config: dict[str, Any]) -> str:
    """Return the anemoi-inference base runner named by a generated prepml config."""
    runner = (prepml_config.get("model") or {}).get("runner", "")
    if isinstance(runner, dict):
        parallel = runner.get("parallel")
        if isinstance(parallel, dict):
            return str(parallel.get("base_runner", ""))
        return str(next(iter(runner), ""))
    return str(runner)


def wrap_pythonpath(venv: str | Path, timeout: int = 120) -> list[str]:
    """PYTHONPATH entries that ``<venv>/bin/activate`` sets, sourced in a clean bash."""
    activate = Path(venv) / "bin" / "activate"
    if not activate.is_file():
        raise FileNotFoundError(f"no bin/activate in runner venv {venv}")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c",
         'source "$1" >/dev/null 2>&1; printf "%s" "${PYTHONPATH:-}"', "_", str(activate)],
        capture_output=True, text=True, timeout=timeout, env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"sourcing {activate} failed (rc={result.returncode}): {result.stderr.strip()}")
    return [p for p in result.stdout.split(":") if p]


def runner_fix_level(path: str | Path) -> str | None:
    """The RUNNER_FIX_LEVEL a runner file declares, or None (read as text)."""
    match = _FIX_LEVEL_RE.search(Path(path).read_text(errors="replace"))
    return match.group(1) if match else None


def check_runner_venv(venv: str | Path) -> RunnerCheck:
    """Check the downscaling_unified runner that ``venv`` puts on PYTHONPATH."""
    venv = str(venv)
    try:
        entries = wrap_pythonpath(venv)
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        return RunnerCheck(False, f"cannot resolve the runner venv PYTHONPATH: {exc}", venv)
    site = Path(venv) / "site"
    for module in EQUIVALENT_PATCH_MODULES:
        if (site / module).is_file() and (site / "sitecustomize.py").is_file() and any(
                os.path.realpath(p) == os.path.realpath(site) for p in entries):
            return RunnerCheck(True, f"wrap applies the fixes through {site / module}", venv,
                               str(site / module), "equivalent-patch")
    for entry in entries:
        candidate = Path(entry) / RUNNER_MODULE_FILE
        if candidate.is_file():
            level = runner_fix_level(candidate)
            if level in ACCEPTED_RUNNER_FIX_LEVELS:
                return RunnerCheck(True, f"runner {candidate} is fixed (RUNNER_FIX_LEVEL={level})",
                                   venv, str(candidate), level)
            return RunnerCheck(
                False,
                f"runner {candidate} has RUNNER_FIX_LEVEL={level!r}, not one of "
                f"{sorted(ACCEPTED_RUNNER_FIX_LEVELS)}: it drops the piecewise sampler keys and "
                "seeds once per process (standing rule 31)",
                venv, str(candidate), level)
    return RunnerCheck(False, f"no {RUNNER_MODULE_FILE} on the PYTHONPATH set by {venv}/bin/activate "
                              f"({entries})", venv)


def enforce_prepml_runner(prepml_config: dict[str, Any], *, allow_unfixed: bool = False) -> RunnerCheck:
    """Raise SystemExit unless the prepml config's runner venv carries a fixed runner."""
    base = base_runner_of(prepml_config)
    venv = str((prepml_config.get("runner") or {}).get("venv", ""))
    if base != UNIFIED_BASE_RUNNER:
        check = RunnerCheck(True, f"base runner {base!r} does not load {RUNNER_MODULE_FILE}; not checked", venv)
        LOG.info("prepml runner guard: %s", check.message)
        return check
    check = check_runner_venv(venv)
    if check.ok:
        LOG.info("prepml runner guard: OK: %s", check.message)
        return check
    message = (f"prepml runner guard: REFUSED runner.venv {venv}: {check.message}. Fix the runner "
               "(season-scale-tools downscaling_unified_runner.py, fix of 2026-11-10) or pass "
               "--allow-unfixed-runner to launch anyway.")
    if allow_unfixed:
        LOG.warning("%s Overridden by --allow-unfixed-runner.", message)
        return check
    raise SystemExit(message)
