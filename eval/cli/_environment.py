"""Process environment a lane command needs before it starts.

Applies the host's ``module load`` list and checks that metview is importable when
the tropical cyclone evaluator is asked for regridded support.
"""
from __future__ import annotations

import os
import subprocess

from eval.cli._common import LOG


def _apply_host_module_loads(host_config: dict) -> None:
    """C5 (a): best-effort apply host ``environment_setup.module_loads``.

    The inline eval subprocess needs the env that ``module load <mod>`` sets up
    (e.g. ``ecmwf-toolbox`` provides metview, used by regridded TC). When this
    process was launched outside the rendered sbatch (which already loads the
    modules), those vars are absent. We source the module system, run the
    loads, dump the resulting env, and import any *new/changed* vars into
    ``os.environ`` so child evaluators inherit them.

    Best-effort: if the module system isn't available or anything fails, we log
    and move on — the regridded assertion in _assert_metview_for_regridded_tc is
    the hard guard against a silent degrade.
    """
    module_loads = (
        host_config.get("environment_setup", {}).get("module_loads", []) or []
    )
    if not module_loads:
        return
    # Skip if modules already appear loaded (LOADEDMODULES is set by the module
    # system); avoids spawning a shell on every invocation inside sbatch.
    if os.environ.get("LOADEDMODULES"):
        return
    load_cmd = " && ".join(f"module load {m}" for m in module_loads)
    script = (
        "source /etc/profile.d/modules.sh 2>/dev/null || "
        "source /usr/share/Modules/init/bash 2>/dev/null || true; "
        f"{load_cmd} >/dev/null 2>&1; env"
    )
    try:
        result = subprocess.run(
            ["bash", "-lc", script],
            capture_output=True, text=True, timeout=120,
        )
    except Exception:
        LOG.warning(
            "Could not apply host module_loads %s (non-fatal); regridded TC will "
            "be guarded by an explicit metview check.", module_loads, exc_info=True,
        )
        return
    if result.returncode != 0:
        LOG.warning("module load returned %d; continuing", result.returncode)
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        # Only import vars not already set by the user/sbatch so explicit
        # overrides win, mirroring the host_exports policy.
        if key and key not in os.environ:
            os.environ[key] = value


def _tc_support_implies_regridded(lane_config: dict) -> bool:
    """Return True when the lane's TC support mode needs metview (regridded path)."""
    tc_cfg = lane_config.get("tc") or {}
    mode = str(tc_cfg.get("support_mode", "")).strip().lower()
    return mode in ("regridded", "both")


def _assert_metview_for_regridded_tc(
    lane_config: dict, evaluators: list[str],
) -> None:
    """C5 (b): hard-fail instead of silently degrading regridded TC to native.

    Regridded/both TC support requires metview (from the ``ecmwf-toolbox``
    module). If the module wasn't loaded the import fails deep inside the TC
    backend and the result quietly degrades to native support (a different,
    incomparable measurement). Assert importability up front with a clear,
    actionable error.
    """
    if "tc" not in evaluators:
        return
    if not _tc_support_implies_regridded(lane_config):
        return
    try:
        import metview  # noqa: F401
    except Exception as exc:
        mode = str((lane_config.get("tc") or {}).get("support_mode", "")).strip()
        raise SystemExit(
            "TC support_mode="
            f"{mode!r} requires metview, but it is not importable: "
            f"{exc}. Load the host's ecmwf-toolbox module (e.g. "
            "`module load ecmwf-toolbox`) before running, or run inside the "
            "rendered sbatch which loads it. Refusing to silently degrade "
            "regridded TC to native."
        ) from exc
