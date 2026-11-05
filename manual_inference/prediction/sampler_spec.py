"""The flow-map sampler block of a lane, and a one-line description of any sampler block.

A flow-map run (MeanFlow / iMF) samples by a few jumps from a start sigma instead of an
EDM schedule. Its lane block (``predict.sampler``) is::

    schedule_type: "flow_map"
    num_steps: 2          # number of jumps (1, 2 or 4 network calls)
    sigma_max: 80.0       # start sigma
    sigma_min: 0.0        # end sigma (default 0.0)
    sampler: "jumps"

and the checkpoint's ``predict_step`` receives exactly
``noise_scheduler_params={"schedule_type": "flow_map", "num_steps": N, "sigma_max": 80.0,
"sigma_min": 0.0}`` and ``sampler_params={"sampler": "jumps"}``.

No other key is allowed in a flow-map block: the churn keys (``S_churn``, ``S_min``,
``S_max``, ``S_noise``) have no meaning for jumps, and the EDM-only helpers of
``eval.predict`` (autoguidance, sigma switch) wrap the EDM denoiser, so they are refused
here instead of running silently on a path they were never tested on.

Pure Python on purpose (no torch): ``eval.config.loader`` and ``eval.cli judge`` import it.
"""
from __future__ import annotations

import json
import logging

FLOW_MAP = "flow_map"
FLOW_MAP_SAMPLER = "jumps"
FLOW_MAP_DEFAULT_SIGMA_MAX = 80.0
FLOW_MAP_DEFAULT_SIGMA_MIN = 0.0
FLOW_MAP_TESTED_NUM_STEPS = (1, 2, 4)
FLOW_MAP_KEYS = ("schedule_type", "num_steps", "sigma_max", "sigma_min", "sampler")
CHURN_KEYS = ("S_churn", "S_min", "S_max", "S_noise")

LOG = logging.getLogger(__name__)


class SamplerConfigError(ValueError):
    """A sampler block that cannot be run as written."""


def is_flow_map(cfg) -> bool:
    return isinstance(cfg, dict) and cfg.get("schedule_type") == FLOW_MAP


def normalize_flow_map_sampler(cfg: dict, *, warn: bool = False) -> dict:
    """Return the canonical flow-map block, or a copy of ``cfg`` when it is not one.

    Raises SamplerConfigError on churn keys, unknown keys, a sampler other than "jumps",
    or a jump count / sigma that is not a number in range.
    """
    if not is_flow_map(cfg):
        return dict(cfg or {})
    churn = [k for k in CHURN_KEYS if k in cfg]
    if churn:
        raise SamplerConfigError(
            f"schedule_type flow_map takes no churn keys, got {churn}: a flow-map run jumps "
            "deterministically from sigma_max; remove them from the sampler block"
        )
    unknown = sorted(set(cfg) - set(FLOW_MAP_KEYS))
    if unknown:
        raise SamplerConfigError(
            f"schedule_type flow_map takes only {list(FLOW_MAP_KEYS)}, got also {unknown} "
            "(EDM schedule keys and the autoguidance / sigma-switch keys do not apply to jumps)"
        )
    sampler = cfg.get("sampler", FLOW_MAP_SAMPLER)
    if sampler != FLOW_MAP_SAMPLER:
        raise SamplerConfigError(f"schedule_type flow_map needs sampler {FLOW_MAP_SAMPLER!r}, got {sampler!r}")
    n = cfg.get("num_steps")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise SamplerConfigError(f"schedule_type flow_map needs num_steps, the number of jumps (>= 1), got {n!r}")
    try:
        sigma_max = float(cfg.get("sigma_max", FLOW_MAP_DEFAULT_SIGMA_MAX))
        sigma_min = float(cfg.get("sigma_min", FLOW_MAP_DEFAULT_SIGMA_MIN))
    except (TypeError, ValueError) as exc:
        raise SamplerConfigError(f"schedule_type flow_map: sigma_max / sigma_min must be numbers ({exc})") from None
    if not 0.0 <= sigma_min < sigma_max:
        raise SamplerConfigError(f"schedule_type flow_map needs 0 <= sigma_min < sigma_max, got {sigma_min}, {sigma_max}")
    if warn and n not in FLOW_MAP_TESTED_NUM_STEPS:
        LOG.warning("flow_map num_steps=%d: only %s jumps have been read so far", n, FLOW_MAP_TESTED_NUM_STEPS)
    return {
        "schedule_type": FLOW_MAP,
        "num_steps": int(n),
        "sigma_max": sigma_max,
        "sigma_min": sigma_min,
        "sampler": FLOW_MAP_SAMPLER,
    }


def normalize_sampler_json(raw: str | None, *, warn: bool = False) -> str | None:
    """``normalize_flow_map_sampler`` on a JSON string; a non-flow-map string comes back unchanged."""
    if not raw:
        return raw
    cfg = json.loads(raw)
    if not is_flow_map(cfg):
        return raw
    return json.dumps(normalize_flow_map_sampler(cfg, warn=warn), sort_keys=True)


def _num(x) -> str:
    return f"{float(x):g}"


def describe_sampler(cfg: dict) -> str:
    """One line for a read header, e.g. "jumps 2 from sigma 80" or
    "heun karras 30 steps from sigma 80 to 0.03, rho 7, churn 0"."""
    if is_flow_map(cfg):
        line = f"jumps {cfg.get('num_steps', '?')} from sigma {_num(cfg.get('sigma_max', FLOW_MAP_DEFAULT_SIGMA_MAX))}"
        sigma_min = float(cfg.get("sigma_min", FLOW_MAP_DEFAULT_SIGMA_MIN))
        return line if sigma_min == 0.0 else f"{line} to {_num(sigma_min)}"
    parts = [str(cfg.get("sampler", "?")), str(cfg.get("schedule_type", "?"))]
    if "num_steps" in cfg:
        parts.append(f"{cfg['num_steps']} steps")
    if "sigma_max" in cfg:
        parts.append(f"from sigma {_num(cfg['sigma_max'])}" + (f" to {_num(cfg['sigma_min'])}" if "sigma_min" in cfg else ""))
    line = " ".join(parts)
    if "rho" in cfg:
        line += f", rho {_num(cfg['rho'])}"
    if "S_churn" in cfg:
        line += f", churn {_num(cfg['S_churn'])}"
    return line
