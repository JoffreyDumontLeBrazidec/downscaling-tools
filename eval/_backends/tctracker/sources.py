"""Multi-source (model / ctrl / target / input) resolution for tctracker runs.

A *source* is one track-producing dataset: the model expver under evaluation,
a ctrl expver, or an operational reference stream (target ENFO, input EEFO).
All sources are tracked with the SAME tracker settings (grid support, steps,
vorticity) so their tracks live on one support — the comparison contract.

Reference sources depend only on (class, stream, expver, grid, dates, members),
never on the model expver, so their tars are cached once under a shared root

    <scratch_root>/eval/tcrefs/<class>_<stream>_<expver>_o<grid>/

and reused by every lane, campaign and arm. Model/ctrl expvers keep the
existing per-expver roots ``<scratch>/eval/<lane_short>/tctracker/<expver>``.

Default reference specs are derived from the lane's ``prepml.input`` /
``prepml.output`` blocks when present (the lane already declares its stream
schema there), overridable via ``tctracker.sources`` in the lane YAML.
"""
from __future__ import annotations

import calendar
import logging
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)

KNOWN_ROLES = ("model", "ctrl", "target", "input")


def expand_months(months_spec: str, cadence: str = "daily") -> tuple[str, ...]:
    """Expand ``202509`` / ``202508,202509`` into daily YYYYMMDD date tuples."""
    if cadence != "daily":
        raise SystemExit(f"unsupported months cadence: {cadence}")
    dates: list[str] = []
    for token in str(months_spec).split(","):
        token = token.strip()
        if not token:
            continue
        if len(token) != 6 or not token.isdigit():
            raise SystemExit(f"--months entries must be YYYYMM, got {token!r}")
        year, month = int(token[:4]), int(token[4:6])
        _, n_days = calendar.monthrange(year, month)
        dates.extend(f"{year:04d}{month:02d}{day:02d}" for day in range(1, n_days + 1))
    return tuple(dates)


def default_source_specs(lane_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Reference/source specs for a lane: lane ``tctracker.sources`` wins,
    else derive target/input from the ``prepml`` output/input blocks."""
    specs: dict[str, dict[str, Any]] = {}
    prepml = lane_config.get("prepml") or {}
    pin = prepml.get("input") or {}
    if pin.get("stream"):
        specs["input"] = {
            "class": str(pin.get("class", "od")),
            "stream": str(pin["stream"]),
            "type": str(pin.get("type", "pf")),
            "expver": str(pin.get("expver", "0001")),
        }
    pout = prepml.get("output") or {}
    if pout.get("stream"):
        # The target is the OPERATIONAL archive of the lane's output stream
        # (class od), not the rd expver itself.
        specs["target"] = {
            "class": "od",
            "stream": str(pout["stream"]),
            "type": str(pout.get("type", "pf")),
            "expver": "0001",
        }
    for role, spec in ((lane_config.get("tctracker") or {}).get("sources") or {}).items():
        base = specs.get(role, {})
        base.update({str(k): v for k, v in (spec or {}).items()})
        specs[role] = base
    return specs


def source_id_for(role: str, spec: dict[str, Any]) -> str:
    """Stable identifier: rd expvers are just the expver; operational refs are
    ``<class>_<stream>_<expver>``."""
    expver = str(spec.get("expver", ""))
    fdb_class = str(spec.get("class", "rd"))
    if fdb_class == "rd":
        return expver
    return f"{fdb_class}_{spec.get('stream')}_{expver}"


def refs_cache_root(host_config: dict[str, Any]) -> Path:
    return Path(host_config["scratch_root"]) / "eval" / "tcrefs"


def source_output_dir(
    role: str,
    spec: dict[str, Any],
    config: Any,
    host_config: dict[str, Any],
    lane: str,
    lane_config: dict[str, Any],
) -> Path:
    """Refs (class != rd) go to the shared cache; rd expvers keep the
    per-expver lane root."""
    from .pipeline import default_output_dir

    fdb_class = str(spec.get("class", "rd"))
    if fdb_class != "rd":
        sid = source_id_for(role, spec)
        return refs_cache_root(host_config) / f"{sid}_o{config.grid}"
    return default_output_dir(host_config, lane, lane_config, str(spec["expver"]))


def parse_sources_arg(raw: str | None) -> dict[str, str | None]:
    """Parse ``--track-sources model,ctrl=j95z,target,input`` into
    {role: expver_override_or_None}."""
    out: dict[str, str | None] = {}
    if not raw:
        return out
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        role, _, value = token.partition("=")
        out[role.strip()] = value.strip() or None
    return out


def resolve_source_configs(
    base_config: Any,
    roles: dict[str, str | None],
    lane_config: dict[str, Any],
    host_config: dict[str, Any],
) -> list[tuple[str, str, Any]]:
    """Build one TCTrackerConfig per requested role from the model-role base
    config. Returns [(role, source_id, config)]. The base config's tracker
    settings (grid, steps, vorticity, dates, members) are shared by every
    source — one support."""
    specs = default_source_specs(lane_config)
    resolved: list[tuple[str, str, Any]] = []
    for role, override in roles.items():
        if role == "model":
            resolved.append((role, base_config.expver, base_config))
            continue
        spec = dict(specs.get(role) or {})
        if override:
            spec["expver"] = override
        if not spec.get("expver"):
            raise SystemExit(
                f"source role {role!r} has no expver: pass --track-sources {role}=<expver> "
                f"or define lane tctracker.sources.{role}"
            )
        spec.setdefault("class", "rd")
        spec.setdefault("stream", base_config.stream)
        spec.setdefault("type", base_config.fdb_type)
        sid = source_id_for(role, spec)
        out_dir = source_output_dir(
            role, spec, base_config, host_config, base_config.lane, lane_config,
        )
        cfg = replace(
            base_config,
            expver=str(spec["expver"]),
            fdb_class=str(spec["class"]),
            stream=str(spec["stream"]),
            fdb_type=str(spec["type"]),
            output_dir=out_dir,
        )
        resolved.append((role, sid, cfg))
    return resolved


# ---------------------------------------------------------------------------
# FDB completeness preflight (warn-only — correct defaults, never a gate)
# ---------------------------------------------------------------------------

def fdb_date_counts(config: Any) -> dict[str, int]:
    """Per-init-date field counts from ``fdb list --porcelain`` for an rd
    expver. Returns {} (with a warning) if fdb is unavailable."""
    key = f"class={config.fdb_class},expver={config.expver},stream={config.stream}"
    counts: dict[str, int] = {date: 0 for date in config.dates}
    try:
        proc = subprocess.run(
            ["bash", "-lc", f"fdb list --porcelain {key}"],
            capture_output=True, text=True, timeout=900, check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        LOG.warning("fdb preflight skipped (%s)", exc)
        return {}
    if proc.returncode != 0:
        LOG.warning("fdb preflight skipped (rc=%s): %s", proc.returncode, proc.stderr[:200])
        return {}
    for line in proc.stdout.splitlines():
        if "date=" not in line:
            continue
        date = line.split("date=", 1)[1].split(",", 1)[0]
        if date in counts:
            counts[date] += 1
    return counts


def completeness_report(config: Any) -> dict[str, Any]:
    """Classify requested dates as complete/partial/empty against the modal
    per-date field count. Warn-only; the caller decides what to track."""
    counts = fdb_date_counts(config)
    if not counts:
        return {"checked": False, "complete": list(config.dates), "partial": [], "empty": []}
    nonzero = sorted(v for v in counts.values() if v > 0)
    modal = max(set(nonzero), key=nonzero.count) if nonzero else 0
    complete = [d for d, v in counts.items() if v == modal and v > 0]
    partial = [d for d, v in counts.items() if 0 < v < modal]
    over = [d for d, v in counts.items() if v > modal]
    empty = [d for d, v in counts.items() if v == 0]
    report = {
        "checked": True,
        "modal_fields_per_date": modal,
        "counts": counts,
        "complete": sorted(complete + over),
        "partial": sorted(partial),
        "empty": sorted(empty),
    }
    if partial or empty:
        LOG.warning(
            "FDB completeness for %s/%s: %d complete, %d partial %s, %d empty %s "
            "(modal %d fields/date). Partial/empty dates will be SKIPPED for "
            "tracking unless --track-incomplete.",
            config.fdb_class, config.expver, len(report["complete"]),
            len(partial), partial[:5], len(empty), empty[:5], modal,
        )
    return report
