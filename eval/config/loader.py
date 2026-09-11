"""Configuration loader with validation for the evaluation framework."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

_CONFIG_DIR = Path(__file__).parent

_LANE_REQUIRED_KEYS = {"predict", "evaluator_groups"}
_LANE_ALLOWED_KEYS = {
    "predict", "tc", "spectra", "spectra_ecmwf", "spectra_ecmwf_v2", "surface", "regions",
    "evaluator_groups", "sigma", "sigma_loss", "mechanistic", "intermediate",
    "resource_profiles", "region_plot", "prepare", "prepml",
    "default_host", "allowed_hosts", "lineage",
    "precip", "precip_dist", "precip_events", "precip_scores", "probabilistic", "quaver", "obs_crps", "local_global",
    "tctracker", "lane_diagnostics", "texture", "wind_extremes", "displacement",
    "membermaps",
}

_HOST_REQUIRED_KEYS = {"code_root", "scratch_root", "scheduler", "environment_setup"}

_EVENT_REQUIRED_KEYS = {"name", "lat_min", "lat_max", "lon_min", "lon_max", "dates"}


class ConfigValidationError(Exception):
    pass


def _deep_merge(base: dict, overrides: dict) -> dict:
    result = dict(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = {**result[key], **value}
        else:
            result[key] = value
    return result


def _sampler_keys(config: dict) -> set:
    """Keys of `predict.sampler` in a config, or an empty set when there is none."""
    predict = config.get("predict")
    if not isinstance(predict, dict):
        return set()
    sampler = predict.get("sampler")
    return set(sampler) if isinstance(sampler, dict) else set()


def _warn_on_wholesale_sampler_replacement(
    base_config: dict, child_config: dict, lane_name: str
) -> None:
    """Advisory guard for a child that restates a full `predict.sampler` block.

    The top-level merge performed by `_deep_merge` is two levels deep, while the
    sampler sits three levels down at `predict.sampler`. A child that sets any key
    under `predict.sampler` therefore REPLACES the base sampler block wholesale, and
    every base key it does not restate is dropped. Those dropped keys are later
    filled from the checkpoint's own inference defaults rather than from the base
    lane, so the result is a silent mixture and not an error. This warns about that
    and points at `predict.sampler_overrides`, which merges key by key instead.

    This only ever warns. The project has a standing rule against blocking
    validators, so the whole body is wrapped so that a bug in the check itself can
    never stop a run.
    """
    try:
        base_keys = _sampler_keys(base_config)
        child_keys = _sampler_keys(child_config)
        if not base_keys or not child_keys:
            return
        # Keys the child restores through `sampler_overrides` are not dropped.
        child_predict = child_config.get("predict")
        if isinstance(child_predict, dict):
            child_overrides = child_predict.get("sampler_overrides")
            if isinstance(child_overrides, dict):
                child_keys = child_keys | set(child_overrides)
        dropped = sorted(base_keys - child_keys)
        if not dropped:
            return
        print(
            f"WARNING [load_lane {lane_name}]: 'predict.sampler' restates the sampler "
            f"block, which replaces the base block wholesale, so base key(s) {dropped} "
            f"are dropped here and will be filled from the checkpoint's inference "
            f"defaults instead of from the base lane. If you meant to change only the "
            f"keys you named, use 'predict.sampler_overrides', which merges key by key.",
            file=sys.stderr,
        )
    except Exception as exc:  # noqa: BLE001 - advisory only, must never stop a run
        print(
            f"WARNING [load_lane {lane_name}]: sampler-replacement check failed "
            f"({type(exc).__name__}: {exc}); continuing.",
            file=sys.stderr,
        )


def _apply_sampler_overrides(config: dict, lane_name: str) -> None:
    """Fold `predict.sampler_overrides` key by key onto the resolved sampler.

    `predict.sampler_overrides` is the merging counterpart of `predict.sampler`. Its
    keys are applied one by one to the sampler resolved from the base chain, so every
    key the child does not name keeps the value it inherited. `predict.sampler` is
    left exactly as it was, replacing the base block wholesale, because all existing
    lane files depend on that behaviour.

    DECISION, both keys present in one config: `predict.sampler` is applied first, as
    a wholesale replacement of the base block (this already happened in `_deep_merge`
    by the time we get here), and `predict.sampler_overrides` is then merged on top of
    that result. So `sampler` decides the block and `sampler_overrides` adjusts
    individual keys within it, and `sampler_overrides` wins on any key both name.

    The key is consumed at each level of the base chain, so a lane that is itself used
    as a base hands its successors an already resolved `predict.sampler` and never
    re-applies its own overrides further down the chain.

    When `predict.sampler_overrides` is absent this is a no-op, which is what keeps
    every existing lane resolving bit-exactly as it did before.
    """
    predict = config.get("predict")
    if not isinstance(predict, dict) or "sampler_overrides" not in predict:
        return
    predict = dict(predict)  # never mutate a dict shared with a base config
    overrides = predict.pop("sampler_overrides")
    config["predict"] = predict
    if not isinstance(overrides, dict):
        print(
            f"WARNING [load_lane {lane_name}]: 'predict.sampler_overrides' must be a "
            f"mapping of sampler key to value, got {type(overrides).__name__}; "
            f"ignoring it and keeping the inherited sampler.",
            file=sys.stderr,
        )
        return
    sampler = predict.get("sampler")
    sampler = dict(sampler) if isinstance(sampler, dict) else {}
    if not sampler:
        print(
            f"WARNING [load_lane {lane_name}]: 'predict.sampler_overrides' is set but "
            f"no 'predict.sampler' was inherited or declared, so the overrides become "
            f"the whole sampler block; unnamed keys will come from the checkpoint's "
            f"inference defaults.",
            file=sys.stderr,
        )
    sampler.update(overrides)
    predict["sampler"] = sampler


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigValidationError(f"{path}: expected a YAML mapping, got {type(data).__name__}")
    return data


def _validate_lane(config: dict[str, Any], path: Path) -> None:
    for key in _LANE_REQUIRED_KEYS:
        if key not in config:
            raise ConfigValidationError(f"{path}: missing required top-level key '{key}'")

    unknown = set(config.keys()) - _LANE_ALLOWED_KEYS
    if unknown:
        raise ConfigValidationError(
            f"{path}: unknown top-level key(s) {sorted(unknown)} "
            f"(allowed: {', '.join(sorted(_LANE_ALLOWED_KEYS))})"
        )

    predict = config["predict"]
    if not isinstance(predict, dict):
        raise ConfigValidationError(f"{path}: 'predict' must be a mapping, got {type(predict).__name__}")

    if "members" not in predict:
        raise ConfigValidationError(f"{path}: missing required key 'predict.members' (expected list of int)")
    if not isinstance(predict["members"], list) or not all(isinstance(m, int) for m in predict["members"]):
        raise ConfigValidationError(
            f"{path}: 'predict.members' must be a list of int, got {predict['members']!r}"
        )

    if "steps" not in predict:
        raise ConfigValidationError(f"{path}: missing required key 'predict.steps' (expected list of int)")
    if not isinstance(predict["steps"], list) or not all(isinstance(s, int) for s in predict["steps"]):
        raise ConfigValidationError(
            f"{path}: 'predict.steps' must be a list of int, got {predict['steps']!r}"
        )

    if "dates" not in predict:
        raise ConfigValidationError(f"{path}: missing required key 'predict.dates' (expected list of str)")
    if not isinstance(predict["dates"], list) or not all(isinstance(d, str) for d in predict["dates"]):
        raise ConfigValidationError(
            f"{path}: 'predict.dates' must be a list of str, got {predict['dates']!r}"
        )

    evaluator_groups = config["evaluator_groups"]
    if not isinstance(evaluator_groups, dict):
        raise ConfigValidationError(
            f"{path}: 'evaluator_groups' must be a mapping, got {type(evaluator_groups).__name__}"
        )
    if "default" not in evaluator_groups:
        raise ConfigValidationError(f"{path}: missing required key 'evaluator_groups.default' (expected list of str)")
    if not isinstance(evaluator_groups["default"], list):
        raise ConfigValidationError(
            f"{path}: 'evaluator_groups.default' must be a list of str, got {evaluator_groups['default']!r}"
        )

    allowed_hosts = config.get("allowed_hosts")
    if allowed_hosts is not None:
        if isinstance(allowed_hosts, list):
            valid_allowed_hosts = all(isinstance(item, str) for item in allowed_hosts)
        elif isinstance(allowed_hosts, dict):
            valid_allowed_hosts = all(
                isinstance(stage, str)
                and isinstance(hosts, list)
                and all(isinstance(item, str) for item in hosts)
                for stage, hosts in allowed_hosts.items()
            )
        else:
            valid_allowed_hosts = False
        if not valid_allowed_hosts:
            raise ConfigValidationError(
                f"{path}: 'allowed_hosts' must be a list of str or a mapping "
                f"of CLI stage to list of str, got {allowed_hosts!r}"
            )


def _validate_host(config: dict[str, Any], path: Path) -> None:
    for key in _HOST_REQUIRED_KEYS:
        if key not in config:
            raise ConfigValidationError(f"{path}: missing required key '{key}'")

    code_root = config["code_root"]
    if not isinstance(code_root, str) or not code_root.startswith("/"):
        raise ConfigValidationError(
            f"{path}: key 'code_root' must be an absolute path (starts with '/'), got {code_root!r}"
        )

    scratch_root = config["scratch_root"]
    if not isinstance(scratch_root, str) or not scratch_root.startswith("/"):
        raise ConfigValidationError(
            f"{path}: key 'scratch_root' must be an absolute path (starts with '/'), got {scratch_root!r}"
        )

    scheduler = config["scheduler"]
    if not isinstance(scheduler, dict):
        raise ConfigValidationError(f"{path}: 'scheduler' must be a mapping, got {type(scheduler).__name__}")
    if "qos" not in scheduler or not isinstance(scheduler["qos"], str):
        raise ConfigValidationError(f"{path}: 'scheduler.qos' must be a str")
    if "default_time" not in scheduler or not isinstance(scheduler["default_time"], str):
        raise ConfigValidationError(f"{path}: 'scheduler.default_time' must be a str")

    env = config["environment_setup"]
    if not isinstance(env, dict):
        raise ConfigValidationError(f"{path}: 'environment_setup' must be a mapping, got {type(env).__name__}")
    if "module_loads" not in env or not isinstance(env["module_loads"], list):
        raise ConfigValidationError(f"{path}: 'environment_setup.module_loads' must be a list")
    if "venv_activate" not in env or not isinstance(env["venv_activate"], str):
        raise ConfigValidationError(f"{path}: 'environment_setup.venv_activate' must be a str")


def _validate_event(config: dict[str, Any], path: Path) -> None:
    for key in _EVENT_REQUIRED_KEYS:
        if key not in config:
            raise ConfigValidationError(f"{path}: missing required key '{key}'")

    if not isinstance(config["name"], str):
        raise ConfigValidationError(f"{path}: 'name' must be a str, got {type(config['name']).__name__}")

    for coord in ("lat_min", "lat_max", "lon_min", "lon_max"):
        val = config[coord]
        if not isinstance(val, (int, float)):
            raise ConfigValidationError(f"{path}: '{coord}' must be numeric, got {type(val).__name__}")

    if not isinstance(config["dates"], list):
        raise ConfigValidationError(f"{path}: 'dates' must be a list, got {type(config['dates']).__name__}")


_ANEMOI_REF_DIR = _CONFIG_DIR / "anemoi_inference_reference"


def _apply_canonical_anemoi_reference(config: dict, lane_name: str) -> None:
    """Overwrite prepml.input/output MARS identity (class/stream/type/grid) from the
    canonical per-input-grid reference in anemoi_inference_reference/<GRID>.yaml.

    This is the single source of truth for downscaling input/output streams, so lane
    YAMLs cannot silently drift them (see the eecdb127 eefo/enfo incident, 2026-07-01).
    Keyed by prepml.input.grid because each downscaling input grid maps to exactly one
    task family (O48/O96/O320/O1280). Warns loudly on any disagreement, then wins.
    """
    prepml = config.get("prepml")
    if not isinstance(prepml, dict):
        return
    grid = (prepml.get("input") or {}).get("grid")
    if not grid:
        return
    ref_path = _ANEMOI_REF_DIR / f"{grid}.yaml"
    if not ref_path.exists():
        print(
            f"WARNING [load_lane {lane_name}]: no canonical anemoi-inference reference "
            f"for input grid {grid} ({ref_path}); prepml.input/output taken from the lane "
            f"YAML (stream-drift risk). Add {ref_path.name} to lock it down.",
            file=sys.stderr,
        )
        return
    ref = _load_yaml(ref_path)
    for block in ("input", "output"):
        canon = ref.get(block)
        if not isinstance(canon, dict):
            continue
        cur = dict(prepml.get(block) or {})
        for key, val in canon.items():
            if key in cur and cur[key] != val:
                print(
                    f"WARNING [load_lane {lane_name}]: prepml.{block}.{key}={cur[key]!r} "
                    f"disagrees with canonical {val!r} (input grid {grid}); using canonical. "
                    f"Remove {block}.{key} from the lane YAML.",
                    file=sys.stderr,
                )
        cur.update(canon)
        prepml[block] = cur


def load_lane(name: str, overrides: dict | None = None) -> dict:
    path = _CONFIG_DIR / "lanes" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Lane config not found: {path}")
    config = _load_yaml(path)
    if "base" in config:
        base_config = load_lane(config.pop("base"))
        _warn_on_wholesale_sampler_replacement(base_config, config, name)
        config = _deep_merge(base_config, config)
    if overrides:
        config = _deep_merge(config, overrides)
    # Fold `predict.sampler_overrides` onto the sampler resolved from the base chain
    # (and from any programmatic `overrides`) before validation, so the key is fully
    # consumed here and never reaches the rest of the framework.
    _apply_sampler_overrides(config, name)
    _apply_canonical_anemoi_reference(config, name)
    _validate_lane(config, path)
    return config


def load_host(name: str, overrides: dict | None = None) -> dict:
    path = _CONFIG_DIR / "hosts" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Host config not found: {path}")
    config = _load_yaml(path)
    if overrides:
        config = _deep_merge(config, overrides)
    _validate_host(config, path)
    return config


def allowed_hosts_for_stage(lane_config: dict[str, Any], stage: str | None = None) -> list[str] | None:
    """Return allowed hosts for a lane, optionally scoped to a CLI stage."""

    allowed_hosts = lane_config.get("allowed_hosts")
    if allowed_hosts is None or isinstance(allowed_hosts, list):
        return allowed_hosts
    if not isinstance(allowed_hosts, dict):
        return None
    if stage and stage in allowed_hosts:
        return allowed_hosts[stage]
    return allowed_hosts.get("default")


def default_host_for_stage(lane_config: dict[str, Any], stage: str | None = None) -> str | None:
    """Return the stage-specific default host when a lane declares one."""

    allowed_hosts = allowed_hosts_for_stage(lane_config, stage)
    if allowed_hosts:
        return allowed_hosts[0]
    return lane_config.get("default_host")


def validate_lane_host_compatible(
    lane_name: str,
    lane_config: dict[str, Any],
    host_name: str,
    stage: str | None = None,
) -> None:
    """Fail fast when a lane/stage is rendered or run on a forbidden host."""

    allowed_hosts = allowed_hosts_for_stage(lane_config, stage)
    if allowed_hosts is None:
        return
    stage_text = f" stage '{stage}'" if stage else ""
    if not allowed_hosts:
        raise ConfigValidationError(
            f"Lane '{lane_name}'{stage_text} cannot be run as a single-host stage. "
            "Run prepare/predict and evaluate/scoreboard on their stage-specific hosts."
        )
    if host_name not in allowed_hosts:
        default_host = default_host_for_stage(lane_config, stage)
        suffix = f" Default host: {default_host}." if default_host else ""
        raise ConfigValidationError(
            f"Lane '{lane_name}'{stage_text} must be run on host(s) "
            f"{allowed_hosts}; got '{host_name}'.{suffix}"
        )


def load_event(name: str) -> dict:
    path = _CONFIG_DIR / "events" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Event config not found: {path}")
    config = _load_yaml(path)
    _validate_event(config, path)
    return config
