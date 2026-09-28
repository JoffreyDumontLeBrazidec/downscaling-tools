"""Facts about one evaluator, gathered from the places that already hold them.

There is no second catalogue. The role and the one-sentence question come from
``eval/evaluators/registry.py``; the method summary is the package docstring;
what it needs and writes come from its ``EVALUATOR_SPEC``; the lane configuration
keys are read from the evaluator's own code (``eval_config`` lookups) and from the
lane YAML files that set them. ``eval.cli list`` and ``eval.cli describe`` only
format what this module returns.
"""
from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

from eval.evaluators import registry

EVAL_ROOT = Path(__file__).resolve().parents[1]
LANES_DIR = EVAL_ROOT / "config" / "lanes"

REQUIREMENT_TEXT = {
    "predictions": "prediction files (predictions_<date>_step<NNN>.nc) written by `predict`",
    "checkpoint": "the model checkpoint (--checkpoint); the evaluator runs the model itself",
}

# Names under which evaluator code holds its own lane block.
_CONFIG_NAMES = {"eval_config", "cfg", "ecfg", "config"}


def _first_docstring(path: Path) -> str | None:
    try:
        return ast.get_docstring(ast.parse(path.read_text()))
    except (OSError, SyntaxError):
        return None


def _package_dir(name: str) -> Path | None:
    entry = registry.get(name)
    if entry is None:
        return None
    if entry.group == registry.RETIRED:
        return EVAL_ROOT / "_quarantine" / str(entry.retired_on) / name
    return EVAL_ROOT / "evaluators" / name


def config_keys_read(name: str) -> list[str]:
    """Keys the evaluator's own code looks up in its lane block, found by reading the source.

    Matches ``eval_config.get("key")`` and ``eval_config["key"]`` (and the same on
    the other names in ``_CONFIG_NAMES``) in every non-test module of the package.
    It cannot see keys passed on to a backend as a whole dict, so it can miss some;
    ``lane_keys_set`` shows what the lane files actually set.
    """
    pkg = _package_dir(name)
    keys: set[str] = set()
    if pkg is None or not pkg.is_dir():
        return []
    for path in sorted(pkg.rglob("*.py")):
        if "tests" in path.relative_to(pkg).parts:
            continue
        try:
            tree = ast.parse(path.read_text())
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("get", "pop", "setdefault")
                    and isinstance(node.func.value, ast.Name) and node.func.value.id in _CONFIG_NAMES
                    and node.args and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                keys.add(node.args[0].value)
            elif (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                  and node.value.id in _CONFIG_NAMES and isinstance(node.slice, ast.Constant)
                  and isinstance(node.slice.value, str)):
                keys.add(node.slice.value)
    return sorted(keys)


def lane_keys_set(name: str) -> dict[str, int]:
    """For each key of the ``<name>:`` block in the tracked lane YAML files, how many lanes set it."""
    import yaml

    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    counts: dict[str, int] = {}
    for path in sorted(LANES_DIR.glob("*.yaml")):
        try:
            data = yaml.load(path.read_text(), Loader=loader) or {}
        except Exception:
            continue
        block = data.get(name) if isinstance(data, dict) else None
        if isinstance(block, dict):
            for key in block:
                counts[str(key)] = counts.get(str(key), 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def lanes_running(name: str) -> list[str]:
    """Tracked lanes whose evaluator groups list this evaluator (default or diagnostics)."""
    import yaml

    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    found: list[str] = []
    for path in sorted(LANES_DIR.glob("*.yaml")):
        try:
            data = yaml.load(path.read_text(), Loader=loader) or {}
        except Exception:
            continue
        groups = (data.get("evaluator_groups") or {}) if isinstance(data, dict) else {}
        if any(name in (v or []) for v in groups.values() if isinstance(v, list)):
            found.append(path.stem)
    return found


def example_command(name: str, requires: list[str], lane: str) -> str:
    parts = [f"python -m eval.cli evaluate --lane {lane} --only {name}",
             "--predictions-dir <run>/predictions" if "predictions" in requires else None,
             "--checkpoint <checkpoint>" if "checkpoint" in requires else None]
    return " ".join(p for p in parts if p)


def summaries() -> list[dict]:
    """One dict per registry entry, in registry order: what ``eval.cli list`` prints."""
    rows = []
    for name in registry.names():
        e = registry.get(name)
        rows.append({
            "name": e.name, "group": e.group, "feeds_scoreboard": e.feeds_scoreboard,
            "question": e.question, "host_prefix": e.host_prefix,
            "replacement": e.replacement, "retired_on": e.retired_on,
        })
    return rows


def describe(name: str) -> dict:
    """Everything ``eval.cli describe <name>`` prints, as plain data. Raises KeyError for unknown names."""
    entry = registry.get(name)
    if entry is None:
        raise KeyError(name)
    info: dict = {
        "name": name, "group": entry.group, "feeds_scoreboard": entry.feeds_scoreboard,
        "question": entry.question, "host_prefix": entry.host_prefix,
        "replacement": entry.replacement, "retired_on": entry.retired_on,
        "package": str(_package_dir(name).relative_to(EVAL_ROOT.parent)),
    }
    if entry.group == registry.RETIRED:
        pkg = _package_dir(name)
        info["method"] = _first_docstring(pkg / "__init__.py")
        info["tombstone"] = registry.retired_message(name)
        return info

    mod = importlib.import_module(f"eval.evaluators.{name}")
    spec = getattr(mod, "EVALUATOR_SPEC", {})
    requires = list(spec.get("requires", []))
    info["method"] = inspect.getdoc(mod)
    info["requires"] = requires
    info["requires_text"] = [REQUIREMENT_TEXT.get(r, r) for r in requires]
    info["outputs"] = list(spec.get("outputs", []))
    info["deliverables"] = spec.get("deliverables") or {}
    info["results_dir"] = f"<run>/evaluators/{name}/"
    info["config_keys_read"] = config_keys_read(name)
    info["lane_keys_set"] = lane_keys_set(name)
    lanes = lanes_running(name)
    info["lanes_running"] = lanes
    example_lane = next((l for l in ("o320_o1280", "o96_o320", "o1280_o2560") if l in lanes),
                        lanes[0] if lanes else "o96_o320")
    info["example"] = example_command(name, requires, example_lane)
    if entry.feeds_scoreboard:
        info["scoreboard_example"] = f"python -m eval.cli scoreboard --lane {example_lane} --eval-dir <run> --only {name}"
    return info
