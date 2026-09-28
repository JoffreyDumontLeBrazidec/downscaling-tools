"""One definition of how to obtain the external binaries the eval pipeline needs.

The recipes live in ``eval/config/toolchains.yaml``.  Both the Python evaluators
and the hand-written job scripts render their module setup from here, so a
module version moves in one place rather than in every caller.

Each rendered block ends in a positive probe.  Callers used to write
``module load X 2>/dev/null || true``, which hid why a module failed and let the
failure surface much later as a confusing downstream error.  A missing binary is
now reported where it goes missing, naming the toolchain, the probe and the host.
"""
from __future__ import annotations

import shlex
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "toolchains.yaml"


@lru_cache(maxsize=1)
def _recipes() -> dict[str, dict[str, Any]]:
    with open(_CONFIG_PATH, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{_CONFIG_PATH} must contain a mapping of toolchain names")
    return data


def available() -> list[str]:
    """Names of the toolchains defined in the config."""
    return sorted(_recipes())


def recipe(name: str) -> dict[str, Any]:
    """The raw recipe for one toolchain."""
    recipes = _recipes()
    try:
        return recipes[name]
    except KeyError:
        raise KeyError(
            f"unknown toolchain {name!r}; defined toolchains are {', '.join(sorted(recipes))}"
        ) from None


def render_module_block(name: str) -> str:
    """Render the bash lines that make one toolchain available.

    The block is safe under ``set -euo pipefail``: ``module unload`` tolerates a
    module that was never loaded, ``module load`` does not tolerate failure, and
    the closing probe turns a silently missing binary into an immediate error.
    """
    spec = recipe(name)
    lines: list[str] = [f"# toolchain: {name}"]

    for mod in spec.get("module_unloads") or []:
        lines.append(f"module unload {mod} 2>/dev/null || true")
    for mod in spec.get("module_loads") or []:
        lines.append(f"module load {mod}")
    for key, value in (spec.get("exports") or {}).items():
        lines.append(f'export {key}="{value}"')
    lines.extend(spec.get("shell_post") or [])

    probe = str(spec.get("probe") or "").strip()
    if probe:
        loads = " ".join(spec.get("module_loads") or []) or "(no modules)"
        # The hostname is deliberately left for the shell to expand: this block is
        # rendered on one host and often runs on another, and knowing which
        # compute node lacked the binary is usually the whole diagnosis.
        message = (
            f"FATAL: toolchain {name}: required command {probe} is not on PATH after "
            f"module load [{loads}]. Nothing downstream will work, so stopping here "
            f"rather than failing later. Host:"
        )
        lines.append(
            f"command -v {shlex.quote(probe)} >/dev/null 2>&1 || "
            f'{{ echo {shlex.quote(message)} "$(hostname)" >&2; exit 1; }}'
        )

    return "\n".join(lines)


def resolve(name: str) -> dict[str, Any]:
    """Actually load the toolchain in a login shell and report what it resolved to.

    Used to record provenance in run summaries and to diagnose a host quickly.
    Raises ``RuntimeError`` if the probe fails.
    """
    spec = recipe(name)
    probe = str(spec.get("probe") or "").strip()
    version_env = str(spec.get("version_env") or "").strip()

    script = "\n".join(
        [
            "set -euo pipefail",
            render_module_block(name),
            f'echo "__PATH__=$(command -v {shlex.quote(probe)})"' if probe else 'echo "__PATH__="',
            f'echo "__VERSION__=${{{version_env}:-}}"' if version_env else 'echo "__VERSION__="',
        ]
    )
    proc = subprocess.run(
        ["bash", "--login", "-c", script],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"toolchain {name!r} could not be resolved on this host:\n"
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )

    out: dict[str, Any] = {
        "toolchain": name,
        "modules": list(spec.get("module_loads") or []),
        "path": "",
        "version": "",
    }
    for line in proc.stdout.splitlines():
        if line.startswith("__PATH__="):
            out["path"] = line.split("=", 1)[1].strip()
        elif line.startswith("__VERSION__="):
            out["version"] = line.split("=", 1)[1].strip()
    return out


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_render = sub.add_parser("render", help="print the bash block for a toolchain")
    p_render.add_argument("name")
    p_resolve = sub.add_parser("resolve", help="load a toolchain and report what it resolved to")
    p_resolve.add_argument("name")
    sub.add_parser("list", help="list defined toolchains")

    args = parser.parse_args()
    if args.command == "list":
        for name in available():
            print(name)
    elif args.command == "render":
        print(render_module_block(args.name))
    else:
        for key, value in resolve(args.name).items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
