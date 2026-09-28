"""``eval.cli config``: print a lane's fully resolved configuration.

The command module is named ``lane_config`` (not ``config``) so it is not confused
with the ``eval.config`` package that holds the YAML files and the loader.
"""
from __future__ import annotations

import argparse

from eval.cli._common import Command
from eval.config.loader import load_lane

SUMMARY = "Print a lane's fully resolved configuration."
DESCRIPTION = (
    "Resolve a lane YAML the way every other command resolves it, following its base: "
    "chain and merging each level, and print the result. This answers 'what will actually "
    "be used', which a chain of base: lookups does not make obvious. It only reads "
    "configuration; it never predicts, scores or submits anything. Loader warnings (for "
    "example a sampler block that drops keys from its base) go to stderr, so the printed "
    "configuration stays clean and pipeable."
)


def register(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser("config", help=SUMMARY, description=DESCRIPTION)
    p.add_argument("lane", nargs="?", default=None, help="Lane to resolve. Omit it when using --samplers.")
    p.add_argument("--sampler", action="store_true", default=False, help="Print only predict.sampler.")
    p.add_argument(
        "--samplers", action="store_true", default=False,
        help="Print the sampler of the four canonical lanes side by side and say which keys differ.",
    )
    p.add_argument("--json", dest="as_json", action="store_true", default=False, help="Emit JSON instead of YAML.")
    return p


CANONICAL_LANES = ("o48_o96", "o96_o320", "o320_o1280", "o1280_o2560")


def _run_config_subcommand(args) -> int:
    """Print a lane's fully resolved configuration, or the canonical lane samplers.

    Read-only. The lane YAMLs are the authority for what a run uses; this prints what
    they resolve to so nobody has to follow a base: chain by hand or trust a number
    remembered from somewhere else.
    """
    import json as _json
    import yaml as _yaml

    def _emit(obj) -> None:
        if args.as_json:
            print(_json.dumps(obj, indent=2, sort_keys=True, default=str))
        else:
            print(_yaml.safe_dump(obj, sort_keys=True, default_flow_style=False).rstrip())

    if args.samplers:
        resolved = {}
        for lane in CANONICAL_LANES:
            try:
                resolved[lane] = (load_lane(lane).get("predict") or {}).get("sampler") or {}
            except Exception as exc:  # a broken lane must not hide the others
                resolved[lane] = {"ERROR": f"{type(exc).__name__}: {exc}"}
        if args.as_json:
            _emit(resolved)
            return 0
        keys = sorted({k for v in resolved.values() for k in v})
        shared = {k: resolved[CANONICAL_LANES[0]].get(k) for k in keys
                  if len({repr(resolved[l].get(k)) for l in CANONICAL_LANES}) == 1}
        differing = [k for k in keys if k not in shared]
        width = max((len(k) for k in keys), default=12) + 2
        print("Canonical lane samplers, resolved from eval/config/lanes/<lane>.yaml")
        print("(the YAML files are the authority; this is printed from them, not remembered)\n")
        print("differing across lanes:")
        print(" " * width + "".join(f"{l:>18}" for l in CANONICAL_LANES))
        for k in differing:
            print(f"{k:<{width}}" + "".join(f"{str(resolved[l].get(k)):>18}" for l in CANONICAL_LANES))
        if shared:
            print("\nidentical on all four lanes:")
            for k, v in sorted(shared.items()):
                print(f"  {k} = {v}")
        return 0

    if not args.lane:
        raise SystemExit("config: give a lane name, or use --samplers for the four canonical lanes.")
    try:
        cfg = load_lane(args.lane)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Lane config not found: '{args.lane}'. "
            f"Available lanes are YAML files in eval/config/lanes/. Error: {exc}"
        ) from exc
    except Exception as exc:
        raise SystemExit(f"Failed to load lane config '{args.lane}': {exc}") from exc
    _emit((cfg.get("predict") or {}).get("sampler") or {} if args.sampler else cfg)
    return 0


def run(args: argparse.Namespace) -> None:
    raise SystemExit(_run_config_subcommand(args))


COMMANDS = (Command("config", "maintenance", SUMMARY, register, run),)
