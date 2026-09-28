"""``eval.cli list`` and ``eval.cli describe``: find out what can be evaluated.

Both read the evaluator registry (``eval/evaluators/registry.py``) and the
evaluator packages; they read no lane and no host configuration and run nothing.
"""
from __future__ import annotations

import argparse
import json
import textwrap

from eval.cli._common import Command
from eval.evaluators import describe as describe_module
from eval.evaluators import registry

LIST_SUMMARY = "List every evaluator with its question, role and constraints."
DESCRIBE_SUMMARY = "Explain one evaluator: question, method, inputs, outputs, lane keys, example."

_GROUP_TEXT = {
    registry.SCORED: "scored: produce the rows that the lane scoreboard ranks runs by",
    registry.STANDARD: "standard: run by default on a lane, but produce no scoreboard row",
    registry.DIAGNOSTIC: "diagnostic: asked for explicitly, to explain a result rather than rank it",
    registry.RETIRED: "retired: can no longer be run (naming one with --only prints its replacement)",
}


def _wrap(text: str, indent: int) -> str:
    return textwrap.fill(" ".join(text.split()), width=100, initial_indent=" " * indent,
                         subsequent_indent=" " * indent, break_on_hyphens=False)


def _bullet(text: str) -> str:
    return textwrap.fill(" ".join(text.split()), width=100, initial_indent="  - ", subsequent_indent="    ", break_on_hyphens=False)


def register_list(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser("list", help=LIST_SUMMARY, description=(
        LIST_SUMMARY + " The list is derived from eval/evaluators/registry.py, the one "
        "place evaluators are declared. Use `describe <name>` for the details of one."))
    p.add_argument("--json", dest="as_json", action="store_true", default=False,
                   help="Print JSON instead of text.")
    return p


def run_list(args: argparse.Namespace) -> None:
    rows = describe_module.summaries()
    if args.as_json:
        print(json.dumps(rows, indent=2))
        return
    print("Evaluators, from eval/evaluators/registry.py. "
          "`python -m eval.cli describe <name>` explains one.\n")
    for group in registry.GROUPS:
        members = [r for r in rows if r["group"] == group]
        print(_GROUP_TEXT[group])
        for r in members:
            print(f"  {r['name']}")
            print(_wrap(r["question"], 6))
            facts = []
            if group != registry.RETIRED:
                facts.append("feeds the scoreboard: " + ("yes" if r["feeds_scoreboard"] else "no"))
                facts.append("host: " + (f"Atos {r['host_prefix'].upper()} only (host name starts with '{r['host_prefix']}')"
                                         if r["host_prefix"] else "any"))
            else:
                facts.append(f"retired on {r['retired_on']}")
                facts.append("replacement: " + (r["replacement"] or "none"))
            print("      " + "; ".join(facts))
        print()


def register_describe(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser("describe", help=DESCRIBE_SUMMARY, description=(
        DESCRIBE_SUMMARY + " Reads the registry, the evaluator package and the lane files."))
    p.add_argument("evaluator", help="Evaluator name, as printed by `list`.")
    p.add_argument("--json", dest="as_json", action="store_true", default=False,
                   help="Print JSON instead of text.")
    return p


def _section(title: str) -> None:
    print(f"\n{title}")


def run_describe(args: argparse.Namespace) -> None:
    try:
        info = describe_module.describe(args.evaluator)
    except KeyError:
        raise SystemExit(
            f"Unknown evaluator {args.evaluator!r}. Known evaluators: {', '.join(registry.names())}. "
            "Run `python -m eval.cli list` for what each one answers."
        )
    if args.as_json:
        print(json.dumps(info, indent=2, default=str))
        return
    print(f"{info['name']}  [{info['group']}]")
    print(_wrap(info["question"], 2))
    if info["group"] == registry.RETIRED:
        _section("Retired")
        print(_wrap(info["tombstone"], 2))
        if info.get("method"):
            _section("What it did")
            print(textwrap.indent(info["method"], "  "))
        return
    _section("Method")
    print(textwrap.indent(info["method"] or "(no package docstring)", "  "))
    _section("Role")
    print("  feeds the scoreboard: " + ("yes" if info["feeds_scoreboard"] else "no"))
    print("  host: " + (f"Atos {info['host_prefix'].upper()} only (host name starts with '{info['host_prefix']}'); "
                        "elsewhere it is skipped with a warning" if info["host_prefix"] else "any host"))
    _section("Inputs it needs")
    for text in info["requires_text"]:
        print(_bullet(text))
    _section(f"Outputs it writes, below {info['results_dir']}")
    for text in info["outputs"] or ["(not documented in EVALUATOR_SPEC['outputs'])"]:
        print(_bullet(text))
    deliverables = info["deliverables"]
    if deliverables.get("top_level"):
        _section("Promoted to the run root (eval.lean_layout)")
        for item in deliverables["top_level"]:
            print(f"  - {item['src']} as {item['as']}")
    _section("Lane configuration it reads (the `" + info["name"] + ":` block, eval_config)")
    read = info["config_keys_read"]
    print("  looked up by the code: " + (", ".join(read) if read else "none found by reading the source"))
    setting = info["lane_keys_set"]
    print("  set by tracked lanes:  " + (", ".join(f"{k} ({n})" for k, n in setting.items())
                                         if setting else "no lane sets a block for it"))
    lanes = info["lanes_running"]
    print(f"  listed in the evaluator groups of {len(lanes)} tracked lane(s)"
          + (f", for example {', '.join(lanes[:4])}" if lanes else ""))
    _section("Example")
    print("  " + info["example"])
    if info.get("scoreboard_example"):
        print("  " + info["scoreboard_example"])


COMMANDS = (
    Command("list", "discovery", LIST_SUMMARY, register_list, run_list),
    Command("describe", "discovery", DESCRIBE_SUMMARY, register_describe, run_describe),
)
