"""Cleanup tracked prepml experiments via the ecFlow `run/delete/*` tasks.

The prepml-deployed ecFlow suite for each experiment carries a `run/delete`
family with tasks `{quaver, s3, mars, fdb, catalogue, workdir}`. The family
sits at `defstatus complete`, so force-running a task triggers the actual
delete. Running just `run/delete/fdb` matches the announcement promise of
"delete all data from FDB" while keeping the catalogue entry intact.

Ledger source: ~/.config/eval/prepml_consumed.jsonl (written by
`eval.predict.prepml.record_consumed`).
"""
from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from eval.predict.prepml import (
    ECFLOW_BIN,
    ECFLOW_ENV,
    LEDGER_PATH,
    discover_owner,
    ecflow_client,
    mark_cleaned,
    read_ledger,
)

LOG = logging.getLogger(__name__)

# Tasks under <owner>/<expver>/run/delete/<task>. The announcement preserves
# the catalogue entry so a cleaned experiment can be re-submitted; we never
# include it in either scope.
DELETE_TASKS_FDB_ONLY = ("fdb",)
DELETE_TASKS_ALL = ("quaver", "s3", "mars", "fdb", "workdir")  # catalogue intentionally omitted


@dataclass
class LedgerRow:
    expver: str
    owner: str
    lane: str
    last_consumed_utc: str
    run_dir: str
    cleaned_utc: str | None
    ecflow_state: str | None
    fdb_state: str | None

    @property
    def is_cleaned(self) -> bool:
        return bool(self.cleaned_utc) or self.fdb_state == "complete-after-cleanup"


def _state_from_ecflow_output(stdout: str) -> str | None:
    """Parse an ecflow_client --get_state output line and return the state token."""
    text = stdout.strip()
    if not text:
        return None
    m = re.search(r"\bstate:(\w+)", text)
    if m:
        return m.group(1)
    last = text.splitlines()[-1].strip()
    return last or None


def _ecflow_state(path: str) -> str | None:
    try:
        result = ecflow_client(["--get_state", path], check=False, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        LOG.debug("ecflow_client --get_state %s failed: %s", path, exc)
        return None
    if result.returncode != 0:
        return None
    return _state_from_ecflow_output(result.stdout)


def _collapse_records(records: Iterable[dict]) -> list[LedgerRow]:
    """Collapse multiple ledger lines per (owner, expver) into one row.

    Keeps the newest `ts_utc` as last_consumed; cleaned_utc is the latest
    non-null cleaned_ts_utc seen for that (owner, expver). Other fields
    come from the newest record.
    """
    by_key: dict[tuple[str, str], list[dict]] = {}
    for rec in records:
        key = (rec.get("owner", ""), rec.get("expver", ""))
        by_key.setdefault(key, []).append(rec)
    rows: list[LedgerRow] = []
    for (owner, expver), recs in by_key.items():
        recs.sort(key=lambda r: r.get("ts_utc", ""))
        newest = recs[-1]
        cleaned = next(
            (r.get("cleaned_ts_utc") for r in reversed(recs) if r.get("cleaned_ts_utc")),
            None,
        )
        rows.append(
            LedgerRow(
                expver=expver,
                owner=owner,
                lane=newest.get("lane", ""),
                last_consumed_utc=newest.get("ts_utc", ""),
                run_dir=newest.get("run_dir", ""),
                cleaned_utc=cleaned,
                ecflow_state=None,
                fdb_state=None,
            )
        )
    rows.sort(key=lambda r: r.last_consumed_utc, reverse=True)
    return rows


def load_rows(*, probe_ecflow: bool = True, ledger_path: Path | None = None) -> list[LedgerRow]:
    """Load the ledger and (optionally) annotate each row with ecFlow state."""
    rows = _collapse_records(read_ledger(ledger_path or LEDGER_PATH))
    if not probe_ecflow:
        return rows
    for row in rows:
        row.ecflow_state = _ecflow_state(f"/{row.owner}/{row.expver}")
        row.fdb_state = _ecflow_state(f"/{row.owner}/{row.expver}/run/delete/fdb")
    return rows


def render_table(rows: list[LedgerRow]) -> str:
    """Format rows as a fixed-width table for terminal output."""
    if not rows:
        return "(ledger empty — no prepml experiments tracked yet)"
    header = ("idx", "expver", "lane", "last consumed (UTC)", "cleaned?", "suite", "fdb task")
    body = [
        (
            str(i + 1),
            r.expver,
            r.lane or "-",
            r.last_consumed_utc.split(".")[0],
            "yes" if r.is_cleaned else "no",
            r.ecflow_state or "?",
            r.fdb_state or "?",
        )
        for i, r in enumerate(rows)
    ]
    widths = [max(len(h), *(len(b[i]) for b in body)) for i, h in enumerate(header)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*header), fmt.format(*("-" * w for w in widths))]
    lines.extend(fmt.format(*r) for r in body)
    return "\n".join(lines)


def _select_interactive(rows: list[LedgerRow]) -> list[LedgerRow]:
    if not rows:
        print("Ledger is empty; nothing to clean.")
        return []
    print(render_table(rows))
    print(
        "\nEnter the indices to clean (comma-separated, e.g. '1,3,4'), "
        "'all' for every uncleaned row, or empty to abort."
    )
    try:
        raw = input("> ").strip()
    except EOFError:
        return []
    if not raw:
        return []
    if raw.lower() == "all":
        return [r for r in rows if not r.is_cleaned]
    picks: list[LedgerRow] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            idx = int(tok) - 1
        except ValueError:
            print(f"  ! ignoring non-integer '{tok}'", file=sys.stderr)
            continue
        if 0 <= idx < len(rows):
            picks.append(rows[idx])
        else:
            print(f"  ! out-of-range index {tok}", file=sys.stderr)
    return picks


def _run_delete_task(task_path: str, *, dry_run: bool) -> bool:
    """Force a task to queued, then force-run it. Returns True on success."""
    queue_args = ["--force=queued", task_path]
    run_args = ["--run", "--force", task_path]
    for args in (queue_args, run_args):
        printable = f"ECF_HOST={ECFLOW_ENV['ECF_HOST']} ECF_PORT={ECFLOW_ENV['ECF_PORT']} {ECFLOW_BIN} {' '.join(args)}"
        if dry_run:
            print(f"DRY-RUN: {printable}")
            continue
        try:
            result = ecflow_client(args, check=False, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"  ! ecflow_client {args[0]} failed: {exc}", file=sys.stderr)
            return False
        if result.returncode != 0:
            print(
                f"  ! ecflow_client {args[0]} exited {result.returncode}: "
                f"{result.stderr.strip()}",
                file=sys.stderr,
            )
            return False
    return True


def clean_expver(
    row: LedgerRow,
    *,
    tasks: tuple[str, ...],
    dry_run: bool,
) -> bool:
    """Force-run the delete tasks for one experiment. Returns True if all succeeded."""
    print(f"\n=> {row.owner}/{row.expver} (lane={row.lane or '-'}; scope tasks={list(tasks)})")
    ok = True
    for task in tasks:
        path = f"/{row.owner}/{row.expver}/run/delete/{task}"
        if not _run_delete_task(path, dry_run=dry_run):
            ok = False
    if ok and not dry_run:
        updated = mark_cleaned(row.expver, owner=row.owner)
        if updated:
            print(f"   ledger updated ({updated} record(s) marked cleaned)")
    return ok


def _confirm(prompt: str) -> bool:
    try:
        answer = input(f"{prompt} [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eval.predict.prepml_cleanup",
        description="List and clean tracked prepml experiments via ecFlow.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print the ledger (with ecFlow state) and exit. Default if no --expver given.",
    )
    parser.add_argument(
        "--expver", action="append", default=[],
        help="Clean a specific expver. Repeat for multiple.",
    )
    parser.add_argument(
        "--scope", choices=("fdb", "all"), default="fdb",
        help="Which run/delete tasks to force-run. fdb (default) matches the announcement; "
             "all = fdb + mars + s3 + quaver + workdir (catalogue is always preserved).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the ecflow_client commands that would fire; do not execute.",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Skip the final confirmation prompt.",
    )
    parser.add_argument(
        "--no-ecflow", action="store_true",
        help="Skip the ecFlow state probe when listing (faster, offline-safe).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    rows = load_rows(probe_ecflow=not args.no_ecflow)

    if args.list and not args.expver:
        print(render_table(rows))
        return 0

    tasks = DELETE_TASKS_FDB_ONLY if args.scope == "fdb" else DELETE_TASKS_ALL

    if args.expver:
        chosen = [r for r in rows if r.expver in args.expver]
        missing = sorted(set(args.expver) - {r.expver for r in chosen})
        if missing:
            print(f"warning: not in ledger: {missing}", file=sys.stderr)
            owner = discover_owner()
            for ev in missing:
                chosen.append(
                    LedgerRow(
                        expver=ev, owner=owner, lane="", last_consumed_utc="",
                        run_dir="", cleaned_utc=None, ecflow_state=None, fdb_state=None,
                    )
                )
    else:
        chosen = _select_interactive(rows)

    if not chosen:
        print("Nothing selected; exiting.")
        return 0

    print(f"\nSelected {len(chosen)} experiment(s) for cleanup (scope={args.scope}):")
    for r in chosen:
        print(f"  - {r.owner}/{r.expver}  lane={r.lane or '-'}  cleaned={'yes' if r.is_cleaned else 'no'}")

    if not args.dry_run and not args.yes:
        if not _confirm("Proceed with cleanup?"):
            print("Aborted.")
            return 1

    failures = 0
    for r in chosen:
        if not clean_expver(r, tasks=tasks, dry_run=args.dry_run):
            failures += 1

    if failures:
        print(f"\nDone with {failures} failure(s).", file=sys.stderr)
        return 2
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
