"""``eval.cli verdict``: gates and the one VERDICT line of a finished run (trusted evaluation suite, docs note
epics/misc/20261119_trusted_evaluation_suite.md s4).

    verdict --reader P --out DIR --manifest RUN/MANIFEST.json [--ref-manifest REF/MANIFEST.json] [--check-md5]
            [--run S --ref S --ens9 S --input S]                                    -> DIR/gates_P.csv
    verdict --reader A --out DIR --run S --ref S --ens9 S --input S                 -> DIR/gates_A.csv
    verdict --reader B --out DIR --spectra-run D --spectra-ref D --battery-run D --battery-ref D
            [--shape-run CSV --shape-ref CSV] [--texture-run JSON --texture-ref JSON] -> DIR/gates_B.csv
    verdict --reader C --out DIR --xset-run D --xset-ref D --xset-ens9 D [--hist D --hist-labels RUN,REF,ENS9]
                                                                                    -> DIR/gates_C.csv
    verdict --out DIR [--row NAME]                    merge: DIR/verdict.md, verdict.json, prints the VERDICT line

S = an obs_points scores_per_date.csv, optionally ``::ROW`` (default row ``model``; ``input`` for --input).
REF = GL512 450k (standard, rule 2). Readers compute from tables; a reader agent never computes a number by hand.
Thresholds are proposals until the calibration (note s7).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from eval.cli._common import Command

SUMMARY = "Gates of the trusted evaluation suite and the one VERDICT line of a finished run."
DESCRIPTION = __doc__.split("\n\n", 1)[1] if __doc__ else SUMMARY


def register(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser("verdict", help=SUMMARY, description=DESCRIPTION,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reader", choices=("P", "A", "B", "C"), default=None, help="write one gates csv; omit to merge")
    p.add_argument("--out", type=Path, required=True, help="verdict folder (gates_*.csv, verdict.md)")
    p.add_argument("--row", default="", help="run name on verdict.md")
    p.add_argument("--blocks", choices=("fixed5", "chain", "date"), default=None,
                   help="date-block rule (default fixed5 for P/A/B on MI months, chain for C on XSET-1)")
    p.add_argument("--members", type=int, default=10, help="ensemble size for the SSR correction (default 10)")
    g = p.add_argument_group("preflight (P)")
    g.add_argument("--manifest", type=Path)
    g.add_argument("--ref-manifest", type=Path)
    g.add_argument("--check-md5", action="store_true")
    g = p.add_argument_group("stations (A, and P5)")
    for r in ("run", "ref", "ens9", "input"):
        g.add_argument(f"--{r}", default=None)
    g = p.add_argument_group("spectra, battery, structure (B)")
    for r in ("spectra-run", "spectra-ref", "battery-run", "battery-ref", "shape-run", "shape-ref", "texture-run",
              "texture-ref"):
        g.add_argument(f"--{r}", type=Path, default=None)
    g.add_argument("--spectra-leads", default="24 72 120 240")
    g = p.add_argument_group("extremes, cyclones (C)")
    for r in ("xset-run", "xset-ref", "xset-ens9", "hist"):
        g.add_argument(f"--{r}", default=None)
    g.add_argument("--hist-labels", default=None, help="RUN,REF,ENS9 labels of the histogram files")
    return p


def _need(args, *names):
    miss = [n for n in names if not getattr(args, n.replace("-", "_"))]
    if miss:
        raise SystemExit(f"eval.cli verdict --reader {args.reader}: missing {', '.join('--' + n for n in miss)}")


def run(args: argparse.Namespace) -> None:
    from eval.verdict import gates, merge, preflight, reader_a, reader_b, reader_c

    out = args.out
    r = args.reader
    if r is None:
        page, line = merge.write(out, args.row)
        print(page)
        print(line)
        return
    rule = args.blocks or ("chain" if r == "C" else "fixed5")
    if r == "P":
        _need(args, "manifest")
        sc = {k: getattr(args, k.lower()) for k in ("RUN", "REF", "ENS9", "INPUT") if getattr(args, k.lower())}
        lines = preflight.run(args.manifest, args.ref_manifest, args.check_md5, sc or None)
    elif r == "A":
        _need(args, "run", "ref", "ens9", "input")
        lines = reader_a.run(args.run, args.ref, args.ens9, args.input, rule, args.members)
    elif r == "B":
        lines = []
        if args.spectra_run and args.spectra_ref:
            lines += reader_b.spectra(args.spectra_run, args.spectra_ref, rule,
                                      tuple(int(x) for x in args.spectra_leads.split()))
        else:
            lines += [gates.GateLine(g, "spectra", True, "MISSING", note="--spectra-run/--spectra-ref not given")
                      for g in ("G-B1", "G-B2")]
        if args.battery_run and args.battery_ref:
            lines += reader_b.battery(args.battery_run, args.battery_ref)
        else:
            lines += [gates.GateLine(g, "battery", True, "MISSING", note="--battery-run/--battery-ref not given")
                      for g in ("G-B3", "G-B5")]
        b4 = []
        if args.shape_run and args.shape_ref:
            b4 += reader_b.shape(args.shape_run, args.shape_ref, rule)
        if args.texture_run and args.texture_ref:
            b4 += reader_b.texture(args.texture_run, args.texture_ref, rule)
        lines += b4 or [gates.GateLine("G-B4", "texture, shape", True, "MISSING", note="no shape/texture outputs given")]
    else:
        _need(args, "xset-run", "xset-ref", "xset-ens9")
        lines = reader_c.cyclones(args.xset_run, args.xset_ref, args.xset_ens9)
        lines += reader_c.rain_tails(args.xset_run, args.xset_ref, args.xset_ens9, rule)
        labs = (args.hist_labels or "").split(",")
        if args.hist and len(labs) == 3:
            lines += reader_c.tails(Path(args.hist), *labs, rule=rule)
        else:
            lines.append(gates.GateLine("G-C4", "tail shares", True, "MISSING", note="--hist and --hist-labels RUN,REF,ENS9"))
    path = gates.write_csv(lines, out / f"gates_{r}.csv")
    hard = [ln for ln in lines if ln.hard]
    print(path)
    for gid in sorted({ln.gate for ln in hard}):
        from eval.verdict import stats
        st = stats.combine([ln.state for ln in hard if ln.gate == gid])
        print(f"{gid} {st} ({sum(ln.gate == gid for ln in hard)} lines)")


COMMANDS = (Command("verdict", "final", SUMMARY, register, run),)
