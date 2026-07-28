"""Ladder grid: 3 weather states (rows) x 3 metrics (columns), one curve per experiment.

Layout requested 2026-07-27, adapted from the CRPS version to the metrics this work produces:

  columns   RMSE(ens mean)  |  spread  |  spectra rel-L2
  rows      10u | 10v | 2t | tp (when it exists on the lane)
  curves    the experiment(s) of interest, the REFERENCE experiment (best working, currently
            SOAP pristine 200k), and ENFO as a flat black line (it does not train).

Everything must be on the SAME SUPPORT: same lane, dates, leads, members, sampler. Cards scored
on different budgets are NOT comparable and the script refuses to draw them together unless
--allow-mixed-support is passed (it still labels the figure loudly).

Usage:
  ladder_grid.py --out FIG.png \
      --exp  LABEL=/path/to/ladder.json  [--exp LABEL=...] \
      --ref  LABEL=/path/to/ladder.json[:STEP] \
      [--enfo /path/to/enfo.json] [--region n.hem]

ENFO json (when it exists) is a flat mapping of the same metric keys used below, e.g.
  {"probabilistic_2t_n.hem_rmse_ens_mean_mean": 1.02, "spectra_2t_relative_l2": 0.031}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (row label, probabilistic weather_state, spectra field, unit)
# 10u and 10v are STORED weather states, so both evaluators can score them. 10ff is not usable
# here: the probabilistic scorer derives it as hypot(10u,10v) but spectra reads stored states
# only and cannot see it -- which is why the rows are the components, not the speed.
# A None entry means "this evaluator has no such field" and renders as an empty panel.
ROWS = [
    ("10u", "10u", "10u", "m/s"),
    ("10v", "10v", "10v", "m/s"),
    ("2t", "2t", "2t", "K"),
    ("tp", "tp", "tp", "mm"),
]
# (column label, key template, "lower is better", which field naming the column uses)
COLS = [
    ("RMSE (ens mean)", "probabilistic_{f}_{region}_rmse_ens_mean_mean", True, "ws"),
    ("spread", "probabilistic_{f}_{region}_spread_mean", None, "ws"),  # no "good" direction
    ("spectra rel-L2", "spectra_{f}_relative_l2", True, "sf"),
]
CURVE_COLORS = ["#1f77b4", "#ff7f0e", "#9467bd", "#8c564b", "#17becf"]


def load_card(spec: str):
    """LABEL=/path/to/ladder.json[:STEP] -> (label, ladder, step_or_None)."""
    label, _, rest = spec.partition("=")
    if not rest:
        raise SystemExit(f"expected LABEL=path, got {spec!r}")
    step = None
    if ":" in rest[2:]:
        rest, _, s = rest.rpartition(":")
        step = int(s)
    ladder = json.loads(Path(rest).read_text())
    return label, ladder, step


def support_of(ladder: dict) -> str:
    pins = ladder.get("profile_pins") or {}
    b = pins.get("budget") or {}
    return "%s | dates=%s steps=%s members=%s" % (
        ladder.get("lane", "?"), b.get("dates", "?"), b.get("steps", "?"), b.get("members", "?"))


def series(ladder: dict, key: str):
    rows = sorted(ladder.get("rows", []), key=lambda r: r["step"])
    st = np.array([r["step"] for r in rows], float)
    v = np.array([r["metrics"].get(key, np.nan) for r in rows], float)
    return st, v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", action="append", required=True,
                    help="LABEL=/path/to/ladder.json (repeatable)")
    ap.add_argument("--ref", help="LABEL=/path/to/ladder.json[:STEP] -- flat reference line "
                                  "(default: that card's LAST rung)")
    ap.add_argument("--enfo", help="json of flat ENFO values keyed by the same metric names")
    ap.add_argument("--region", default="n.hem")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-mixed-support", action="store_true")
    args = ap.parse_args()

    exps = [load_card(s) for s in args.exp]
    ref = load_card(args.ref) if args.ref else None
    enfo = json.loads(Path(args.enfo).read_text()) if args.enfo else {}

    supports = {support_of(l) for _, l, _ in exps} | ({support_of(ref[1])} if ref else set())
    mixed = len(supports) > 1
    if mixed and not args.allow_mixed_support:
        raise SystemExit("cards are on DIFFERENT support and are not comparable:\n  " +
                         "\n  ".join(sorted(supports)) +
                         "\nRe-score onto one budget, or pass --allow-mixed-support.")

    fig, axes = plt.subplots(len(ROWS), len(COLS), figsize=(6.0 * len(COLS), 3.6 * len(ROWS)),
                             squeeze=False)
    missing_enfo = False
    legend_done = False
    for ri, (row_label, ws, sf, unit) in enumerate(ROWS):
        for ci, (col_label, tpl, lower_better, kind) in enumerate(COLS):
            ax = axes[ri][ci]
            field = ws if kind == "ws" else sf
            key = tpl.format(f=field, region=args.region) if field else None
            drew = False

            for ei, (label, ladder, _) in enumerate(exps):
                if key is None:
                    break
                st, v = series(ladder, key)
                if np.isfinite(v).any():
                    ax.plot(st, v, "-o", ms=5, lw=1.8, color=CURVE_COLORS[ei % len(CURVE_COLORS)],
                            label=label, zorder=3)
                    drew = True

            if ref is not None and key is not None:
                rlabel, rladder, rstep = ref
                st, v = series(rladder, key)
                ok = np.isfinite(v)
                if ok.any():
                    rv = (v[list(st).index(rstep)] if rstep in list(st) else v[ok][-1])
                    rs = rstep if rstep in list(st) else int(st[ok][-1])
                    ax.axhline(rv, ls="--", lw=1.6, color="#d62728", zorder=2,
                               label="ref: %s @%dk" % (rlabel, round(rs / 1000)))
                    drew = True

            ev = enfo.get(key) if key else None
            if ev is not None:
                ax.axhline(float(ev), ls="-", lw=2.0, color="black", zorder=4, label="ENFO")
                drew = True
            elif key is not None:
                missing_enfo = True

            if not drew:
                ax.text(0.5, 0.5, "not available\non this lane", transform=ax.transAxes,
                        ha="center", va="center", fontsize=11, color="#888888")
                ax.set_facecolor("#f5f5f5")
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.xaxis.set_major_formatter(
                    matplotlib.ticker.FuncFormatter(lambda x, _: "%gk" % (x / 1000)))
                ax.grid(alpha=0.25)
                ax.set_xlabel("training step", fontsize=9)
                ax.set_ylabel(unit if kind == "ws" else "relative L2", fontsize=9)
            arrow = "" if lower_better is None else "  (lower = better)"
            ax.set_title("%s — %s%s" % (col_label, field or row_label, arrow), fontsize=11)
            if drew and not legend_done:
                ax.legend(fontsize=8.5, loc="best")
                legend_done = True
            if ci == 0:
                ax.text(-0.20, 0.5, row_label, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=13, fontweight="bold")

    sup = ("SUPPORT  " + sorted(supports)[0]) if not mixed else \
          ("MIXED SUPPORT - curves are NOT comparable: " + " || ".join(sorted(supports)))
    note = ("" if not missing_enfo else
            "\nENFO line absent: no ENFO score exists on this support yet")
    fig.suptitle("Ladder grid  |  region %s\n%s%s" % (args.region, sup, note), fontsize=11)
    fig.tight_layout(rect=[0.012, 0, 1, 0.93])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
