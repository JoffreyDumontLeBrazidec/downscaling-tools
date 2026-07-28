"""Ladder grid: 3 weather states (rows) x 3 metrics (columns), one curve per experiment.

Layout requested 2026-07-27, adapted from the CRPS version to the metrics this work produces:

  columns   RMSE(ens mean)  |  spectra rel-L2
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
    ("spectra rel-L2", "spectra_{f}_relative_l2", True, "sf"),
]

# ---------------------------------------------------------------------------------------
# TC-score proxy component. Rows are EVENTS (discovered from the card, so a lane with more
# events grows the panel by itself); columns are the two eye numbers that must be read
# together plus peak wind.
#
# `eye_deepest` alone is an instance lottery at ladder budgets -- it can swing many hPa while
# the distribution does not move. `eye_casemin_mean` is the same physical quantity averaged
# over cases, so it moves only when the whole distribution moves. A trend is believable only
# where BOTH move the same way. Measured 2026-07-27; see
# epics/training-diagnostics/metric-skill-gap/in-progress/20260727_ladder_tc_proxy_studyA.md
#
# These read `tcproxy_*`, NOT `tc_*`: the `tc` evaluator scores on the lane's `support_mode`
# (regridded on most lanes) while the proxy uses the model's native grid. TC extremes are
# support-dependent, so the two families must never share a column.
TC_COLS = [
    ("deepest eye", "tcproxy_{f}_eye_deepest", True, "hPa"),
    ("avg per-case deepest eye", "tcproxy_{f}_eye_casemin_mean", True, "hPa"),
    ("peak 10 m wind", "tcproxy_{f}_wind_peak", None, "m/s"),
]
# non-training curves that ride inside the SAME prediction files, so they cannot drift in
# support: label -> metric infix, plus the line style used for them.
TC_ANCHORS = [("ENFO target", "enfo", "black", "-"), ("EEFO input", "eefo", "#777777", ":")]


def tc_rows(cards):
    """Discover the events present in the cards -> one row per event, in a stable order."""
    events = set()
    for _, ladder, _ in cards:
        for row in ladder.get("rows", []):
            for key in row.get("metrics", {}):
                if key.startswith("tcproxy_") and key.endswith("_eye_deepest"):
                    stem = key[len("tcproxy_"):-len("_eye_deepest")]
                    # skip the anchor variants: <event>_enfo, <event>_eefo
                    if not stem.endswith("_enfo") and not stem.endswith("_eefo"):
                        events.add(stem)
    return [(ev, ev, ev, "hPa") for ev in sorted(events)]
CURVE_COLORS = ["#1f77b4", "#ff7f0e", "#9467bd", "#8c564b", "#17becf"]
# non-training anchors: solid black reads as "the target", grey dash-dot as "the raw input"
HLINE_STYLES = [("black", "-"), ("#777777", "-."), ("#2ca02c", "-.")]


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
    ap.add_argument("--ref", help="LABEL=/path/to/ladder.json -- the reference RUN, drawn as "
                                  "its own curve vs step (not a frozen value)")
    ap.add_argument("--hline", action="append", default=[],
                    help="LABEL=/path/to/flat.json -- a non-training reference drawn as a "
                         "horizontal line (repeatable), e.g. the EEFO input or the ENFO target")
    ap.add_argument("--region", default="n.hem")
    ap.add_argument("--component", default="surface", choices=("surface", "tc"),
                    help="surface = RMSE/spectra over weather states; "
                         "tc = the TC-score proxy over events")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", dest="title", action="store_true", default=False,
                    help="draw the heading block (off by default: the dashboard supplies its "
                         "own header). Support provenance is ALWAYS drawn as a footer.")
    ap.add_argument("--allow-mixed-support", action="store_true")
    args = ap.parse_args()

    exps = [load_card(s) for s in args.exp]
    ref = load_card(args.ref) if args.ref else None
    hlines = []
    for spec in args.hline:
        lab, _, path = spec.partition("=")
        hlines.append((lab, json.loads(Path(path).read_text())))

    supports = {support_of(l) for _, l, _ in exps} | ({support_of(ref[1])} if ref else set())
    mixed = len(supports) > 1
    if mixed and not args.allow_mixed_support:
        raise SystemExit("cards are on DIFFERENT support and are not comparable:\n  " +
                         "\n  ".join(sorted(supports)) +
                         "\nRe-score onto one budget, or pass --allow-mixed-support.")

    if args.component == "tc":
        rows, cols = tc_rows(exps + ([ref] if ref else [])), TC_COLS
        if not rows:
            raise SystemExit(
                "no tcproxy_* metrics in these cards -- run the `tc_proxy` evaluator first "
                "(eval.cli evaluate --only tc_proxy), then re-collect.")
    else:
        rows, cols = ROWS, COLS

    fig, axes = plt.subplots(len(rows), len(cols), figsize=(6.6 * len(cols), 3.6 * len(rows)),
                             squeeze=False)
    missing_enfo = False
    legend_done = False
    drift_notes: list[str] = []
    for ri, (row_label, ws, sf, unit) in enumerate(rows):
        for ci, (col_label, tpl, lower_better, kind) in enumerate(cols):
            ax = axes[ri][ci]
            field = row_label if args.component == "tc" else (ws if kind == "ws" else sf)
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
                # the reference is another RUN: plot its whole trajectory, not one frozen value
                rlabel, rladder, _ = ref
                st, v = series(rladder, key)
                if np.isfinite(v).any():
                    ax.plot(st, v, "--s", ms=4, lw=1.6, color="#d62728", zorder=2,
                            label="ref: %s" % rlabel)
                    drew = True

            if args.component == "tc" and key is not None:
                # the target/input curves travel inside the same prediction files, so they are
                # invariant across rungs by construction. If they are NOT, the support moved
                # and the whole panel is meaningless -> say so on the figure.
                for alabel, infix, acolor, als in TC_ANCHORS:
                    akey = key.replace(f"tcproxy_{field}_", f"tcproxy_{field}_{infix}_")
                    vals = []
                    for _, ladder, _ in exps + ([ref] if ref else []):
                        _, av = series(ladder, akey)
                        vals.extend(v for v in av if np.isfinite(v))
                    if not vals:
                        continue
                    if max(vals) - min(vals) > 1e-6:
                        drift_notes.append(
                            "%s %s %s drifts %.3g across rungs" %
                            (row_label, col_label, alabel, max(vals) - min(vals)))
                    ax.axhline(float(vals[0]), lw=2.0, zorder=4, label=alabel,
                               color=acolor, ls=als)
                    drew = True
            for hi, (hlabel, hvals) in enumerate(hlines):
                hv = hvals.get(key) if key else None
                if hv is None:
                    continue
                ax.axhline(float(hv), lw=2.0, zorder=4, label=hlabel,
                           color=HLINE_STYLES[hi % len(HLINE_STYLES)][0],
                           ls=HLINE_STYLES[hi % len(HLINE_STYLES)][1])
                drew = True
            if key is not None and not hlines and args.component != "tc":
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
                # the ENFO/EEFO anchors bracket the model, so without headroom they sit flush
                # against the frame and read as clipped
                ax.margins(y=0.12)
                if args.component == "tc" and "eye" in (tpl or ""):
                    ax.invert_yaxis()   # deeper eye = lower hPa -> put "better" upwards
                ax.set_xlabel("training step", fontsize=9)
                ax.set_ylabel(kind if args.component == "tc"
                              else (unit if kind == "ws" else "relative L2"), fontsize=9)
            arrow = "" if lower_better is None else "  (lower = better)"
            ax.set_title(("%s%s" % (col_label, arrow)) if args.component == "tc"
                         else ("%s — %s%s" % (col_label, field or row_label, arrow)), fontsize=11)
            if drew and not legend_done:
                ax.legend(fontsize=8.5, loc="best")
                legend_done = True
            if ci == 0:
                ax.text(-0.20, 0.5, row_label, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=13, fontweight="bold")

    sup = ("SUPPORT  " + sorted(supports)[0]) if not mixed else \
          ("MIXED SUPPORT - curves are NOT comparable: " + " || ".join(sorted(supports)))
    note = ("" if not missing_enfo else
            "\nno non-training reference supplied (--hline)")
    if args.title:
        head = "Ladder grid" if args.component == "surface" else "TC-score proxy"
        fig.suptitle("%s  |  region %s\n%s%s" % (head, args.region, sup, note), fontsize=11)
        fig.tight_layout(rect=[0.012, 0, 1, 0.93])
    else:
        fig.tight_layout(rect=[0.012, 0.045, 1, 1])
    # provenance is NOT decoration: it is the same-support guarantee, so it is drawn whether or
    # not the heading is. Any anchor drift is appended in red -- it invalidates the panel.
    foot = sup + ("" if args.component == "tc" else note.replace("\n", "  "))
    fig.text(0.012, 0.012, foot, fontsize=7.5, color="#c0392b" if mixed else "#555555")
    if drift_notes:
        fig.text(0.012, 0.028,
                 "ANCHOR DRIFT (support moved, panel not comparable): " + "; ".join(drift_notes[:3]),
                 fontsize=7.5, color="#c0392b", fontweight="bold")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
