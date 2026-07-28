"""Experiment-evolution grid: how a training run is doing, against reference / input / target.

One row per weather state, one column per metric family, one curve per experiment. Reached from
`eval.cli evolution`; the dashboard calls the same entry point so a card click renders exactly
this figure.

    rows      the weather states (10u, 10v, 2t, tp ...)
    columns   metric families, chosen from the COLUMNS registry below
    curves    each --exp experiment, plus the --ref reference ML run as its own dashed curve
    lines     --input and --target, which do NOT train and are drawn flat: the coarse input
              (no downscaling) and the target ensemble's own member-to-member distance

All three references are REQUIRED. A bare trajectory invites over-reading -- on o96->o320 the
wind RMSE panels look like steady improvement until the anchors reveal the whole span is 3.5%
wide. --allow-missing-references exists for bootstrapping and stamps the gap on the figure.

ADDING A COLUMN
---------------
Register it in COLUMNS and it becomes selectable via --columns. A column only needs to say how
to build its metric key from a field name:

    COLUMNS["crps"] = Column(
        label="fair CRPS", key="probabilistic_{f}_{region}_fcrps_mean",
        field="ws", lower_better=True)

`field` picks which naming the row supplies: "ws" = the probabilistic weather_state, "sf" = the
spectra field name. A row whose entry for that naming is None renders an explicit empty panel
rather than a metric key that can never match. Nothing else needs touching -- no plotting code,
no CLI changes -- so several people can add families independently.

SAME SUPPORT IS ENFORCED
------------------------
Cards scored on different lanes/dates/leads/members are not comparable, and silently overlaying
them is the mistake this figure exists to prevent. Mixing refuses unless
--allow-mixed-support, which then stamps a warning across the figure.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Column:
    label: str
    key: str            # format string over {f} and {region}
    field: str          # "ws" (probabilistic naming) or "sf" (spectra naming)
    lower_better: bool | None
    unit: str | None = None   # None -> take the row's unit


@dataclass(frozen=True)
class Row:
    label: str
    ws: str | None      # probabilistic weather_state, None if the evaluator has no such field
    sf: str | None      # spectra field name
    unit: str


# 10u and 10v are STORED weather states, so both evaluators can score them. 10ff is deliberately
# absent: the probabilistic scorer derives it as hypot(10u,10v) but spectra reads stored states
# only and never sees it, which would leave its spectra panel permanently blank.
ROWS: dict[str, Row] = {
    "10u": Row("10u", "10u", "10u", "m/s"),
    "10v": Row("10v", "10v", "10v", "m/s"),
    "2t": Row("2t", "2t", "2t", "K"),
    "tp": Row("tp", "tp", "tp", "mm"),
    "msl": Row("msl", "msl", "msl", "Pa"),
    "10ff": Row("10ff", "10ff", None, "m/s"),
}
DEFAULT_ROWS = "10u,10v,2t,tp"

COLUMNS: dict[str, Column] = {
    "rmse": Column("RMSE (ens mean)", "probabilistic_{f}_{region}_rmse_ens_mean_mean", "ws", True),
    "spectra": Column("spectra rel-L2", "spectra_{f}_relative_l2", "sf", True,
                      unit="relative L2"),
    # spread has no "better" direction, so it carries lower_better=None
    "spread": Column("spread", "probabilistic_{f}_{region}_spread_mean", "ws", None),
    # CRPS family. Prefer `fcrps`: it is the ensemble-size-FAIR form, and the anchors do not
    # all carry the same member count -- the ENFO-target hline drops its verifying member, so
    # it is scored with one member fewer than the model. Plain `crps` is biased by that
    # difference; fair CRPS is not, which makes it the honest column against these hlines.
    "fcrps": Column("fair CRPS", "probabilistic_{f}_{region}_fcrps_mean", "ws", True),
    "crps": Column("CRPS", "probabilistic_{f}_{region}_crps_mean", "ws", True),
}
DEFAULT_COLUMNS = "rmse,spectra"

CURVE_COLORS = ["#1f77b4", "#ff7f0e", "#9467bd", "#8c564b", "#17becf"]
REF_COLOR = "#d62728"
# non-training anchors: solid black reads as "the target", grey dash-dot as "the raw input"
HLINE_STYLES = [("black", "-"), ("#777777", "-."), ("#2ca02c", "-.")]


def load_card(spec: str) -> tuple[str, dict]:
    """LABEL=/path/to/ladder.json"""
    label, sep, path = spec.partition("=")
    if not sep:
        raise SystemExit(f"expected LABEL=path, got {spec!r}")
    return label, json.loads(Path(path).expanduser().read_text())


def load_flat(spec: str) -> tuple[str, dict]:
    label, sep, path = spec.partition("=")
    if not sep:
        raise SystemExit(f"expected LABEL=path, got {spec!r}")
    return label, json.loads(Path(path).expanduser().read_text())


def support_of(ladder: dict) -> str:
    b = (ladder.get("profile_pins") or {}).get("budget") or {}
    return "%s | dates=%s steps=%s members=%s" % (
        ladder.get("lane", "?"), b.get("dates", "?"), b.get("steps", "?"), b.get("members", "?"))


def series(ladder: dict, key: str) -> tuple[np.ndarray, np.ndarray]:
    rows = sorted(ladder.get("rows", []), key=lambda r: r["step"])
    st = np.array([r["step"] for r in rows], dtype=float)
    v = np.array([r["metrics"].get(key, np.nan) for r in rows], dtype=float)
    return st, v


def render(
    experiments: list[tuple[str, dict]],
    out: Path,
    *,
    reference: tuple[str, dict] | None = None,
    input_ref: tuple[str, dict] | None = None,
    target_ref: tuple[str, dict] | None = None,
    hlines: list[tuple[str, dict]] | None = None,
    allow_missing_references: bool = False,
    rows: list[str] | None = None,
    columns: list[str] | None = None,
    region: str = "n.hem",
    title: str | None = None,
    allow_mixed_support: bool = False,
) -> Path:
    # target first so it takes the solid-black style, then input, then any extras
    supplied = [x for x in (target_ref, input_ref) if x is not None]
    # supplied but NOT APPLICABLE: reported, never drawn as a line
    not_applicable = [(lab, v["_absent"]) for lab, v in supplied if "_absent" in v]
    hlines = [h for h in supplied if "_absent" not in h[1]] + list(hlines or [])
    missing = [n for n, v in (("reference run (--ref)", reference),
                              ("input (--input)", input_ref),
                              ("target (--target)", target_ref)) if v is None]
    if missing and not allow_missing_references:
        raise SystemExit(
            "an evolution figure must carry all three references; missing: "
            + ", ".join(missing)
            + "\n  --ref    the reference ML experiment (a ladder.json)"
            + "\n  --input  the INPUT anchor   (flat json from eval.jobs.ladder_references)"
            + "\n  --target the TARGET anchor  (flat json from eval.jobs.ladder_references)"
            + "\nPass --allow-missing-references only while bootstrapping a lane; the figure "
              "is then stamped with what is missing.")
    row_specs = [ROWS[r] for r in (rows or DEFAULT_ROWS.split(","))]
    col_specs = [COLUMNS[c] for c in (columns or DEFAULT_COLUMNS.split(","))]

    supports = {support_of(l) for _, l in experiments}
    if reference is not None:
        supports.add(support_of(reference[1]))
    mixed = len(supports) > 1
    if mixed and not allow_mixed_support:
        raise SystemExit(
            "cards are on DIFFERENT support and are not comparable:\n  "
            + "\n  ".join(sorted(supports))
            + "\nRe-score onto one budget, or pass --allow-mixed-support.")

    fig, axes = plt.subplots(len(row_specs), len(col_specs),
                             figsize=(6.6 * len(col_specs), 3.6 * len(row_specs)), squeeze=False)
    legend_done = False
    for ri, row in enumerate(row_specs):
        for ci, col in enumerate(col_specs):
            ax = axes[ri][ci]
            field = row.ws if col.field == "ws" else row.sf
            key = col.key.format(f=field, region=region) if field else None
            drew = False

            if key is not None:
                for ei, (label, ladder) in enumerate(experiments):
                    st, v = series(ladder, key)
                    if np.isfinite(v).any():
                        ax.plot(st, v, "-o", ms=5, lw=1.8, zorder=3, label=label,
                                color=CURVE_COLORS[ei % len(CURVE_COLORS)])
                        drew = True

                if reference is not None:
                    # the reference is another RUN: its score moves with step, so it is a curve
                    rlabel, rladder = reference
                    st, v = series(rladder, key)
                    if np.isfinite(v).any():
                        ax.plot(st, v, "--s", ms=4, lw=1.6, color=REF_COLOR, zorder=2,
                                label="ref: %s" % rlabel)
                        drew = True

                for hi, (hlabel, hvals) in enumerate(hlines):
                    hv = hvals.get(key)
                    if hv is None:
                        continue
                    color, ls = HLINE_STYLES[hi % len(HLINE_STYLES)]
                    ax.axhline(float(hv), lw=2.0, color=color, ls=ls, zorder=4, label=hlabel)
                    drew = True

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
                ax.set_ylabel(col.unit or row.unit, fontsize=9)

            arrow = "" if col.lower_better is None else "  (lower = better)"
            ax.set_title("%s — %s%s" % (col.label, field or row.label, arrow), fontsize=11)
            # the legend belongs on the first panel that HAS content; pinning it to [0][0]
            # loses it entirely whenever that panel is one of the empty ones
            if drew and not legend_done:
                ax.legend(fontsize=8.5, loc="best")
                legend_done = True
            if ci == 0:
                ax.text(-0.20, 0.5, row.label, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=13, fontweight="bold")

    # No title by default. The one exception is a figure that would otherwise mislead: when
    # mixed support has been forced, that warning is stamped on regardless.
    if mixed:
        banner = "MIXED SUPPORT — these curves are NOT comparable: " + " || ".join(sorted(supports))
    elif missing:
        banner = "INCOMPLETE — missing " + ", ".join(missing)
    elif not_applicable:
        banner = "  |  ".join("%s not applicable on this lane — %s" % (lab, why)
                              for lab, why in not_applicable)
    else:
        banner = title
    if banner:
        fig.suptitle(banner, fontsize=9.5, wrap=True,
                     color="#b00020" if (mixed or missing) else "#555555")
        fig.tight_layout(rect=[0.012, 0, 1, 0.95])
    else:
        fig.tight_layout(rect=[0.012, 0, 1, 1])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("support: %s" % " || ".join(sorted(supports)))
    print("wrote %s" % out)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="eval.cli evolution", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", action="append", required=True,
                    help="LABEL=/path/to/ladder.json (repeatable)")
    ap.add_argument("--ref", help="LABEL=/path/to/ladder.json -- the reference ML experiment, "
                                  "drawn as its own curve vs step (REQUIRED)")
    ap.add_argument("--input", dest="input_ref",
                    help="LABEL=/path/to/flat.json -- the INPUT anchor, drawn flat (REQUIRED)")
    ap.add_argument("--target", dest="target_ref",
                    help="LABEL=/path/to/flat.json -- the TARGET anchor, drawn flat (REQUIRED)")
    ap.add_argument("--hline", action="append", default=[],
                    help="LABEL=/path/to/flat.json -- any further flat anchor (repeatable)")
    ap.add_argument("--allow-missing-references", action="store_true",
                    help="bootstrap a lane that has no reference yet; stamps the gap on the figure")
    ap.add_argument("--rows", default=DEFAULT_ROWS,
                    help="comma-separated, from: " + ",".join(ROWS))
    ap.add_argument("--columns", default=DEFAULT_COLUMNS,
                    help="comma-separated, from: " + ",".join(COLUMNS))
    ap.add_argument("--region", default="n.hem")
    ap.add_argument("--title", default=None, help="optional figure title (default: none)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-mixed-support", action="store_true")
    args = ap.parse_args(argv)

    for name, valid in (("rows", ROWS), ("columns", COLUMNS)):
        bad = [x for x in getattr(args, name).split(",") if x not in valid]
        if bad:
            raise SystemExit("unknown %s: %s (valid: %s)" % (name, ",".join(bad), ",".join(valid)))

    render(
        [load_card(s) for s in args.exp],
        Path(args.out),
        reference=load_card(args.ref) if args.ref else None,
        input_ref=load_flat(args.input_ref) if args.input_ref else None,
        target_ref=load_flat(args.target_ref) if args.target_ref else None,
        hlines=[load_flat(s) for s in args.hline],
        allow_missing_references=args.allow_missing_references,
        rows=args.rows.split(","),
        columns=args.columns.split(","),
        region=args.region,
        title=args.title,
        allow_mixed_support=args.allow_mixed_support,
    )


if __name__ == "__main__":
    main()
