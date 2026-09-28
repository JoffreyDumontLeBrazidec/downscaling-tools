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
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from eval.plotting import convert_difference  # noqa: E402


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
    "rmse": Column("RMSE of the ensemble mean", "probabilistic_{f}_{region}_rmse_ens_mean_mean", "ws", True),
    # spectra_ecmwf_v2 (ECMWF transform, complete grid). The retired HEALPix proxy's rows,
    # present only on cards scored before 2026-09-28, stay readable under their own column;
    # the two instruments give different numbers and are never drawn as one column.
    "spectra": Column("Spectral relative L2 distance", "spectra_v2_{f}_relative_l2", "sf", True,
                      unit="relative L2 distance"),
    "spectra_proxy": Column("Spectral relative L2 distance (retired proxy)", "spectra_{f}_relative_l2", "sf", True,
                            unit="relative L2 distance"),
    # spread has no "better" direction, so it carries lower_better=None
    "spread": Column("Ensemble spread", "probabilistic_{f}_{region}_spread_mean", "ws", None),
    # CRPS family. Prefer `fcrps`: it is the ensemble-size-FAIR form, and the anchors do not
    # all carry the same member count -- the ENFO-target hline drops its verifying member, so
    # it is scored with one member fewer than the model. Plain `crps` is biased by that
    # difference; fair CRPS is not, which makes it the honest column against these hlines.
    "fcrps": Column("Fair CRPS", "probabilistic_{f}_{region}_fcrps_mean", "ws", True),
    "crps": Column("CRPS", "probabilistic_{f}_{region}_crps_mean", "ws", True),
}
DEFAULT_COLUMNS = "rmse,spectra"

# Line styles come from the house role table (eval.plotting.roles), applied in render():
#   one experiment      -> "model" (red, solid);  several -> sequence_style(i)
#   --ref run           -> "baseline" (dark grey, dash-dot)
#   --target anchor     -> "truth" (black, solid, thick);  --input anchor -> "input" (blue, dashed)
#   further --hline     -> reference_style(i)


def _absent_reason(payload: dict) -> str | None:
    """Reason this anchor does not apply to the lane, or None if it carries real values.

    Two conventions exist and both are honoured: {"absent": true, "reason": ...} written by
    anchor_scores.py as `target.ABSENT.json`, and {"_absent": "<reason>"} written by
    ladder_references.py. Same meaning, so the plot accepts either.
    """
    if not isinstance(payload, dict):
        return None
    if payload.get("absent"):
        return str(payload.get("reason") or "not applicable on this lane")
    if "_absent" in payload:
        return str(payload["_absent"])
    return None


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


def _panel_has_data(row: Row, col: Column, experiments, reference, hlines, region: str) -> bool:
    """True when at least one curve or line would be drawn in the panel of ``row`` and ``col``."""
    field = row.ws if col.field == "ws" else row.sf
    if field is None:
        return False
    key = col.key.format(f=field, region=region)
    for _label, ladder in experiments:
        if np.isfinite(series(ladder, key)[1]).any():
            return True
    if reference is not None and np.isfinite(series(reference[1], key)[1]).any():
        return True
    return any(hvals.get(key) is not None for _label, hvals in hlines)


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
    keep_empty: bool = False,
) -> Path:
    """Draw the grid. Rows and columns that have no data at all on this lane (every panel of
    the row or column would read "not available") are left out and named in the footer;
    ``keep_empty=True`` keeps them as explicit empty panels."""
    # target first so it takes the solid-black style, then input, then any extras
    supplied = [x for x in (target_ref, input_ref) if x is not None]
    # supplied but NOT APPLICABLE on this lane: reported, never drawn as a line
    not_applicable = [(lab, _absent_reason(v)) for lab, v in supplied if _absent_reason(v)]
    hlines = [h for h in supplied if not _absent_reason(h[1])] + list(hlines or [])
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
    dropped_rows: list[str] = []
    dropped_cols: list[str] = []
    if not keep_empty:
        # judge availability on the full request, then drop what is empty in every panel
        has = [[_panel_has_data(r, c, experiments, reference, hlines, region) for c in col_specs]
               for r in row_specs]
        keep_r = [any(h) for h in has]
        keep_c = [any(has[ri][ci] for ri in range(len(row_specs))) for ci in range(len(col_specs))]
        dropped_rows = [r.label for r, k in zip(row_specs, keep_r) if not k]
        dropped_cols = [c.label for c, k in zip(col_specs, keep_c) if not k]
        if not any(keep_r):
            raise SystemExit("none of the requested rows and columns has data on this lane; "
                             "pass --keep-empty to draw the empty panels anyway")
        row_specs = [r for r, k in zip(row_specs, keep_r) if k]
        col_specs = [c for c, k in zip(col_specs, keep_c) if k]

    supports = {support_of(l) for _, l in experiments}
    if reference is not None:
        supports.add(support_of(reference[1]))
    mixed = len(supports) > 1
    if mixed and not allow_mixed_support:
        raise SystemExit(
            "cards are on DIFFERENT support and are not comparable:\n  "
            + "\n  ".join(sorted(supports))
            + "\nRe-score onto one budget, or pass --allow-mixed-support.")

    from eval.plotting import (
        AXIS, WORSE_COLOR, eval_style, reference_style, role_style, save_figure, sequence_style,
        variable_spec,
    )
    from eval.plotting.probabilistic import DOMAIN_NAMES
    from eval.plotting.spec_helpers import format_steps

    # Styles by role: one experiment is "the model" (red); several are arms told apart by the
    # colour-blind-safe sequence. The --ref run is the lane's baseline; the target anchor is the
    # truth line, the input anchor the input line, any further anchor a reference style.
    if len(experiments) == 1:
        exp_styles = [role_style("model")]
    else:
        exp_styles = [sequence_style(i) for i in range(len(experiments))]
    anchor_styles: list[dict] = []
    n_extra = 0
    for h in hlines:
        if target_ref is not None and h is target_ref:
            anchor_styles.append(role_style("truth"))
        elif input_ref is not None and h is input_ref:
            anchor_styles.append(role_style("input"))
        else:
            anchor_styles.append(reference_style(n_extra))
            n_extra += 1

    with eval_style():
        fig, axes = plt.subplots(len(row_specs), len(col_specs),
                                 figsize=(6.2 * len(col_specs), 3.5 * len(row_specs)),
                                 squeeze=False)
        legend_done = False
        for ri, row in enumerate(row_specs):
            var_name = variable_spec(row.ws or row.sf or row.label).name
            for ci, col in enumerate(col_specs):
                ax = axes[ri][ci]
                field = row.ws if col.field == "ws" else row.sf
                key = col.key.format(f=field, region=region) if field else None
                drew = False
                # scores in the variable's own unit are shown in its display unit (hPa, not Pa);
                # dimensionless columns carry their own `unit` and are left alone
                if col.unit is None:
                    spec = variable_spec(row.ws or row.label)
                    unit = spec.unit or row.unit

                    def disp(v, _row=row):
                        return np.asarray(convert_difference(_row.ws or _row.label, v,
                                                             native_unit=_row.unit), dtype=float)
                else:
                    unit = None

                    def disp(v):
                        return np.asarray(v, dtype=float)

                if key is not None:
                    for ei, (label, ladder) in enumerate(experiments):
                        st, v = series(ladder, key)
                        if np.isfinite(v).any():
                            ax.plot(st, disp(v), marker="o", markersize=5, label=label,
                                    **exp_styles[ei])
                            drew = True

                    if reference is not None:
                        # the reference is another RUN: its score moves with step, so it is a curve
                        rlabel, rladder = reference
                        st, v = series(rladder, key)
                        if np.isfinite(v).any():
                            ax.plot(st, disp(v), marker="s", markersize=4,
                                    label="Baseline run: %s" % rlabel, **role_style("baseline"))
                            drew = True

                    for hi, (hlabel, hvals) in enumerate(hlines):
                        hv = hvals.get(key)
                        if hv is None:
                            continue
                        ax.axhline(float(disp(float(hv))), label=hlabel.replace("_", " "),
                                   **anchor_styles[hi])
                        drew = True

                if not drew:
                    ax.text(0.5, 0.5, "not available\non this lane", transform=ax.transAxes,
                            ha="center", va="center", fontsize=11, color="0.45")
                    ax.set_facecolor("#f5f5f5")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.grid(False)
                else:
                    format_steps(ax)
                    ax.set_xlabel(AXIS["step"])
                    ax.set_ylabel(f"{col.label} ({unit})" if unit else
                                  (col.unit[:1].upper() + col.unit[1:] if col.unit else col.label))

                arrow = "" if col.lower_better is None else " (lower is better)"
                ax.set_title("%s%s" % (col.label, arrow))
                # the legend belongs on the first panel that HAS content; pinning it to [0][0]
                # loses it entirely whenever that panel is one of the empty ones
                if drew and not legend_done:
                    ax.legend(loc="best")
                    legend_done = True
                if ci == 0:
                    ax.text(-0.24, 0.5, var_name, transform=ax.transAxes, rotation=90,
                            va="center", ha="center", fontsize=12, fontweight="bold")

        # No title by default. The one exception is a figure that would otherwise mislead: when
        # mixed support has been forced, that warning is stamped on regardless.
        if mixed:
            banner = ("MIXED SUPPORT: these curves are NOT comparable: "
                      + " || ".join(sorted(supports)))
        elif missing:
            banner = "INCOMPLETE: missing " + ", ".join(missing)
        elif not_applicable:
            banner = "  |  ".join("%s not applicable on this lane: %s" % (lab, why)
                                  for lab, why in not_applicable)
        else:
            banner = title
        # the scoring support (sample counts) always goes in a small footer
        foot = ("Scored on: " + " || ".join(sorted(supports))
                + "  |  region: %s" % DOMAIN_NAMES.get(region, region))
        if dropped_rows or dropped_cols:
            omitted = []
            if dropped_rows:
                omitted.append("rows " + ", ".join(dropped_rows))
            if dropped_cols:
                omitted.append("columns " + ", ".join(dropped_cols))
            foot += "\nLeft out, no data on this lane: " + "; ".join(omitted)
        fig.text(0.01, 0.002, foot, fontsize=8, color="0.35", ha="left", va="bottom")
        bottom = 0.025 + (0.018 if "\n" in foot else 0.0)
        if banner:
            fig.suptitle(banner, fontsize=10, wrap=True,
                         color=WORSE_COLOR if (mixed or missing) else "0.35",
                         fontweight="bold" if (mixed or missing) else "normal")
            fig.tight_layout(rect=[0.02, bottom, 1, 0.95])
        else:
            fig.tight_layout(rect=[0.02, bottom, 1, 1])
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        # the requested file plus its sibling format (PNG at 150 dpi and PDF)
        save_figure(fig, out, close=True)
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
    ap.add_argument("--keep-empty", action="store_true",
                    help="keep rows and columns that have no data on this lane, as "
                         "'not available' panels (default: leave them out and say so)")
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
        keep_empty=args.keep_empty,
    )


if __name__ == "__main__":
    main()
