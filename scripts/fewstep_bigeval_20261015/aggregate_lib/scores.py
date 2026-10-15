"""Part 4: fair CRPS, spread, ensemble-mean RMSE, spread over RMSE, and the paired per-date test.

Inputs, per row and set: ``evaluators/probabilistic/summary_by_lead.csv`` (columns step,
weather_state, domain, metric, mean, std, stderr, n_dates, n_points_total; metric in crps,
fcrps, spread, rmse_ens_mean) and ``evaluators/probabilistic/scores_by_lead.csv`` (columns
date, step, weather_state, domain, metric, value, ...): the per-date values of the paired
test. When a (variable, domain) has no per-date value for the row or the reference, both are
recomputed from the prediction files by ``fcrps.per_date_fcrps`` and the table says so. Variables
in ``recompute_vars`` (default tp: probabilistic does not clip tp at 0) always take the fallback.

Paired test (pre-registered, note 20261015 section 5): per date d,
p_d = 100 (row_d - ref_d) / ref_d; mean, t = mean / (sd / sqrt(n)); k = dates on which the
reference is better (ref_d < row_d). Established when |t| >= t_crit (2.03) and the better side
wins at least 2/3 of the dates. Established and |mean| <= 1.0 % is "equal in practice";
established and worse is "worse", better is "better"; otherwise "tie".
"""
from __future__ import annotations

import math
from collections import defaultdict

from . import fcrps as F
from .common import (Context, SectionResult, canon_var, fmt, fnum, guarded, na, pct,
                     read_csv, table)

DEFAULT_VARS = ["10u", "10v", "ws", "2t", "msl", "t_850", "z_500", "tp"]
DEFAULT_DOMAINS = ["global", "n.hem", "tropics", "s.hem", "europe"]
TEST_DOMAINS = ["global", "europe"]
METRICS = ("fcrps", "spread", "rmse_ens_mean")


def paired_test(row_vals: dict, ref_vals: dict, *, t_crit: float = 2.03,
                win_frac: float = 2.0 / 3.0, practical_pct: float = 1.0) -> dict:
    """The paired per-date test of the row against the reference (keys: dates)."""
    keys = sorted(k for k in set(row_vals) & set(ref_vals)
                  if math.isfinite(fnum(row_vals[k])) and math.isfinite(fnum(ref_vals[k]))
                  and fnum(ref_vals[k]) != 0)
    n = len(keys)
    out = {"n": n, "mean_pct": math.nan, "sd": math.nan, "se": math.nan, "t": math.nan,
           "k_ref_better": 0, "k_row_better": 0, "established": False, "status": "",
           "dates": keys}
    if n < 2:
        out["status"] = na(f"{n} common date(s) with finite values")
        return out
    p = [100.0 * (fnum(row_vals[k]) - fnum(ref_vals[k])) / fnum(ref_vals[k]) for k in keys]
    mean = sum(p) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in p) / (n - 1))
    se = sd / math.sqrt(n)
    if se > 0:
        t = mean / se
    else:
        t = 0.0 if mean == 0 else math.copysign(math.inf, mean)
    k_ref = sum(1 for k in keys if fnum(ref_vals[k]) < fnum(row_vals[k]))
    k_row = sum(1 for k in keys if fnum(row_vals[k]) < fnum(ref_vals[k]))
    wins = k_ref if mean > 0 else k_row
    established = abs(t) >= t_crit and mean != 0 and wins >= win_frac * n - 1e-9
    if not established:
        status = "tie"
    elif abs(mean) <= practical_pct:
        status = "equal in practice"
    else:
        status = "worse" if mean > 0 else "better"
    out.update(mean_pct=mean, sd=sd, se=se, t=t, k_ref_better=k_ref, k_row_better=k_row,
               established=established, status=status)
    return out


# --------------------------------------------------------------------------------------

def _summary(found) -> dict:
    """{(var, domain, metric): (mean, stderr, n_dates, step)} from summary_by_lead.csv
    (one lead per set; several leads are averaged and the step column says 'all')."""
    acc = defaultdict(list)
    for r in read_csv(found.path / "summary_by_lead.csv"):
        acc[(canon_var(r["weather_state"]), r["domain"], r["metric"])].append(
            (fnum(r["mean"]), fnum(r.get("stderr")), fnum(r.get("n_dates")), r.get("step", "")))
    out = {}
    for k, v in acc.items():
        if len(v) == 1:
            out[k] = v[0]
        else:
            out[k] = (sum(x[0] for x in v) / len(v), math.nan, sum(x[2] for x in v), "all")
    return out


def _per_date(found) -> dict:
    """{(var, domain, metric): {(date, step): value}} from scores_by_lead.csv."""
    out: dict = defaultdict(dict)
    p = found.path / "scores_by_lead.csv"
    if not p.exists():
        return {}
    for r in read_csv(p):
        out[(canon_var(r["weather_state"]), r["domain"], r["metric"])][
            (r["date"], str(int(fnum(r["step"])) if math.isfinite(fnum(r["step"])) else r["step"]))
        ] = fnum(r["value"])
    return dict(out)


def _fallback(ctx: Context, label: str, set_name: str, variables, domains) -> dict | str:
    """{(var, domain): {(date, step): value}} from the prediction files, or a reason."""
    key = ("fcrps_fallback", label, set_name)
    if key in ctx.cache:
        return ctx.cache[key]
    row = ctx.by_label[label]
    pdir = ctx.pred_dir(row, set_name)
    if pdir is None:
        res: dict | str = (f"no per-date value in probabilistic and no prediction files found for "
                           f"{label} in set {set_name} (arm {row.arm!r}, sampler {row.sampler!r})")
    else:
        recs = F.per_date_fcrps(pdir, variables, domains,
                                clip_vars=ctx.opt("clip_vars", F.DEFAULT_CLIP))
        res = defaultdict(dict)
        for r in recs:
            res[(r["variable"], r["domain"])][(r["date"], str(r["step"]))] = r["value"]
        res = dict(res)
    ctx.cache[key] = res
    return res


def section(ctx: Context) -> SectionResult:
    res = SectionResult("4. Scores", "")
    ref = ctx.reference
    variables = ctx.opt("score_vars", DEFAULT_VARS)
    domains = ctx.opt("score_domains", DEFAULT_DOMAINS)
    test_domains = ctx.opt("test_domains", TEST_DOMAINS)
    t_crit = float(ctx.opt("t_crit", 2.03))
    recompute = set(ctx.opt("recompute_vars", ["tp"]))
    md = ["# 4. Scores\n",
          "Fair CRPS, spread (pointwise member standard deviation, ddof 1, area-weighted mean), "
          "ensemble-mean RMSE and spread over RMSE from `probabilistic` (mean over dates of the "
          f"per-date values), with the % change against the reference `{ref}`.\n"]
    ref_found = ctx.results_by_key(ref, "probabilistic", "summary_by_lead.csv")

    def scores_table() -> str:
        if not ref_found:
            return "## 4a. Scores per variable and domain\n\n" + na(
                f"no probabilistic output for the reference under {ctx.where(ref, 'probabilistic')}") + "\n"
        header = ["set", "row", "variable", "domain", "fCRPS", "fCRPS ref", "fCRPS %",
                  "spread", "spread %", "RMSE", "RMSE %", "spread/RMSE", "spread/RMSE ref", "n dates"]
        rows = []
        for key, rf in ref_found.items():
            ref_s = _summary(rf)
            for row in ctx.rows:
                found = ctx.results_by_key(row.label, "probabilistic", "summary_by_lead.csv").get(key)
                for var in variables:
                    for dom in domains:
                        r_ref = {m: ref_s.get((var, dom, m)) for m in METRICS}
                        if not any(r_ref.values()):
                            continue
                        if found is None:
                            rows.append([key, row.label, var, dom] + [na("no probabilistic output")] * 10)
                            continue
                        s = ctx.cache.setdefault(("prob_summary", row.label, key), _summary(found))
                        r = {m: s.get((var, dom, m)) for m in METRICS}
                        g = lambda d, m: d[m][0] if d.get(m) else math.nan  # noqa: E731
                        ratio = g(r, "spread") / g(r, "rmse_ens_mean") if g(r, "rmse_ens_mean") else math.nan
                        ratio_ref = (g(r_ref, "spread") / g(r_ref, "rmse_ens_mean")
                                     if g(r_ref, "rmse_ens_mean") else math.nan)
                        rows.append([key, row.label, var, dom,
                                     fmt(g(r, "fcrps"), 4), fmt(g(r_ref, "fcrps"), 4),
                                     fmt(pct(g(r, "fcrps"), g(r_ref, "fcrps")), 2, signed=True),
                                     fmt(g(r, "spread"), 4),
                                     fmt(pct(g(r, "spread"), g(r_ref, "spread")), 2, signed=True),
                                     fmt(g(r, "rmse_ens_mean"), 4),
                                     fmt(pct(g(r, "rmse_ens_mean"), g(r_ref, "rmse_ens_mean")), 2, signed=True),
                                     fmt(ratio, 3), fmt(ratio_ref, 3),
                                     fmt(r["fcrps"][2], 0) if r.get("fcrps") else "n/a"])
        note = ""
        if recompute:
            note = (f"Values of {', '.join(sorted(recompute))} in this table are probabilistic's own (it "
                    "does not clip tp at 0); the paired test below recomputes them from the prediction "
                    "files with tp clipped at 0 in members and truth.\n\n")
        return "## 4a. Scores per variable and domain\n\n" + note + table(ctx, "scores_by_variable_domain", header, rows)

    def paired_table() -> str:
        head = (f"## 4b. Paired per-date test against `{ref}` on fair CRPS\n\n"
                "p_d = 100 (row - ref) / ref per date; t = mean / (sd / sqrt(n)); "
                f"established when |t| >= {t_crit} and the better side wins >= 2/3 of the dates; "
                "established and |mean| <= 1.0 % is 'equal in practice'. "
                "'source' says whether the per-date values come from probabilistic's "
                "scores_by_lead.csv or from the fallback over the prediction files.\n\n")
        sets = sorted(ref_found) or sorted({s for r in ctx.rows for s in ctx.set_dirs(r.label)})
        if not sets:
            return head + na("no probabilistic output and no evaluation directory for any row") + "\n"
        header = ["set", "row", "calls", "variable", "domain", "n", "mean % (row-ref)/ref", "SE %",
                  "t", "ref better (k of n)", "established", "status", "source"]
        rows = []
        for key in sets:
            set_name = key.split("/")[0]
            rf = ref_found.get(key)
            ref_pd = _per_date(rf) if rf else {}
            for row in ctx.rows:
                if row.label == ref:
                    continue
                fr = ctx.results_by_key(row.label, "probabilistic", "summary_by_lead.csv").get(key)
                row_pd = _per_date(fr) if fr else {}
                for var in variables:
                    for dom in test_domains:
                        a = row_pd.get((var, dom, "fcrps"))
                        b = ref_pd.get((var, dom, "fcrps"))
                        if var in recompute:      # e.g. tp: the evaluator does not clip at 0
                            a = b = None
                        source = "probabilistic scores_by_lead.csv"
                        if not a or not b:
                            fa = _fallback(ctx, row.label, set_name, variables, test_domains)
                            fb = _fallback(ctx, ref, set_name, variables, test_domains)
                            if isinstance(fa, str) or isinstance(fb, str):
                                reason = fa if isinstance(fa, str) else fb
                                rows.append([key, row.label, row.calls, var, dom] +
                                            [na(reason)] * 7 + ["none"])
                                continue
                            a, b = fa.get((var, dom)), fb.get((var, dom))
                            source = "fallback: prediction files"
                            if not a or not b:
                                rows.append([key, row.label, row.calls, var, dom] +
                                            [na(f"{var} not in the prediction files")] * 7 + [source])
                                continue
                        t = paired_test(a, b, t_crit=t_crit)
                        ctx.cache.setdefault("paired", {})[(row.label, key, var, dom)] = t
                        rows.append([key, row.label, row.calls, var, dom, t["n"],
                                     fmt(t["mean_pct"], 2, signed=True), fmt(t["se"], 2),
                                     fmt(t["t"], 2), f"{t['k_ref_better']} of {t['n']}",
                                     "yes" if t["established"] else "no", t["status"], source])
        return head + table(ctx, "scores_paired_test", header, rows)

    guarded(md, "## 4a. Scores per variable and domain", scores_table)
    guarded(md, "## 4b. Paired per-date test", paired_table)
    res.md = "\n".join(md)

    # summary fragments: global and europe paired-test status per variable
    paired = ctx.cache.get("paired", {})
    for row in ctx.rows:
        if row.label == ref:
            res.add(row.label, "reference")
            continue
        frags = []
        for dom in test_domains:
            items = [(k[2], v) for k, v in paired.items() if k[0] == row.label and k[3] == dom]
            if not items:
                continue
            parts = []
            for var, t in items:
                if t["status"].startswith("Not available"):
                    continue
                tag = {"worse": "W", "better": "B", "equal in practice": "=", "tie": "~"}.get(t["status"], "?")
                parts.append(f"{var} {fmt(t['mean_pct'], 1, signed=True)}% t={fmt(t['t'], 1)} "
                             f"{t['k_ref_better']}/{t['n']} [{tag}]")
            if parts:
                frags.append(f"{dom}: " + "; ".join(parts))
        res.add(row.label, " | ".join(frags) if frags else na("no paired test"))
    return res
