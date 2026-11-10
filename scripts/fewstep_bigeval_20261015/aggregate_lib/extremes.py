"""Part 3: extremes (tc raw extremes, precipitation distribution).

- tc: ``evaluators/tc/stats.json``, ``events[<event>]["extreme_tail"]["rows"]``: one row per
  curve with ``exp`` and mslp_min, mslp_p001 (0.01st percentile), mslp_p01 (0.1st), wind_p9999,
  wind_max. The analysis row is ``events[<event>]["analysis_key"]`` (else the OPER-prefixed
  row); the model row is the one that is neither analysis, input, target nor an ENFO/EEFO/
  IEKM/IP6Y reference. Event keys ``<event>__<mode>`` (a second support mode) are kept apart.
  Pre-registered rule: a loss is > 2 hPa on minimum msl or > 3 % on peak wind away from the
  analysis AND beyond 2 SE; tc pools members, so no SE exists and the status says
  "beyond threshold (no SE)".
- precipitation: ``evaluators/precip_dist/tail_stats.csv`` (precip_dist with
  ``tail_stats: true``; columns scope, step, series in pred/truth/input, variable, n, wet_frac,
  mean_mm, p99, p99_9, p99_99, max_mm, neg_share, min_mm, n_members; the scope "all" rows are
  read), or another statistics file with the same keys; otherwise (precip_dist without
  ``tail_stats``) the same numbers straight from the prediction files:
  tp x 1000 (m to mm), share of negative raw values, then clipped at 0; wet fraction = share
  of values > 0.1 mm; percentiles pooled over dates, members and points (exact, see
  ``precip_stats``), every member of y_pred, y and x_interp (as tail_stats.csv). Percentiles of
  values clipped at 0 equal tail_stats' raw-value percentiles whenever they are positive.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from .common import (Context, SectionResult, fmt, fnum, guarded, na, pct, read_csv,
                     read_json, row_guard, table)

TC_KEYS = [("mslp_min", "min msl hPa", "msl"), ("mslp_p01", "msl p0.1 hPa", "msl"),
           ("mslp_p001", "msl p0.01 hPa", "msl"), ("wind_p9999", "wind p99.99 m/s", "wind"),
           ("wind_max", "wind max m/s", "wind")]
NON_MODEL_PREFIXES = ("OPER", "ENFO", "EEFO", "IEKM", "IP6Y", "TARGET", "INPUT")
PRECIP_KEYS = [("wet_fraction", "wet fraction > 0.1 mm"), ("p99", "p99 mm"), ("p99.9", "p99.9 mm"),
               ("p99.99", "p99.99 mm"), ("max", "max mm"), ("negative_share", "negative share (raw)")]
QUANTILES = {"p99": 99.0, "p99.9": 99.9, "p99.99": 99.99}


# --------------------------------------------------------------------------------------
# tc
# --------------------------------------------------------------------------------------

def tc_rows(found, label: str) -> dict:
    """{event_key: {"model": row|None, "analysis": row|None}}"""
    data = read_json(found.path / "stats.json")
    out = {}
    for ev, st in (data.get("events") or {}).items():
        rows = [r for r in (st.get("extreme_tail") or {}).get("rows", []) if isinstance(r, dict)]
        akey = st.get("analysis_key")
        analysis = next((r for r in rows if akey and r.get("exp") == akey), None) or \
            next((r for r in rows if str(r.get("exp", "")).upper().startswith("OPER")), None)
        others = [r for r in rows if r is not analysis
                  and not str(r.get("exp", "")).strip().upper().startswith(NON_MODEL_PREFIXES)
                  and str(r.get("exp", "")).strip().lower() not in ("input", "target", "truth")]
        model = next((r for r in others if label and label in str(r.get("exp", ""))), None) or \
            (others[0] if others else None)
        out[ev] = {"model": model, "analysis": analysis, "prediction_only": bool(st.get("prediction_only"))}
    return out


def tc_section(ctx: Context, res: SectionResult) -> str:
    ref = ctx.reference
    head = ("## 3a. Storm raw extremes (tc)\n\nPer event, row against the OPER analysis and "
            f"against `{ref}`. Wind differences also in %. Status: 'beyond threshold' when "
            "|row - ref| > 2 hPa (msl) or > 3 % (wind), with the side relative to the analysis "
            "(tc pools members: no SE, the 2 SE part of the rule cannot be applied).\n\n")
    founds = {r.label: ctx.results_by_key(r.label, "tc", "stats.json") for r in ctx.rows}
    keys = sorted({k for d in founds.values() for k in d})
    if not keys:
        for r in ctx.rows:
            res.add(r.label, "tc " + na("no output"))
        return head + na(f"no tc/stats.json under {ctx.eval_root}/<set>/<row>/") + "\n"
    header = ["view", "row", "event", "quantity", "row value", "analysis", "ref value", "row - analysis",
              "row - ref", "row - ref %", "status"]
    rows = []
    for key in keys:
        rf = founds[ref].get(key)
        ref_ev = tc_rows(rf, ref) if rf else {}
        for row in ctx.rows:
            f = founds[row.label].get(key)
            if f is None:
                rows.append([key, row.label] + [na("no tc output")] * 9)
                res.add(row.label, f"tc {key}: " + na("no output"))
                continue
            with row_guard(rows, [key, row.label], len(header) - 2, res, row.label, f"tc {key}"):
                frags = []
                for ev, d in sorted(tc_rows(f, row.label).items()):
                    m, a = d["model"] or {}, d["analysis"] or {}
                    rm = (ref_ev.get(ev) or {}).get("model") or {}
                    for k, name, kind in TC_KEYS:
                        rv, av, fv = fnum(m.get(k)), fnum(a.get(k)), fnum(rm.get(k))
                        diff = rv - fv
                        dpct = pct(rv, fv)
                        status = ""
                        if row.label != ref and math.isfinite(diff):
                            over = abs(diff) > 2.0 if kind == "msl" else abs(dpct) > 3.0
                            if over:
                                side = ("towards analysis" if math.isfinite(av) and abs(rv - av) < abs(fv - av)
                                        else "away from analysis" if math.isfinite(av) else "analysis n/a")
                                status = f"beyond threshold, {side} (no SE)"
                                if k in ("mslp_min", "wind_max", "wind_p9999"):
                                    frags.append(f"{ev} {k} {fmt(diff, 1, signed=True)}"
                                                 f"{' (' + fmt(dpct, 1, signed=True) + '%)' if kind == 'wind' else ''}"
                                                 f" {side}")
                            else:
                                status = "within threshold"
                        cell_a = fmt(av, 2) if a else na("no analysis row")
                        rows.append([key, row.label, ev, name, fmt(rv, 2) if m else na("no model row"), cell_a,
                                     fmt(fv, 2) if rm else na(f"no {ref} row"), fmt(rv - av, 2, signed=True),
                                     fmt(diff, 2, signed=True), fmt(dpct, 2, signed=True), status])
                if row.label == ref:
                    res.add(row.label, f"tc {key}: reference")
                else:
                    res.add(row.label, f"tc {key}: " + ("; ".join(frags) if frags else
                                                        "all within 2 hPa / 3 % of ref"))
    return head + table(ctx, "extremes_tc", header, rows)


# --------------------------------------------------------------------------------------
# precipitation
# --------------------------------------------------------------------------------------

class _Tail:
    """Exact pooled percentiles from the top fraction of every file (see ``value``)."""

    def __init__(self, keep_frac: float = 0.03):
        self.keep_frac = keep_frac
        self.kept: list[np.ndarray] = []
        self.thresholds: list[float] = []
        self.n = 0
        self.n_wet = 0
        self.n_neg = 0
        self.max = -math.inf

    def update(self, raw_mm: np.ndarray) -> None:
        v = raw_mm[np.isfinite(raw_mm)].astype(np.float64)
        if v.size == 0:
            return
        self.n_neg += int((v < 0).sum())
        v = np.maximum(v, 0.0)
        self.n += v.size
        self.n_wet += int((v > 0.1).sum())
        self.max = max(self.max, float(v.max()))
        k = max(int(math.ceil(self.keep_frac * v.size)), 1)
        top = np.partition(v, v.size - k)[v.size - k:]
        self.thresholds.append(float(top.min()))
        self.kept.append(top)

    def percentile(self, q: float) -> tuple[float, bool]:
        """(value, exact). numpy's linear percentile of all values seen; exact when every
        value at or above the needed order statistics was kept (above every file's cut)."""
        if self.n == 0:
            return math.nan, False
        thr = max(self.thresholds)
        kept = np.sort(np.concatenate(self.kept))
        above = kept[kept > thr]
        pos = q / 100.0 * (self.n - 1)
        lo = int(math.floor(pos))
        start = self.n - above.size           # index of the first value > thr in the full order
        if lo >= start:
            vals = above
            i = lo - start
            exact = True
        else:  # fall back on all kept values (approximate)
            vals = kept
            i = max(lo - (self.n - kept.size), 0)
            exact = False
        hi = min(i + 1, vals.size - 1)
        frac = pos - lo
        return float(vals[i] + (vals[hi] - vals[i]) * frac), exact

    def stats(self) -> dict:
        out = {"n": self.n,
               "wet_fraction": self.n_wet / self.n if self.n else math.nan,
               "negative_share": self.n_neg / self.n if self.n else math.nan,
               "max": self.max if self.n else math.nan, "exact": True}
        for k, q in QUANTILES.items():
            v, ex = self.percentile(q)
            out[k] = v
            out["exact"] = out["exact"] and ex
        return out


def precip_stats(pred_dir: Path, *, var: str = "tp", scale: float = 1000.0,
                 keep_frac: float = 0.03) -> dict:
    """{"model"|"truth"|"input": stats} pooled over the files of ``pred_dir``."""
    import xarray as xr

    tails = {"model": _Tail(keep_frac), "truth": _Tail(keep_frac), "input": _Tail(keep_frac)}
    n_files = 0
    for path in sorted(Path(pred_dir).glob("predictions_*.nc")):
        with xr.open_dataset(path, cache=False, decode_timedelta=False) as ds:
            names = [v.decode() if isinstance(v, bytes) else str(v) for v in ds["weather_state"].values.tolist()]
            if var not in names:
                continue
            i = names.index(var)
            n_files += 1
            # every member of every series, as precip_dist's tail_stats.csv pools them
            for src, name, first_only in (("model", "y_pred", False), ("truth", "y", False),
                                          ("input", "x_interp", False)):
                if name not in ds:
                    continue
                da = ds[name].isel(weather_state=i)
                if "sample" in da.dims:
                    da = da.isel(sample=0)
                if first_only and "ensemble_member" in da.dims:
                    da = da.isel(ensemble_member=0)
                tails[src].update(np.asarray(da.values, dtype=np.float64).reshape(-1) * scale)
    out = {src: t.stats() for src, t in tails.items() if t.n}
    out["_n_files"] = n_files
    return out


_PRECIP_FILES = ("tail_stats.csv", "precip_stats.json", "precip_stats.csv", "tp_stats.json", "tp_stats.csv",
                 "stats.json", "stats.csv", "summary.json", "summary.csv")


def _norm_key(k: str) -> str | None:
    s = str(k).lower().replace("_", "").replace(".", "").replace("-", "")
    if s.startswith("wet"):
        return "wet_fraction"
    if s.startswith("neg"):
        return "negative_share"
    if s in ("max", "maximum", "tpmax", "maxmm"):
        return "max"
    m = re.fullmatch(r"(?:tp)?(?:p|q|percentile)(99|999|9999)(?:mm)?", s)
    if m:
        return {"99": "p99", "999": "p99.9", "9999": "p99.99"}[m.group(1)]
    return None


def _norm_source(s: str) -> str:
    s = str(s).lower()
    if s in ("y", "truth", "target"):
        return "truth"
    if s in ("x_interp", "input", "interp"):
        return "input"
    return "model" if s in ("y_pred", "model", "prediction", "pred") else s


def precip_from_evaluator(found) -> dict | None:
    """Per-source stats from a precip_dist statistics file, or None if there is none."""
    for name in _PRECIP_FILES:
        p = found.path / name
        if not p.exists():
            continue
        out: dict = {}
        if p.suffix == ".csv":
            for r in read_csv(p):
                if r.get("scope") not in (None, "", "all"):
                    continue          # tail_stats.csv: pooled rows only (per-date rows exist too)
                if r.get("variable") not in (None, "", "tp"):
                    continue
                src = _norm_source(r.get("source") or r.get("series") or r.get("field") or "model")
                for k, v in r.items():
                    nk = _norm_key(k)
                    if nk:
                        out.setdefault(src, {})[nk] = fnum(v)
        else:
            data = read_json(p)
            block = data.get("sources") or data.get("stats") or data
            if isinstance(block, dict):
                for src, d in block.items():
                    if isinstance(d, dict):
                        for k, v in d.items():
                            nk = _norm_key(k)
                            if nk:
                                out.setdefault(_norm_source(src), {})[nk] = fnum(v)
        if any(out.values()):
            out["_file"] = name
            return out
    return None


def precip_section(ctx: Context, res: SectionResult) -> str:
    ref = ctx.reference
    head = ("## 3b. Precipitation distribution (tp, 6 h, mm)\n\nWet fraction > 0.1 mm, p99, "
            "p99.9, p99.99, max (after clipping at 0) and the share of negative raw values; row, "
            "truth, reference. Pre-registered: a one-call wet fraction more than 0.03 above the "
            "reference's is a named deficit; p99.9 and p99.99 within 5 % of the truth are right.\n\n")
    header = ["set", "row", "calls", "quantity", "row value", "truth value", "ref value", "row - ref", "row/truth - 1 %",
              "status", "source"]
    rows = []
    sets_opt = ctx.opt("precip_sets") or []

    def stats_for(label: str):
        out = {}
        row = ctx.by_label[label]
        evs = ctx.results_by_key(label, "precip_dist", ("plots", "tail_stats.csv", "precip_stats.json", "precip_stats.csv",
                                                        "tp_stats.json", "tp_stats.csv", "stats.json",
                                                        "stats.csv"))
        sets = sorted({k.split("/")[0] for k in evs}) or list(sets_opt) or row.sets[:1] or ctx.set_dirs(label)[:1]
        for s in sets:
            key = ("precip", label, s)
            if key in ctx.cache:
                out[s] = ctx.cache[key]
                continue
            got = None
            for k, f in evs.items():
                if k.split("/")[0] == s:
                    got = precip_from_evaluator(f)
                    if got:
                        got = (got, f"precip_dist {got.pop('_file')}")
                        break
            if not got:
                pdir = ctx.pred_dir(row, s)
                if pdir is None:
                    got = (None, f"no precip_dist statistics and no prediction files for set {s}")
                else:
                    st = precip_stats(pdir, scale=float(ctx.opt("tp_scale", 1000.0)))
                    got = (st, "fallback: prediction files" + ("" if all(
                        v.get("exact", True) for k2, v in st.items() if isinstance(v, dict)) else " (approximate percentiles)"))
            ctx.cache[key] = got
            out[s] = got
        return out

    try:
        ref_stats = stats_for(ref)
    except Exception as exc:  # noqa: BLE001
        ref_stats = {}
        rows.append(["", ref, ctx.ref_row.calls] + [na(f"unreadable reference input: {exc}")] * 8)
    for row in ctx.rows:
        with row_guard(rows, ["", row.label, row.calls], len(header) - 3, res, row.label, "precip"):
            mine = stats_for(row.label)
            if not mine:
                rows.append(["", row.label, row.calls] + [na("no set to read")] * 8)
                res.add(row.label, "precip " + na("no set to read"))
                continue
            for s, (st, src) in sorted(mine.items()):
                if st is None or "model" not in st:
                    rows.append([s, row.label, row.calls] + [na(src if st is None else "no tp in the files")] * 7 + [src])
                    res.add(row.label, f"precip {s}: " + na(src if st is None else "no tp"))
                    continue
                rst = (ref_stats.get(s) or (None, ""))[0] or {}
                m, t, r = st.get("model", {}), (rst.get("truth") or st.get("truth") or {}), rst.get("model", {})
                frags = []
                for k, name in PRECIP_KEYS:
                    mv, tv, rv = fnum(m.get(k)), fnum(t.get(k)), fnum(r.get(k))
                    status = ""
                    if row.label != ref:
                        if k == "wet_fraction" and math.isfinite(mv - rv):
                            status = ("named deficit (> 0.03 above ref)" if (mv - rv > 0.03 and row.calls == 1)
                                      else "above ref by > 0.03" if mv - rv > 0.03 else "within 0.03 of ref")
                            frags.append(f"wet {fmt(mv, 3)} vs ref {fmt(rv, 3)}")
                        if k in ("p99.9", "p99.99") and math.isfinite(tv) and tv:
                            status = "within 5 % of truth" if abs(mv / tv - 1) <= 0.05 else "outside 5 % of truth"
                            frags.append(f"{k} {fmt(100 * (mv / tv - 1), 1, signed=True)}% vs truth "
                                         f"(ref {fmt(100 * (rv / tv - 1), 1, signed=True) if math.isfinite(rv) else 'n/a'}%)")
                        if k == "negative_share" and math.isfinite(mv):
                            frags.append(f"neg {fmt(mv, 4)}")
                    rows.append([s, row.label, row.calls, name, fmt(mv, 4), fmt(tv, 4),
                                 fmt(rv, 4) if r else na(f"no {ref} stats"), fmt(mv - rv, 4, signed=True),
                                 fmt(100 * (mv / tv - 1) if tv else math.nan, 2, signed=True), status, src])
                res.add(row.label, f"precip {s}: " + ("; ".join(frags) if frags else "reference"))
    return head + table(ctx, "extremes_precip", header, rows)


def section(ctx: Context) -> SectionResult:
    res = SectionResult("3. Extremes", "")
    md = ["# 3. Extremes\n"]
    guarded(md, "## 3a. Storm raw extremes", lambda: tc_section(ctx, res))
    guarded(md, "## 3b. Precipitation", lambda: precip_section(ctx, res))
    res.md = "\n".join(md)
    return res
