#!/usr/bin/env python3
"""o2560 Europe-box read (2 Nov 2026): scores per date of every row at one lead, from the prediction files, and the pairs.

Copy of the E2 seed-fix kit's paired_scores.py (DB vs CA wide read, 23 Oct) adapted to the o2560 box and to more than two
rows. For every date that all rows hold, and for each row (A100, B24, B50) plus the input row IN (x_interp of the same files,
a one-member forecast, so its fair CRPS is its mean absolute error and its spread is 0):
  * fair CRPS, member spread (ddof 1), ensemble-mean RMSE and the corrected spread/RMSE ratio sqrt((M+1)/M) x spread / RMSE
    (rule 32, M = 10 members), area-weighted (cos lat), for 10ff (from 10u, 10v), tp (mm per 6 h, clipped at 0), 10u, 10v,
    2t, 2d, msl, skt, tcw, cp, z_500, t_850, over three domains, all without the 1-degree edge strip of the box:
    'box' (10-70 N, 80 W-40 E), 'europe' (35-70 N, 25 W-40 E) and 'alps' (43-49 N, 4-16 E);
  * the tp tail per date over the box without the strip, members pooled: p99, p99.9, p99.99 and the maximum, for each row,
    the input and the truth; and the pooled count of points at or above 10, 25, 50, 100 mm per 6 h per field.
Pairs (A - B, mean over dates, SE = standard deviation of the per-date differences / sqrt(n), dates A better):
A100-B24 (equal GPU-hours), A100-B50 (latest steps), B50-B24, and each row minus IN.
Output: per_date.csv (row, date, domain, var, metric, value), paired.md, tail.md, meta.json.
Usage: python paired_scores_o2560.py <lead> <out dir> <label>=<pred dir> [<label>=<pred dir> ...]"""
import csv
import json
import math
import sys
from pathlib import Path

import netCDF4
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kit"))
from o2560_common import edge_strip, fcrps_terms  # noqa: E402

lead = int(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
ROWS = dict(a.split("=", 1) for a in sys.argv[3:])
ROWS = {k: Path(v) for k, v in ROWS.items()}
VARS = ("10ff", "tp", "10u", "10v", "2t", "2d", "msl", "skt", "tcw", "cp", "z_500", "t_850")
THR = (10, 25, 50, 100)
PCTS = (99.0, 99.9, 99.99)
DOMAINS = {"box": None, "europe": (35.0, 70.0, -25.0, 40.0), "alps": (43.0, 49.0, 4.0, 16.0)}


def read(path):
    with netCDF4.Dataset(str(path)) as ds:
        ds.set_auto_mask(False)
        st = [str(s) for s in ds["weather_state"][:]]
        lat = np.asarray(ds["lat_hres"][:], dtype=np.float64); lon = np.asarray(ds["lon_hres"][:], dtype=np.float64)
        lon = ((lon + 180.0) % 360.0) - 180.0
        yp = np.asarray(ds["y_pred"][0], dtype=np.float32)
        y = np.asarray(ds["y"][0], dtype=np.float64); xi = np.asarray(ds["x_interp"][0], dtype=np.float64)
        attrs = {k: ds.getncattr(k) for k in ("model", "ckpt_step", "sampler", "seeds", "weights_sha256", "code_pin")}
    return st, lat, lon, yp, y, xi, attrs


def read_seeds(path):
    with netCDF4.Dataset(str(path)) as ds:
        return ds.getncattr("seeds")


def fields(st, a):
    """a (..., P, S) -> dict of (..., P) float64 fields; tp and cp in mm per 6 h, clipped at 0."""
    i = {v: st.index(v) for v in st}
    g = lambda v: np.asarray(a[..., i[v]], dtype=np.float64)
    d = {v: g(v) for v in ("10u", "10v", "2t", "2d", "msl", "skt", "tcw", "z_500", "t_850")}
    d["10ff"] = np.hypot(d["10u"], d["10v"])
    d["tp"] = np.clip(g("tp") * 1000.0, 0, None); d["cp"] = np.clip(g("cp") * 1000.0, 0, None)
    return d


labels = list(ROWS)
sets = [{p.name[12:20] for p in d.glob(f"predictions_*_step{lead:03d}.nc")} for d in ROWS.values()]
dates = sorted(set.intersection(*sets))
print("rows", labels, "dates", len(dates), dates[:1], dates[-1:], flush=True)
if not dates:
    sys.exit("no common dates")
rows = []; meta = {}
counts = {k: np.zeros(len(THR)) for k in labels + ["IN", "truth"]}; nfield = {k: 0 for k in counts}
M = None
for d in dates:
    data = {}
    for lab, dd in ROWS.items():
        st, lat, lon, yp, y, xi, attrs = read(dd / f"predictions_{d}_step{lead:03d}.nc")
        meta.setdefault(lab, attrs); data[lab] = (st, lat, lon, yp, y, xi)
    st, lat, lon, _, y0, xi0 = data[labels[0]]
    for lab in labels[1:]:
        s2, la2, lo2, _, y2, xi2 = data[lab]
        assert s2 == st and np.array_equal(la2, lat) and np.array_equal(lo2, lon), f"{d}: grid or states differ ({lab})"
        assert np.array_equal(y2, y0) and np.array_equal(xi2, xi0), f"{d}: truth or input differ ({lab})"
    seed_sets = {lab: json.loads(str(read_seeds(ROWS[lab] / f"predictions_{d}_step{lead:03d}.nc"))) for lab in labels}
    assert len({json.dumps(s) for s in seed_sets.values()}) == 1, f"{d}: seeds differ across rows {seed_sets}"
    strip = edge_strip(lat, lon)
    w = np.cos(np.deg2rad(lat))
    doms = {}
    for dn, b in DOMAINS.items():
        m = ~strip if b is None else (~strip & (lat >= b[0]) & (lat <= b[1]) & (lon >= b[2]) & (lon <= b[3]))
        doms[dn] = m
    T = fields(st, y0)
    F = {lab: fields(st, data[lab][3]) for lab in labels}
    F["IN"] = {k: v[None] for k, v in fields(st, xi0).items()}
    M = data[labels[0]][3].shape[0]
    for lab in labels + ["IN"]:
        m_here = F[lab]["10u"].shape[0]
        for v in VARS:
            fc, sd, se = fcrps_terms(F[lab][v], T[v])
            for dn, msk in doms.items():
                ww = w * msk; s = ww.sum()
                rmse = math.sqrt(float((ww * se).sum() / s)); spr = float((ww * sd).sum() / s)
                rows += [dict(row=lab, date=d, domain=dn, var=v, metric="fcrps", value=float((ww * fc).sum() / s)),
                         dict(row=lab, date=d, domain=dn, var=v, metric="spread", value=spr),
                         dict(row=lab, date=d, domain=dn, var=v, metric="rmse", value=rmse)]
                if m_here > 1:
                    rows.append(dict(row=lab, date=d, domain=dn, var=v, metric="spread_rmse_corr",
                                     value=math.sqrt((m_here + 1) / m_here) * spr / rmse if rmse > 0 else float("nan")))
        tp = F[lab]["tp"][:, ~strip].ravel()
        rows += [dict(row=lab, date=d, domain="box", var="tp", metric=k, value=float(x)) for k, x in
                 zip([f"p{p:g}" for p in PCTS] + ["max"], (*np.percentile(tp, PCTS), tp.max()))]
        counts[lab] += [(tp >= t).sum() for t in THR]; nfield[lab] += F[lab]["tp"].shape[0]
    tt = T["tp"][~strip]
    rows += [dict(row="truth", date=d, domain="box", var="tp", metric=k, value=float(x)) for k, x in
             zip([f"p{p:g}" for p in PCTS] + ["max"], (*np.percentile(tt, PCTS), tt.max()))]
    counts["truth"] += [(tt >= t).sum() for t in THR]; nfield["truth"] += 1
    print("done", d, flush=True)

with open(out / "per_date.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
by = {}
for r in rows:
    by.setdefault((r["row"], r["domain"], r["var"], r["metric"]), {})[r["date"]] = r["value"]
n = len(dates)
seeds_equal = True   # asserted per date above (the run stops on any difference)
L = [f"# o2560 box read, +{lead} h: scores and paired differences (n = {n} dates x {M} members; IN = o1280 input interpolated)", "",
     "Rows: " + "; ".join(f"{lab} = {meta[lab]['model']} step {meta[lab]['ckpt_step']} (weights {str(meta[lab]['weights_sha256'])[:8]}, code {str(meta[lab]['code_pin'])[:9]})" for lab in labels),
     f"Sampler identical across rows: {len({meta[lab]['sampler'] for lab in labels}) == 1}; seeds identical across rows on every date: {seeds_equal}.",
     "Domains exclude the 1-degree edge strip. fcrps/rmse: lower is better; diff = first minus second, SE over dates; 'better' = dates the first is lower.",
     "spread_rmse_corr = sqrt((M+1)/M) x spread / RMSE, 1 is calibrated. tp and cp in mm per 6 h.", "",
     "## Per-row means over dates", "", "| domain | var | metric | " + " | ".join(labels + ["IN"]) + " |", "|---|---|---|" + "---|" * (len(labels) + 1)]
for dn in DOMAINS:
    for v in VARS:
        for mt in ("fcrps", "rmse", "spread", "spread_rmse_corr"):
            cells = []
            for lab in labels + ["IN"]:
                vals = by.get((lab, dn, v, mt))
                cells.append(f"{np.mean([vals[x] for x in dates]):.4g}" if vals else "-")
            L.append(f"| {dn} | {v} | {mt} | " + " | ".join(cells) + " |")
pairs = [(a, b) for a, b in (("A100", "B24"), ("A100", "B50"), ("B50", "B24")) if a in labels and b in labels]
pairs += [(lab, "IN") for lab in labels]
L += ["", "## Paired differences (first minus second)", "", "| pair | domain | var | metric | diff | SE | diff % | dates first better |", "|---|---|---|---|---|---|---|---|"]
for a, b in pairs:
    for dn in DOMAINS:
        for v in VARS:
            for mt in ("fcrps", "rmse") + (("spread",) if b != "IN" else ()):
                A = by[(a, dn, v, mt)]; B = by[(b, dn, v, mt)]
                x = np.array([A[q] for q in dates]); y = np.array([B[q] for q in dates]); dif = x - y
                se = dif.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
                better = int((dif < 0).sum()) if mt != "spread" else int((dif > 0).sum())
                L.append(f"| {a}-{b} | {dn} | {v} | {mt} | {dif.mean():+.4g} | {se:.2g} | {100 * dif.mean() / y.mean():+.2f} % | {better}/{n} |")
(out / "paired.md").write_text("\n".join(L) + "\n")
T2 = [f"# tp tail at +{lead} h, box without the edge strip, members pooled (n = {n} dates; mm per 6 h)", "",
      "Mean over dates of the per-date statistic; in brackets the paired difference to the truth and its SE.", "",
      "| statistic | " + " | ".join(labels + ["IN", "truth"]) + " |", "|---|" + "---|" * (len(labels) + 2)]
for mt in [f"p{p:g}" for p in PCTS] + ["max"]:
    t = np.array([by[("truth", "box", "tp", mt)][q] for q in dates]); cells = []
    for lab in labels + ["IN"]:
        x = np.array([by[(lab, "box", "tp", mt)][q] for q in dates]); dd = x - t
        cells.append(f"{x.mean():.2f} ({dd.mean():+.2f} +- {dd.std(ddof=1) / math.sqrt(n) if n > 1 else float('nan'):.2f})")
    T2.append(f"| {mt} | " + " | ".join(cells) + f" | {t.mean():.2f} |")
T2 += ["", "Per-date maximum (mm per 6 h): " + "; ".join(f"{q}: " + ", ".join(f"{lab} {by[(lab, 'box', 'tp', 'max')][q]:.1f}" for lab in labels + ['IN', 'truth']) for q in dates), "",
       "## Points at or above a threshold per field (model: per member field; IN and truth: per date)", "",
       "| mm per 6 h | " + " | ".join(labels + ["IN", "truth"]) + " |", "|---|" + "---|" * (len(labels) + 2)]
for i, t in enumerate(THR):
    T2.append(f"| {t} | " + " | ".join(f"{counts[k][i] / max(nfield[k], 1):.1f}" for k in labels + ["IN", "truth"]) + " |")
(out / "tail.md").write_text("\n".join(T2) + "\n")
json.dump(dict(lead=lead, dates=dates, members=M, rows={k: str(v) for k, v in ROWS.items()},
               meta={k: {kk: str(vv) for kk, vv in v.items()} for k, v in meta.items()}), open(out / "meta.json", "w"), indent=1)
print("PAIRED_SCORES_O2560_DONE", lead, labels, n)
