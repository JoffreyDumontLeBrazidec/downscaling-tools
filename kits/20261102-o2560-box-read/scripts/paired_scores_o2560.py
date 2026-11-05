#!/usr/bin/env python3
"""o2560 Europe-box read (2 Nov 2026): scores per date of every row at one lead, from the prediction files, and the pairs.

Adapted from the E2 seed-fix kit's paired_scores.py (DB vs CA wide read, 23 Oct) to the o2560 box and to more than two
rows; revised after the 2 Nov review (area weights, pooled spread, bootstrap ratio, per-member tp max, strict row checks).
For every date and each row (A100, B24, B50) plus the input row IN (x_interp of the same files, a one-member forecast, so
its fair CRPS is its mean absolute error and its variance is 0):
  * area weights cos(lat) / (points on the full octahedral ring), the point area of the o2560 reduced Gaussian grid;
  * per date and domain: fair CRPS, mean ensemble variance (ddof 1, 'var'), mean squared error of the ensemble mean
    ('mse'), and from them spread = sqrt(var) and RMSE = sqrt(mse), for 10ff (per member from 10u, 10v), tp (mm per 6 h,
    clipped at 0), 10u, 10v, 2t, 2d, msl, skt, tcw, cp, z_500, t_850, over three domains, all with the 1-degree edge strip
    of the box EXCLUDED: 'box' (10-70 N, 80 W-40 E), 'europe' (35-70 N, 25 W-40 E) and 'alps' (43-49 N, 4-16 E);
  * the corrected spread/RMSE ratio sqrt((M+1)/M) x sqrt(mean over dates of var) / sqrt(mean over dates of mse), pooled
    over dates (rule 32, as eval/evaluators/probabilistic/core/calibration.py), with a 95 % date bootstrap (seed 32, 2000
    resamples) and, between rows, the paired bootstrap of the ratio difference;
  * the tp tail per date over the box without the strip: pooled percentiles p99, p99.9, p99.99 (members pooled) and the
    mean over members of each member's maximum; the same for IN and the truth; counts at or above 10, 25, 50, 100 mm.
Pairs: A100-B24 (equal GPU-hours), A100-B50 (latest steps), B50-B24, each row minus IN. All rows must hold the same dates
and the same member count, with identical truth, input, grid and seeds on every date; the script stops otherwise.
Output: per_date.csv (row, date, domain, var, metric, value), paired.md, tail.md, ratio.csv, meta.json.
Usage: python paired_scores_o2560.py <lead> <out dir> <label>=<pred dir> [<label>=<pred dir> ...]"""
import csv
import json
import math
import sys
from pathlib import Path

import netCDF4
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kit"))
from o2560_common import area_weights, edge_strip, fcrps_terms, ratio_bootstrap  # noqa: E402

lead = int(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
ROWS = {k: Path(v) for k, v in (a.split("=", 1) for a in sys.argv[3:])}
VARS = ("10ff", "tp", "10u", "10v", "2t", "2d", "msl", "skt", "tcw", "cp", "z_500", "t_850")
THR = (10, 25, 50, 100)
PCTS = (99.0, 99.9, 99.99)
PCT_NAMES = [f"p{p:g}_pooled" for p in PCTS]
DOMAINS = {"box": None, "europe": (35.0, 70.0, -25.0, 40.0), "alps": (43.0, 49.0, 4.0, 16.0)}


def read(path):
    with netCDF4.Dataset(str(path)) as ds:
        ds.set_auto_mask(False)
        st = [str(s) for s in ds["weather_state"][:]]
        lat = np.asarray(ds["lat_hres"][:], dtype=np.float64); lon = np.asarray(ds["lon_hres"][:], dtype=np.float64)
        yp = np.asarray(ds["y_pred"][0], dtype=np.float32)
        y = np.asarray(ds["y"][0], dtype=np.float64); xi = np.asarray(ds["x_interp"][0], dtype=np.float64)
        attrs = {k: ds.getncattr(k) for k in ("model", "ckpt_step", "sampler", "seeds", "weights_sha256", "code_pin")}
    return st, lat, lon, yp, y, xi, attrs


def field(st, a, v):
    """One variable of a (..., P, S) as float64 (..., P); 10ff per member from that member's 10u and 10v; tp, cp in mm."""
    i = {s: k for k, s in enumerate(st)}
    g = lambda s: np.asarray(a[..., i[s]], dtype=np.float64)
    if v == "10ff":
        return np.hypot(g("10u"), g("10v"))
    if v in ("tp", "cp"):
        return np.clip(g(v) * 1000.0, 0, None)
    return g(v)


labels = list(ROWS)
sets = {lab: sorted(p.name[12:20] for p in d.glob(f"predictions_*_step{lead:03d}.nc")) for lab, d in ROWS.items()}
dates = sets[labels[0]]
for lab in labels[1:]:
    if sets[lab] != dates:
        sys.exit(f"FATAL: date sets differ: {labels[0]} {dates} vs {lab} {sets[lab]}")
print("rows", labels, "dates", len(dates), dates[:1], dates[-1:], flush=True)
if not dates:
    sys.exit("FATAL: no dates")
rows = []; meta = {}; M = None
counts = {k: np.zeros(len(THR)) for k in labels + ["IN", "truth"]}; nfield = {k: 0 for k in counts}
W = DOM = None
for d in dates:
    data = {}
    for lab, dd in ROWS.items():
        st, lat, lon, yp, y, xi, attrs = read(dd / f"predictions_{d}_step{lead:03d}.nc")
        meta.setdefault(lab, attrs); data[lab] = (st, lat, lon, yp, y, xi, attrs)
    st, lat, lon, yp0, y0, xi0, a0 = data[labels[0]]
    for lab in labels[1:]:
        s2, la2, lo2, yp2, y2, xi2, a2 = data[lab]
        assert s2 == st and np.array_equal(la2, lat) and np.array_equal(lo2, lon), f"{d}: grid or states differ ({lab})"
        assert np.array_equal(y2, y0) and np.array_equal(xi2, xi0), f"{d}: truth or input differ ({lab})"
        assert yp2.shape[0] == yp0.shape[0], f"{d}: member count differs ({lab} {yp2.shape[0]} vs {yp0.shape[0]})"
        assert json.loads(a2["seeds"]) == json.loads(a0["seeds"]), f"{d}: seeds differ ({lab})"
    if M is None:
        M = yp0.shape[0]
    assert yp0.shape[0] == M, f"{d}: member count changed across dates"
    if W is None:
        lon180 = ((lon + 180.0) % 360.0) - 180.0
        strip = edge_strip(lat, lon180)
        W = area_weights(lat, lon)
        DOM = {}
        for dn, b in DOMAINS.items():
            DOM[dn] = ~strip if b is None else (~strip & (lat >= b[0]) & (lat <= b[1]) & (lon180 >= b[2]) & (lon180 <= b[3]))
        lat0, lon0 = lat, lon
    assert np.array_equal(lat, lat0) and np.array_equal(lon, lon0), f"{d}: grid differs from the first date"
    for v in VARS:
        T = field(st, y0, v)
        fc_by = {}
        for lab in labels + ["IN"]:
            F = field(st, xi0, v)[None] if lab == "IN" else field(st, data[lab][3], v)
            fc, var, se = fcrps_terms(F, T)
            for dn, msk in DOM.items():
                ww = W * msk; s = ww.sum()
                mv = float((ww * var).sum() / s); ms = float((ww * se).sum() / s)
                rows += [dict(row=lab, date=d, domain=dn, var=v, metric="fcrps", value=float((ww * fc).sum() / s)),
                         dict(row=lab, date=d, domain=dn, var=v, metric="var", value=mv),
                         dict(row=lab, date=d, domain=dn, var=v, metric="mse", value=ms),
                         dict(row=lab, date=d, domain=dn, var=v, metric="spread", value=math.sqrt(max(mv, 0.0))),
                         dict(row=lab, date=d, domain=dn, var=v, metric="rmse", value=math.sqrt(ms))]
    nostrip = DOM["box"]
    for lab in labels + ["IN"]:
        tpf = field(st, xi0, "tp")[None] if lab == "IN" else field(st, data[lab][3], "tp")   # (m, P)
        tp = tpf[:, nostrip]
        rows += [dict(row=lab, date=d, domain="box", var="tp", metric=k, value=float(x)) for k, x in
                 zip(PCT_NAMES + ["max_member_mean"], (*np.percentile(tp.ravel(), PCTS), tp.max(axis=1).mean()))]
        counts[lab] += [(tp >= t).sum() for t in THR]; nfield[lab] += tp.shape[0]
    tt = field(st, y0, "tp")[nostrip]
    rows += [dict(row="truth", date=d, domain="box", var="tp", metric=k, value=float(x)) for k, x in
             zip(PCT_NAMES + ["max_member_mean"], (*np.percentile(tt, PCTS), tt.max()))]
    counts["truth"] += [(tt >= t).sum() for t in THR]; nfield["truth"] += 1
    print("done", d, flush=True)

with open(out / "per_date.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
by = {}
for r in rows:
    by.setdefault((r["row"], r["domain"], r["var"], r["metric"]), {})[r["date"]] = r["value"]
n = len(dates)
arr = lambda lab, dn, v, mt: np.array([by[(lab, dn, v, mt)][q] for q in dates])
pairs = [(a, b) for a, b in (("A100", "B24"), ("A100", "B50"), ("B50", "B24")) if a in labels and b in labels]

# corrected spread/RMSE pooled over dates, date bootstrap, paired bootstrap between rows
ratio_rows = []
for dn in DOMAINS:
    for v in VARS:
        for lab in labels:
            bs = ratio_bootstrap(arr(lab, dn, v, "var"), arr(lab, dn, v, "mse"), M)
            ratio_rows.append(dict(kind="row", row=lab, domain=dn, var=v, ratio=bs["ratio"], lo95=bs["lo"], hi95=bs["hi"], n_dates=n))
        for a, b in pairs:
            bs = ratio_bootstrap(arr(a, dn, v, "var"), arr(a, dn, v, "mse"), M, arr(b, dn, v, "var"), arr(b, dn, v, "mse"))
            ratio_rows.append(dict(kind="pair", row=f"{a}-{b}", domain=dn, var=v, ratio=bs["diff"], lo95=bs["diff_lo"], hi95=bs["diff_hi"], n_dates=n))
with open(out / "ratio.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(ratio_rows[0])); wr.writeheader(); wr.writerows(ratio_rows)
R = {(r["kind"], r["row"], r["domain"], r["var"]): r for r in ratio_rows}

L = [f"# o2560 box read, +{lead} h: scores and paired differences (n = {n} dates x {M} members; IN = o1280 input interpolated)", "",
     "Rows: " + "; ".join(f"{lab} = {meta[lab]['model']} step {meta[lab]['ckpt_step']} (weights {str(meta[lab]['weights_sha256'])[:8]}, code {str(meta[lab]['code_pin'])[:9]})" for lab in labels),
     f"Sampler identical across rows: {len({meta[lab]['sampler'] for lab in labels}) == 1}; seeds, truth, input and grid identical across rows on every date (checked).",
     "The 1-degree edge strip of the box IS excluded from every domain. Area weights cos(lat) / points on the full octahedral ring.",
     "fcrps, rmse: lower is better; diff = first minus second, SE = std of per-date differences / sqrt(n); 'better' = dates the first is lower.",
     "spread = sqrt(area-mean ensemble variance), rmse = sqrt(area-mean squared error of the ensemble mean), means over dates of the per-date values.",
     "ratio = sqrt((M+1)/M) x sqrt(mean var) / sqrt(mean mse) pooled over dates, 1 is calibrated, [95 % date-bootstrap interval, seed 32].",
     "tp and cp in mm per 6 h.", "",
     "## Per-row values", "", "| domain | var | metric | " + " | ".join(labels + ["IN"]) + " |", "|---|---|---|" + "---|" * (len(labels) + 1)]
for dn in DOMAINS:
    for v in VARS:
        for mt in ("fcrps", "rmse", "spread"):
            L.append(f"| {dn} | {v} | {mt} | " + " | ".join(f"{arr(lab, dn, v, mt).mean():.4g}" for lab in labels + ["IN"]) + " |")
        L.append(f"| {dn} | {v} | ratio | " + " | ".join(
            f"{R[('row', lab, dn, v)]['ratio']:.3f} [{R[('row', lab, dn, v)]['lo95']:.3f}, {R[('row', lab, dn, v)]['hi95']:.3f}]" for lab in labels) + " | - |")
L += ["", "## Paired differences (first minus second)", "",
      "| pair | domain | var | metric | diff | SE or [95 % CI] | diff % | dates first better |", "|---|---|---|---|---|---|---|---|"]
for a, b in pairs + [(lab, "IN") for lab in labels]:
    for dn in DOMAINS:
        for v in VARS:
            for mt in ("fcrps", "rmse"):
                x = arr(a, dn, v, mt); y = arr(b, dn, v, mt); dif = x - y
                se = dif.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
                L.append(f"| {a}-{b} | {dn} | {v} | {mt} | {dif.mean():+.4g} | {se:.2g} | {100 * dif.mean() / y.mean():+.2f} % | {int((dif < 0).sum())}/{n} |")
            if b != "IN":
                rp = R[("pair", f"{a}-{b}", dn, v)]
                closer = sum(abs(1 - math.sqrt((M + 1) / M) * math.sqrt(by[(a, dn, v, 'var')][q] / by[(a, dn, v, 'mse')][q]))
                             < abs(1 - math.sqrt((M + 1) / M) * math.sqrt(by[(b, dn, v, 'var')][q] / by[(b, dn, v, 'mse')][q])) for q in dates)
                L.append(f"| {a}-{b} | {dn} | {v} | ratio | {rp['ratio']:+.3f} | [{rp['lo95']:+.3f}, {rp['hi95']:+.3f}] | - | {closer}/{n} closer to 1 |")
(out / "paired.md").write_text("\n".join(L) + "\n")

T2 = [f"# tp tail at +{lead} h, box without the edge strip (n = {n} dates; mm per 6 h)", "",
      "p..._pooled: percentile over all points and members pooled; max_member_mean: mean over members of each member's maximum "
      "(the truth and IN have one field). Mean over dates of the per-date statistic; in brackets the paired difference to "
      "the truth and its SE over dates.", "",
      "| statistic | " + " | ".join(labels + ["IN", "truth"]) + " |", "|---|" + "---|" * (len(labels) + 2)]
for mt in PCT_NAMES + ["max_member_mean"]:
    t = arr("truth", "box", "tp", mt); cells = []
    for lab in labels + ["IN"]:
        x = arr(lab, "box", "tp", mt); dd = x - t
        cells.append(f"{x.mean():.2f} ({dd.mean():+.2f} +- {dd.std(ddof=1) / math.sqrt(n) if n > 1 else float('nan'):.2f})")
    T2.append(f"| {mt} | " + " | ".join(cells) + f" | {t.mean():.2f} |")
T2 += ["", "Per-date maximum (mm per 6 h; rows: mean over members of the member maximum): " + "; ".join(
    f"{q}: " + ", ".join(f"{lab} {by[(lab, 'box', 'tp', 'max_member_mean')][q]:.1f}" for lab in labels + ['IN', 'truth']) for q in dates), "",
       "## Points at or above a threshold per field (rows: per member field; IN and truth: per date)", "",
       "| mm per 6 h | " + " | ".join(labels + ["IN", "truth"]) + " |", "|---|" + "---|" * (len(labels) + 2)]
for i, t in enumerate(THR):
    T2.append(f"| {t} | " + " | ".join(f"{counts[k][i] / max(nfield[k], 1):.1f}" for k in labels + ["IN", "truth"]) + " |")
(out / "tail.md").write_text("\n".join(T2) + "\n")
json.dump(dict(lead=lead, dates=dates, members=M, rows={k: str(v) for k, v in ROWS.items()},
               meta={k: {kk: str(vv) for kk, vv in v.items()} for k, v in meta.items()}), open(out / "meta.json", "w"), indent=1)
print("PAIRED_SCORES_O2560_DONE", lead, labels, n)
