#!/usr/bin/env python
"""Follow-up tests Z1-Z4 on GLB450e2_m (realism t2m, 17 Nov 2026). CPU only, reuses realism_t2m.py statics.
  case DATE LEAD | run [--workers N] | aggregate
Z1 ensemble std (ddof 1) of the low-pass and the high-pass part (judge 0.6-degree operator) of 2t msl sp 10ff,
   over sea / land zhr<500 / land zhr>1500, model vs truth pf vs input members.
Z2 low-pass msl move draw k - x_interp k (and truth ctrl - input ctrl) by land/sea x local-solar-time bin.
Z3 sp diagnosed from each field's own msl and 2t: sp_diag = msl exp(-g zhr / (Rd (2t + 0.00325 zhr))); calibrated with
   c = mean(truth pf sp - truth pf sp_diag) per point and valid hour (00/12 UTC), leave-one-date-out.
Z4 2t ensemble std by signed coast distance, inland-water proxy and dz band.
"""
import os, sys, glob, pickle, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import realism_t2m as R

OUTZ = R.OUT + "/z"
ZV = ["2t", "msl", "sp", "10ff"]
MASKS = ["sea", "land_lt500", "land_gt1500"]


def masks(st):
    return {"sea": st.sea, "land_lt500": st.L & (st.zhr < 500), "land_gt1500": st.L & (st.zhr > 1500)}


def zcase(hdate, lead, st, M):
    import zarr, netCDF4 as nc
    t0 = time.time()
    fn = f"{R.X}/pred/{R.ROW}_L{lead:03d}/predictions_{hdate}_step{lead:03d}.nc"
    with nc.Dataset(fn) as d:
        d.set_auto_mask(False)
        iv = [R.states(d).index(v) for v in ("10u", "10v", "2t", "msl", "sp")]
        yp = np.asarray(d["y_pred"][0])[:, :, iv].astype(np.float64)
        xm = np.asarray(d["x_interp_members"][0])[:, :, iv].astype(np.float64)
        yc = np.asarray(d["y"][0])[:, iv].astype(np.float64)
        xc = np.asarray(d["x_interp"][0])[:, iv].astype(np.float64)
    zi, zo, r = R.store_row(lead, hdate)
    ov = list(zo.attrs["variables"])
    tp = np.moveaxis(zo["data"][r][[ov.index(v) for v in ("10u", "10v", "2t", "msl", "sp")]].astype(np.float64), 0, 2)  # (10,N,5)

    def fields(a):  # (10|N..., 5) -> dict
        return {"2t": a[..., 2], "msl": a[..., 3], "sp": a[..., 4], "10ff": np.hypot(a[..., 0], a[..., 1])}
    F = {"model": fields(yp), "truth_pf": fields(tp), "input": fields(xm)}
    mk = masks(st)
    res = {"hdate": hdate, "lead": lead, "z1": [], "z2": {}, "z4": {}}
    low = {}
    for v in ZV:
        for src, f in F.items():
            lo = st.lowpass(f[v]); hp = f[v] - lo
            if v == "msl":
                low[src] = lo
            vl, vh, vt = lo.var(axis=0, ddof=1), hp.var(axis=0, ddof=1), f[v].var(axis=0, ddof=1)
            for mn, m in mk.items():
                res["z1"].append((src, v, mn, int(m.sum()), float(vl[m].mean()), float(vh[m].mean()), float(vt[m].mean())))
    # Z2
    lab, names = st.lst_scheme(lead)
    dm = low["model"] - low["input"]
    dc = st.lowpass(yc[:, 3]) - st.lowpass(xc[:, 3])
    res["z2"]["model_minus_input"] = R.binacc(lab, len(names), dm).sum(axis=0)
    res["z2"]["truth_ctrl_minus_input_ctrl"] = R.binacc(lab, len(names), dc)
    res["z2_names"] = names
    # Z3 (land interior points only)
    L = st.L
    z = st.zhr

    def spdiag(msl, t2):
        return msl * np.exp(-R.G * z / (R.RD * (t2 + 0.00325 * z)))
    mdiag = spdiag(F["model"]["msl"], F["model"]["2t"])
    tdiag = spdiag(F["truth_pf"]["msl"], F["truth_pf"]["2t"])
    tsp, msp = F["truth_pf"]["sp"], F["model"]["sp"]
    res["z3_rt"] = (tsp - tdiag).mean(axis=0)[L].astype(np.float32)
    e = mdiag - tsp
    res["z3_Se"] = e.sum(axis=0)[L].astype(np.float32)
    res["z3_See"] = (e ** 2).sum(axis=0)[L].astype(np.float32)
    zlab, znames = st.schemes["zhr_band"]
    res["z3_direct"] = R.binacc(zlab, len(znames), msp - tsp).sum(axis=0)
    res["z3_floor"] = R.binacc(zlab, len(znames), tsp - tsp[np.roll(np.arange(10), -1)]).sum(axis=0)
    res["z3_spread"] = {k: R.binacc(zlab, len(znames), a.var(axis=0, ddof=1)) for k, a in
                        (("direct", msp), ("diag", mdiag), ("truth_pf", tsp))}
    # Z4
    for sn in ("coast_km", "dz_band"):
        lab4, n4 = st.schemes[sn]
        res["z4"][sn] = {src: R.binacc(lab4, len(n4), F[src]["2t"].var(axis=0, ddof=1)) for src in F}
    os.makedirs(OUTZ + "/parts", exist_ok=True)
    p = f"{OUTZ}/parts/{hdate}_L{lead:03d}.pkl"
    with open(p + ".tmp", "wb") as f:
        pickle.dump(res, f)
    os.replace(p + ".tmp", p)
    open(p + ".done", "w").write(f"{time.time() - t0:.1f}\n")
    print(f"ZCASE_DONE {hdate} L{lead:03d} {time.time() - t0:.0f}s", flush=True)
    return p


_ST = _M = None
def _w(c):
    global _ST, _M
    if _ST is None:
        _ST, _M = R.Stat(), R.load_matrix()
    try:
        return zcase(c[0], c[1], _ST, _M)
    except Exception as e:
        import traceback
        print(f"ZCASE_FAIL {c} {e!r}\n{traceback.format_exc()}", flush=True)
        return None


def run(workers):
    from multiprocessing import Pool
    todo = [c for c in R.all_cases() if not os.path.exists(f"{OUTZ}/parts/{c[0]}_L{c[1]:03d}.pkl.done")]
    print(f"ZRUN todo={len(todo)} workers={workers}", flush=True)
    with Pool(workers) as p:
        r = list(p.imap_unordered(_w, todo))
    nf = sum(x is None for x in r)
    print(f"ZRUN_{'DONE' if nf == 0 else 'FAIL'} failed={nf}", flush=True)


def aggregate():
    import pandas as pd
    st = R.Stat()
    parts = sorted(glob.glob(OUTZ + "/parts/*.pkl"))
    tabs = OUTZ + "/tables"; os.makedirs(tabs, exist_ok=True)
    Z1, Z2, Z4, Z3d = [], [], [], []
    rt_sum, rt_n, rt_date = {}, {}, {}
    cache = []
    zl, zn = st.schemes["zhr_band"]
    zlabL = zl[st.L]
    for p in parts:
        r = pickle.load(open(p, "rb"))
        h, L = r["hdate"], r["lead"]; vh = L % 24
        Z1 += [(h, L, *x) for x in r["z1"]]
        for src, a in r["z2"].items():
            for b, bn in enumerate(r["z2_names"]):
                n, s, ss = a[b]
                if n > 0:
                    Z2.append((h, L, src, bn, int(n), np.sqrt(ss / n)))
        for sn, dct in r["z4"].items():
            names = st.schemes[sn][1]
            for src, a in dct.items():
                for b, bn in enumerate(names):
                    n, s, _ = a[b]
                    if n > 0:
                        Z4.append((h, L, sn, src, bn, int(n), np.sqrt(s / n)))
        for kind, a in (("direct", r["z3_direct"]), ("floor", r["z3_floor"])):
            for b, bn in enumerate(zn):
                n, s, ss = a[b]
                if n > 0:
                    Z3d.append((h, L, kind, bn, int(n), s / n, np.sqrt(ss / n)))
        for kind, a in r["z3_spread"].items():
            for b, bn in enumerate(zn):
                n, s, _ = a[b]
                if n > 0:
                    Z3d.append((h, L, "spread_" + kind, bn, int(n), np.nan, np.sqrt(s / n)))
        rt = r["z3_rt"].astype(np.float64)
        rt_sum[vh] = rt_sum.get(vh, 0) + rt; rt_n[vh] = rt_n.get(vh, 0) + 1
        rt_date[(vh, h)] = rt_date.get((vh, h), 0) + rt
        rt_date[(vh, h, "n")] = rt_date.get((vh, h, "n"), 0) + 1
        cache.append((h, L, vh, p))
    for h, L, vh, p in cache:
        r = pickle.load(open(p, "rb"))
        c = (rt_sum[vh] - rt_date[(vh, h)]) / (rt_n[vh] - rt_date[(vh, h, "n")])  # leave-one-date-out
        Se, See = r["z3_Se"].astype(np.float64), r["z3_See"].astype(np.float64)
        for kind, s1, s2 in (("diag", Se, See), ("diag_cal", Se + 10 * c, See + 2 * c * Se + 10 * c * c)):
            for b, bn in enumerate(zn[:-1]):  # land bands only
                m = zlabL == b
                n = 10 * m.sum()
                if n > 0:
                    Z3d.append((h, L, kind, bn, int(n), s1[m].sum() / n, np.sqrt(s2[m].sum() / n)))
    c0 = ["date", "lead"]
    z1 = pd.DataFrame(Z1, columns=c0 + ["source", "var", "mask", "n", "var_low", "var_high", "var_total"])
    z2 = pd.DataFrame(Z2, columns=c0 + ["source", "lst_bin", "n", "lowpass_rms"])
    z3 = pd.DataFrame(Z3d, columns=c0 + ["kind", "zhr_band", "n", "bias", "rmse_or_spread"])
    z4 = pd.DataFrame(Z4, columns=c0 + ["scheme", "source", "bin", "n", "spread"])
    for nm, df in (("z1", z1), ("z2", z2), ("z3", z3), ("z4", z4)):
        df.to_csv(f"{tabs}/{nm}_per_date.csv.gz", index=False, float_format="%.6g")
    # Z1 summary: std per date, ratio model/truth per date, mean/sd over dates
    for part in ("low", "high", "total"):
        z1[f"std_{part}"] = np.sqrt(z1[f"var_{part}"])
    w = z1.pivot_table(index=["date", "lead", "var", "mask"], columns="source", values=["std_low", "std_high", "std_total"])
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.reset_index()
    rows = []
    for part in ("low", "high", "total"):
        w[f"ratio_{part}"] = w[f"std_{part}_model"] / w[f"std_{part}_truth_pf"]
        w[f"ratio_in_{part}"] = w[f"std_{part}_input"] / w[f"std_{part}_truth_pf"]
    for (lead, v, m), g in w.groupby(["lead", "var", "mask"]):
        d = dict(lead=lead, var=v, mask=m, n_dates=len(g))
        for part in ("low", "high", "total"):
            d.update({f"std_{part}_model": g[f"std_{part}_model"].mean(), f"std_{part}_truth_pf": g[f"std_{part}_truth_pf"].mean(),
                      f"std_{part}_input": g[f"std_{part}_input"].mean(), f"ratio_{part}_mean": g[f"ratio_{part}"].mean(),
                      f"ratio_{part}_sd_dates": g[f"ratio_{part}"].std(), f"ratio_{part}_dates_gt1": int((g[f"ratio_{part}"] > 1).sum()),
                      f"ratio_in_{part}_mean": g[f"ratio_in_{part}"].mean()})
        rows.append(d)
    pd.DataFrame(rows).to_csv(f"{tabs}/Z1_spread_by_scale.csv", index=False, float_format="%.4g")
    z2s = z2.groupby(["lead", "source", "lst_bin"]).agg(lowpass_rms=("lowpass_rms", "mean"), sd_dates=("lowpass_rms", "std"),
                                                       n_dates=("date", "nunique"), n_points=("n", "mean")).reset_index()
    z2s.to_csv(f"{tabs}/Z2_msl_move_by_lst.csv", index=False, float_format="%.4g")
    z3s = z3.groupby(["lead", "kind", "zhr_band"]).agg(bias=("bias", "mean"), bias_sd_dates=("bias", "std"),
                                                      bias_npos=("bias", lambda x: int((x > 0).sum())),
                                                      rmse_or_spread=("rmse_or_spread", "mean"), n_dates=("date", "nunique")).reset_index()
    z3a = z3.groupby(["date", "kind", "zhr_band"]).apply(lambda d: pd.Series({"bias": d.bias.mean(), "rmse_or_spread": np.sqrt((d.rmse_or_spread ** 2).mean())}),
                                                         include_groups=False).reset_index()
    z3b = z3a.groupby(["kind", "zhr_band"]).agg(bias=("bias", "mean"), bias_sd_dates=("bias", "std"), bias_npos=("bias", lambda x: int((x > 0).sum())),
                                               rmse_or_spread=("rmse_or_spread", "mean"), n_dates=("date", "nunique")).reset_index()
    z3b["lead"] = 0
    pd.concat([z3s, z3b]).to_csv(f"{tabs}/Z3_sp_diagnosed.csv", index=False, float_format="%.4g")
    z4s = z4.pivot_table(index=["date", "lead", "scheme", "bin"], columns="source", values="spread").reset_index()
    z4s["ratio"] = z4s.model / z4s.truth_pf
    o = z4s.groupby(["lead", "scheme", "bin"]).agg(spread_model=("model", "mean"), spread_truth_pf=("truth_pf", "mean"), spread_input=("input", "mean"),
                                                  ratio_mean=("ratio", "mean"), ratio_sd_dates=("ratio", "std"),
                                                  ratio_dates_gt1=("ratio", lambda x: int((x > 1).sum())), n_dates=("date", "nunique")).reset_index()
    oa = z4s.groupby(["scheme", "bin"]).agg(spread_model=("model", "mean"), spread_truth_pf=("truth_pf", "mean"), spread_input=("input", "mean"),
                                           ratio_mean=("ratio", "mean"), ratio_sd_dates=("ratio", "std"),
                                           ratio_dates_gt1=("ratio", lambda x: int((x > 1).sum())), n_dates=("date", "count")).reset_index()
    oa["lead"] = 0
    pd.concat([o, oa]).to_csv(f"{tabs}/Z4_2t_spread_coast_dz.csv", index=False, float_format="%.4g")
    for f in sorted(glob.glob(tabs + "/*.csv")):
        print("ZTABLE", os.path.basename(f), os.path.getsize(f), flush=True)
    print("ZAGG_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("args", nargs="*"); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.cmd == "case":
        st, M = R.Stat(), R.load_matrix(); zcase(a.args[0], int(a.args[1]), st, M)
    elif a.cmd == "run":
        run(a.workers)
    elif a.cmd == "aggregate":
        aggregate()
