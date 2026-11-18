#!/usr/bin/env python
"""2d / skt extension of the realism t2m read on the 10-output rows GLB450e2_m10_L012 / _L024 (job 2959525).
  bitcheck | case DATE LEAD | run [--workers N] | aggregate
Truth for 2d/skt/tcw = 9 km enfh control y (no pf member truth): ensemble-mean(model) - truth ctrl is labelled unpaired.
Reference rows: the same quantities for 2t, where truth pf exists (truth pf mean - ctrl gives the size of a 10-member
truth mean vs the control, the right yardstick for model mean - ctrl).
"""
import os, sys, glob, pickle, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import realism_t2m as R

ROW10 = "GLB450e2_m10"
OUT2 = R.OUT + "/d2"
LEADS10 = [12, 24]
V10 = ["10u", "10v", "2t", "msl", "sp", "tp", "cp", "2d", "skt", "tcw"]
VS = ["2t", "2d", "skt", "tcw"]


def fname(row, hdate, lead):
    return f"{R.X}/pred/{row}_L{lead:03d}/predictions_{hdate}_step{lead:03d}.nc"


def cases10():
    out = []
    for L in LEADS10:
        for f in sorted(glob.glob(f"{R.X}/pred/{ROW10}_L{L:03d}/predictions_*_step{L:03d}.nc")):
            out.append((os.path.basename(f).split("_")[1], L))
    return out


def bitcheck():
    import netCDF4 as nc
    for h, L in cases10():
        a, b = nc.Dataset(fname(ROW10, h, L)), nc.Dataset(fname(R.ROW, h, L))
        a.set_auto_mask(False); b.set_auto_mask(False)
        sa, sb = list(a["weather_state"][:]), list(b["weather_state"][:])
        out = []
        for v in sb:
            ia, ib = sa.index(v), sb.index(v)
            for arr in ("y_pred", "x_interp_members", "y", "x_interp"):
                d = np.nanmax(np.abs(np.asarray(a[arr][..., ia], np.float64) - np.asarray(b[arr][..., ib], np.float64)))
                out.append((v, arr, d))
        worst = max(out, key=lambda x: x[2])
        print(f"BITCHECK {h} L{L:03d} states={sa} worst={worst[0]}/{worst[1]} {worst[2]:.3g} "
              f"y_pred_max={max(d for v, a_, d in out if a_ == 'y_pred'):.3g} seeds_equal={a.getncattr('seeds') == b.getncattr('seeds')}", flush=True)
        a.close(); b.close()


def dcase(hdate, lead, st, M):
    import netCDF4 as nc
    t0 = time.time()
    with nc.Dataset(fname(ROW10, hdate, lead)) as d:
        d.set_auto_mask(False)
        sa = list(d["weather_state"][:])
        iv = [sa.index(v) for v in VS]
        mod = np.moveaxis(np.asarray(d["y_pred"][0])[:, :, iv], 2, 0).astype(np.float64)       # (4, 10, N)
        inp = np.moveaxis(np.asarray(d["x_interp_members"][0])[:, :, iv], 2, 0).astype(np.float64)
        tc = np.asarray(d["y"][0])[:, iv].T.astype(np.float64)                                   # (4, N)
        ic = np.asarray(d["x_interp"][0])[:, iv].T.astype(np.float64)
    zi, zo, r = R.store_row(lead, hdate)
    ov = list(zo.attrs["variables"])
    tpf2t = zo["data"][r, ov.index("2t")].astype(np.float64)                                      # (10, N)
    res = {"hdate": hdate, "lead": lead, "season": R.season(int(hdate[4:6]))}
    sch = dict(st.schemes); sch["lst"] = st.lst_scheme(lead)
    t1 = {}
    for sn, (lab, names) in sch.items():
        nb = len(names)
        t1[sn] = {"model_mean_minus_ctrl": R.binacc(lab, nb, mod.mean(axis=1) - tc),
                  "input_mean_minus_ctrl": R.binacc(lab, nb, inp.mean(axis=1) - tc),
                  "model_member_minus_ctrl": R.binacc(lab, nb, mod - tc[:, None]).sum(axis=1),
                  "input_member_minus_ctrl": R.binacc(lab, nb, inp - tc[:, None]).sum(axis=1),
                  "truthpf2t_mean_minus_ctrl": R.binacc(lab, nb, tpf2t.mean(axis=0) - tc[0]),
                  "truthpf2t_member_minus_ctrl": R.binacc(lab, nb, tpf2t - tc[0]).sum(axis=0),
                  "model_spread": R.binacc(lab, nb, mod.var(axis=1, ddof=1)),
                  "input_spread": R.binacc(lab, nb, inp.var(axis=1, ddof=1)),
                  "truthpf2t_spread": R.binacc(lab, nb, tpf2t.var(axis=0, ddof=1))}
    res["t1"] = t1
    # lapse slopes (2d, skt): model k - x_interp k vs dz ; truth ctrl - input ctrl
    t2 = []
    for reg, rm in st.regions.items():
        if reg == "box_land":
            continue
        m = rm & (np.abs(st.dz) > 100)
        n = int(m.sum())
        if n < 30:
            continue
        x = st.dz[m]
        for vi, v in enumerate(VS):
            if v == "tcw":
                continue
            for k in range(10):
                s, r2 = R.slope_fit(x, (mod[vi, k] - inp[vi, k])[m]); t2.append(("model", reg, v, k, s * 1000, r2, n))
            s, r2 = R.slope_fit(x, (tc[vi] - ic[vi])[m]); t2.append(("truth_ctrl", reg, v, -1, s * 1000, r2, n))
    res["t2"] = t2
    # 2d > 2t and skt - 2t
    t4 = []
    for sname, sm in (("land", st.L), ("sea", st.sea)):
        def frac(t2m, d2m, src, k):
            ex = (d2m - t2m)[sm]
            t4.extend([(src, k, sname, "2d_gt_2t+0.1", float((ex > 0.1).mean())), (src, k, sname, "2d_gt_2t+1", float((ex > 1).mean())),
                       (src, k, sname, "max_2d_minus_2t", float(ex.max())), (src, k, sname, "p99.9_2d_minus_2t", float(np.percentile(ex, 99.9)))])
        frac(tc[0], tc[1], "truth_ctrl", -1); frac(ic[0], ic[1], "input_ctrl", -1)
        for k in range(10):
            frac(mod[0, k], mod[1, k], "model", k); frac(inp[0, k], inp[1, k], "input_member", k)
    res["t4"] = t4
    lab, names = st.lst_scheme(lead)
    t4s = []
    for b, nm in enumerate(names):
        m = lab == b
        if m.sum() < 100:
            continue
        for src, e in (("truth_ctrl", tc[2] - tc[0]), ("input_ctrl", ic[2] - ic[0])):
            x = e[m]; t4s.append((src, -1, nm, int(m.sum()), x.mean(), *np.percentile(x, [1, 99])))
        for k in range(10):
            x = (mod[2, k] - mod[0, k])[m]; t4s.append(("model", k, nm, int(m.sum()), x.mean(), *np.percentile(x, [1, 99])))
    res["t4s"] = t4s
    # fine-scale amplitude of 2d / skt (mean |hp|) by zhr band, model members vs truth ctrl vs input
    zl, zn = st.schemes["zhr_band"]
    hpv = {}
    for vi, v in enumerate(VS[:3]):
        hm = mod[vi] - st.lowpass(mod[vi]); ht = tc[vi] - st.lowpass(tc[vi]); hi = inp[vi] - st.lowpass(inp[vi])
        hpv[v] = {"model": R.binacc(zl, len(zn), np.abs(hm)).sum(axis=0), "truth_ctrl": R.binacc(zl, len(zn), np.abs(ht)),
                  "input": R.binacc(zl, len(zn), np.abs(hi)).sum(axis=0)}
    res["hp"] = hpv
    os.makedirs(OUT2 + "/parts", exist_ok=True)
    p = f"{OUT2}/parts/{hdate}_L{lead:03d}.pkl"
    with open(p + ".tmp", "wb") as f:
        pickle.dump(res, f)
    os.replace(p + ".tmp", p)
    print(f"DCASE_DONE {hdate} L{lead:03d} {time.time() - t0:.0f}s", flush=True)
    return p


_ST = _M = None
def _w(c):
    global _ST, _M
    if _ST is None:
        _ST, _M = R.Stat(), R.load_matrix()
    try:
        return dcase(c[0], c[1], _ST, _M)
    except Exception as e:
        import traceback
        print(f"DCASE_FAIL {c} {e!r}\n{traceback.format_exc()}", flush=True)
        return None


def run(workers):
    from multiprocessing import Pool
    todo = [c for c in cases10() if not os.path.exists(f"{OUT2}/parts/{c[0]}_L{c[1]:03d}.pkl")]
    print(f"DRUN todo={len(todo)} workers={workers}", flush=True)
    with Pool(workers) as p:
        r = list(p.imap_unordered(_w, todo))
    nf = sum(x is None for x in r)
    print(f"DRUN_{'DONE' if nf == 0 else 'FAIL'} failed={nf}", flush=True)


def aggregate():
    import pandas as pd
    st = R.Stat()
    names = {k: v[1] for k, v in st.schemes.items()}; names["lst"] = st.lst_scheme(0)[1]
    tabs = OUT2 + "/tables"; os.makedirs(tabs, exist_ok=True)
    A1, A2, A4, A4s, AH = [], [], [], [], []
    for p in sorted(glob.glob(OUT2 + "/parts/*.pkl")):
        r = pickle.load(open(p, "rb")); h, L, se = r["hdate"], r["lead"], r["season"]
        for sn, dct in r["t1"].items():
            for q, acc in dct.items():
                vs = ["2t"] if q.startswith("truthpf2t") else VS
                for vi, v in enumerate(vs):
                    a = acc[vi] if acc.ndim == 3 else acc
                    for b, bn in enumerate(names[sn]):
                        n, s, ss = a[b]
                        if n > 0:
                            A1.append((h, L, se, q, v, sn, bn, int(n), s / n, np.sqrt(ss / n)))
        A2 += [(h, L, se, *x) for x in r["t2"]]
        A4 += [(h, L, se, *x) for x in r["t4"]]
        A4s += [(h, L, se, *x) for x in r["t4s"]]
        for v, dct in r["hp"].items():
            for src, a in dct.items():
                for b, bn in enumerate(names["zhr_band"]):
                    n, s, _ = a[b]
                    if n > 0:
                        AH.append((h, L, v, src, bn, int(n), s / n))
    c0 = ["date", "lead", "season"]
    d1 = pd.DataFrame(A1, columns=c0 + ["quantity", "var", "scheme", "bin", "n", "mean", "rms"])
    d2 = pd.DataFrame(A2, columns=c0 + ["source", "region", "var", "member", "slope", "r2", "n"])
    d4 = pd.DataFrame(A4, columns=c0 + ["source", "member", "surface", "metric", "value"])
    d4s = pd.DataFrame(A4s, columns=c0 + ["source", "member", "lst_bin", "n", "mean", "p1", "p99"])
    dh = pd.DataFrame(AH, columns=["date", "lead", "var", "source", "zhr_band", "n", "mean_abs_hp"])
    for nm, df in (("d1", d1), ("d2", d2), ("d4", d4), ("d4s", d4s), ("dh", dh)):
        df.to_csv(f"{tabs}/{nm}_per_date.csv.gz", index=False, float_format="%.6g")
    # spread rows: binacc summed the per-point variance, so mean = mean variance; spread = sqrt(mean variance)
    sp = d1.quantity.str.endswith("spread")
    d1.loc[sp, "rms"] = np.sqrt(d1.loc[sp, "mean"]); d1.loc[sp, "mean"] = np.nan
    # D1: bias of means / member rms / spread (spread rows: rms = sqrt(mean var)) per lead, over dates
    D1 = d1.groupby(["lead", "quantity", "var", "scheme", "bin"]).agg(
        mean=("mean", "mean"), mean_sd_dates=("mean", "std"), mean_npos=("mean", lambda x: int((x > 0).sum())),
        rms=("rms", "mean"), n_dates=("date", "nunique"), n_points=("n", "mean")).reset_index()
    D1.to_csv(f"{tabs}/D1_2d_skt_binned.csv", index=False, float_format="%.4g")
    g = d2.groupby(["date", "lead", "season", "region", "var", "source"]).agg(slope=("slope", "mean"), r2=("r2", "mean"), n=("n", "first")).reset_index()
    D2 = g.pivot_table(index=["date", "lead", "season", "region", "var", "n"], columns="source", values=["slope", "r2"]).reset_index()
    D2.columns = ["_".join([c for c in col if c]).strip("_") for col in D2.columns.values]
    D2.to_csv(f"{tabs}/D2_lapse_2d_skt.csv", index=False, float_format="%.4g")
    D4 = d4.groupby(["lead", "source", "surface", "metric"]).value.agg(["mean", "std", "max", "count"]).reset_index()
    D4.to_csv(f"{tabs}/D4_2d_vs_2t.csv", index=False, float_format="%.5g")
    D4s = d4s.groupby(["lead", "source", "lst_bin"]).agg(n_cases=("date", "count"), mean=("mean", "mean"), p1=("p1", "mean"), p99=("p99", "mean")).reset_index()
    D4s.to_csv(f"{tabs}/D4_skt_minus_2t.csv", index=False, float_format="%.4g")
    DH = dh.groupby(["lead", "var", "source", "zhr_band"]).agg(mean_abs_hp=("mean_abs_hp", "mean"), sd_dates=("mean_abs_hp", "std"), n_dates=("date", "nunique")).reset_index()
    DH.to_csv(f"{tabs}/DH_fine_amplitude.csv", index=False, float_format="%.4g")
    for f in sorted(glob.glob(tabs + "/*.csv")):
        print("DTABLE", os.path.basename(f), os.path.getsize(f), flush=True)
    print("DAGG_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("args", nargs="*"); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.cmd == "bitcheck":
        bitcheck()
    elif a.cmd == "case":
        st, M = R.Stat(), R.load_matrix(); dcase(a.args[0], int(a.args[1]), st, M)
    elif a.cmd == "run":
        run(a.workers)
    elif a.cmd == "aggregate":
        aggregate()
