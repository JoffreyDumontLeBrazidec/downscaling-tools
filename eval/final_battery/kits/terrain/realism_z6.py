#!/usr/bin/env python
"""Z6: sp residual baseline on the paired control (truth y vs input x_interp), all 180 cases.
  run [--workers N] | aggregate
Residuals (Pa), per zhr band (land) and sea:
  a     = y_sp - xi_sp                                     (current residual target)
  b     = y_sp - xi_sp exp(-g dz / (Rd Tm)), Tm = xi_2t - 0.00325 dz   (layer mean going up dz from zlr to zhr)
  b_brief = same with Tm = xi_2t + 0.00325 dz              (sign as written in the brief)
  c     = same with Tm = xi_2t - 0.5 gamma dz, gamma = (t850 - t700) / ((z700 - z850)/g) from the input control (interp)
dz = zhr - zlr. All inputs are input fields (available at inference). Std and high-pass std (judge 0.6-degree operator).
Model reference per band: high-pass ensemble std of the model sp (10 draws).
"""
import os, sys, glob, pickle, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import realism_t2m as R

OUT6 = R.OUT + "/z6"


def case6(hdate, lead, st, M):
    import zarr, netCDF4 as nc
    t0 = time.time()
    with nc.Dataset(f"{R.X}/pred/{R.ROW}_L{lead:03d}/predictions_{hdate}_step{lead:03d}.nc") as d:
        d.set_auto_mask(False)
        ysp = np.asarray(d["y"][0, :, R.W.index("sp")], np.float64)
        xsp = np.asarray(d["x_interp"][0, :, R.W.index("sp")], np.float64)
        x2t = np.asarray(d["x_interp"][0, :, R.W.index("2t")], np.float64)
        msp = np.asarray(d["y_pred"][0, :, :, R.W.index("sp")], np.float64)
        xmsp = np.asarray(d["x_interp_members"][0, :, :, R.W.index("sp")], np.float64)
        xm2t = np.asarray(d["x_interp_members"][0, :, :, R.W.index("2t")], np.float64)
    zi, zo, r = R.store_row(lead, hdate)
    er = int(zo["e2_row"][r])
    i4 = zarr.open(R.INV4, mode="r"); iv = list(i4.attrs["variables"])
    g = lambda v: M @ i4["data"][er, iv.index(v), 0, :].astype(np.float64)
    t850, t700, z850, z700 = g("t_850"), g("t_700"), g("z_850"), g("z_700")
    gamma = (t850 - t700) / ((z700 - z850) / R.G)          # K/m, positive when T falls with height
    dz = st.dz
    res_ = {
        "a": ysp - xsp,
        "b": ysp - xsp * np.exp(-R.G * dz / (R.RD * (x2t - 0.00325 * dz))),
        "b_brief": ysp - xsp * np.exp(-R.G * dz / (R.RD * (x2t + 0.00325 * dz))),
        "c": ysp - xsp * np.exp(-R.G * dz / (R.RD * (x2t - 0.5 * gamma * dz))),
    }
    zl, zn = st.schemes["zhr_band"]
    out = {"hdate": hdate, "lead": lead, "raw": {}, "hp": {}}
    for k, v in res_.items():
        out["raw"][k] = R.binacc(zl, len(zn), v)
        out["hp"][k] = R.binacc(zl, len(zn), v - st.lowpass(v))
    bm = msp - xmsp * np.exp(-R.G * dz / (R.RD * (xm2t - 0.00325 * dz)))   # model draw k vs baseline of its own input k
    out["raw"]["b_model_draws"] = R.binacc(zl, len(zn), bm).sum(axis=0)
    out["hp"]["b_model_draws"] = R.binacc(zl, len(zn), bm - st.lowpass(bm)).sum(axis=0)
    hm = msp - st.lowpass(msp)
    out["model_hp_spread"] = R.binacc(zl, len(zn), hm.var(axis=0, ddof=1))
    out["gamma"] = R.binacc(zl, len(zn), gamma * 1000)
    os.makedirs(OUT6 + "/parts", exist_ok=True)
    p = f"{OUT6}/parts/{hdate}_L{lead:03d}.pkl"
    pickle.dump(out, open(p + ".tmp", "wb")); os.replace(p + ".tmp", p)
    print(f"Z6_DONE {hdate} L{lead:03d} {time.time() - t0:.0f}s", flush=True)
    return p


_ST = _M = None
def _w(c):
    global _ST, _M
    if _ST is None:
        _ST, _M = R.Stat(), R.load_matrix()
    try:
        return case6(c[0], c[1], _ST, _M)
    except Exception as e:
        import traceback
        print(f"Z6_FAIL {c} {e!r}\n{traceback.format_exc()}", flush=True)
        return None


def run(workers):
    from multiprocessing import Pool
    todo = [c for c in R.all_cases() if not os.path.exists(f"{OUT6}/parts/{c[0]}_L{c[1]:03d}.pkl")]
    print(f"Z6RUN todo={len(todo)}", flush=True)
    with Pool(workers) as p:
        r = list(p.imap_unordered(_w, todo))
    nf = sum(x is None for x in r)
    print(f"Z6RUN_{'DONE' if nf == 0 else 'FAIL'} failed={nf}", flush=True)


def aggregate():
    import pandas as pd
    st = R.Stat(); zn = st.schemes["zhr_band"][1]
    rows = []
    for p in sorted(glob.glob(OUT6 + "/parts/*.pkl")):
        o = pickle.load(open(p, "rb")); h, L = o["hdate"], o["lead"]
        for part in ("raw", "hp"):
            for k, a in o[part].items():
                for b, bn in enumerate(zn):
                    n, s, ss = a[b]
                    if n > 0:
                        m = s / n
                        rows.append((h, L, part, k, bn, int(n), m, np.sqrt(max(ss / n - m * m, 0)), np.sqrt(ss / n)))
        for b, bn in enumerate(zn):
            n, s, _ = o["model_hp_spread"][b]
            if n > 0:
                rows.append((h, L, "hp", "model_spread", bn, int(n), np.nan, np.sqrt(s / n), np.nan))
            n, s, _ = o["gamma"][b]
            if n > 0:
                rows.append((h, L, "raw", "gamma_K_per_km", bn, int(n), s / n, np.nan, np.nan))
    d = pd.DataFrame(rows, columns=["date", "lead", "part", "residual", "zhr_band", "n", "mean", "std", "rms"])
    os.makedirs(OUT6 + "/tables", exist_ok=True)
    d.to_csv(OUT6 + "/tables/z6_per_date.csv.gz", index=False, float_format="%.6g")
    s = d.groupby(["part", "residual", "zhr_band"]).agg(mean=("mean", "mean"), std=("std", "mean"), std_sd_cases=("std", "std"),
                                                       rms=("rms", "mean"), n_cases=("date", "count"), n_points=("n", "mean")).reset_index()
    for part in ("raw", "hp"):
        a = s[(s.part == part) & (s.residual == "a")].set_index("zhr_band")["std"]
        for k in ("b", "b_brief", "c"):
            m = (s.part == part) & (s.residual == k)
            s.loc[m, "ratio_a_over_this"] = a.reindex(s.loc[m, "zhr_band"]).values / s.loc[m, "std"].values
    s.to_csv(OUT6 + "/tables/Z6_sp_residual_baseline.csv", index=False, float_format="%.4g")
    order = ["0-200", "200-500", "500-1000", "1000-1500", "1500-2000", "2000-3000", "3000+", "sea"]
    pd.set_option("display.width", 250)
    for part in ("raw", "hp"):
        print(part); print(s[s.part == part].pivot_table(index="zhr_band", columns="residual", values="std").reindex(order).round(2).to_string())
    print("ratio a/x"); print(s.pivot_table(index=["part", "zhr_band"], columns="residual", values="ratio_a_over_this").round(2))
    print("mean residual"); print(s[s.part == "raw"].pivot_table(index="zhr_band", columns="residual", values="mean").reindex(order).round(2))
    print("Z6AGG_DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    run(a.workers) if a.cmd == "run" else aggregate()
