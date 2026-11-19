#!/usr/bin/env python
"""Headline numbers for the 2d / skt extension (out/d2/tables). Usage: danalyze.py TABLES"""
import sys
import numpy as np, pandas as pd
T = sys.argv[1]
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
d1 = pd.read_csv(f"{T}/D1_2d_skt_binned.csv")
order = ["0-200", "200-500", "500-1000", "1000-1500", "1500-2000", "2000-3000", "3000+", "sea"]
for sc, idx in (("all", None), ("zhr_band", order), ("coast_km", None), ("dz_band", None)):
    e = d1[d1.scheme == sc]
    print(f"== D1 {sc}: mean (K) of ensemble-mean minus truth ctrl [npos/30]")
    p = e[e.quantity.isin(["model_mean_minus_ctrl", "input_mean_minus_ctrl", "truthpf2t_mean_minus_ctrl"])].copy()
    p["txt"] = p["mean"].round(3).astype(str) + " [" + p.mean_npos.astype(int).astype(str) + "]"
    print(p.pivot_table(index=["lead", "bin"], columns=["quantity", "var"], values="txt", aggfunc="first").to_string())
    print(f"-- D1 {sc}: member rms vs truth ctrl (K) and ensemble std (K)")
    q = e[e.quantity.isin(["model_member_minus_ctrl", "input_member_minus_ctrl", "truthpf2t_member_minus_ctrl", "model_spread", "input_spread", "truthpf2t_spread"])]
    print(q.pivot_table(index=["lead", "bin"], columns=["quantity", "var"], values="rms").round(3).to_string())
d2 = pd.read_csv(f"{T}/D2_lapse_2d_skt.csv")
print("== D2 lapse slopes (K/km): model member mean vs truth ctrl")
for v in ("2d", "skt", "2t"):
    e = d2[d2["var"] == v]
    if len(e) == 0:
        continue
    diff = e.slope_model - e.slope_truth_ctrl
    print(v, f"n={len(e)} dates={e.date.nunique()} corr={np.corrcoef(e.slope_model, e.slope_truth_ctrl)[0,1]:.3f} mean diff={diff.mean():+.3f} mean|diff|={diff.abs().mean():.3f} "
             f"reg={np.polyfit(e.slope_truth_ctrl, e.slope_model, 1).round(3)} truth mean {e.slope_truth_ctrl.mean():.2f} model {e.slope_model.mean():.2f}")
    print(e.assign(diff=diff).groupby("region")[["slope_truth_ctrl", "slope_model", "diff"]].mean().round(2).to_string())
d4 = pd.read_csv(f"{T}/D4_2d_vs_2t.csv")
print("== D4 2d vs 2t"); print(d4.pivot_table(index=["surface", "metric"], columns=["source", "lead"], values="mean").round(5).to_string())
d4s = pd.read_csv(f"{T}/D4_skt_minus_2t.csv")
print("== D4 skt - 2t by LST bin"); print(d4s.pivot_table(index=["lst_bin"], columns=["source", "lead"], values=["mean", "p1", "p99"]).round(2).to_string())
dh = pd.read_csv(f"{T}/DH_fine_amplitude.csv")
print("== DH mean |high-pass| (K)"); print(dh.pivot_table(index=["var", "zhr_band"], columns=["source", "lead"], values="mean_abs_hp").round(3).to_string())
