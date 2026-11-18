#!/usr/bin/env python
"""Z5 figures + headline printout from out/z/tables. Usage: zplots.py TABLES OUTDIR"""
import sys, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
T, O = sys.argv[1], sys.argv[2]; os.makedirs(O, exist_ok=True)
z1 = pd.read_csv(f"{T}/Z1_spread_by_scale.csv")
print("== Z1 ratio model/truth pf (mean over dates; dates>1 of 30)")
for part in ("low", "high", "total"):
    p = z1.pivot_table(index=["var", "mask"], columns="lead", values=f"ratio_{part}_mean").round(3)
    q = z1.pivot_table(index=["var", "mask"], columns="lead", values=f"ratio_{part}_dates_gt1")
    print(part); print(p.astype(str) + " (" + q.astype(int).astype(str) + ")")
print("input/truth ratio, total"); print(z1.pivot_table(index=["var", "mask"], columns="lead", values="ratio_in_total_mean").round(3))
print("std values low/high model truth input (msl, 2t)")
print(z1[z1["var"].isin(["msl", "2t"])][["lead", "var", "mask", "std_low_model", "std_low_truth_pf", "std_low_input", "std_high_model", "std_high_truth_pf", "std_high_input"]].round(3).to_string())
fig, axs = plt.subplots(2, 4, figsize=(15, 6.5), sharex=True)
for i, v in enumerate(("2t", "msl", "sp", "10ff")):
    for j, part in enumerate(("low", "high")):
        ax = axs[j, i]
        for m, c in (("sea", "tab:blue"), ("land_lt500", "tab:green"), ("land_gt1500", "tab:brown")):
            e = z1[(z1["var"] == v) & (z1["mask"] == m)].sort_values("lead")
            ax.errorbar(e.lead, e[f"ratio_{part}_mean"], yerr=2 * e[f"ratio_{part}_sd_dates"] / np.sqrt(e.n_dates), marker="o", capsize=2, color=c, label=m)
        ax.axhline(1, color="k", lw=0.8); ax.set_title(f"{v}: {'low-pass (>0.6 deg)' if part == 'low' else 'high-pass (<0.6 deg)'} spread ratio", fontsize=9)
        ax.set_xticks([12, 24, 36, 48, 60, 72])
        if j == 1: ax.set_xlabel("lead (h)")
axs[0, 0].legend(fontsize=7); axs[0, 0].set_ylabel("model / truth pf ensemble std"); axs[1, 0].set_ylabel("model / truth pf ensemble std")
fig.suptitle("Z1: ensemble std ratio GLB450e2_m / truth pf by scale and lead (mean over 30 dates, bars 2 SE); 00 UTC valid at 24/48/72 h", fontsize=9)
fig.tight_layout(); fig.savefig(f"{O}/z1_spread_ratio_by_scale.png", dpi=130); plt.close(fig)

z2 = pd.read_csv(f"{T}/Z2_msl_move_by_lst.csv")
print("== Z2 low-pass msl move RMS (Pa) by local solar time bin")
print(z2.pivot_table(index=["source", "lst_bin"], columns="lead", values="lowpass_rms").round(1))
z3 = pd.read_csv(f"{T}/Z3_sp_diagnosed.csv")
print("== Z3 (lead 0 = leads pooled)")
print(z3[z3.lead == 0].pivot_table(index="zhr_band", columns="kind", values="rmse_or_spread").round(1))
print(z3[z3.lead == 0].pivot_table(index="zhr_band", columns="kind", values="bias").round(2))
print(z3[z3.lead == 0].pivot_table(index="zhr_band", columns="kind", values="bias_npos"))
print(z3[(z3.lead > 0) & z3.zhr_band.isin(["1500-2000", "2000-3000", "3000+"])].pivot_table(index=["zhr_band", "lead"], columns="kind", values="rmse_or_spread").round(1))
order = ["0-200", "200-500", "500-1000", "1000-1500", "1500-2000", "2000-3000", "3000+"]
d = z3[z3.lead == 0].pivot_table(index="zhr_band", columns="kind", values="rmse_or_spread").reindex(order)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
x = np.arange(len(order))
for k, sty in (("direct", dict(color="tab:red", marker="o", label="direct sp: draw k - truth pf k")),
               ("diag_cal", dict(color="tab:purple", marker="s", label="sp_diag_cal (from draw msl, 2t, calibrated)")),
               ("diag", dict(color="tab:purple", ls=":", marker="s", mfc="none", label="sp_diag uncalibrated")),
               ("floor", dict(color="k", marker="^", label="floor: truth pf k - pf k+1"))):
    ax[0].plot(x, d[k], **sty)
for k, sty in (("spread_direct", dict(color="tab:red", marker="o", label="direct sp")), ("spread_diag", dict(color="tab:purple", marker="s", label="sp_diag")),
               ("spread_truth_pf", dict(color="k", marker="^", label="truth pf"))):
    ax[1].plot(x, d[k], **sty)
for a, t in ((ax[0], "sp RMSE (Pa), land, leads pooled, 30 dates"), (ax[1], "sp ensemble std (Pa)")):
    a.set_xticks(x); a.set_xticklabels(order, rotation=30); a.set_xlabel("O1280 orography band (m)"); a.set_title(t); a.legend(fontsize=7)
ax[0].set_yscale("log")
fig.suptitle("Z3: direct sp vs sp diagnosed from the draw's own msl and 2t (calibration c from truth pf, leave-one-date-out, per point and valid hour)", fontsize=9)
fig.tight_layout(); fig.savefig(f"{O}/z3_sp_diagnosed.png", dpi=130); plt.close(fig)
z4 = pd.read_csv(f"{T}/Z4_2t_spread_coast_dz.csv")
print("== Z4 2t spread ratio model/truth pf (lead 0 = pooled; n_dates there = cases)")
print(z4.pivot_table(index=["scheme", "bin"], columns="lead", values="ratio_mean").round(3))
print(z4[z4.lead == 0][["scheme", "bin", "spread_model", "spread_truth_pf", "spread_input", "ratio_mean", "ratio_dates_gt1", "n_dates"]].round(3).to_string())
