#!/usr/bin/env python
"""Three summary figures from out/tables (house colours: truth black, input blue dashed, model red).
Usage: plot_summary.py TABLES_DIR OUT_DIR"""
import sys, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

T, O = sys.argv[1], sys.argv[2]
os.makedirs(O, exist_ok=True)
STY = {"model": dict(color="tab:red", ls="-", marker="o", label=os.environ.get("FB_ROW", "GLB450e2") + " draw k - truth pf k"),
       "input": dict(color="tab:blue", ls="--", marker="s", label="input x_interp k - truth pf k"),
       "floor": dict(color="k", ls="-", marker="^", label="truth pf k - truth pf k+1 (floor)")}


def save(fig, p):
    fig.savefig(p, dpi=130); plt.close(fig)
    Image.open(p).convert("RGB").quantize(colors=256).save(p, optimize=True)
    print("FIG", p, os.path.getsize(p) // 1024, "KB")


# F1 bias and RMSE vs signed distance to coast, all leads pooled
t1 = pd.read_csv(f"{T}/T1_bias_rmse_binned.csv")
order = ["sea<-100", "sea-100..-30", "sea-30..-10", "sea-10..0", "land0..10", "land10..30", "land30..100", "land>100"]
fig, axs = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
for j, (v, u, f) in enumerate((("2t", "K", 1), ("msl", "hPa", 0.01), ("sp", "hPa", 0.01))):
    for src in ("model", "input", "floor"):
        d = t1[(t1.scheme == "coast_km") & (t1["var"] == v) & (t1.lead == 0) & (t1.source == src)].set_index("bin").reindex(order)
        x = np.arange(len(order)); se = d.bias_sd_dates / np.sqrt(d.n_dates)
        axs[0, j].errorbar(x, d.bias_mean * f, yerr=2 * se * f, capsize=2, **STY[src])
        axs[1, j].plot(x, d.rmse_mean * f, **STY[src])
    axs[0, j].axhline(0, color="0.6", lw=0.6); axs[0, j].axvline(3.5, color="0.6", lw=0.6, ls=":")
    axs[1, j].axvline(3.5, color="0.6", lw=0.6, ls=":")
    axs[0, j].set_title(f"{v}: bias ({u}), mean over 30 dates, bars 2 SE"); axs[1, j].set_title(f"{v}: RMSE ({u})")
    axs[1, j].set_xticks(np.arange(len(order))); axs[1, j].set_xticklabels([o.replace("..", "-") for o in order], rotation=40, fontsize=7)
axs[0, 0].legend(fontsize=7)
fig.suptitle("Error vs signed distance to coast (km, land positive), E2 box interior, leads 12-72 h pooled; "
             "members unpaired (eefh k vs enfh k), so compare model with the floor", fontsize=9)
fig.tight_layout(); save(fig, f"{O}/f1_coast_bias_rmse.png")

# F2 lapse slope scatter, 2t, model member mean vs truth control
t2 = pd.read_csv(f"{T}/T2_lapse_slopes.csv")
d = t2[t2["var"] == "2t"]
fig, axs = plt.subplots(1, 2, figsize=(12, 5.6))
regs = sorted(d.region.unique()); cols = plt.cm.tab10(np.linspace(0, 1, 10))
for i, r in enumerate(regs):
    e = d[d.region == r]
    night = (e.lead % 24) == 0
    axs[0].scatter(e.slope_truth_ctrl[night], e.slope_model[night], s=10, color=cols[i], marker="o", label=r)
    axs[0].scatter(e.slope_truth_ctrl[~night], e.slope_model[~night], s=10, color=cols[i], marker="x")
    axs[1].scatter(e.slope_truth_ctrl, e.slope_model - e.slope_truth_ctrl, s=8, color=cols[i], marker="o")
lim = [min(d.slope_truth_ctrl.min(), d.slope_model.min()) - 0.5, max(d.slope_truth_ctrl.max(), d.slope_model.max()) + 0.5]
axs[0].plot(lim, lim, "k-", lw=0.8); axs[0].axvline(-2, color="0.5", ls=":", lw=0.8); axs[0].axhline(-2, color="0.5", ls=":", lw=0.8)
axs[0].set_xlabel("truth slope (K/km): control y - x_interp control vs dz"); axs[0].set_ylabel("model slope (K/km): draw k - x_interp k vs dz, member mean")
axs[0].set_title("2t effective lapse rate, land |dz|>100 m; o = 00 UTC valid, x = 12 UTC valid"); axs[0].legend(fontsize=7, ncol=2)
axs[1].axhline(0, color="k", lw=0.8); axs[1].axvline(-2, color="0.5", ls=":", lw=0.8)
axs[1].set_xlabel("truth slope (K/km)"); axs[1].set_ylabel("model minus truth slope (K/km)")
axs[1].set_title("slope error vs truth slope (inversions to the right of -2 K/km)")
fig.suptitle(f"{os.environ.get('FB_ROW', 'GLB450e2_m')}, 30 dates x 6 leads x 8 regions ({len(d)} points)", fontsize=9)
fig.tight_layout(); save(fig, f"{O}/f2_lapse_scatter.png")

# F3 O320 imprint: mean |hp 2t| and RMSE of err vs distance to nearest O320 point
t5 = pd.read_csv(f"{T}/T5_o320_imprint.csv"); t5 = t5[t5.lead == 0]
order = ["0-3", "3-6", "6-9", "9-12", "12-15", "15-18", "18+"]
fig, axs = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
for j, sub in enumerate(("land", "mountain", "flat")):
    e = t5[t5.subset == sub]
    for q, sty in (("abs_hp_model", dict(color="tab:red", marker="o", label="model draw k")),
                   ("abs_hp_truth_pf", dict(color="k", marker="^", label="truth pf k")),
                   ("abs_hp_truth_ctrl", dict(color="k", ls=":", marker="v", label="truth control")),
                   ("abs_hp_input", dict(color="tab:blue", ls="--", marker="s", label="input x_interp k"))):
        f = e[e.quantity == q].set_index("d320_km").reindex(order)
        axs[0, j].plot(np.arange(7), f["mean"], **sty)
    for q, sty in (("err_model", STY["model"]), ("err_floor", STY["floor"])):
        f = e[e.quantity == q].set_index("d320_km").reindex(order)
        axs[1, j].plot(np.arange(7), f["rms"], **sty)
    n = e[e.quantity == "err_model"].set_index("d320_km").reindex(order)["n_points"]
    axs[0, j].set_title(f"{sub}: mean |high-pass 2t| (K)"); axs[1, j].set_title(f"{sub}: RMSE of 2t error (K)")
    axs[1, j].set_xticks(np.arange(7)); axs[1, j].set_xticklabels([f"{o}\nn={int(x / 10)}" for o, x in zip(order, n)], fontsize=7)
    axs[1, j].set_xlabel("distance from O1280 point to nearest O320 input point (km)")
axs[0, 0].legend(fontsize=7); axs[1, 0].legend(fontsize=7)
fig.suptitle("Input-grid imprint test (land, interior, leads pooled; mountain = |dz|>100 m): a flat truth curve with a sloped model curve would be an O320 imprint", fontsize=9)
fig.tight_layout(); save(fig, f"{O}/f3_o320_imprint.png")
