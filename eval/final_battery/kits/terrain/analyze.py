#!/usr/bin/env python
"""Headline numbers from out/tables (prints text; used for the README and the report)."""
import sys
import numpy as np, pandas as pd
T = sys.argv[1]
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
t1 = pd.read_csv(f"{T}/T1_bias_rmse_binned.csv")
print("== T1 all-leads, scheme all")
a = t1[(t1.lead == 0) & (t1.scheme == "all")]
print(a.pivot_table(index=["var", "bin"], columns="source", values="rmse_mean").round(3))
print(a[a.source == "model"][["var", "bin", "bias_mean", "bias_sd_dates", "bias_npos", "floor_bias_sd_dates", "bias_flag", "n_dates"]].round(4).to_string())
print("== T1 model rmse / floor rmse, all leads, per scheme bin (2t)")
for sc in ("zhr_band", "dz_band", "coast_km", "lst"):
    b = t1[(t1.lead == 0) & (t1.scheme == sc) & (t1["var"] == "2t")]
    p = b.pivot_table(index="bin", columns="source", values=["rmse_mean", "bias_mean"])
    p.columns = [f"{x}_{y}" for x, y in p.columns]
    p["ratio_model_floor"] = p["rmse_mean_model"] / p["rmse_mean_floor"]
    n = b[b.source == "model"].set_index("bin")[["n_points_per_date", "bias_flag", "bias_npos"]]
    ca = t1[(t1.lead > 0) & (t1.scheme == sc) & (t1["var"] == "2t") & (t1.source == "model")].groupby("bin").rmse_dates_above_floor.sum().rename("cases_above_floor_of180")
    print(sc); print(p.round(3).join(n).join(ca).to_string())
print("== T1 flagged bins (model), per lead incl. 0")
f = t1[(t1.source == "model") & (t1.bias_flag == True)]
print(f[["var", "lead", "scheme", "bin", "bias_mean", "bias_sd_dates", "floor_bias_sd_dates", "bias_npos", "n_dates", "rmse_mean", "floor_rmse_mean", "rmse_dates_above_floor"]].round(4).to_string())
print("== T1 msl/sp by zhr and dz (model vs floor rmse, all leads)")
for v in ("msl", "sp"):
    for sc in ("zhr_band", "dz_band", "coast_km"):
        b = t1[(t1.lead == 0) & (t1.scheme == sc) & (t1["var"] == v)]
        p = b.pivot_table(index="bin", columns="source", values="rmse_mean")
        print(v, sc); print((p[["model", "floor", "input"]]).round(2).T.to_string())
print("== T1 ctrl_pair 2d/skt (truth ctrl - input ctrl), all leads, scheme all/zhr")
c = t1[(t1.lead == 0) & (t1.source == "ctrl_pair") & (t1.scheme.isin(["all", "zhr_band"]))]
print(c.pivot_table(index=["scheme", "bin"], columns="var", values="bias_mean").round(3))
print("== T1 per lead, scheme all land, 2t model/floor rmse")
b = t1[(t1.scheme == "all") & (t1.bin == "land") & (t1["var"] == "2t") & (t1.lead > 0)]
print(b.pivot_table(index="lead", columns="source", values="rmse_mean").round(3))
s = pd.read_csv(f"{T}/T1e_season.csv")
print("== T1e season, all, land")
print(s[(s.scheme == "all") & (s.bin == "land")].pivot_table(index=["var", "season"], columns="source", values=["bias_mean", "rmse_mean"]).round(3))

t2 = pd.read_csv(f"{T}/T2_lapse_slopes.csv")
print("== T2 2t slopes: model member mean vs truth ctrl")
d = t2[t2["var"] == "2t"].copy()
d["diff"] = d.slope_model - d.slope_truth_ctrl
d["night"] = (d.lead % 24) == 0
print("n", len(d), "corr", round(np.corrcoef(d.slope_model, d.slope_truth_ctrl)[0, 1], 4), "mean diff", round(d["diff"].mean(), 3),
      "mean |diff|", round(d["diff"].abs().mean(), 3), "member sd mean", round(d.slope_sd_members_model.mean(), 3),
      "truth_pf_unpaired vs ctrl mean|diff|", round((d.slope_truth_pf_unpaired - d.slope_truth_ctrl).abs().mean(), 3))
print(d.groupby("region")[["slope_truth_ctrl", "slope_model", "diff"]].agg(["mean"]).round(3))
inv = d[d.slope_truth_ctrl > -2]
print("inversions truth>-2:", len(inv), "of", len(d), "mean diff", round(inv["diff"].mean(), 3), "sd", round(inv["diff"].std(), 3),
      "model also >-2:", int((inv.slope_model > -2).sum()), "dates", inv.date.nunique())
inv0 = d[d.slope_truth_ctrl > 0]
print("truth>0:", len(inv0), "mean truth", round(inv0.slope_truth_ctrl.mean(), 2), "mean model", round(inv0.slope_model.mean(), 2),
      "frac model/truth", round(inv0.slope_model.sum() / inv0.slope_truth_ctrl.sum(), 3), "model>0:", int((inv0.slope_model > 0).sum()))
print(inv.groupby(["region", "season", "night"]).agg(n=("diff", "size"), truth=("slope_truth_ctrl", "mean"), model=("slope_model", "mean"), diff=("diff", "mean")).round(2))
nw = d[(d.season == "DJF") & d.night]
print("winter nights n", len(nw), "mean truth", round(nw.slope_truth_ctrl.mean(), 2), "model", round(nw.slope_model.mean(), 2))
fit = np.polyfit(d.slope_truth_ctrl, d.slope_model, 1); print("regression model = a*truth + b", fit.round(3))
for v in ("sp", "2d", "skt"):
    e = t2[t2["var"] == v]
    cols = [c for c in e.columns if c.startswith("slope_") and "sd" not in c]
    print(v, e[cols].describe().loc[["mean", "std", "min", "max"]].round(2).to_string())
jan = d[(d.region == "alps") & (d.date.astype(str).str[4:6] == "01") & (d.lead == 24)].sort_values("slope_truth_ctrl", ascending=False)
print("alps jan +24 strongest inversion:"); print(jan[["date", "slope_truth_ctrl", "slope_model", "r2_truth_ctrl"]].head(3))
t3 = pd.read_csv(f"{T}/T3_hydrostatic_summary.csv")
print("== T3 (all leads mean over cases)")
print(t3.groupby(["metric", "source"])["mean"].mean().round(6).to_string())
t4 = pd.read_csv(f"{T}/T4_2d_vs_2t.csv")
print("== T4"); print(t4.groupby(["surface", "metric", "source"])["mean"].mean().round(4).to_string())
t4s = pd.read_csv(f"{T}/T4_skt_minus_2t.csv"); print(t4s.round(2).to_string())
t5 = pd.read_csv(f"{T}/T5_o320_imprint.csv"); t5 = t5[t5.lead == 0]
print("== T5"); print(t5.pivot_table(index=["subset", "d320_km"], columns="quantity", values="mean")[["abs_hp_model", "abs_hp_truth_pf", "abs_hp_input"]].round(3))
print(t5.pivot_table(index=["subset", "d320_km"], columns="quantity", values="rms")[["err_model", "err_floor"]].round(3))
t6 = pd.read_csv(f"{T}/T6_tails.csv")
print("== T6 tails: mean over cases of member-mean quantile, model - truth_pf")
q = ["q0.01", "q0.1", "q1", "q99", "q99.9", "q99.99"]
m = t6[t6.source == "model"].set_index(["date", "lead", "region", "field"])[q]
tr = t6[t6.source == "truth_pf"].set_index(["date", "lead", "region", "field"])[q]
dd = (m - tr).reset_index()
g = dd.groupby(["field", "region"])[q]
print(g.mean().round(2)); print("dates with model<truth (q0.01, q99.99 sign):")
print(dd.groupby(["field", "region"]).apply(lambda x: pd.Series({"n": len(x), "q0.01_neg": int((x["q0.01"] < 0).sum()), "q99.99_neg": int((x["q99.99"] < 0).sum())}), include_groups=False))
t7 = pd.read_csv(f"{T}/T7_large_scale.csv")
print("== T7"); print(t7.pivot_table(index=["var", "subset", "lead"], columns="source", values="lowpass_rms").round(3))
t8 = pd.read_csv(f"{T}/T8_spread.csv")
p = t8.groupby(["var", "zhr_band", "source"]).spread.mean().unstack()
p["ratio_model_truth"] = p.model / p.truth_pf
print("== T8"); print(p.round(3))
