import sys, json, zipfile, numpy as np, pandas as pd
t2 = pd.read_csv(sys.argv[1])
d = t2[t2["var"] == "2t"].copy()
for col in ("slope_model", "slope_truth_pf_unpaired"):
    e = d[col] - d.slope_truth_ctrl
    inv = d.slope_truth_ctrl > -2; pos = d.slope_truth_ctrl > 0
    print(col, "all: mean|diff| %.3f reg %s" % (e.abs().mean(), np.polyfit(d.slope_truth_ctrl, d[col], 1).round(3)),
          "| truth>-2 n=%d dates=%d diff %.3f also>-2 %d" % (inv.sum(), d[inv].date.nunique(), e[inv].mean(), (d[col][inv] > -2).sum()),
          "| truth>0 n=%d dates=%d mean %.2f vs %.2f frac %.3f" % (pos.sum(), d[pos].date.nunique(), d[col][pos].mean(), d.slope_truth_ctrl[pos].mean(), d[col][pos].sum() / d.slope_truth_ctrl[pos].sum()))
    for reg in ("norway", "appalachians", "alps"):
        m = inv & (d.region == reg)
        print("   ", reg, "truth>-2 n=%d dates=%d diff %.3f" % (m.sum(), d[m].date.nunique(), e[m].mean()))
# model vs pf paired by case on the inversion subset
inv = d.slope_truth_ctrl > -2
dd = (d.slope_model - d.slope_truth_pf_unpaired)[inv]
print("model - truth_pf on truth>-2: mean %.3f, cases model<pf %d of %d, dates %d" % (dd.mean(), (dd < 0).sum(), len(dd), d[inv].date.nunique()))
g = d[inv].assign(x=dd).groupby("date").x.mean(); print("per-date mean model-pf: neg dates %d of %d" % ((g < 0).sum(), len(g)))
s = t2[t2["var"] == "sp"].copy(); s["x"] = s.slope_model - s.slope_truth_ctrl
print("sp slope model - truth ctrl (hPa/km) by region: mean, dates with mean>0 / n dates")
for reg, e in s.groupby("region"):
    g = e.groupby("date").x.mean(); print("   ", reg, round(e.x.mean(), 3), int((g > 0).sum()), len(g))
