import csv, glob, json, math, os, re
from collections import defaultdict
O = "/e/data1/jureap-data/ecmwf/users/jdumont/outputs/o2560_box_read_20261102"
LEADS = [12, 24, 36, 48, 60, 72]
ROWS = ["A100", "B24", "B50"]
# 1 spectra: selected bins, ws and tp, box and europe
TARGET = [125, 62, 31, 20, 15.6, 10]
for reg in ("box", "europe"):
    for L in (24, 72):
        rs = list(csv.DictReader(open(f"{O}/pairs/L{L:03d}/spectra_{reg}.csv")))
        for v in ("ws", "tp", "2t", "msl"):
            sub = [r for r in rs if r["variable"] == v]
            kms = sorted({float(r["bin_centre_km"]) for r in sub}, reverse=True)
            pick = []
            for t in TARGET:
                k = min(kms, key=lambda x: abs(math.log(x / t)))
                if k not in pick: pick.append(k)
            n = sub[0]["n_draws"]
            print(f"SPEC {reg} +{L} {v} (draws/row {n}) km: " + " | ".join(
                f"{k:.1f}: " + " ".join(f"{row} r{float(next(r for r in sub if r['row']==row and float(r['bin_centre_km'])==k)['ratio_resid']):.2f}/f{float(next(r for r in sub if r['row']==row and float(r['bin_centre_km'])==k)['ratio_full']):.2f}" for row in ROWS)
                + f" IN f{float(next(r for r in sub if r['row']=='A100' and float(r['bin_centre_km'])==k)['ratio_interp_to_truth']):.2f}" for k in pick))
# 2 tp tail per lead
for L in LEADS:
    per = defaultdict(dict)
    for r in csv.DictReader(open(f"{O}/pairs/L{L:03d}/per_date.csv")):
        if r["var"] == "tp" and r["domain"] == "box" and r["metric"] in ("p99.9_pooled", "p99.99_pooled", "max_member_mean"):
            per[(r["row"], r["metric"])][r["date"]] = float(r["value"])
    n = len(per[("truth", "max_member_mean")])
    line = f"TAIL +{L} (n={n} dates, mm/6h):"
    for m in ("p99.9_pooled", "p99.99_pooled", "max_member_mean"):
        line += f" {m}: " + " ".join(f"{row} {sum(per[(row, m)].values())/n:.1f}" for row in ROWS + ["IN", "truth"]) + ";"
    print(line)
# 3 scores box + europe: fcrps rmse for 10ff tp 2t msl; diffs with SE; ratio
for dom in ("box", "europe"):
    for L in LEADS:
        by = defaultdict(dict)
        for r in csv.DictReader(open(f"{O}/pairs/L{L:03d}/per_date.csv")):
            if r["domain"] == dom:
                by[(r["row"], r["var"], r["metric"])][r["date"]] = float(r["value"])
        rat = {(r["kind"], r["row"], r["var"]): r for r in csv.DictReader(open(f"{O}/pairs/L{L:03d}/ratio.csv")) if r["domain"] == dom}
        for v in ("10ff", "tp", "2t", "msl"):
            out = []
            for m in ("fcrps", "rmse"):
                vals = {row: by[(row, v, m)] for row in ROWS + ["IN"]}
                ds = sorted(vals["A100"]); n = len(ds)
                mean = {row: sum(vals[row][d] for d in ds) / n for row in vals}
                def diff(a, b):
                    x = [vals[a][d] - vals[b][d] for d in ds]; mu = sum(x) / n
                    se = math.sqrt(sum((y - mu) ** 2 for y in x) / (n - 1) / n)
                    return f"{100*mu/mean[b]:+.1f}%±{100*se/mean[b]:.1f} ({sum(y<0 for y in x)}/{n})"
                out.append(f"{m} A100 {mean['A100']:.4g} B24 {mean['B24']:.4g} B50 {mean['B50']:.4g} IN {mean['IN']:.4g}; A-B24 {diff('A100','B24')} A-B50 {diff('A100','B50')} B50-B24 {diff('B50','B24')} A-IN {diff('A100','IN')}")
            rr = " ".join(f"{row} {float(rat[('row',row,v)]['ratio']):.2f}[{float(rat[('row',row,v)]['lo95']):.2f},{float(rat[('row',row,v)]['hi95']):.2f}]" for row in ROWS)
            print(f"SCORE {dom} +{L} {v} (n={n}): " + " || ".join(out) + f" || ratio {rr}")
# 4 judge verdicts
for f in sorted(glob.glob(f"{O}/judge/*_L0*")):
    if os.path.isdir(f):
        md = sorted(glob.glob(f"{f}/*.md"))
        print("JUDGE", os.path.basename(f), [os.path.basename(x) for x in md][:5])
        for x in md[:1]:
            for ln in open(x):
                if re.search(r"verdict|better|worse|undecided|label|VERDICT", ln, re.I):
                    print("   ", ln.strip()[:220])
print("MAPS", sorted(glob.glob(f"{O}/maps/L0*/*.png")))
for L in (24, 72):
    print("CASES", L, open(f"{O}/maps/L{L:03d}/cases.json").read().replace("\n", " "))
