#!/usr/bin/env python3
"""o2560 box read: one CSV and one markdown table per domain of the per-row means over dates, per lead and variable, for
A100, B24, B50 and IN, from pairs/L<lead>/per_date.csv; plus the paired differences A100-B24, A100-B50, B50-B24 with SE.
Usage: python summarize.py <out root> "<leads>". Writes summary/summary_scores.csv, summary/summary_<domain>.md."""
import csv
import math
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1]); leads = [int(x) for x in sys.argv[2].split()]
S = root / "summary"; S.mkdir(exist_ok=True)
recs = []
for lead in leads:
    f = root / "pairs" / f"L{lead:03d}" / "per_date.csv"
    if not f.exists():
        print("MISSING", f); continue
    for r in csv.DictReader(open(f)):
        r["lead"] = lead; r["value"] = float(r["value"]); recs.append(r)
by = {}
for r in recs:
    by.setdefault((r["lead"], r["domain"], r["var"], r["metric"], r["row"]), {})[r["date"]] = r["value"]
rows_out = []
for (lead, dn, v, mt, row), d in sorted(by.items()):
    x = np.array(list(d.values()))
    rows_out.append(dict(lead=lead, domain=dn, var=v, metric=mt, row=row, n_dates=len(x), mean=x.mean(),
                         se=x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else float("nan")))
with open(S / "summary_scores.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows_out[0])); w.writeheader(); w.writerows(rows_out)
for dn in ("box", "europe", "alps"):
    L = [f"# o2560 box read summary, domain {dn}: mean over dates (n dates per cell in brackets on the first column)", "",
         "| lead | var | metric | A100 | B24 | B50 | IN | A100-B24 (SE) | A100-B50 (SE) | B50-B24 (SE) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for lead in leads:
        for v in ("10ff", "tp", "10u", "10v", "2t", "2d", "msl", "skt", "tcw", "cp", "z_500", "t_850"):
            for mt in ("fcrps", "rmse", "spread", "spread_rmse_corr"):
                cell = {row: by.get((lead, dn, v, mt, row)) for row in ("A100", "B24", "B50", "IN")}
                if not cell["A100"]:
                    continue
                def pd(a, b):
                    if not cell[a] or not cell[b]:
                        return "-"
                    ks = sorted(set(cell[a]) & set(cell[b])); dd = np.array([cell[a][k] - cell[b][k] for k in ks])
                    return f"{dd.mean():+.4g} ({dd.std(ddof=1) / math.sqrt(len(dd)):.2g})"
                vals = [f"{np.mean(list(cell[r].values())):.4g}" if cell[r] else "-" for r in ("A100", "B24", "B50", "IN")]
                L.append(f"| +{lead} (n={len(cell['A100'])}) | {v} | {mt} | " + " | ".join(vals) + f" | {pd('A100', 'B24')} | {pd('A100', 'B50')} | {pd('B50', 'B24')} |")
    (S / f"summary_{dn}.md").write_text("\n".join(L) + "\n")
print("SUMMARY_DONE", len(rows_out), "rows", S)
