#!/usr/bin/env python3
"""o2560 box read: summary tables per lead and variable for A100, B24, B50 and IN, from pairs/L<lead>/per_date.csv (fair
CRPS, RMSE, spread: means over dates; paired differences with SE over dates) and pairs/L<lead>/ratio.csv (corrected
spread/RMSE pooled over dates with its 95 % date-bootstrap interval). Stops if the rows of a lead hold different dates.
Usage: python summarize.py <out root> "<leads>". Writes summary/summary_scores.csv, summary/summary_<domain>.md."""
import csv
import math
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1]); leads = [int(x) for x in sys.argv[2].split()]
S = root / "summary"; S.mkdir(exist_ok=True)
by = {}; ratio = {}; done = []
for lead in leads:
    f = root / "pairs" / f"L{lead:03d}" / "per_date.csv"
    if not f.exists():
        print("MISSING", f); continue
    for r in csv.DictReader(open(f)):
        by.setdefault((lead, r["domain"], r["var"], r["metric"], r["row"]), {})[r["date"]] = float(r["value"])
    for r in csv.DictReader(open(root / "pairs" / f"L{lead:03d}" / "ratio.csv")):
        ratio[(lead, r["kind"], r["row"], r["domain"], r["var"])] = r
    ds = {row: tuple(sorted(by[(lead, "box", "2t", "fcrps", row)])) for row in ("A100", "B24", "B50", "IN") if (lead, "box", "2t", "fcrps", row) in by}
    if len(set(ds.values())) != 1:
        sys.exit(f"FATAL: lead {lead}: rows hold different dates {ds}")
    done.append(lead)
rows_out = []
for (lead, dn, v, mt, row), d in sorted(by.items()):
    x = np.array(list(d.values()))
    rows_out.append(dict(lead=lead, domain=dn, var=v, metric=mt, row=row, n_dates=len(x), mean=x.mean(),
                         se=x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else float("nan"), lo95="", hi95=""))
for (lead, kind, row, dn, v), r in sorted(ratio.items()):
    rows_out.append(dict(lead=lead, domain=dn, var=v, metric="ratio_corr" if kind == "row" else "ratio_corr_diff", row=row,
                         n_dates=r["n_dates"], mean=r["ratio"], se="", lo95=r["lo95"], hi95=r["hi95"]))
with open(S / "summary_scores.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows_out[0])); w.writeheader(); w.writerows(rows_out)
VARS = ("10ff", "tp", "10u", "10v", "2t", "2d", "msl", "skt", "tcw", "cp", "z_500", "t_850")
for dn in ("box", "europe", "alps"):
    L = [f"# o2560 box read summary, domain {dn} (1-degree edge strip excluded): means over dates, n dates in the first column", "",
         "| lead | var | metric | A100 | B24 | B50 | IN | A100-B24 (SE) | A100-B50 (SE) | B50-B24 (SE) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for lead in done:
        for v in VARS:
            for mt in ("fcrps", "rmse", "spread"):
                cell = {row: by.get((lead, dn, v, mt, row)) for row in ("A100", "B24", "B50", "IN")}
                if not cell["A100"]:
                    continue
                def pd(a, b):
                    ks = sorted(cell[a]); dd = np.array([cell[a][k] - cell[b][k] for k in ks])
                    return f"{dd.mean():+.4g} ({dd.std(ddof=1) / math.sqrt(len(dd)):.2g})"
                vals = [f"{np.mean(list(cell[r].values())):.4g}" for r in ("A100", "B24", "B50", "IN")]
                L.append(f"| +{lead} (n={len(cell['A100'])}) | {v} | {mt} | " + " | ".join(vals) + f" | {pd('A100', 'B24')} | {pd('A100', 'B50')} | {pd('B50', 'B24')} |")
            rc = lambda kind, row: ratio.get((lead, kind, row, dn, v))
            if rc("row", "A100"):
                f3 = lambda r: f"{float(r['ratio']):.3f} [{float(r['lo95']):.3f}, {float(r['hi95']):.3f}]" if r else "-"
                L.append(f"| +{lead} | {v} | ratio | {f3(rc('row', 'A100'))} | {f3(rc('row', 'B24'))} | {f3(rc('row', 'B50'))} | - | "
                         f"{f3(rc('pair', 'A100-B24'))} | {f3(rc('pair', 'A100-B50'))} | {f3(rc('pair', 'B50-B24'))} |")
    (S / f"summary_{dn}.md").write_text("\n".join(L) + "\n")
print("SUMMARY_DONE", len(rows_out), "rows", done, S)
