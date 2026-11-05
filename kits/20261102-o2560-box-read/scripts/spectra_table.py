#!/usr/bin/env python3
"""o2560 box read: one per-bin table per variable from the box_spectra bins.csv of several rows at one lead (pooled scope).
Columns per row: residual amplitude ratio (pred - input over truth - input) and full-field ratio (pred over truth); plus the
input's full-field ratio (input over truth). Amplitude ratios, per bin, no band average (rule 14). The o2560 truth holds
content down to about 15.6 km; bins below are model against model only and are marked. box_spectra takes one dx, at the
middle latitude of the region (40 N for the whole box, 52.5 N for Europe), so the wavelength labels are approximate over
a region that spans many degrees of latitude; the ratios between rows, the truth and the input stay valid (one operator
for every field). Writes spectra_<region>.md and spectra_<region>.csv.
Usage: python spectra_table.py <lead> <out dir> <region label> <label>=<bins.csv> [...]"""
import csv
import sys
from pathlib import Path

lead = int(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True); region = sys.argv[3]
rows = dict(a.split("=", 1) for a in sys.argv[4:])
data = {lab: [r for r in csv.DictReader(open(p)) if r["scope"] == "all"] for lab, p in rows.items()}
labs = list(rows)
variables = [v for v in ("ws", "tp", "10u", "10v", "2t", "msl") if any(r["variable"] == v for r in data[labs[0]])]
L = [f"# Per-bin amplitude ratios at +{lead} h, o2560 read, region {region} (pooled over dates x members; draws per row in brackets)", "",
     "resid = (pred - input) / (truth - input); full = pred / truth; IN full = input / truth. Target 1. Rows below 15.6 km: the "
     "truth's content ends, model against model only. Wavelengths use one dx at the region's middle latitude: approximate "
     "across latitudes; the ratios are valid.", ""]
flat = []
for v in variables:
    sub = {lab: [r for r in data[lab] if r["variable"] == v] for lab in labs}
    n = {lab: sub[lab][0]["n_draws"] if sub[lab] else "0" for lab in labs}
    L += [f"## {v} (" + ", ".join(f"{lab} {n[lab]} draws" for lab in labs) + ")", "",
          "| km | " + " | ".join(f"{lab} resid | {lab} full" for lab in labs) + " | IN full |", "|---|" + "---|---|" * len(labs) + "---|"]
    for i, r0 in enumerate(sub[labs[0]]):
        km = float(r0["bin_centre_km"])
        cells = []
        for lab in labs:
            r = sub[lab][i]
            assert abs(float(r["bin_centre_km"]) - km) < 1e-6 * km, "bins differ between rows"
            cells += [f"{float(r['ratio_resid']):.3f}", f"{float(r['ratio_full']):.3f}"]
            flat.append(dict(lead=lead, region=region, variable=v, row=lab, bin_centre_km=km, ratio_resid=r["ratio_resid"], ratio_full=r["ratio_full"],
                             ratio_interp_to_truth=r["ratio_interp_to_truth"], coherence=r["coherence"], n_draws=r["n_draws"]))
        mark = " (model only)" if km < 15.6 else ""
        L.append(f"| {km:.1f}{mark} | " + " | ".join(cells) + f" | {float(r0['ratio_interp_to_truth']):.3f} |")
    L.append("")
(out / f"spectra_{region}.md").write_text("\n".join(L) + "\n")
with open(out / f"spectra_{region}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
print("SPECTRA_TABLE_DONE", lead, labs, len(variables))
