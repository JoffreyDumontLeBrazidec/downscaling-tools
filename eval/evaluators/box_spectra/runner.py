"""box_spectra: per-bin amplitude ratios of a box, full field AND residual AND input (rule 14).

For every prediction file (date, lead) and every member, each variable is sampled on the same
regular grid (core/spectrum.py) and three fields are compared with the truth, bin by bin:

    full field   y_pred                 over  y                (``ratio_full``)
    residual     y_pred - x_interp      over  y - x_interp     (``ratio_resid``)
    input        x_interp               over  y                (``ratio_interp_to_truth``)

plus the coherence of the prediction with the truth, cross / sqrt(power x power), and the
aligned part = coherence x full-field amplitude ratio (judge a gain on the aligned part, never
on the coherence share alone). Powers are averaged over the draws (files x members) before the
square root, as in the docs script. Rows are written pooled (``scope`` = all) and per date, so
a gain can be checked case by case. No band average and no "distance" is computed (rule 14).

eval_config keys (all optional):

- box: [lat_min, lat_max, lon_min, lon_max] or a name of eval/shared/regions.py (e2,
  north_atlantic, tc_atlantic, alps). Default: the file's own bounds for a regional file,
  the E2 box 10-70 N, 80 W-40 E for a global file (the protocol cuts global predictions to E2);
- steps (default [24]), dates, members (labels, default all), variables
  (default 10u, 10v, ws, 2t, msl; those absent from the files are skipped);
- method (linear; nearest only to reproduce old nearest-neighbour reads), res_deg (0.07),
  margin_deg (1.0).
"""
from __future__ import annotations

import csv
import json
import logging
import subprocess
import time
from pathlib import Path

import numpy as np

from eval.discovery.predictions import find_predictions
from eval.shared import regions

from .core.spectrum import NBINS, SPEC_EDGE_MARGIN_DEG, SPEC_RES_DEG, Sampler, Spectrum

LOG = logging.getLogger(__name__)

DEFAULT_VARIABLES = ["10u", "10v", "ws", "2t", "msl"]
COMPOSITES = {"ws": ("10u", "10v")}
BIN_COLS = ["scope", "variable", "bin_lo_km", "bin_hi_km", "bin_centre_km",
            "amp_resid_pred", "amp_resid_true", "ratio_resid",
            "amp_pred_full", "amp_truth_full", "ratio_full",
            "amp_interp", "ratio_interp_to_truth", "coherence", "aligned", "n_draws"]
# accumulated powers per (scope, variable)
TERMS = ("pred", "truth", "interp", "resid_pred", "resid_true", "pred_x_truth")
NAMED_BOXES = {"e2": regions.E2, "north_atlantic": regions.NORTH_ATLANTIC,
               "tc_atlantic": regions.TC_ATLANTIC, "alps": regions.ALPS}


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _decode_states(var) -> list[str]:
    out = []
    for v in np.asarray(var[:]).reshape(-1):
        if isinstance(v, bytes):
            out.append(v.decode())
        elif isinstance(v, np.ndarray) and v.dtype.kind == "S":
            out.append(b"".join(x for x in v.tolist() if x).decode())
        else:
            out.append(str(v))
    return out


class _File:
    """One prediction file; reads one weather state of one member at a time, on the box only."""

    def __init__(self, path):
        import netCDF4

        self.ds = netCDF4.Dataset(str(path))
        self.ds.set_auto_mask(False)
        self.lat = np.asarray(self.ds["lat_hres"][:], dtype=np.float64).reshape(-1)
        self.lon = np.asarray(self.ds["lon_hres"][:], dtype=np.float64).reshape(-1)
        self.states = _decode_states(self.ds["weather_state"])
        n = len(self.ds.dimensions["ensemble_member"])
        if "ensemble_member" in self.ds.variables:
            self.members = [int(v) for v in np.asarray(self.ds["ensemble_member"][:]).reshape(-1)]
        else:
            self.members = list(range(1, n + 1))

    def column(self, name: str, member: int, state: str, idx: np.ndarray) -> np.ndarray:
        v = self.ds[name]
        key = []
        for d in v.dimensions:
            if d == "sample":
                key.append(0)
            elif d == "ensemble_member":
                key.append(self.members.index(member))
            elif d == "weather_state":
                key.append(self.states.index(state))
            else:
                key.append(slice(None))
        return np.asarray(v[tuple(key)], dtype=np.float64).reshape(-1)[idx]

    def field(self, name: str, member: int, var: str, idx: np.ndarray) -> np.ndarray:
        if var in COMPOSITES:
            u, w = (self.column(name, member, s, idx) for s in COMPOSITES[var])
            return np.hypot(u, w)
        return self.column(name, member, var, idx)

    def has(self, var: str) -> bool:
        return all(s in self.states for s in COMPOSITES.get(var, (var,)))

    def close(self):
        self.ds.close()


def resolve_box(spec, lat: np.ndarray, lon: np.ndarray) -> tuple[list[float], str]:
    """[lat_min, lat_max, lon_min, lon_max] of the read and how it was chosen."""
    if isinstance(spec, str) and spec in NAMED_BOXES:
        return list(NAMED_BOXES[spec]), spec
    if spec:
        if isinstance(spec, str):
            spec = spec.replace(",", " ").split()
        return [float(v) for v in spec], "given"
    lon_n = (lon + 180.0) % 360.0 - 180.0
    if float(lat.max() - lat.min()) > 100.0:    # a global file (E2 spans 60 degrees)
        return list(regions.E2), "e2 (global file)"
    return [float(lat.min()), float(lat.max()), float(lon_n.min()), float(lon_n.max())], "file bounds"


def _contiguous(lon: np.ndarray, west: float) -> np.ndarray:
    return west + np.mod(lon - west, 360.0)


def accumulate(files, variables, members, method, res_deg, margin_deg, box_spec, clip_nonnegative=()):
    """Sum of the powers per (scope, variable) over draws; returns (spectrum, sums, counts, meta)."""
    sums: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    counts: dict[tuple[str, str], int] = {}
    smp = spec = None
    meta: dict = {}
    for pf in files:
        f = _File(pf.path)
        try:
            if smp is None:
                box, how = resolve_box(box_spec, f.lat, f.lon)
                la0, la1, lo0, lo1 = box
                extent = (la0 + margin_deg, la1 - margin_deg, lo0 + margin_deg,
                          lo0 + ((lo1 - lo0) % 360.0 or 360.0) - margin_deg)
                lon_c = _contiguous(f.lon, lo0)
                smp = Sampler(f.lat, lon_c, extent, res_deg, method=method)
                dx, dy = smp.dx_dy_km()
                spec = Spectrum(len(smp.gy), len(smp.gx), dx, dy, nbins=NBINS)
                grid_lat, grid_lon = f.lat, f.lon
                meta = {"box": box, "box_chosen_by": how, "extent_sampled": list(extent),
                        "grid": [spec.ny, spec.nx], "res_deg": res_deg, "method": method,
                        "dx_km": dx, "dy_km": dy, "dx_at_lat": smp.lat0,
                        "n_bins": int(spec.wavelength_km.size)}
                if la1 - la0 > 35.0:
                    meta["note"] = (f"the box spans {la1 - la0:.0f} degrees of latitude and dx is taken at "
                                    f"{smp.lat0:.0f} N, so the wavelength axis is approximate; model/truth "
                                    "ratios stay fair (one operator for every field)")
                LOG.info("box_spectra: %s box %s, grid %d x %d", how, box, spec.ny, spec.nx)
            elif f.lat.shape != grid_lat.shape or np.abs(f.lat - grid_lat).max() > 1e-6 \
                    or np.abs(f.lon - grid_lon).max() > 1e-6:
                raise ValueError(f"{pf.path}: point set differs from the first file's")
            idx = smp.src_idx
            mems = [m for m in f.members if not members or m in members]
            for var in variables:
                if not f.has(var):
                    continue
                for m in mems:
                    # the sampler indexes global point numbers; give it a full-length array
                    def grid(name, _m=m, _v=var):
                        full = np.zeros(f.lat.size)
                        full[idx] = f.field(name, _m, _v, idx)
                        if _v in clip_nonnegative:   # rule 7: negative precipitation clipped
                            full[idx] = np.maximum(full[idx], 0.0)
                        return smp(full)
                    y, xi, yp = grid("y"), grid("x_interp"), grid("y_pred")
                    Fy, Fx, Fp = spec.fft(y), spec.fft(xi), spec.fft(yp)
                    terms = {"pred": spec.power(Fp), "truth": spec.power(Fy),
                             "interp": spec.power(Fx), "resid_pred": spec.power(Fp - Fx),
                             "resid_true": spec.power(Fy - Fx), "pred_x_truth": spec.cross(Fp, Fy)}
                    for scope in ("all", pf.date):
                        acc = sums.setdefault((scope, var), {t: 0.0 for t in TERMS})
                        for t in TERMS:
                            acc[t] = acc[t] + terms[t]
                        counts[(scope, var)] = counts.get((scope, var), 0) + 1
            LOG.info("box_spectra: %s done (%d members)", Path(pf.path).name, len(mems))
        finally:
            f.close()
    return spec, sums, counts, meta


def bin_rows(spec, sums, counts) -> list[dict]:
    rows = []
    wl = spec.wavelength_km
    for (scope, var), acc in sorted(sums.items(), key=lambda kv: (kv[0][0] != "all", kv[0])):
        n = counts[(scope, var)]
        p = {t: np.asarray(acc[t]) / n for t in TERMS}
        amp = {t: np.sqrt(p[t]) for t in TERMS if t != "pred_x_truth"}
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_full = amp["pred"] / amp["truth"]
            coherence = p["pred_x_truth"] / np.sqrt(p["pred"] * p["truth"])
            cols = {"ratio_resid": amp["resid_pred"] / amp["resid_true"], "ratio_full": ratio_full,
                    "ratio_interp_to_truth": amp["interp"] / amp["truth"],
                    "coherence": coherence, "aligned": coherence * ratio_full}
        for i in np.argsort(-wl):
            rows.append({"scope": scope, "variable": var, "bin_lo_km": spec.band_lo_km[i],
                         "bin_hi_km": spec.band_hi_km[i], "bin_centre_km": wl[i],
                         "amp_resid_pred": amp["resid_pred"][i], "amp_resid_true": amp["resid_true"][i],
                         "ratio_resid": cols["ratio_resid"][i], "amp_pred_full": amp["pred"][i],
                         "amp_truth_full": amp["truth"][i], "ratio_full": cols["ratio_full"][i],
                         "amp_interp": amp["interp"][i],
                         "ratio_interp_to_truth": cols["ratio_interp_to_truth"][i],
                         "coherence": cols["coherence"][i], "aligned": cols["aligned"][i],
                         "n_draws": n})
    return rows


def write_bins(rows, path) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=BIN_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def read_bins(path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def write_table(rows, path, label: str, meta: dict) -> None:
    """The per-bin table of the pooled rows, one block per variable (no band average)."""
    lines = [f"# Per-bin amplitude ratios, {label}", "",
             f"Box {meta.get('box')} ({meta.get('box_chosen_by')}), {meta.get('method')} sampling at "
             f"{meta.get('res_deg')} deg, grid {meta.get('grid')}. AMPLITUDE ratios (not power), per bin; "
             "target 1 from 125 km to 31 km; below 31 km the truth is empty (model against model only).", ""]
    if meta.get("note"):
        lines += [f"Note: {meta['note']}.", ""]
    for var in sorted({r["variable"] for r in rows if r["scope"] == "all"}):
        sub = [r for r in rows if r["scope"] == "all" and r["variable"] == var]
        lines += [f"## {var} (draws: {sub[0]['n_draws']})", "",
                  "| km | residual | full field | input/truth | coherence | aligned |",
                  "|---|---|---|---|---|---|"]
        for r in sub:
            lines.append(f"| {float(r['bin_centre_km']):.1f} | {float(r['ratio_resid']):.3f} | "
                         f"{float(r['ratio_full']):.3f} | {float(r['ratio_interp_to_truth']):.3f} | "
                         f"{float(r['coherence']):.3f} | {float(r['aligned']):.3f} |")
        lines.append("")
    Path(path).write_text("\n".join(lines))


def run(predictions_dir, lane_config: dict, eval_config: dict, *, output_dir=None,
        overwrite: bool = False, run_label: str = "", **kwargs) -> Path:
    t0 = time.time()
    predictions_dir = Path(predictions_dir).expanduser()
    output_dir = Path(output_dir) if output_dir else predictions_dir / "evaluators" / "box_spectra"
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = [int(s) for s in (eval_config.get("steps") or [24])]
    dates = {str(d) for d in (eval_config.get("dates") or [])}
    members = [int(m) for m in (eval_config.get("members") or [])]
    variables = list(eval_config.get("variables") or DEFAULT_VARIABLES)
    method = str(eval_config.get("method", "linear"))
    res_deg = float(eval_config.get("res_deg", SPEC_RES_DEG))
    margin_deg = float(eval_config.get("margin_deg", SPEC_EDGE_MARGIN_DEG))
    label = run_label or predictions_dir.name

    files = [p for p in find_predictions(predictions_dir)
             if p.step in steps and (not dates or p.date in dates)]
    if not files:
        raise FileNotFoundError(f"box_spectra: no prediction files for steps {steps} under {predictions_dir}")
    clip = tuple(eval_config.get("clip_nonnegative") or ())
    spec, sums, counts, meta = accumulate(files, variables, members, method, res_deg, margin_deg,
                                          eval_config.get("box"), clip_nonnegative=clip)
    meta["clip_nonnegative"] = list(clip)
    if not sums:
        raise ValueError(f"box_spectra: none of {variables} is in the files of {predictions_dir}")
    rows = bin_rows(spec, sums, counts)
    write_bins(rows, output_dir / "bins.csv")
    write_table(rows, output_dir / "bins.md", label, meta)
    meta.update({"evaluator": "box_spectra", "git_commit": _git_commit(), "label": label,
                 "predictions_dir": str(predictions_dir), "steps": steps,
                 "files": [str(p.path) for p in files], "members": members or "all",
                 "variables": sorted({v for _, v in sums}), "seconds": time.time() - t0})
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    return output_dir
