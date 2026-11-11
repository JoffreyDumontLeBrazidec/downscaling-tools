"""box_spectra: per-bin amplitude ratios of a box, full field AND residual AND input (rule 14).

For every prediction file (date, lead) and every member, each variable is sampled on the same
regular grid (core/spectrum.py) and three fields are compared with the truth, bin by bin:

    full field   y_pred                 over  y                (``ratio_full``)
    residual     y_pred - x_interp      over  y - x_interp     (``ratio_resid``)
    input        x                      over  y                (``ratio_interp_to_truth``)

The residual of member k is taken against member k's own interpolated input (``x_interp_members[k]``)
when the file carries it, else against ``x_interp`` (run_meta ``residual_reference``). The input row is
the native input ``x`` when the file has it (rule 41), sampled on a regular grid at its own point spacing
and put on the same wavelength bins (core/spectrum.py ``InputSpectrum``); it is reported only in bins of
at least 3 native grid lengths and NaN below (run_meta ``input_spectrum``). Without a native x it is
``x_interp`` on the fine grid, with a warning.

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
  margin_deg (1.0);
- target_note: the sentence naming the bins where the ratio should be 1, written into bins.md
  and the judge's read (default: the o320 -> o1280 band, TARGET_NOTE). Another lane sets its own.
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
from eval.shared.edge_strip import inner_bounds, resolve_box_edge_deg
from eval.shared import regions
from eval.shared.native_input import native_spacing_km

from .core.spectrum import NBINS, SPEC_EDGE_MARGIN_DEG, SPEC_RES_DEG, InputSpectrum, Sampler, Spectrum

LOG = logging.getLogger(__name__)

DEFAULT_VARIABLES = ["10u", "10v", "ws", "2t", "msl"]
TARGET_NOTE = "target 1 from 125 km to 31 km; below 31 km the truth is empty (model against model only)"
COMPOSITES = {"ws": ("10u", "10v")}
BIN_COLS = ["scope", "variable", "bin_lo_km", "bin_hi_km", "bin_centre_km",
            "amp_resid_pred", "amp_resid_true", "ratio_resid",
            "amp_pred_full", "amp_truth_full", "ratio_full",
            "amp_interp", "ratio_interp_to_truth", "coherence", "aligned", "n_draws"]
# accumulated powers per (scope, variable)
TERMS = ("pred", "truth", "interp", "resid_pred", "resid_true", "pred_x_truth")
INPUT_MIN_GRID_LENGTHS = 3.0   # the native input spectrum is reported at wavelengths of at least 3 grid lengths
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
        # the native input (rule 41): x on lat_lres/lon_lres, when the file carries it
        self.native = "x" in self.ds.variables and "lat_lres" in self.ds.variables
        if self.native:
            self.lat_in = np.asarray(self.ds["lat_lres"][:], dtype=np.float64).reshape(-1)
            self.lon_in = np.asarray(self.ds["lon_lres"][:], dtype=np.float64).reshape(-1)
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


def accumulate(files, variables, members, method, res_deg, margin_deg, box_spec, clip_nonnegative=(),
               box_edge_deg: float | None = None):
    """Sum of the powers per (scope, variable) over draws; returns (spectrum, sums, counts, meta)."""
    sums: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    counts: dict[tuple[str, str], int] = {}
    smp = spec = smp_in = spec_in = None
    meta: dict = {}
    input_sources: set[str] = set()
    resid_refs: set[str] = set()
    for pf in files:
        f = _File(pf.path)
        try:
            if smp is None:
                box, how = resolve_box(box_spec, f.lat, f.lon)
                la0, la1, lo0, lo1 = box
                extent = (la0 + margin_deg, la1 - margin_deg, lo0 + margin_deg,
                          lo0 + ((lo1 - lo0) % 360.0 or 360.0) - margin_deg)
                ib = inner_bounds(f.lat, f.lon, box_edge_deg)
                if ib is not None:
                    # regional file: the sampled extent never reaches into the box-edge strip (edge_strip),
                    # whatever box was asked for (the default file-bounds box with margin 1 is unchanged)
                    ilo = ib[2] + 360.0 * round((lo0 - ib[2]) / 360.0)
                    extent = (max(extent[0], ib[0]), min(extent[1], ib[1]),
                              max(extent[2], ilo), min(extent[3], ilo + (ib[3] - ib[2])))
                    if extent[0] >= extent[1] or extent[2] >= extent[3]:
                        raise ValueError(f"box_spectra: box {box} lies inside the {box_edge_deg:g}-degree "
                                         "box-edge strip of the regional file")
                lon_c = _contiguous(f.lon, lo0)
                smp = Sampler(f.lat, lon_c, extent, res_deg, method=method)
                dx, dy = smp.dx_dy_km()
                spec = Spectrum(len(smp.gy), len(smp.gx), dx, dy, nbins=NBINS)
                grid_lat, grid_lon = f.lat, f.lon
                meta = {"box": box, "box_chosen_by": how, "extent_sampled": list(extent),
                        "grid": [spec.ny, spec.nx], "res_deg": res_deg, "method": method,
                        "dx_km": dx, "dy_km": dy, "dx_at_lat": smp.lat0,
                        "n_bins": int(spec.wavelength_km.size), "source_pad_deg": smp.pad_deg}
                if la1 - la0 > 35.0:
                    meta["note"] = (f"the box spans {la1 - la0:.0f} degrees of latitude and dx is taken at "
                                    f"{smp.lat0:.0f} N, so the wavelength axis is approximate; model/truth "
                                    "ratios stay fair (one operator for every field)")
                LOG.info("box_spectra: %s box %s, grid %d x %d", how, box, spec.ny, spec.nx)
            if f.native and smp_in is None:
                # the input on a regular grid at ITS OWN spacing, nearest point, on the same bins (rule 41)
                # steps of about the native spacing, chosen so that the grid spans exactly the fine grid's
                # length (n_in x step_in = n x step): both FFTs then share their fundamental and low wavenumbers
                km_in = native_spacing_km(f.lat_in, f.lon_in)
                ny_in = max(4, round(spec.ny * meta["dy_km"] / km_in))
                nx_in = max(4, round(spec.nx * meta["dx_km"] / km_in))
                smp_in = Sampler(f.lat_in, _contiguous(f.lon_in, meta["box"][2]), tuple(meta["extent_sampled"]),
                                 spec.ny * res_deg / ny_in, method="nearest", res_lon=spec.nx * res_deg / nx_in)
                dxi, dyi = smp_in.dx_dy_km()
                spec_in = InputSpectrum(spec, len(smp_in.gy), len(smp_in.gx), dxi, dyi,
                                        min_wavelength_km=INPUT_MIN_GRID_LENGTHS * km_in,
                                        ref_dx_km=meta["dx_km"], ref_dy_km=meta["dy_km"])
                meta["input_spectrum"] = {
                    "grid": "regular grid at the native spacing, nearest native point", "native_spacing_km": km_in,
                    "dx_km": dxi, "dy_km": dyi, "shape": [len(smp_in.gy), len(smp_in.gx)],
                    "min_wavelength_km": spec_in.min_wavelength_km,
                    "note": f"input/truth NaN below {INPUT_MIN_GRID_LENGTHS:g} native grid lengths"}
            elif f.lat.shape != grid_lat.shape or np.abs(f.lat - grid_lat).max() > 1e-6 \
                    or np.abs(f.lon - grid_lon).max() > 1e-6:
                raise ValueError(f"{pf.path}: point set differs from the first file's")
            idx = smp.src_idx
            mems = [m for m in f.members if not members or m in members]
            # the residual of member k against its own interpolated input when the file carries one per member
            ref_name = "x_interp_members" if "x_interp_members" in f.ds.variables else "x_interp"
            resid_refs.add(ref_name)
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
                    y, xi, yp = grid("y"), grid(ref_name), grid("y_pred")
                    Fy, Fx, Fp = spec.fft(y), spec.fft(xi), spec.fft(yp)
                    # input spectrum: from the native input when the file has it (rule 41); the residual
                    # terms stay defined against x_interp (y - x_interp is what the model learns)
                    if f.native and smp_in is not None:
                        full_in = np.zeros(f.lat_in.size)
                        full_in[smp_in.src_idx] = f.field("x", m, var, smp_in.src_idx)
                        if var in clip_nonnegative:
                            full_in = np.maximum(full_in, 0.0)
                        p_in = spec_in.power(spec_in.fft(smp_in(full_in)))
                        input_sources.add("x (native input grid)")
                    else:
                        p_in = spec.power(Fx)
                        input_sources.add(f"{ref_name} (input interpolated to the output grid)")
                    terms = {"pred": spec.power(Fp), "truth": spec.power(Fy),
                             "interp": p_in, "resid_pred": spec.power(Fp - Fx),
                             "resid_true": spec.power(Fy - Fx), "pred_x_truth": spec.cross(Fp, Fy)}
                    for scope in ("all", pf.date):
                        acc = sums.setdefault((scope, var), {t: 0.0 for t in TERMS})
                        for t in TERMS:
                            acc[t] = acc[t] + terms[t]
                        counts[(scope, var)] = counts.get((scope, var), 0) + 1
            LOG.info("box_spectra: %s done (%d members)", Path(pf.path).name, len(mems))
        finally:
            f.close()
    meta["input_source"] = sorted(input_sources)
    meta["residual_reference"] = sorted(resid_refs)
    if any("interpolated" in s_ for s_ in input_sources):
        LOG.warning("box_spectra: input spectrum from x_interp (no native x in the files); rule 41: the input/truth "
                    "ratio is then a smoothed-input diagnostic, not the input's spectrum")
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
             f"{meta.get('target_note', TARGET_NOTE)}.", ""]
    if meta.get("input_source"):
        lines += [f"Input spectrum from: {'; '.join(meta['input_source'])} (rule 41: the input is never "
                  "interpolated to the output grid before scoring).", ""]
    if meta.get("input_spectrum"):
        i = meta["input_spectrum"]
        lines += [f"input/truth: the native input on a regular grid at its own spacing ({i['native_spacing_km']:.0f} km, "
                  f"nearest native point), on the same bins; blank below {i['min_wavelength_km']:.0f} km "
                  "(3 native grid lengths).", ""]
    if meta.get("residual_reference"):
        lines += [f"Residuals against: {'; '.join(meta['residual_reference'])} (member k against its own input "
                  "member when the file carries x_interp_members).", ""]
    if meta.get("note"):
        lines += [f"Note: {meta['note']}.", ""]
    for var in sorted({r["variable"] for r in rows if r["scope"] == "all"}):
        sub = [r for r in rows if r["scope"] == "all" and r["variable"] == var]
        lines += [f"## {var} (draws: {sub[0]['n_draws']})", "",
                  "| km | residual | full field | input/truth | coherence | aligned |",
                  "|---|---|---|---|---|---|"]
        for r in sub:
            ri = float(r['ratio_interp_to_truth'])
            lines.append(f"| {float(r['bin_centre_km']):.1f} | {float(r['ratio_resid']):.3f} | "
                         f"{float(r['ratio_full']):.3f} | {'' if np.isnan(ri) else f'{ri:.3f}'} | "
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
                                          eval_config.get("box"), clip_nonnegative=clip,
                                          box_edge_deg=resolve_box_edge_deg(eval_config))
    meta["clip_nonnegative"] = list(clip)
    meta["target_note"] = str(eval_config.get("target_note") or TARGET_NOTE)
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
