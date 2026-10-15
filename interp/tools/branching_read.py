"""CPU read of the branching runs (U3, TD-221): features per final field, undecided shares, sigma*.

Input: the case folders written by ``interp.tools.branching`` (case.npz, finals/*.npy, meta.json).
Nothing here needs torch or a GPU.

Features of every final field (trunk, branches, independent full draws):
  storm     per storm event box present in the case (Franklin 15-38 N 78-58 W, Idalia 10-40 N 100-80 W),
            with the tc_structure evaluator's own functions (eval/evaluators/tc_structure/core.py):
            first guess = the TRUTH's closed low in the event box (minimum 50 km from the box edge, at
            most 1005 hPa, as the evaluator's storm_edge_km / storm_max_pmin_hpa), then per draw
            find_centre within 250 km of it and measure_structure:
              pmin_hpa (central pressure), maxwind300_ms (maximum 10 m wind within 300 km),
              rmw_km (radius of maximum azimuthal-mean tangential wind = the ring radius).
  spectra   the per-bin tool of the box instrument (plot_sampler_texture.py, --spectra-method linear:
            barycentric interpolation onto a 0.07 degree grid, mean removed, 2-D Hann window, FFT,
            azimuthal average in 40 log-spaced bins of which the populated ones are kept):
              amp_resid_<var>_<km>: amplitude of the RESIDUAL (prediction minus interpolated input)
              per bin, for 10u, 10v, 2t, msl; and per sub-band (amplitude = square root of the mean
              power of the bins whose centre lies in the sub-band): 31.3-40, 40-60, 60-100, 100-125.5,
              140-200 and 200-400 km. No band-averaged number over 32-125 or 140-400 km is formed.
              coh_2t_<km>: coherence of the 2 m temperature field with the truth per bin, cross power
              over the square root of the two powers (the definition of the spectra_coherence and
              box_spectra evaluators), and per sub-band (summed cross over summed powers).
  fronts    grad2t_p99: 99th percentile over the box grid of |grad 2t| in K per 100 km (central
            differences on the same 0.07 degree grid, one cell of border dropped). No front metric
            exists in eval; this is the simplest definition and is labelled as such.
  rain      tp_max_mm and tp_area50_km2 (area at or above 50 mm, O1280 nominal cell area), only when
            the checkpoint outputs tp; s397434 does not, so these rows are absent for it.

Shares: for feature F, case unit u (a case, or a case and storm for the storm features) and branch
node k: share_u(F, k) = var(F over the n_branch branches at k) / var(F over the n_full independent
draws) (ddof 1, draws with a missing value dropped, at least 4 values per set). Pooled over units:
sum_u var_branch / sum_u var_full; also the median of the per-unit shares. Bootstrap (draws resampled
with replacement inside each unit and set) gives the 10-90 % range of the pooled share and of sigma*.
sigma*(F): going down the ladder, the first node where the pooled share is below 0.5, interpolated
linearly in log sigma with the node above; "above" when already below 0.5 at the first node, "below"
when it never falls below 0.5. The same rule is also applied to the least-squares fit of the curve
that does not increase as sigma decreases (pool adjacent violators), reported beside it. With 8 + 8 draws the ratio of two variances of one unit has a 90 %
interval of about x0.27 to x3.8 (F distribution, 7 and 7 degrees of freedom): read single units with
that in mind; the pooled curves are the result.

Usage:
    python -m interp branching_read --root OUT --out OUT/read \
        --feature-box alps_20230226_024=43,49,4,16
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import logging
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np

_DT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_DT_ROOT) not in sys.path:
    sys.path.insert(0, str(_DT_ROOT))

LOG = logging.getLogger("interp.branching_read")

# The box instrument's spectra tool, copied verbatim from the section 7.2 work folder
# (agent-work/20261007-sec7-forward-reverse/reads/plot_sampler_texture.py).
PST_SHA256 = "6daaebf7bab2df2468c299657d9ebd695924894453fcf8b69a99f18919e78c4a"
STORM_EVENTS = {"franklin": (15.0, 38.0, -78.0, -58.0), "idalia": (10.0, 40.0, -100.0, -80.0)}
STORM_EDGE_KM = 50.0
STORM_MAX_PMIN_HPA = 1005.0
SPEC_VARS = ("10u", "10v", "2t", "msl")
SUB_BANDS = ((31.3, 40.0), (40.0, 60.0), (60.0, 100.0), (100.0, 125.5), (140.0, 200.0), (200.0, 400.0))
MIN_VALID = 4
SHARE_THRESHOLD = 0.5


def load_pst(path: Path, check_sha: bool = True):
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if check_sha and sha != PST_SHA256:
        raise SystemExit(f"{path}: sha256 {sha} is not the box instrument's {PST_SHA256}")
    spec = importlib.util.spec_from_file_location("plot_sampler_texture", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def norm_lon(lon):
    return ((np.asarray(lon, dtype=np.float64) + 180.0) % 360.0) - 180.0


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

class CaseFeatures:
    """Feature extractor for one case (fixed point set, truth and interpolated input)."""

    def __init__(self, case_dir: Path, pst, feature_box=None, spectra_method: str = "linear"):
        from eval.evaluators.tc_structure.core import StructureParams, find_centre

        z = np.load(case_dir / "case.npz", allow_pickle=False)
        self.lat = np.asarray(z["lat"], np.float64)
        self.lon = norm_lon(z["lon"])
        self.vars = [str(v) for v in z["var_names"]]
        self.truth = np.asarray(z["truth"], np.float64)
        self.xi = np.asarray(z["x_interp"], np.float64)
        self.node_index = [int(v) for v in z["node_index"]]
        self.node_sigma = [float(v) for v in z["node_sigma"]]
        self.node_label = [str(v) for v in z["node_label"]]
        self.vi = {v: i for i, v in enumerate(self.vars)}
        if feature_box is None:
            m = pst.SPEC_EDGE_MARGIN_DEG
            feature_box = (self.lat.min() + m, self.lat.max() - m, self.lon.min() + m, self.lon.max() - m)
        self.box = tuple(float(v) for v in feature_box)
        la0, la1, lo0, lo1 = self.box
        self.in_box = (self.lat >= la0) & (self.lat <= la1) & (self.lon >= lo0) & (self.lon <= lo1)
        self.smp = pst.Sampler(self.lat, self.lon, self.box, pst.SPEC_RES_DEG, method=spectra_method)
        dx, dy = self.smp.dx_dy_km()
        self.dx, self.dy = dx, dy
        self.spec = pst.Spectrum(len(self.smp.gy), len(self.smp.gx), dx, dy)
        self.wl = np.asarray(self.spec.wavelength_km)
        self.sub = [(lo, hi, (self.wl >= lo) & (self.wl < hi)) for lo, hi in SUB_BANDS]
        self.F_truth_2t = self.spec.fft(self.smp(self.truth[:, self.vi["2t"]])) if "2t" in self.vi else None

        # storms: first guess from the truth, as the tc_structure evaluator's first lead
        self.params = StructureParams()
        self.storms = {}
        if "msl" in self.vi and "10u" in self.vi and "10v" in self.vi:
            from dataclasses import replace
            msl_t = self._hpa(self.truth[:, self.vi["msl"]])
            for ev, (s, n, w, e) in STORM_EVENTS.items():
                ev_mask = (self.lat >= s) & (self.lat <= n) & (self.lon >= w) & (self.lon <= e)
                if ev_mask.sum() < 100:
                    continue
                fg = find_centre(self.lat, self.lon, msl_t, ev_mask, bbox=(s, n, w, e),
                                 params=replace(self.params, box_edge_km=STORM_EDGE_KM))
                ok = bool(fg["found"] and fg["pmin_hpa"] <= STORM_MAX_PMIN_HPA)
                self.storms[ev] = {"mask": ev_mask, "fg": fg, "ok": ok, "bbox": (s, n, w, e)}
                LOG.info("%s %s: truth low %.1f hPa at %.2f N %.2f E (%s)", case_dir.name, ev, fg["pmin_hpa"],
                         fg["lat"], fg["lon"], "storm case" if ok else f"no storm case: {fg['reason']}")
        LOG.info("%s: %d points, feature box %s (%d points), grid %d x %d, %d bins %.1f-%.1f km, storms %s",
                 case_dir.name, self.lat.size, self.box, int(self.in_box.sum()), len(self.smp.gy),
                 len(self.smp.gx), self.wl.size, self.wl.min(), self.wl.max(),
                 [k for k, v in self.storms.items() if v["ok"]])

    @staticmethod
    def _hpa(p):
        p = np.asarray(p, np.float64)
        return p / 100.0 if np.nanmedian(p) > 5000.0 else p

    def features(self, field: np.ndarray) -> dict[tuple[str, str], float]:
        """{(feature, unit suffix): value}; unit suffix "" for case-level features, the storm name for storms."""
        from eval.evaluators.tc_structure.core import find_centre, measure_structure

        f = np.asarray(field, np.float64)
        out: dict[tuple[str, str], float] = {}
        for ev, st in self.storms.items():
            if not st["ok"]:
                continue
            msl = self._hpa(f[:, self.vi["msl"]])
            c = find_centre(self.lat, self.lon, msl, st["mask"], bbox=st["bbox"],
                            first_guess=(st["fg"]["lat"], st["fg"]["lon"]), params=self.params)
            vals = {"pmin_hpa": np.nan, "maxwind300_ms": np.nan, "rmw_km": np.nan}
            if c["found"]:
                sc, _ = measure_structure(self.lat, self.lon, f[:, self.vi["10u"]], f[:, self.vi["10v"]],
                                          c["lat"], c["lon"], self.params)
                vals = {"pmin_hpa": c["pmin_hpa"], "maxwind300_ms": sc["maxwind300_ms"], "rmw_km": sc["rmw_km"]}
            for k, v in vals.items():
                out[(f"storm_{k}", ev)] = float(v)
        for v in SPEC_VARS:
            if v not in self.vi:
                continue
            p = self.spec.power(self.spec.fft(self.smp(f[:, self.vi[v]] - self.xi[:, self.vi[v]])))
            for wl, val in zip(self.wl, np.sqrt(p)):
                out[(f"amp_resid_{v}_{wl:.1f}km", "")] = float(val)
            for lo, hi, m in self.sub:
                if m.any():
                    out[(f"amp_resid_{v}_sub{lo:g}-{hi:g}km", "")] = float(np.sqrt(p[m].mean()))
        if self.F_truth_2t is not None:
            g = self.smp(f[:, self.vi["2t"]])
            F = self.spec.fft(g)
            pp, pt, cr = self.spec.power(F), self.spec.power(self.F_truth_2t), self.spec.cross(F, self.F_truth_2t)
            coh = cr / np.sqrt(pp * pt)
            for wl, val in zip(self.wl, coh):
                out[(f"coh_2t_{wl:.1f}km", "")] = float(val)
            for lo, hi, m in self.sub:
                if m.any():
                    out[(f"coh_2t_sub{lo:g}-{hi:g}km", "")] = float(cr[m].sum() / np.sqrt(pp[m].sum() * pt[m].sum()))
            gy, gx = np.gradient(g, self.dy, self.dx)
            mag = np.hypot(gx, gy)[1:-1, 1:-1] * 100.0
            out[("grad2t_p99_K_per_100km", "")] = float(np.percentile(mag, 99.0))
        if "tp" in self.vi:
            from eval.evaluators.tc_structure.core import O1280_CELL_AREA_KM2
            tp = f[self.in_box, self.vi["tp"]]
            tp_mm = tp * 1000.0 if np.nanmax(np.abs(tp)) < 5.0 else tp
            out[("tp_max_mm", "")] = float(np.nanmax(tp_mm))
            out[("tp_area50_km2", "")] = float((tp_mm >= 50.0).sum() * O1280_CELL_AREA_KM2)
        return out


def parse_draw(name: str) -> tuple[str, int, int]:
    if name == "trunk":
        return "trunk", -1, 0
    if name.startswith("full_j"):
        return "full", -1, int(name[6:])
    if name.startswith("branch_n"):
        node, j = name[8:].split("_j")
        return "branch", int(node), int(j)
    raise ValueError(name)


# ---------------------------------------------------------------------------
# shares and sigma*
# ---------------------------------------------------------------------------

def _var(x) -> float:
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    return float(np.var(x, ddof=1)) if x.size >= MIN_VALID else np.nan


def _var_rows(x) -> np.ndarray:
    """Row-wise sample variance (ddof 1) ignoring NaN; NaN where a row has fewer than MIN_VALID values."""
    x = np.asarray(x, np.float64)
    n = np.isfinite(x).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        v = np.nanvar(x, axis=1, ddof=1) if x.size else np.full(x.shape[0], np.nan)
    return np.where(n >= MIN_VALID, v, np.nan)


def monotone_fit(sigmas, shares) -> np.ndarray:
    """Least-squares fit of the share curve that does not increase as sigma decreases (pool adjacent
    violators, equal weights); NaN entries are left NaN. Returned in the input order."""
    s = np.asarray(sigmas, np.float64)
    r = np.asarray(shares, np.float64)
    out = np.full(r.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(r))
    if ok.size == 0:
        return out
    order = ok[np.argsort(s[ok])]                 # ascending sigma: the fit must be non-decreasing
    blocks = []                                   # [sum, count]
    for v in r[order]:
        blocks.append([v, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            a = blocks.pop()
            blocks[-1][0] += a[0]
            blocks[-1][1] += a[1]
    fit = np.concatenate([[b[0] / b[1]] * b[1] for b in blocks])
    out[order] = fit
    return out


def sigma_star(sigmas, shares, threshold: float = SHARE_THRESHOLD) -> tuple[float, str]:
    """First crossing below ``threshold`` going down the ladder; (sigma*, flag) with flag in
    {"crossed", "above", "below", "nan"}. Log-linear interpolation with the node above."""
    s = np.asarray(sigmas, np.float64)
    r = np.asarray(shares, np.float64)
    order = np.argsort(-s)
    s, r = s[order], r[order]
    ok = np.isfinite(r)
    s, r = s[ok], r[ok]
    if s.size == 0:
        return float("nan"), "nan"
    below = np.flatnonzero(r < threshold)
    if below.size == 0:
        return float(s[-1]), "below"
    m = int(below[0])
    if m == 0:
        return float(s[0]), "above"
    t = (r[m - 1] - threshold) / (r[m - 1] - r[m])
    return float(math.exp(math.log(s[m - 1]) + t * (math.log(s[m]) - math.log(s[m - 1])))), "crossed"


def compute_shares(table: dict, n_boot: int, seed: int):
    """table[feature][unit] = {"nodes": {node: (sigma, label, [values])}, "full": [values]}.
    Returns (rows, star_rows)."""
    rng = np.random.default_rng(seed)
    rows, stars = [], []
    for feat in sorted(table):
        units = table[feat]
        node_ids = sorted({n for u in units.values() for n in u["nodes"]})
        sig = {n: next(u["nodes"][n][0] for u in units.values() if n in u["nodes"]) for n in node_ids}
        lab = {n: next(u["nodes"][n][1] for u in units.values() if n in u["nodes"]) for n in node_ids}
        pooled, med = [], []
        for n in node_ids:
            vb_sum = vf_sum = 0.0
            per = []
            nu = 0
            for uname, u in units.items():
                if n not in u["nodes"]:
                    continue
                vb, vf = _var(u["nodes"][n][2]), _var(u["full"])
                if not (np.isfinite(vb) and np.isfinite(vf)) or vf <= 0:
                    continue
                per.append(vb / vf)
                vb_sum += vb
                vf_sum += vf
                nu += 1
                rows.append(dict(feature=feat, unit=uname, node=n, sigma=sig[n], label=lab[n], var_branch=vb,
                                 var_full=vf, share=vb / vf, share_p10="", share_p90="", n_units=1))
            pooled.append(vb_sum / vf_sum if nu and vf_sum > 0 else np.nan)
            med.append(float(np.median(per)) if per else np.nan)
        # bootstrap: draws resampled inside each unit and set (vectorised over replicates)
        boot = np.full((n_boot, len(node_ids)), np.nan)
        stars_b = []
        if n_boot:
            vb_tot = np.zeros((n_boot, len(node_ids)))
            vf_tot = np.zeros((n_boot, len(node_ids)))
            for un, u in units.items():
                xf = np.asarray(u["full"], np.float64)
                vf = _var_rows(xf[rng.integers(0, xf.size, (n_boot, xf.size))])
                for ki, n in enumerate(node_ids):
                    if n not in u["nodes"]:
                        continue
                    xb = np.asarray(u["nodes"][n][2], np.float64)
                    vb = _var_rows(xb[rng.integers(0, xb.size, (n_boot, xb.size))])
                    ok = np.isfinite(vb) & np.isfinite(vf) & (vf > 0)
                    vb_tot[ok, ki] += vb[ok]
                    vf_tot[ok, ki] += vf[ok]
            with np.errstate(divide="ignore", invalid="ignore"):
                boot = np.where(vf_tot > 0, vb_tot / vf_tot, np.nan)
            stars_b = [sigma_star([sig[n] for n in node_ids], boot[b])[0] for b in range(n_boot)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            p10 = np.nanpercentile(boot, 10, axis=0) if n_boot else [np.nan] * len(node_ids)
            p90 = np.nanpercentile(boot, 90, axis=0) if n_boot else [np.nan] * len(node_ids)
        n_units = len(units)
        for ki, n in enumerate(node_ids):
            rows.append(dict(feature=feat, unit="pooled", node=n, sigma=sig[n], label=lab[n], var_branch="",
                             var_full="", share=pooled[ki], share_p10=p10[ki], share_p90=p90[ki], n_units=n_units))
            rows.append(dict(feature=feat, unit="median_of_units", node=n, sigma=sig[n], label=lab[n],
                             var_branch="", var_full="", share=med[ki], share_p10="", share_p90="",
                             n_units=n_units))
        sl = [sig[n] for n in node_ids]
        s_star, flag = sigma_star(sl, pooled)
        s_mono, flag_mono = sigma_star(sl, monotone_fit(sl, pooled))
        sb = np.asarray([v for v in stars_b if np.isfinite(v)])
        stars.append(dict(feature=feat, sigma_star=s_star, flag=flag, sigma_star_monotone=s_mono,
                          flag_monotone=flag_mono,
                          sigma_star_p10=float(np.exp(np.percentile(np.log(sb), 10))) if sb.size else np.nan,
                          sigma_star_p90=float(np.exp(np.percentile(np.log(sb), 90))) if sb.size else np.nan,
                          n_units=n_units, units=";".join(sorted(units))))
    return rows, stars


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


def write_markdown(path: Path, stars, rows, meta_lines: list[str]) -> None:
    by = {s["feature"]: s for s in stars}
    pooled = {}
    for r in rows:
        if r["unit"] == "pooled":
            pooled.setdefault(r["feature"], []).append(r)

    def star_txt(s, key="sigma_star", fkey="flag"):
        if s[fkey] == "crossed":
            return _fmt(s[key])
        if s[fkey] == "above":
            return f"above {_fmt(s[key])}"
        if s[fkey] == "below":
            return f"below {_fmt(s[key])}"
        return "n/a"

    def curve(feat):
        rs = sorted(pooled.get(feat, []), key=lambda r: -r["sigma"])
        return " ".join(f"{_fmt(r['share'], 2)}" for r in rs)

    lines = ["# U3 branching read: sigma* per feature", "",
             "share(F, sigma_k) = variance of F over the branches re-noised at sigma_k / variance over the independent "
             "full draws. sigma* = where the pooled share first falls below 0.5 going down the ladder (log-linear "
             "interpolation); the monotone-fit column applies the same rule to the least-squares non-increasing fit of "
             "the curve, which is less sensitive to the sampling noise of 8 + 8 draws. 'above X': below 0.5 already at "
             "the first branch node X; 'below X': never below 0.5, X the last node. The 10-90 % range is the bootstrap "
             "of the first-crossing sigma* (draws resampled inside each unit).", ""] + meta_lines + [""]
    groups = [
        ("Storm features (tc_structure)", lambda f: f.startswith("storm_")),
        ("Fronts and rain", lambda f: f.startswith("grad2t") or f.startswith("tp_")),
        ("Residual amplitude per sub-band", lambda f: f.startswith("amp_resid_") and "_sub" in f),
        ("2 m temperature coherence with the truth per sub-band", lambda f: f.startswith("coh_2t_sub")),
    ]
    any_sig = next(iter(pooled.values()), [])
    sig_hdr = " ".join(_fmt(r["sigma"], 3) for r in sorted(any_sig, key=lambda r: -r["sigma"]))
    for title, pred in groups:
        feats = [f for f in sorted(by) if pred(f)]
        if not feats:
            continue
        lines += [f"## {title}", "", "| feature | sigma* (pooled, first crossing) | sigma* (monotone fit) | 10-90 % (bootstrap) | units | pooled share, high to low sigma |",
                  "|---|---|---|---|---|---|"]
        for f in feats:
            s = by[f]
            lines.append(f"| {f} | {star_txt(s)} | {star_txt(s, 'sigma_star_monotone', 'flag_monotone')} | "
                         f"{_fmt(s['sigma_star_p10'])}-{_fmt(s['sigma_star_p90'])} | {s['n_units']} | {curve(f)} |")
        lines.append("")
    # per-bin tables, one per variable and box
    per_bin = [f for f in sorted(by) if (f.startswith("amp_resid_") or f.startswith("coh_2t_")) and "_sub" not in f]
    if per_bin:
        lines += ["## Per wavelength bin", "", "Residual amplitude (amp) and 2 m temperature coherence (coh); "
                  "the storm-date box and the Alps box have different bins, so each bin row pools only the "
                  "cases that share it.", "", "| feature | sigma* (first crossing) | sigma* (monotone fit) | 10-90 % | units |",
                  "|---|---|---|---|---|"]

        def key(f):
            head, km = f.rsplit("_", 1)
            return head, -float(km[:-2])
        for f in sorted(per_bin, key=key):
            s = by[f]
            lines.append(f"| {f} | {star_txt(s)} | {star_txt(s, 'sigma_star_monotone', 'flag_monotone')} | "
                         f"{_fmt(s['sigma_star_p10'])}-{_fmt(s['sigma_star_p90'])} | {s['n_units']} |")
        lines.append("")
    lines += [f"Branch-node sigmas, high to low: {sig_hdr}", ""]
    path.write_text("\n".join(lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="folder holding the case folders")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", nargs="*", default=None, help="case labels (default: every folder with case.npz)")
    ap.add_argument("--feature-box", action="append", default=[],
                    help="CASE=lat_min,lat_max,lon_min,lon_max (default: the points' bounds minus 1 degree)")
    ap.add_argument("--pst", default=str(Path(__file__).resolve().parent / "_box_instrument" / "plot_sampler_texture.py"))
    ap.add_argument("--no-pst-sha-check", action="store_true")
    ap.add_argument("--spectra-method", default="linear", choices=["linear", "nearest"])
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261015)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout,
                        force=True)
    pst = load_pst(Path(args.pst), check_sha=not args.no_pst_sha_check)
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    boxes = {}
    for item in args.feature_box:
        k, v = item.split("=")
        boxes[k] = tuple(float(x) for x in v.split(","))
    cases = args.cases or sorted(p.name for p in root.iterdir() if (p / "case.npz").exists())
    if not cases:
        LOG.error("no case folders under %s", root)
        return 2

    table: dict = {}
    t0 = time.time()
    meta_lines = []
    with open(out / "features.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "draw", "kind", "node", "sigma", "j", "feature", "unit", "value"])
        for case in cases:
            cdir = root / case
            cf = CaseFeatures(cdir, pst, boxes.get(case), args.spectra_method)
            sig_of = dict(zip(cf.node_index, cf.node_sigma))
            lab_of = dict(zip(cf.node_index, cf.node_label))
            files = sorted((cdir / "finals").glob("*.npy"))
            files = [f for f in files if not f.name.endswith(".tmp.npy")]
            counts = {"trunk": 0, "full": 0, "branch": 0}
            for fpath in files:
                kind, node, j = parse_draw(fpath.stem)
                counts[kind] += 1
                feats = cf.features(np.load(fpath))
                for (feat, unit_sfx), val in feats.items():
                    w.writerow([case, fpath.stem, kind, node, sig_of.get(node, ""), j, feat, unit_sfx, val])
                    unit = f"{case}:{unit_sfx}" if unit_sfx else case
                    ent = table.setdefault(feat, {}).setdefault(unit, {"nodes": {}, "full": []})
                    if kind == "full":
                        ent["full"].append(val)
                    elif kind == "branch":
                        ent["nodes"].setdefault(node, (sig_of[node], lab_of[node], []))[2].append(val)
            meta = json.loads((cdir / "meta.json").read_text()) if (cdir / "meta.json").exists() else {}
            last = (meta.get("runs") or [{}])[-1]
            meta_lines.append(f"- {case}: {counts['branch']} branch, {counts['full']} full, {counts['trunk']} trunk "
                              f"draws; box {tuple(round(v, 2) for v in cf.box)}; storms "
                              f"{[k for k, v in cf.storms.items() if v['ok']] or 'none'}; source "
                              f"{last.get('source', '?')}; production check {last.get('production_check')}")
            LOG.info("%s: %s draws read (%.0f s so far)", case, counts, time.time() - t0)

    rows, stars = compute_shares(table, args.n_boot, args.seed)
    with open(out / "shares.csv", "w", newline="") as fh:
        cols = ["feature", "unit", "node", "sigma", "label", "var_branch", "var_full", "share", "share_p10",
                "share_p90", "n_units"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(out / "sigma_star.csv", "w", newline="") as fh:
        cols = ["feature", "sigma_star", "flag", "sigma_star_monotone", "flag_monotone", "sigma_star_p10",
                "sigma_star_p90", "n_units", "units"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(stars)
    write_markdown(out / "sigma_star.md", stars, rows, meta_lines)
    LOG.info("READ DONE: %d features, %d share rows -> %s (%.0f s)", len(stars), len(rows), out, time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
