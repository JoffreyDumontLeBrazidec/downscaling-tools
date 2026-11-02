"""End-to-end run of the scoring scripts on synthetic prediction files written by the predictor's own writer: 3 rows,
3 dates, +24 h, 10 members, on 16 real o2560 octahedral rings cut to the box (so the ring-count check of the area weights
runs). Row A100 is a calibrated ensemble (truth and members exchangeable draws), B24 under-dispersive, B50 over-dispersive.
Checks: the DONE lines; the corrected spread/RMSE of A100 within 0.02 of 1 for every variable and below/above 1 for B24/B50;
IN scored as a one-member forecast; tp pooled percentiles and member-max mean and the fair CRPS of 10 m wind speed (per
member from 10u and 10v) against values computed here by hand; the zoom-map case choice; rows with different dates are
refused; the box_spectra runner on the kit's layout; spectra_table column placement with values that differ per row and
per column."""
import csv
import json
import math
import shutil
import subprocess
import sys

import numpy as np
import pytest
from conftest import KIT_DIR
from o2560_common import OUTPUT_VARS, area_weights, case_seeds, edge_strip, fcrps_terms

pytest.importorskip("netCDF4")
from o2560_box_predict import write_nc  # noqa: E402

DATES = ["20241101", "20241102", "20241103"]
ROWS = {"A100": 1.0, "B24": 0.5, "B50": 2.0}   # member noise scale relative to the truth's
RINGS = list(range(600, 2200, 100))           # o2560 ring indices, lat about 69 N to 13 N
BASE = {"10u": 5, "10v": -3, "2d": 280, "2t": 285, "msl": 101000, "skt": 286, "tcw": 20, "cp": 0.0005, "tp": 0.001,
        "z_500": 55000, "t_850": 275}
SIG = {"10u": 2, "10v": 2, "2d": 1.5, "2t": 1.5, "msl": 150, "skt": 1.5, "tcw": 2, "cp": 0.0004, "tp": 0.0008,
       "z_500": 80, "t_850": 1.2}
ALL_VARS = ("10ff", "tp", "10u", "10v", "2t", "2d", "msl", "skt", "tcw", "cp", "z_500", "t_850")


def grid():
    lat, lon = [], []
    for i in RINGS:
        k = 4 * i + 16
        lo = np.arange(k) * 360.0 / k
        l180 = (lo + 180) % 360 - 180
        keep = (l180 >= -80) & (l180 <= 40)
        lon += list(lo[keep]); lat += [90.0 - (i - 0.5) * 90.0 / 2560] * int(keep.sum())
    return np.array(lat), np.array(lon)


def make_rows(root):
    lat, lon = grid(); P = lat.size; S = len(OUTPUT_VARS)
    rng = np.random.default_rng(3)
    l180 = (lon + 180) % 360 - 180
    precip = np.isin(np.arange(S), [OUTPUT_VARS.index("tp"), OUTPUT_VARS.index("cp")])
    fix = lambda a: np.where(precip, np.abs(a), a)
    sig = np.array([SIG[v] for v in OUTPUT_VARS])
    dirs = {}
    for d in DATES:
        mu = np.stack([BASE[v] + 2 * SIG[v] * np.sin(np.deg2rad(3 * lat)) * np.cos(np.deg2rad(2 * l180)) for v in OUTPUT_VARS], -1)
        y = fix(mu + sig * rng.normal(size=(P, S)))
        k = int(np.argmin((lat - 55) ** 2 + (l180 + 20) ** 2)); y[k, OUTPUT_VARS.index("msl")] = 97000.0   # a deep low
        xi = fix(mu + 0.3 * sig * rng.normal(size=(P, S)))
        for r, sc in ROWS.items():
            yp = fix(mu[None] + sc * sig * rng.normal(size=(10, P, S)))
            dd = root / "pred" / f"{r}_L024"; dd.mkdir(parents=True, exist_ok=True); dirs[r] = dd
            write_nc(dd / f"predictions_{d}_step024.nc", lat, lon, list(OUTPUT_VARS), y.astype(np.float32), xi.astype(np.float32),
                     yp.astype(np.float32), 0, 24, dict(model=r, ckpt_step=1, sampler="s", seeds=json.dumps(case_seeds(d, 24, 10)),
                                                        weights_sha256="0" * 64, code_pin="x" * 40, complete="yes"))
    return dirs


def run(args):
    r = subprocess.run([sys.executable] + [str(a) for a in args], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    return r.stdout


def read_nc(path):
    import netCDF4
    with netCDF4.Dataset(str(path)) as d:
        d.set_auto_mask(False)
        return (np.asarray(d["lat_hres"][:]), np.asarray(d["lon_hres"][:]), np.asarray(d["y"][0], np.float64),
                np.asarray(d["y_pred"][0], np.float64))


@pytest.fixture(scope="module")
def scored(tmp_path_factory):
    root = tmp_path_factory.mktemp("o25")
    dirs = make_rows(root)
    P = root / "pairs" / "L024"
    out = run([KIT_DIR / "scripts" / "paired_scores_o2560.py", 24, P] + [f"{r}={d}" for r, d in dirs.items()])
    assert "PAIRED_SCORES_O2560_DONE" in out
    per = {(r["row"], r["date"], r["domain"], r["var"], r["metric"]): float(r["value"]) for r in csv.DictReader(open(P / "per_date.csv"))}
    ratio = {(r["kind"], r["row"], r["domain"], r["var"]): r for r in csv.DictReader(open(P / "ratio.csv"))}
    return root, dirs, P, per, ratio


def test_calibrated_ensemble_reads_one(scored):
    ratio = scored[4]
    for v in ALL_VARS:
        a = float(ratio[("row", "A100", "box", v)]["ratio"])
        assert abs(a - 1) < 0.02, (v, a)
        assert float(ratio[("row", "B24", "box", v)]["ratio"]) < 0.9, v
        assert float(ratio[("row", "B50", "box", v)]["ratio"]) > 1.2, v
        assert float(ratio[("row", "A100", "box", v)]["lo95"]) <= a <= float(ratio[("row", "A100", "box", v)]["hi95"])
        assert float(ratio[("pair", "A100-B24", "box", v)]["lo95"]) > 0, v   # A100 clearly above the under-dispersive B24


def test_input_row_is_one_member(scored):
    per = scored[3]
    assert all(per[("IN", d, "box", "2t", "var")] == 0.0 for d in DATES)


def test_tp_tail_and_10ff_scores_by_hand(scored):
    dirs, per = scored[1], scored[3]
    lat, lon, y, yp = read_nc(dirs["B50"] / "predictions_20241102_step024.nc")
    keep = ~edge_strip(lat, (lon + 180) % 360 - 180)
    tp = np.clip(yp[:, :, OUTPUT_VARS.index("tp")] * 1000, 0, None)[:, keep]
    for p in (99.0, 99.9, 99.99):
        np.testing.assert_allclose(per[("B50", "20241102", "box", "tp", f"p{p:g}_pooled")], np.percentile(tp.ravel(), p), rtol=1e-9)
    np.testing.assert_allclose(per[("B50", "20241102", "box", "tp", "max_member_mean")], tp.max(axis=1).mean(), rtol=1e-9)
    ff = np.hypot(yp[:, :, OUTPUT_VARS.index("10u")], yp[:, :, OUTPUT_VARS.index("10v")])
    tf = np.hypot(y[:, OUTPUT_VARS.index("10u")], y[:, OUTPUT_VARS.index("10v")])
    fc, var, se = fcrps_terms(ff, tf)
    w = area_weights(lat, lon) * keep
    np.testing.assert_allclose(per[("B50", "20241102", "box", "10ff", "fcrps")], (w * fc).sum() / w.sum(), rtol=1e-9)
    np.testing.assert_allclose(per[("B50", "20241102", "box", "10ff", "spread")], math.sqrt((w * var).sum() / w.sum()), rtol=1e-9)
    np.testing.assert_allclose(per[("B50", "20241102", "box", "10ff", "rmse")], math.sqrt((w * se).sum() / w.sum()), rtol=1e-9)


def test_summary_and_zoom_maps(scored):
    root, dirs = scored[0], scored[1]
    out = run([KIT_DIR / "scripts" / "summarize.py", root, "24"])
    assert "SUMMARY_DONE" in out and "| ratio |" in (root / "summary" / "summary_box.md").read_text()
    pytest.importorskip("matplotlib")
    out = run([KIT_DIR / "scripts" / "zoom_maps_o2560.py", 24, root / "maps"] + [f"{r}={d}" for r, d in dirs.items()])
    assert "ZOOM_MAPS_DONE" in out
    cases = json.loads((root / "maps" / "cases.json").read_text())
    assert cases["storm"]["is_storm"] and abs(cases["storm"]["truth_msl_min_hpa"] - 970.0) < 1e-6
    assert len(list((root / "maps").glob("*.png"))) == 4


def test_rows_with_different_dates_are_refused(scored, tmp_path):
    dirs = scored[1]
    short = tmp_path / "B24_short"; shutil.copytree(dirs["B24"], short)
    (short / "predictions_20241103_step024.nc").unlink()
    r = subprocess.run([sys.executable, str(KIT_DIR / "scripts" / "paired_scores_o2560.py"), "24", str(tmp_path / "o"),
                        f"A100={dirs['A100']}", f"B24={short}"], capture_output=True, text=True)
    assert r.returncode != 0 and "date sets differ" in (r.stdout + r.stderr)


def test_box_spectra_runner_reads_the_kit_layout(scored, tmp_path):
    pytest.importorskip("eval.evaluators.box_spectra.runner")
    from eval.evaluators.box_spectra.runner import run as bs_run
    # its own regular 1-degree grid: the sparse rings of the other tests (3.5 degrees apart, 0.04 degrees along a ring)
    # make a degenerate triangulation; this test only checks that the runner reads the kit's file layout
    la, lo = np.meshgrid(np.arange(10.0, 70.01, 1.0), np.arange(-80.0, 40.01, 1.0), indexing="ij")
    lat, lon = la.ravel(), lo.ravel(); P = lat.size; S = len(OUTPUT_VARS)
    rng = np.random.default_rng(5); d = tmp_path / "pred"; d.mkdir()
    for q in DATES:
        y = np.abs(rng.normal(size=(P, S))).astype(np.float32)
        write_nc(d / f"predictions_{q}_step024.nc", lat, lon, list(OUTPUT_VARS), y, y, np.abs(rng.normal(size=(10, P, S))).astype(np.float32),
                 0, 24, dict(seeds=json.dumps(case_seeds(q, 24, 10)), complete="yes"))
    o = bs_run(str(d), {}, {"steps": [24], "variables": ["ws", "tp", "2t"], "res_deg": 0.5},
               output_dir=str(tmp_path / "bs"), overwrite=True, run_label="A100_L024")
    pooled = [r for r in csv.DictReader(open(o / "bins.csv")) if r["scope"] == "all"]
    assert {r["variable"] for r in pooled} == {"ws", "tp", "2t"}
    assert all(int(r["n_draws"]) == 30 for r in pooled)   # 3 dates x 10 members


def test_spectra_table_places_each_column(tmp_path):
    cols = ["scope", "variable", "bin_lo_km", "bin_hi_km", "bin_centre_km", "amp_resid_pred", "amp_resid_true", "ratio_resid",
            "amp_pred_full", "amp_truth_full", "ratio_full", "amp_interp", "ratio_interp_to_truth", "coherence", "aligned", "n_draws"]
    args = []
    for ri, r in enumerate(("A100", "B24", "B50")):
        f = tmp_path / f"{r}.csv"
        with open(f, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
            for vi, v in enumerate(("ws", "tp")):
                for bi, km in enumerate((100.0, 20.0, 10.0)):
                    w.writerow({**{c: 0.5 for c in cols}, "scope": "all", "variable": v, "bin_centre_km": km, "n_draws": 120,
                                "ratio_resid": 0.1 + 0.01 * ri + 0.001 * bi + vi, "ratio_full": 0.6 + 0.01 * ri + 0.001 * bi + vi,
                                "ratio_interp_to_truth": 0.3 + 0.001 * bi + vi})
        args.append(f"{r}={f}")
    out = run([KIT_DIR / "scripts" / "spectra_table.py", 24, tmp_path / "o", "box"] + args)
    assert "SPECTRA_TABLE_DONE" in out
    lines = (tmp_path / "o" / "spectra_box.md").read_text().splitlines()
    row20_tp = [ln for ln in lines if ln.startswith("| 20.0 |")][1]   # second block = tp
    cells = [c.strip() for c in row20_tp.strip("|").split("|")]
    # km | A100 resid | A100 full | B24 resid | B24 full | B50 resid | B50 full | IN full
    assert cells[1:] == ["1.101", "1.601", "1.111", "1.611", "1.121", "1.621", "1.301"], cells
    assert any(ln.startswith("| 10.0 (model only) |") for ln in lines)
