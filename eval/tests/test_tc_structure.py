"""Validation check V0 of the tc_structure evaluator: a synthetic Rankine vortex.

Design note 20260923_physical_realism_scores, check V0: a Rankine vortex with a
radius of maximum wind (RMW) of 30 km, a maximum wind of 50 m/s and a matching
pressure dip is placed on the actual O1280 grid points of the Franklin box at 25 N.
The evaluator must recover the RMW within one bin (10 km), the maximum wind within
5 %, the mean vorticity inside 100 km within 5 % of the analytic value, an asymmetry
below 0.05 for the symmetric vortex, and an asymmetry close to 5 / ring mean when a
uniform 5 m/s translation wind is added.

The azimuthal-mean maximum tangential wind of a Rankine vortex CANNOT be recovered
within 5 % with 10 km bins: the profile has a cusp at the RMW and every bin averages
across it (the two bins next to r = 30 km average about 42-43 m/s, and the parabola
through three bins peaks near 44 m/s). That part of V0 is recorded below as a
strict expected failure, so that the record stays honest and any change of
definition that makes it pass is noticed.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from eval.evaluators.tc_structure.core import (
    KT34_MS, KT50_MS, StructureParams, find_centre, gc_distance_km, measure_structure,
    o1280_points_in_box, outward_bearing_at_point, ring_asymmetry, tangential_profile,
)

FRANKLIN_BOX = (15.0, 38.0, -78.0, -58.0)   # south, north, west, east (eval/config/events/franklin.yaml)
CLAT, CLON = 25.0, -68.0
RMW_KM, VMAX = 30.0, 50.0
RHO = 1.15
P_ENV_HPA = 1010.0


def rankine_speed(r_km):
    r = np.asarray(r_km, dtype=float)
    return np.where(r < RMW_KM, VMAX * r / RMW_KM, VMAX * RMW_KM / np.maximum(r, 1e-9))


def rankine_pressure_hpa(r_km):
    """Gradient-wind balance dp/dr = rho (V^2/r + f V), integrated inwards from 1500 km."""
    f = 2 * 7.292e-5 * math.sin(math.radians(CLAT))
    rr = np.linspace(0.0, 1500.0, 150001)            # km
    V = rankine_speed(rr)
    with np.errstate(divide="ignore", invalid="ignore"):
        dpdr = RHO * (np.where(rr > 0, V ** 2 / (rr * 1000.0), 0.0) + f * V)   # Pa per m
    seg = 0.5 * (dpdr[1:] + dpdr[:-1]) * np.diff(rr) * 1000.0
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    p = P_ENV_HPA * 100.0 - (cum[-1] - cum)           # Pa
    return np.interp(np.asarray(r_km, dtype=float), rr, p) / 100.0


@pytest.fixture(scope="module")
def grid():
    lat, lon = o1280_points_in_box(*FRANKLIN_BOX)
    return lat, lon


def vortex(lat, lon, translation=(0.0, 0.0)):
    r = gc_distance_km(CLAT, CLON, lat, lon)
    beta = outward_bearing_at_point(CLAT, CLON, lat, lon)
    V = rankine_speed(r)
    u = -V * np.cos(beta) + translation[0]
    v = V * np.sin(beta) + translation[1]
    return u, v, rankine_pressure_hpa(r), r


def test_o1280_grid_generator_counts():
    # the full O1280 grid has 6,599,680 points; check the per-row rule on a thin global band
    lat, lon = o1280_points_in_box(-90, 90, -180, 180, n_lat=16)
    assert lat.size == 2 * sum(20 + 4 * i for i in range(16))


def test_v0_centre_and_pmin(grid):
    lat, lon = grid
    _, _, p, _ = vortex(lat, lon)
    box = np.ones(lat.size, dtype=bool)
    c = find_centre(lat, lon, p, box, bbox=FRANKLIN_BOX)
    assert c["found"]
    assert float(gc_distance_km(CLAT, CLON, [c["lat"]], [c["lon"]])[0]) < 3.0
    assert abs(c["pmin_hpa"] - float(rankine_pressure_hpa(0.0))) < 0.5


def test_v0_rmw_within_one_bin(grid):
    lat, lon = grid
    u, v, _, _ = vortex(lat, lon)
    sc, _ = measure_structure(lat, lon, u, v, CLAT, CLON)
    assert abs(sc["rmw_km"] - RMW_KM) <= 10.0


def test_v0_plain_max_wind_within_5pct(grid):
    lat, lon = grid
    u, v, _, _ = vortex(lat, lon)
    sc, _ = measure_structure(lat, lon, u, v, CLAT, CLON)
    assert abs(sc["maxwind300_ms"] / VMAX - 1.0) <= 0.05


@pytest.mark.xfail(strict=True, reason="V0 failure by construction: 10 km bins average across "
                   "the Rankine cusp, so the azimuthal-mean maximum is ~12 % low")
def test_v0_azimuthal_mean_max_tangential_wind_within_5pct(grid):
    lat, lon = grid
    u, v, _, _ = vortex(lat, lon)
    sc, _ = measure_structure(lat, lon, u, v, CLAT, CLON)
    assert abs(sc["vmax_tan_ms"] / VMAX - 1.0) <= 0.05


def test_v0_vorticity_inside_100km_within_5pct(grid):
    lat, lon = grid
    u, v, _, _ = vortex(lat, lon)
    sc, _ = measure_structure(lat, lon, u, v, CLAT, CLON)
    analytic = 2.0 * (VMAX * RMW_KM / 100.0) / 100e3        # 3.0e-4 s^-1
    assert abs(sc["zeta100_s"] / analytic - 1.0) <= 0.05


def test_v0_symmetric_asymmetry_below_005(grid):
    lat, lon = grid
    u, v, _, _ = vortex(lat, lon)
    sc, _ = measure_structure(lat, lon, u, v, CLAT, CLON)
    assert sc["asym_rmw"] < 0.05


def test_v0_translation_asymmetry_matches_analytic(grid):
    lat, lon = grid
    u, v, _, r = vortex(lat, lon, translation=(5.0, 0.0))
    sc, _ = measure_structure(lat, lon, u, v, CLAT, CLON)
    ring = np.abs(r - sc["rmw_km"]) <= 5.0
    analytic = 5.0 / float(rankine_speed(r[ring]).mean())
    assert abs(sc["asym_rmw"] / analytic - 1.0) <= 0.10


def test_v0_extra_wind_radii_within_one_bin(grid):
    # not required by the note; recorded as extra evidence for the wind radii
    lat, lon = grid
    u, v, _, _ = vortex(lat, lon)
    sc, _ = measure_structure(lat, lon, u, v, CLAT, CLON)
    assert abs(sc["r34_km"] - VMAX * RMW_KM / KT34_MS) <= 10.0
    assert abs(sc["r50_km"] - VMAX * RMW_KM / KT50_MS) <= 10.0


def test_v0_report(grid, capsys):
    """Print the V0 numbers (always passes); read them with pytest -s."""
    lat, lon = grid
    u, v, p, r = vortex(lat, lon)
    sc, vt = measure_structure(lat, lon, u, v, CLAT, CLON)
    ut, vt2, _, _ = vortex(lat, lon, translation=(5.0, 0.0))
    sct, _ = measure_structure(lat, lon, ut, vt2, CLAT, CLON)
    ring = np.abs(r - sct["rmw_km"]) <= 5.0
    c = find_centre(lat, lon, p, np.ones(lat.size, bool), bbox=FRANKLIN_BOX)
    with capsys.disabled():
        print("\nV0 REPORT n_points=%d" % lat.size)
        print("V0 centre_error_km=%.3f pmin=%.3f analytic_pmin=%.3f" % (
            float(gc_distance_km(CLAT, CLON, [c["lat"]], [c["lon"]])[0]), c["pmin_hpa"],
            float(rankine_pressure_hpa(0.0))))
        print("V0 rmw_km=%.3f (30) vmax_tan=%.3f vmax_tan_bin=%.3f (50) maxwind300=%.3f (50)" % (
            sc["rmw_km"], sc["vmax_tan_ms"], sc["vmax_tan_bin_ms"], sc["maxwind300_ms"]))
        for rk in (50, 100, 200):
            an = 2.0 * float(rankine_speed(rk)) / (rk * 1e3)
            print("V0 zeta%d=%.4e analytic=%.4e rel=%+.4f" % (rk, sc[f"zeta{rk}_s"], an,
                                                           sc[f"zeta{rk}_s"] / an - 1))
        print("V0 r34=%.2f (%.2f) r50=%.2f (%.2f)" % (sc["r34_km"], VMAX * RMW_KM / KT34_MS,
                                                     sc["r50_km"], VMAX * RMW_KM / KT50_MS))
        an = 5.0 / float(rankine_speed(r[ring]).mean())
        print("V0 asym_symmetric=%.4f asym_translated=%.4f analytic=%.4f rel=%+.4f ring_points=%d" % (
            sc["asym_rmw"], sct["asym_rmw"], an, sct["asym_rmw"] / an - 1, sct["ring_points"]))
        print("V0 profile first 8 bins:", np.round(vt[:8], 2).tolist())


def test_runner_end_to_end_on_synthetic_file(tmp_path):
    """The evaluator's run() on a synthetic prediction file finds and measures the vortex."""
    netCDF4 = pytest.importorskip("netCDF4")
    from eval.evaluators.tc_structure import run, score

    lat, lon = o1280_points_in_box(10.0, 40.0, -100.0, -58.0)
    order = np.lexsort((lon, -lat))
    lat, lon = lat[order], lon[order]
    u, v, p, _ = vortex(lat, lon)
    states = ["10u", "10v", "2t", "msl"]
    nm = 2
    data = np.zeros((1, nm, lat.size, len(states)), dtype=np.float32)
    data[0, :, :, 0] = u
    data[0, :, :, 1] = v
    data[0, :, :, 2] = 300.0
    data[0, :, :, 3] = p * 100.0                    # Pa, as in the prediction files
    pdir = tmp_path / "preds"
    pdir.mkdir()
    path = pdir / "predictions_20230826_step024.nc"
    with netCDF4.Dataset(path, "w") as d:
        d.createDimension("sample", 1)
        d.createDimension("ensemble_member", nm)
        d.createDimension("grid_point_hres", lat.size)
        d.createDimension("weather_state", len(states))
        d.createVariable("lat_hres", "f4", ("grid_point_hres",))[:] = lat
        d.createVariable("lon_hres", "f4", ("grid_point_hres",))[:] = lon
        ws = d.createVariable("weather_state", str, ("weather_state",))
        for i, s in enumerate(states):
            ws[i] = s
        for name in ("y_pred", "y", "x_interp"):
            d.createVariable(name, "f4",
                             ("sample", "ensemble_member", "grid_point_hres", "weather_state"))[:] = data
    out = tmp_path / "out"
    run(pdir, {}, {"events": ["franklin"], "n_boot": 200}, output_dir=out, run_label="synthetic")
    import csv
    rows = list(csv.DictReader(open(out / "cases.csv")))
    assert len(rows) == 3 * nm
    assert all(r["found"] == "1" for r in rows)
    assert all(abs(float(r["rmw_km"]) - RMW_KM) <= 10.0 for r in rows)
    assert all(float(r["displacement_km"]) < 5.0 for r in rows)
    recs = score(out, {}, {})
    names = {r["metric"] for r in recs}
    assert "tcs_franklin_rmw_km_model_24_48" in names


def test_runner_input_first_guess_best_track_and_pairing(tmp_path):
    """No truth in the file, first guess from the input ensemble mean anchored on a best-track fix,
    and member pairing: model member k is 2 hPa deeper than input member k."""
    netCDF4 = pytest.importorskip("netCDF4")
    from eval.evaluators.tc_structure import run

    lat, lon = o1280_points_in_box(10.0, 40.0, -100.0, -58.0)
    order = np.lexsort((lon, -lat))
    lat, lon = lat[order], lon[order]
    u, v, p, r = vortex(lat, lon)
    states = ["10u", "10v", "2t", "msl"]
    nm = 3
    x = np.zeros((1, nm, lat.size, len(states)), dtype=np.float32)
    x[0, :, :, 0], x[0, :, :, 1], x[0, :, :, 2] = u, v, 300.0
    for k in range(nm):
        x[0, k, :, 3] = (p + 0.5 * k) * 100.0          # members differ by 0.5 hPa
    y = x.copy()
    y[0, :, :, 3] -= 200.0 * np.exp(-(r / 60.0) ** 2)  # model 2 hPa deeper at the centre
    pdir = tmp_path / "preds"
    pdir.mkdir()
    with netCDF4.Dataset(pdir / "predictions_20230826_step024.nc", "w") as d:
        d.createDimension("sample", 1)
        d.createDimension("ensemble_member", nm)
        d.createDimension("grid_point_hres", lat.size)
        d.createDimension("weather_state", len(states))
        d.createVariable("lat_hres", "f4", ("grid_point_hres",))[:] = lat
        d.createVariable("lon_hres", "f4", ("grid_point_hres",))[:] = lon
        ws = d.createVariable("weather_state", str, ("weather_state",))
        for i, s in enumerate(states):
            ws[i] = s
        dims = ("sample", "ensemble_member", "grid_point_hres", "weather_state")
        d.createVariable("y_pred", "f4", dims)[:] = y
        # member prediction files: x_interp is the single HRES input, x_interp_members the input ensemble
        d.createVariable("x_interp", "f4", ("sample", "grid_point_hres", "weather_state"))[:] = x[:, 0] + 500.0
        d.createVariable("x_interp_members", "f4", dims)[:] = x
    cache = tmp_path / "tc_cache"
    cache.mkdir()
    (cache / "best_track.csv").write_text(
        "event,storm_id,time,lat,lon,pmin_hpa,wind_ms,wind_src,source\n"
        f"franklin,1,2023082700,{CLAT + 0.8},{CLON - 0.8},990,40,,noaa_ibtracs_csv\n")
    out = tmp_path / "out"
    run(pdir, {}, {"events": ["franklin"], "n_boot": 200, "first_guess_field": "input",
                   "best_track_cache": str(cache)}, output_dir=out, run_label="synthetic")
    import csv
    import json
    rows = list(csv.DictReader(open(out / "cases.csv")))
    assert len(rows) == 3 * nm
    truth = [r for r in rows if r["field"] == "truth"]
    assert all(r["found"] == "0" and r["storm_ok"] == "1" for r in truth)
    found = [r for r in rows if r["field"] != "truth"]
    assert all(r["found"] == "1" for r in found)
    assert all(float(r["displacement_km"]) < 5.0 for r in found)
    paired = json.load(open(out / "summary.json"))["events"]["franklin"]["24"]["paired"]["model_minus_input"]
    assert paired["n_pairs"] == nm
    assert abs(paired["pmin_hpa"]["value"] + 2.0) < 0.3
    inp = [r for r in rows if r["field"] == "input"]
    assert [round(float(r["pmin_hpa"]) - float(inp[0]["pmin_hpa"]), 1) for r in inp] == [0.0, 0.5, 1.0]
    meta = json.load(open(out / "run_meta.json"))
    assert meta["input_source"].startswith("x_interp_members")


def test_best_track_case_without_fix_is_not_a_storm(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    from eval.evaluators.tc_structure import run

    lat, lon = o1280_points_in_box(10.0, 40.0, -100.0, -58.0)
    u, v, p, _ = vortex(lat, lon)
    states = ["10u", "10v", "msl"]
    a = np.stack([u, v, p * 100.0], axis=-1)[None, None].astype(np.float32)
    pdir = tmp_path / "preds"
    pdir.mkdir()
    with netCDF4.Dataset(pdir / "predictions_20230826_step024.nc", "w") as d:
        d.createDimension("sample", 1)
        d.createDimension("ensemble_member", 1)
        d.createDimension("grid_point_hres", lat.size)
        d.createDimension("weather_state", 3)
        d.createVariable("lat_hres", "f4", ("grid_point_hres",))[:] = lat
        d.createVariable("lon_hres", "f4", ("grid_point_hres",))[:] = lon
        ws = d.createVariable("weather_state", str, ("weather_state",))
        for i, s in enumerate(states):
            ws[i] = s
        for name in ("y_pred", "y", "x_interp"):
            d.createVariable(name, "f4", ("sample", "ensemble_member", "grid_point_hres", "weather_state"))[:] = a
    cache = tmp_path / "tc_cache"
    cache.mkdir()
    (cache / "best_track.csv").write_text(
        "event,storm_id,time,lat,lon,pmin_hpa,wind_ms,wind_src,source\n"
        f"franklin,1,2023082900,{CLAT},{CLON},990,40,,noaa_ibtracs_csv\n")
    out = tmp_path / "out"
    run(pdir, {}, {"events": ["franklin"], "n_boot": 50, "best_track_cache": str(cache)},
        output_dir=out, run_label="synthetic")
    import csv
    rows = list(csv.DictReader(open(out / "cases.csv")))
    assert rows and all(r["storm_ok"] == "0" for r in rows)
    assert "best-track" in rows[0]["reason"]


def test_event_base_dates_cross_a_month():
    from eval.evaluators.tc.core.events import EVENTS
    from eval.evaluators.tc.core.loading_predictions import select_prediction_files_for_event
    e = EVENTS["imelda_e2"]
    got = select_prediction_files_for_event([("a", 20250930, 24), ("b", 20251004, 48), ("c", 20251104, 24)], e)
    assert [g[0] for g in got] == ["a", "b"]


def test_runner_input_row_from_the_native_grid(tmp_path, monkeypatch):
    """Rule 41: the input row is measured on the native x (nearest native point), not on x_interp."""
    netCDF4 = pytest.importorskip("netCDF4")
    import json

    from eval.evaluators.tc_structure import run

    monkeypatch.setenv("EVAL_NATIVE_INDEX_CACHE", str(tmp_path / "idx"))
    lat, lon = o1280_points_in_box(10.0, 40.0, -100.0, -58.0)
    order = np.lexsort((lon, -lat))
    lat, lon = lat[order], lon[order]
    u, v, p, _ = vortex(lat, lon)
    states = ["10u", "10v", "2t", "msl"]
    data = np.zeros((1, 2, lat.size, len(states)), dtype=np.float32)
    data[0, :, :, 0], data[0, :, :, 1], data[0, :, :, 2], data[0, :, :, 3] = u, v, 300.0, p * 100.0
    sel = np.arange(0, lat.size, 3)                     # a coarser native set: every third output point
    pdir = tmp_path / "preds"
    pdir.mkdir()
    with netCDF4.Dataset(pdir / "predictions_20230826_step024.nc", "w") as d:
        d.createDimension("sample", 1)
        d.createDimension("ensemble_member", 2)
        d.createDimension("grid_point_hres", lat.size)
        d.createDimension("grid_point_lres", sel.size)
        d.createDimension("weather_state", len(states))
        d.createVariable("lat_hres", "f4", ("grid_point_hres",))[:] = lat
        d.createVariable("lon_hres", "f4", ("grid_point_hres",))[:] = lon
        d.createVariable("lat_lres", "f4", ("grid_point_lres",))[:] = lat[sel]
        d.createVariable("lon_lres", "f4", ("grid_point_lres",))[:] = lon[sel]
        ws = d.createVariable("weather_state", str, ("weather_state",))
        for i, s in enumerate(states):
            ws[i] = s
        full = ("sample", "ensemble_member", "grid_point_hres", "weather_state")
        for name in ("y_pred", "y"):
            d.createVariable(name, "f4", full)[:] = data
        flat = data.copy()
        flat[..., 3] = 101000.0                         # x_interp: no storm at all
        d.createVariable("x_interp", "f4", full)[:] = flat
        d.createVariable("x", "f4", ("sample", "ensemble_member", "grid_point_lres", "weather_state"))[:] = \
            data[:, :, sel]
    out = tmp_path / "out"
    run(pdir, {}, {"events": ["franklin"], "n_boot": 50}, output_dir=out, run_label="native")
    import csv
    rows = [r for r in csv.DictReader(open(out / "cases.csv")) if r["field"] == "input"]
    assert rows and all(r["found"] == "1" for r in rows)            # the storm is in the native x
    assert all(abs(float(r["rmw_km"]) - RMW_KM) <= 15.0 for r in rows)
    meta = json.loads((out / "run_meta.json").read_text())
    assert meta["input_source"] == "x (native input grid)" and meta["native_input"] is True
