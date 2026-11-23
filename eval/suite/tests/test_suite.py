"""``eval.cli suite`` end to end on synthetic files: every page is written, the index lists them."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from eval.cli import build_parser, main
from eval.tests.synthetic_predictions import octahedral_grid, write_predictions_file

DATES = ("20250925", "20250926")
STEPS = (24, 72)
CASES = """
name: synthetic
members: "1-4"
spectra: {steps: [24, 72], box: null, fine_res_deg: 0.05}
tc: {name: storm, date: auto, step: 24, variable: wind10m}
rain: {name: rain, date: auto, step: 24, extent: [-40.0, 20.0, 30.0, 65.0], box_deg: 10, variable: tp}
areas:
  - {name: alps, date: auto, step: 24, extent: [4.0, 16.0, 43.0, 49.0], variable: tp}
  - {name: atl, date: "20250926", step: 72, center: [45.0, -30.0], variable: msl}
stills:
  cases:
    - {name: glob, date: auto, step: 24, variable: tp}
tp_distribution: {steps: [24, 72]}
scores: {region: all, variables: [[tp, raw], [10ff, raw], [2t, lapse], [msl, raw]]}
"""


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("suite")
    grids = (octahedral_grid(96), octahedral_grid(48))
    runs = {}
    for k, label in enumerate(("A_m", "B_m")):
        d = root / label
        for i, date in enumerate(DATES):
            for step in STEPS:
                write_predictions_file(d / f"predictions_{date}_step{step:03d}.nc", members=(1, 2, 3, 4),
                                       per_member_truth=True, seed=100 * k + 10 * i + step, grids=grids)
        runs[label] = d
    obs = {}
    for k, label in enumerate(runs):
        od = root / f"obs_{label}"
        od.mkdir()
        rows = []
        for var, variant in (("tp", "raw"), ("10ff", "raw"), ("2t", "lapse"), ("2t", "raw"), ("msl", "raw")):
            for lead in (12, 24, 48, 72):
                for row, f in (("model", 1.0 - 0.05 * k), ("input", 1.2), ("truth", 0.9)):
                    rows.append(dict(var=var, variant=variant, radius_km=0, region="all", lead=lead, row=row,
                                     fcrps=f * (1 + lead / 100), fcrps_se=0.02, spread=0.8, spread_se=0.02,
                                     ssr_corrected=0.9, ssr_corrected_se=0.03, dates=13, mean_n=900.0,
                                     members=4 if row != "input" else 4))
        pd.DataFrame(rows).to_csv(od / "summary.csv", index=False)
        (od / "run_meta.json").write_text(json.dumps({"run_label": label}))
        obs[label] = od
    tc = {}
    for k, label in enumerate(runs):
        td = root / f"tc_{label}"
        td.mkdir()
        rows = []
        for date in DATES:
            for lead in STEPS:
                for field in ("model", "truth", "input"):
                    for m in (1, 2, 3, 4):
                        rows.append(dict(event="storm", date=date, lead=lead, field=field, member=m, found=1,
                                         storm_ok=1, pmin_hpa=970 + 5 * m + (8 if field == "input" else 0) - k,
                                         maxwind300_ms=40 - 2 * m, centre_lat=30.0, centre_lon=-50.0))
        pd.DataFrame(rows).to_csv(td / "cases.csv", index=False)
        tc[label] = td
    cache = root / "tc_cache"
    cache.mkdir()
    bt = [dict(event="storm", storm_id="X", time=f"{d}{h:02d}", lat=30.0, lon=-50.0, pmin_hpa=950.0, wind_ms=50.0,
               source="anemoi_zarr") for d in ("20250925", "20250926", "20250927", "20250928", "20250929")
          for h in (0, 6, 12, 18)]
    pd.DataFrame(bt).to_csv(cache / "best_track.csv", index=False)
    cases = root / "synthetic.yaml"
    cases.write_text(CASES)
    return root, runs, obs, tc, cache, cases


def test_suite_is_a_figures_command():
    args = build_parser().parse_args(["suite", "--run", "a=x", "--cases", "o2560_tc_2025", "--out", "o"])
    assert args._command.name == "suite" and args._command.group == "figures"


def test_shipped_case_lists_load():
    from eval.suite.cases import available_case_lists, load_case_list

    for name in available_case_lists():
        cl = load_case_list(name)
        assert cl.spectra_steps == [24, 72], name   # exactly two spectra pages
        assert cl.tc is not None and cl.rain is not None and cl.areas and cl.stills


def test_every_page(inputs):
    root, runs, obs, tc, cache, cases = inputs
    out = root / "out"
    argv = ["suite", "--cases", str(cases), "--out", str(out), "--tc-cache", str(cache)]
    for label in runs:
        argv += ["--run", f"{label}={runs[label]}", "--obs", f"{label}={obs[label]}", "--tc", f"{label}={tc[label]}"]
    main(argv)
    rec = json.loads((out / "suite_manifest.json").read_text())
    by = {p["page"]: p for p in rec["pages"]}
    failed = {k: v["note"] + v.get("trace", "") for k, v in by.items() if not v["files"]}
    assert not failed, failed
    assert sorted(by) == ["01", "01b", "02", "02b", "03", "04", "05", "05z", "06", "07", "08", "09"]
    assert "0.05 deg" in by["01b"]["note"] and "0.07 deg" in by["01"]["note"]
    assert len(list(out.glob("05z_zooms_*.png"))) == 2
    assert "close level" in by["05z"]["note"]
    for name in ("01_spectra_L024.png", "02_spectra_L072.png", "01b_spectra_fine_L024.png",
                 "02b_spectra_fine_L072.png", "06_scores_stations.png", "07_tp_stills.png",
                 "05_areas_member1.png", "08_tp_distribution.png", "09_tc_distribution.png"):
        assert (out / name).exists(), name
    assert list(out.glob("03_tc_members_storm_wind10m_*_L024.png"))
    assert list(out.glob("04_rain_members_rain_tp_*_L024.png"))
    assert "pages 10 and 11" in (out / "SUITE.md").read_text().lower()
    assert "deepest target storm" in by["03"]["note"]
    assert "wettest target" in by["04"]["note"]
    assert "10-degree box on its maximum" in by["04"]["note"]


def test_ten_degree_box_and_fitted_range():
    """Owner 23 Nov: ~10-degree map boxes; 2t/msl colour range fitted to the box, wind and tp keep fixed scales."""
    from eval.suite.cases import box_around, load_case_list

    lo0, lo1, la0, la1 = box_around(46.0, 10.0)
    assert la1 - la0 == 10.0 and lo1 - lo0 > 10.0
    for name in ("o2560_tc_2025", "o1280_global"):
        cl = load_case_list(name)
        for c in cl.areas:
            assert abs((c.extent[3] - c.extent[2]) - 10.0) < 1e-9, (name, c.name)
        assert cl.rain.box_deg == 10


def test_rows_without_target_or_x_interp(tmp_path):
    """AIFS-ENS-like rows (x and y_pred only, steps 24 and 72): --attach-target writes y, the pages draw."""
    import xarray as xr

    grids = (octahedral_grid(96), octahedral_grid(48))
    run, truth = tmp_path / "ARMO", tmp_path / "truth"
    run.mkdir()
    truth.mkdir()
    for i, date in enumerate(DATES):
        for step in STEPS:
            full = write_predictions_file(tmp_path / "full" / f"predictions_{date}_step{step:03d}.nc",
                                          members=(1, 2, 3, 4), seed=i + step, grids=grids)
            with xr.open_dataset(full) as ds:
                ds.drop_vars(["y", "x_interp"]).to_netcdf(run / full.name)
                if (date, step) != (DATES[1], 72):   # one valid time without truth: all-NaN target
                    valid = pd.Timestamp(date) + pd.Timedelta(hours=step)
                    y = ds["y"].isel(sample=0, ensemble_member=0).values.copy()
                    y[:, list(ds.weather_state.values).index("tp")] = np.nan   # an analysis has no tp
                    xr.Dataset({"y": (("grid_point", "weather_state"), y)},
                               coords={"weather_state": ds.weather_state.values}).to_netcdf(
                        truth / f"truth_{valid:%Y%m%d%H}.nc")
    out = tmp_path / "out"
    main(["suite", "--cases", "o1280_global", "--run", f"ARMO={run}", "--attach-target", str(truth),
          "--only", "spectra", "rain_members", "stills", "tp_dist", "--out", str(out)])
    by = {p["page"]: p for p in json.loads((out / "suite_manifest.json").read_text())["pages"]}
    failed = {k: v["note"] + v.get("trace", "") for k, v in by.items() if not v["files"]}
    assert not failed, failed
    assert "wettest input" in by["04"]["note"]          # target tp is NaN, so the pick falls back to the input
    with xr.open_dataset(run / f"predictions_{DATES[1]}_step072.nc") as ds:
        assert "y" in ds and bool(np.isnan(ds["y"].values).all())


def test_areas_page_in_real_resolution_cells(inputs):
    """Opt-in grid cells (--cells): page 05 still draws."""
    root, runs, *_ , cases = inputs
    out = root / "out_cells"
    label = next(iter(runs))
    main(["suite", "--cases", str(cases), "--out", str(out), "--run", f"{label}={runs[label]}", "--only", "areas",
          "--cells"])
    rec = json.loads((out / "suite_manifest.json").read_text())
    assert rec["pages"][0]["files"], rec["pages"][0]
    assert (out / "05_areas_member1.png").exists()


def test_zoom_levels_never_tighter_than_close():
    from eval.suite.simple_maps import box_around, widen_to_close, width_km

    close = 1000.0
    tight = [9.0, 12.0, 44.0, 46.0]                      # ~230 km wide: tighter than the close level
    ext, widened = widen_to_close(tight, close)
    assert widened and width_km(ext) >= close - 1 and ext[0] <= 9.0 and ext[1] >= 12.0
    wide = box_around(10.0, 45.0, 4000.0)
    assert widen_to_close(wide, close) == (wide, False)
    assert abs(width_km(box_around(-60.0, 20.0, 1000.0)) - 1000.0) < 1


def test_spacing_of_octahedral_grids():
    from eval.suite.simple_maps import spacing_km
    from eval.tests.synthetic_predictions import octahedral_grid

    lat, lon = octahedral_grid(320)
    k = (lat > 30) & (lat < 60) & (lon > 0) & (lon < 40)
    assert 30 < spacing_km(lat[k], lon[k]) < 40          # O320 ~ 35 km


def test_tc_case_from_ibtracs_csv(inputs):
    """No tc_structure read: page 03 picks the deepest IBTrACS fix at the case step (2026 months, no event yaml)."""
    root, runs, *_, cases = inputs
    csv = root / "ibtracs.csv"
    csv.write_text("SID,SEASON,NAME,ISO_TIME,LAT,LON,WMO_PRES,USA_PRES\n ,Year, , ,deg,deg,mb,mb\n"
                   "2025A,2025,DOLPHIN,2025-09-26 00:00:00,25.0,-60.0,945,\n"
                   "2025B,2025,LALA,2025-09-26 00:00:00,15.0,130.0,,990\n"
                   "2025A,2025,DOLPHIN,2025-09-27 06:00:00,26.0,-61.0,930,\n")
    out = root / "out_ib"
    label = next(iter(runs))
    main(["suite", "--cases", str(cases), "--out", str(out), "--run", f"{label}={runs[label]}",
          "--ibtracs-csv", str(csv), "--only", "tc_members"])
    p = json.loads((out / "suite_manifest.json").read_text())["pages"][0]
    assert p["files"], p
    assert "945 hPa" in p["note"] and "dolphin" in p["note"]


def test_kernel_density_integrates_to_one_and_the_panel_width_comes_from_the_smallest_sample():
    import numpy as np

    from eval.suite.pages import common_kernel_width, kernel_density

    rng = np.random.default_rng(0)
    small, large = rng.normal(960.0, 15.0, 109), rng.normal(965.0, 12.0, 1090)
    width = common_kernel_width([large, small])
    assert width == 1.06 * np.std(small) * 109 ** -0.2
    grid = np.linspace(850.0, 1050.0, 2001)
    for v in (small, large):
        dens = kernel_density(v, grid, width)
        assert abs(float(np.sum(dens) * (grid[1] - grid[0])) - 1.0) < 1e-3
    assert common_kernel_width([np.array([1.0])]) == 1.0
def test_regional_grid_names():
    """A regional box names its real grids (4 km lane: O1280 9 km input, O2560 4.4 km output), not O320/O1280."""
    from eval.evaluators.zoom_maps.core.plot_member_wind_maps import grid_from_lats, panel_titles

    rows = lambda n: 90.0 / n * np.arange(200) + 10.0   # noqa: E731
    assert grid_from_lats(rows(2560)) == "O2560" and grid_from_lats(rows(1280)) == "O1280"
    titles, hres = panel_titles({"enfo": "Target ({hres})", "eefo": "Input ({lres})"}, 50_000, 200_000,
                                rows(1280), rows(2560))
    assert titles["enfo"] == "Target (O2560, 4.4 km)" and titles["eefo"] == "Input (O1280, 9 km)"


def test_target_limit_mark(inputs, tmp_path):
    """Spectra pages mark the target limit: T = N - 1 of the target grid (O96 synthetic -> T95)."""
    import json as _json

    from eval.evaluators.box_spectra.plotter import target_limit_marks

    root, runs, *_ = inputs
    f = sorted(next(iter(runs.values())).glob("predictions_*.nc"))[0]
    (tmp_path / "run_meta.json").write_text(_json.dumps({"files": [str(f)]}))
    (wvn, label), = target_limit_marks(tmp_path)
    assert wvn == 95.0 and label.startswith("Target limit (T95")
