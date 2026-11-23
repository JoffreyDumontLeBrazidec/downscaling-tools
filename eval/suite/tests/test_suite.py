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
spectra: {steps: [24, 72], box: null}
tc: {name: storm, date: auto, step: 24, variable: wind10m}
rain: {name: rain, date: auto, step: 24, extent: [-40.0, 20.0, 30.0, 65.0], variable: tp}
areas:
  - {name: alps, date: auto, step: 24, extent: [4.0, 16.0, 43.0, 49.0], variable: tp}
  - {name: atl, date: "20250926", step: 72, extent: [-40.0, -20.0, 35.0, 55.0], variable: msl}
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
    argv = ["suite", "--cases", str(cases), "--out", str(out), "--tc-cache", str(cache), "--smooth"]
    for label in runs:
        argv += ["--run", f"{label}={runs[label]}", "--obs", f"{label}={obs[label]}", "--tc", f"{label}={tc[label]}"]
    main(argv)
    rec = json.loads((out / "suite_manifest.json").read_text())
    by = {p["page"]: p for p in rec["pages"]}
    failed = {k: v["note"] + v.get("trace", "") for k, v in by.items() if not v["files"]}
    assert not failed, failed
    assert sorted(by) == ["01", "02", "03", "04", "05", "06", "07", "08", "09"]
    for name in ("01_spectra_L024.png", "02_spectra_L072.png", "06_scores_stations.png", "07_tp_stills.png",
                 "05_areas_member1.png", "08_tp_distribution.png", "09_tc_distribution.png"):
        assert (out / name).exists(), name
    assert list(out.glob("03_tc_members_storm_wind10m_*_L024.png"))
    assert list(out.glob("04_rain_members_rain_tp_*_L024.png"))
    assert "pages 10 and 11" in (out / "SUITE.md").read_text().lower()
    assert "deepest target storm" in by["03"]["note"]
    assert "wettest target" in by["04"]["note"]


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
    main(["suite", "--cases", "o1280_global", "--run", f"ARMO={run}", "--attach-target", str(truth), "--smooth",
          "--only", "spectra", "rain_members", "stills", "tp_dist", "--out", str(out)])
    by = {p["page"]: p for p in json.loads((out / "suite_manifest.json").read_text())["pages"]}
    failed = {k: v["note"] + v.get("trace", "") for k, v in by.items() if not v["files"]}
    assert not failed, failed
    assert "wettest input" in by["04"]["note"]          # target tp is NaN, so the pick falls back to the input
    with xr.open_dataset(run / f"predictions_{DATES[1]}_step072.nc") as ds:
        assert "y" in ds and bool(np.isnan(ds["y"].values).all())


def test_areas_page_in_real_resolution_cells(inputs):
    """The default map mode (owner 23 Nov 09:14, 'real resolution'): grid cells, one model, page 05."""
    root, runs, *_ , cases = inputs
    out = root / "out_cells"
    label = next(iter(runs))
    main(["suite", "--cases", str(cases), "--out", str(out), "--run", f"{label}={runs[label]}", "--only", "areas"])
    rec = json.loads((out / "suite_manifest.json").read_text())
    assert rec["pages"][0]["files"], rec["pages"][0]
    assert (out / "05_areas_member1.png").exists()


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
    main(["suite", "--cases", str(cases), "--out", str(out), "--run", f"{label}={runs[label]}", "--smooth",
          "--ibtracs-csv", str(csv), "--only", "tc_members"])
    p = json.loads((out / "suite_manifest.json").read_text())["pages"][0]
    assert p["files"], p
    assert "945 hPa" in p["note"] and "dolphin" in p["note"]
