"""tc_obs on a synthetic cases.csv and reference cache: best-track interpolation, case scores against their
definitions, run() tables, plot() from the saved files, the NOAA CSV reader and the Atos-only fetch."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from eval.evaluators import tc_obs
from eval.evaluators.tc_obs.core.reference import best_track_at, read_noaa_csv
from eval.evaluators.tc_obs.runner import case_scores, gc_km


def _bt():
    rows = []
    for h in range(0, 97, 6):
        t = pd.Timestamp("2023-08-29") + pd.Timedelta(hours=h)
        rows.append({"event": "franklin", "storm_id": "4648", "time": t.strftime("%Y%m%d%H"), "lat": 25 + 0.1 * h,
                     "lon": -70.0 + 0.05 * h, "pmin_hpa": 980.0 - 0.2 * h, "wind_ms": np.nan, "source": "ibtracs_zarr"})
    return pd.DataFrame(rows)


def test_best_track_interpolates_between_fixes():
    bt = _bt()
    b = best_track_at(bt, "Franklin", pd.Timestamp("2023-08-30 03:00").to_pydatetime())
    assert b["interpolated"] and b["lat"] == pytest.approx(25 + 0.1 * 27) and b["pmin_hpa"] == pytest.approx(980 - 0.2 * 27)
    assert not best_track_at(bt, "franklin", pd.Timestamp("2023-08-30 00:00").to_pydatetime())["interpolated"]
    assert best_track_at(bt, "franklin", pd.Timestamp("2023-09-10").to_pydatetime()) is None
    assert best_track_at(bt, "lee", pd.Timestamp("2023-08-30").to_pydatetime()) is None


def test_case_scores_definitions():
    bt = {"lat": 30.0, "lon": -70.0, "pmin_hpa": 960.0, "wind_ms": np.nan}
    p = np.array([950.0, 955.0, 965.0, 970.0])
    lat, lon = np.array([30.0, 30.5, 29.5, 30.0]), np.array([-70.0, -70.0, -70.0, -69.0])
    s = case_scores(p, lat, lon, bt)
    brute = np.abs(p - 960).mean() - np.abs(p[:, None] - p[None]).sum() / (2 * 4 * 3)
    assert s["pmin_crps"] == pytest.approx(brute)
    assert s["pmin_err"] == pytest.approx(p.mean() - 960) and s["n_found"] == 4
    assert s["pos_err_member_km"] == pytest.approx(gc_km(lat, lon, 30, -70).mean())
    one = case_scores(p[:1], lat[:1], lon[:1], bt)
    assert one["pmin_crps"] == pytest.approx(10.0) and one["pos_spread_km"] == 0.0 and one["pos_err_member_km"] < 1e-6


def _cases(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for lead, ok in ((24, 1), (48, 1), (72, 0)):
        t = pd.Timestamp("2023-08-29") + pd.Timedelta(hours=lead)
        for field in ("model", "truth", "input"):
            for m in range(5):
                off = {"model": rng.normal(0, 0.3), "truth": 0.2, "input": 0.8}[field]
                rows.append({"arm": "X", "event": "franklin", "date": "20230829", "lead": lead, "field": field,
                             "member": m, "storm_ok": ok, "found": 1, "pmin_hpa": 980 - 0.2 * lead + 5 * off,
                             "centre_lat": 25 + 0.1 * lead + off, "centre_lon": -70 + 0.05 * lead,
                             "maxwind300_ms": 30.0 + off})
    p = tmp_path / "X" / "cases.csv"
    p.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(p, index=False)
    cache = tmp_path / "cache"
    cache.mkdir()
    _bt().to_csv(cache / "best_track.csv", index=False)
    for stream, members in (("oper", [52]), ("enfo", list(range(1, 53)))):
        tr = []
        for m in members:
            for step in (0, 24, 48, 72):
                tr.append({"storm": "08L", "name": "FRANKLIN", "member": m, "step_h": step, "lat": 25 + 0.1 * step + 0.1,
                           "lon": -70 + 0.05 * step, "msl_hpa": 975 - 0.2 * step + (m % 5), "wind_ms": 30.0})
        pd.DataFrame(tr).to_csv(cache / f"tracker_{stream}_20230829.csv", index=False)
    (cache / "tc_meta.json").write_text(json.dumps({"cache_version": 1}))
    return p, cache


def test_run_then_plot(tmp_path):
    cases, cache = _cases(tmp_path)
    cfg = {"cache": [str(cache)], "cases": str(tmp_path / "{run_label}" / "cases.csv")}
    out = tc_obs.run(tmp_path, {}, cfg, output_dir=tmp_path / "out", run_label="X")
    df = pd.read_csv(out / "cases_obs.csv", dtype={"date": str})
    assert set(df.lead) == {24, 48}  # lead 72 is not a storm case
    assert set(df.row) == {"model", "truth", "input", "hres", "ens"}
    ens = df[df.row == "ens"]
    assert (ens.n_found == 51).all() and (ens.members_total == 51).all()  # member 52 (HRES) is not in the ENS row
    assert (df[df.kind == "single field"].pos_spread_km == 0).all()
    assert (df[df.row.isin(["model", "ens"])].found_frac == 1).all()
    assert "found_frac" in pd.read_csv(out / "summary.csv").columns
    truth = df[(df.row == "truth") & (df.lead == 24)].iloc[0]
    assert truth.pmin_crps == pytest.approx(1.0)  # 5 x 0.2 hPa above the best track
    meta = json.loads((out / "run_meta.json").read_text())
    assert meta["n_cases"] == 2 and meta["n_dates"] == 1 and "not a verdict" in meta["note"]
    assert "not a verdict" in (out / "summary.md").read_text()
    pr = pd.read_csv(out / "paired.csv", dtype={"lead": str})
    assert {"model-truth", "model-ens", "model-hres", "model-input"} <= set(pr.pair)
    names = {p.name for p in map(__import__("pathlib").Path, tc_obs.plot(out, {}, cfg))}
    assert {"tc_obs_by_lead.png", "tc_obs_cases.png"} <= names


def test_noaa_csv_reader(tmp_path):
    p = tmp_path / "ib.csv"
    p.write_text("SID,SEASON,NUMBER,BASIN,SUBBASIN,NAME,ISO_TIME,NATURE,LAT,LON,WMO_WIND,WMO_PRES,USA_WIND,USA_PRES\n"
                 " ,Year, , , , , , ,degrees_north,degrees_east,kts,mb,kts,mb\n"
                 "2023X,2023,1,NA,MM,FRANKLIN,2023-08-30 00:00:00,TS,30.0,-70.0, , ,100,950\n"
                 "2023Y,2023,2,NA,MM,LEE,2023-09-10 00:00:00,TS,20.0,290.0,80,960, , \n")
    x = read_noaa_csv(p, ["franklin", "lee"], 2023)
    assert list(x.event) == ["franklin", "lee"]
    assert x.wind_ms.iloc[0] == pytest.approx(100 * 0.514444 * 0.93) and x.pmin_hpa.iloc[1] == 960
    assert x.wind_ms.iloc[1] == pytest.approx(80 * 0.514444)  # WMO_WIND only where USA_WIND is missing
    assert list(x.wind_src) == ["usa_wind_1min_x0.93", "wmo_wind"]
    assert x.lon.iloc[1] == pytest.approx(-70.0) and x.time.iloc[0] == "202308300000"


def test_fetch_tc_refuses_off_atos(monkeypatch, tmp_path):
    import socket

    from eval.cli import main
    monkeypatch.setattr(socket, "gethostname", lambda: "jrlogin05")
    with pytest.raises(SystemExit, match="Atos only"):
        main(["obs", "fetch-tc", "--storms", "franklin=4648", "--dates", "20230829", "--out", str(tmp_path / "c")])


def test_hres_without_member_number_and_missing_hres_warns(tmp_path):
    cases, cache = _cases(tmp_path)
    tr = pd.read_csv(cache / "tracker_oper_20230829.csv")
    tr["member"] = -1  # a message without ensembleMemberNumber decodes as -1
    tr.to_csv(cache / "tracker_oper_20230829.csv", index=False)
    cfg = {"cache": str(cache), "cases": str(cases)}
    out = tc_obs.run(tmp_path, {}, cfg, output_dir=tmp_path / "out", run_label="X")
    df = pd.read_csv(out / "cases_obs.csv")
    assert (df.row == "hres").sum() == 2
    tr[tr.step_h != 48].to_csv(cache / "tracker_oper_20230829.csv", index=False)
    out = tc_obs.run(tmp_path, {}, cfg, output_dir=tmp_path / "out2", run_label="X")
    log = json.loads((out / "run_meta.json").read_text())["log"]
    assert any(x.startswith("WARN") and "HRES" in x and "+48 h" in x for x in log)


SAMPLE = __import__("pathlib").Path(__file__).parent / "data" / "ibtracs_v04r01_sample.csv"


def test_noaa_csv_reader_on_real_lines():
    """Seven verbatim lines of NOAA NCEI IBTrACS v04r01 (column names, units row, Franklin 2023 at 18, 21 and
    00 UTC, Lee 2023, one provisional track): units row skipped, ' ' read as missing, main tracks only."""
    x = read_noaa_csv(SAMPLE, ["franklin", "lee"], 2023)
    assert list(x.event) == ["franklin"] * 3 + ["lee"]
    f = x[x.time == "202308290000"].iloc[0]
    assert (f.lat, f.lon, f.pmin_hpa) == (28.9, -71.1, 926.0) and f.storm_id == "2023232N13300"
    assert f.wind_ms == pytest.approx(130 * 0.93 * 0.514444) and f.rmw_km == pytest.approx(5 * 1.852)
    three_hourly = x[x.time == "202308282100"].iloc[0]  # WMO_* blank, USA_* filled
    assert three_hourly.wind_src == "usa_wind_1min_x0.93" and three_hourly.wind_ms == pytest.approx(128 * 0.93 * 0.514444)
    assert read_noaa_csv(SAMPLE, ["unnamed"], 2025).empty  # PROVISIONAL track left out
    b = best_track_at(x, "franklin", pd.Timestamp("2023-08-28 22:30").to_pydatetime())
    assert b["interpolated"] and b["pmin_hpa"] == pytest.approx((929 + 926) / 2) and b["rmw_km"] == pytest.approx(6.5 * 1.852)


def test_noaa_csv_from_the_lane_gives_wind_and_rmw_scores(tmp_path):
    cases, cache = _cases(tmp_path)
    c = pd.read_csv(cases, dtype={"date": str})
    c = c[c.lead == 24].assign(date="20230828", rmw_km=12.0)  # valid 2023-08-29 00 UTC: an exact NOAA fix
    c.to_csv(cases, index=False)
    for stream in ("oper", "enfo"):
        tr = pd.read_csv(cache / f"tracker_{stream}_20230829.csv")
        tr.to_csv(cache / f"tracker_{stream}_20230828.csv", index=False)
    cfg = {"cache": str(cache), "cases": str(cases), "noaa_csv": {"jupiter": str(tmp_path / "none.csv"),
                                                                  "atos": str(SAMPLE)}}
    out = tc_obs.run(tmp_path, {}, cfg, output_dir=tmp_path / "out", run_label="X")
    df = pd.read_csv(out / "cases_obs.csv", dtype={"date": str})
    assert (df.bt_source == "noaa_ibtracs_csv").all() and (df.bt_pmin_hpa == 926).all()
    truth = df[df.row == "truth"].iloc[0]
    assert truth.wind_crps == pytest.approx(abs(30.2 - 130 * 0.93 * 0.514444), rel=1e-6)  # maxwind300 = 30 + 0.2
    assert truth.rmw_err == pytest.approx(12.0 - 5 * 1.852)
    assert {"wind_crps", "rmw_crps"} <= set(pd.read_csv(out / "paired.csv").metric)
    md = (out / "summary.md").read_text()
    assert "Maximum 10 m wind" in md and "Radius of maximum wind" in md and "10 km bins" in md
    tc_obs.plot(out, {}, cfg)


def _edited_sample(tmp_path, edits):
    """The real sample with some cells changed: edits = {(line index, column): value}."""
    import csv

    with open(SAMPLE, newline="") as f:
        rows = list(csv.reader(f))
    col = {k: i for i, k in enumerate(rows[0])}
    for (i, c), v in edits.items():
        rows[i][col[c]] = v
    p = tmp_path / "edited.csv"
    with open(p, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    return p


def test_noaa_csv_hurdat_wmo_wind_is_one_minute(tmp_path):
    # line 2 = Franklin 2023-08-28 18:00 (WMO_AGENCY hurdat_atl, WMO_WIND 125): USA_WIND blanked
    p = _edited_sample(tmp_path, {(2, "USA_WIND"): " "})
    x = read_noaa_csv(p, ["franklin"], 2023).set_index("time")
    assert x.loc["202308281800", "wind_src"] == "wmo_wind_hurdat_1min_x0.93"
    assert x.loc["202308281800", "wind_ms"] == pytest.approx(125 * 0.93 * 0.514444)
    p = _edited_sample(tmp_path, {(2, "USA_WIND"): " ", (2, "WMO_AGENCY"): "tokyo"})
    x = read_noaa_csv(p, ["franklin"], 2023).set_index("time")
    assert x.loc["202308281800", "wind_src"] == "wmo_wind" and x.loc["202308281800", "wind_ms"] == pytest.approx(125 * 0.514444)


def test_noaa_csv_time_window_instead_of_season(tmp_path):
    p = _edited_sample(tmp_path, {(5, "SEASON"): "2024"})  # Lee labelled with another season, as for SH storms
    assert read_noaa_csv(p, ["lee"], 2023).empty
    x = read_noaa_csv(p, ["lee", "franklin"], window=("2023-09-09", "2023-09-11"))
    assert list(x.event) == ["lee"] and x.time.iloc[0] == "202309100000"


def test_blank_noaa_pressure_falls_back_to_the_zarr(tmp_path):
    x = read_noaa_csv(_edited_sample(tmp_path, {(4, "USA_PRES"): " ", (4, "WMO_PRES"): " "}), ["franklin"], 2023)
    zarr = pd.DataFrame([{"event": "franklin", "storm_id": "4648", "time": "2023082900", "lat": 29.0, "lon": -71.0,
                          "pmin_hpa": 927.0, "wind_ms": np.nan, "source": "ibtracs_zarr"}])
    b = best_track_at(pd.concat([x, zarr], ignore_index=True), "franklin", pd.Timestamp("2023-08-29").to_pydatetime())
    assert b["source"] == "ibtracs_zarr" and b["pmin_hpa"] == 927.0 and b["noaa_incomplete"]
    b = best_track_at(x, "franklin", pd.Timestamp("2023-08-28 21:00").to_pydatetime())
    assert b["source"] == "noaa_ibtracs_csv" and not b["noaa_incomplete"]


def test_ens_found_fraction_counts_members_that_lost_the_storm(tmp_path):
    cases, cache = _cases(tmp_path)
    tr = pd.read_csv(cache / "tracker_enfo_20230829.csv")
    lost = tr.member.between(40, 51)
    other = tr[lost].assign(storm="09L", name="GERT")  # members 40-51 track another storm only
    pd.concat([tr[~lost], other]).to_csv(cache / "tracker_enfo_20230829.csv", index=False)
    out = tc_obs.run(tmp_path, {}, {"cache": str(cache), "cases": str(cases)}, output_dir=tmp_path / "out",
                     run_label="X")
    ens = pd.read_csv(out / "cases_obs.csv").query("row == 'ens'")
    assert (ens.members_total == 51).all() and (ens.n_found == 39).all()
    assert ens.found_frac.iloc[0] == pytest.approx(39 / 51)
