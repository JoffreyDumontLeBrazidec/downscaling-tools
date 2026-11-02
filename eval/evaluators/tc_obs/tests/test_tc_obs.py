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
    assert x.wind_ms.iloc[0] == pytest.approx(100 * 0.514444) and x.pmin_hpa.iloc[1] == 960
    assert x.lon.iloc[1] == pytest.approx(-70.0) and x.time.iloc[0] == "2023083000"


def test_fetch_tc_refuses_off_atos(monkeypatch, tmp_path):
    import socket

    from eval.cli import main
    monkeypatch.setattr(socket, "gethostname", lambda: "jrlogin05")
    with pytest.raises(SystemExit, match="Atos only"):
        main(["obs", "fetch-tc", "--storms", "franklin=4648", "--dates", "20230829", "--out", str(tmp_path / "c")])
