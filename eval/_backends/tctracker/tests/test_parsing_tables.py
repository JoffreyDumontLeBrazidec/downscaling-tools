from __future__ import annotations

import tarfile
from types import SimpleNamespace

from eval._backends.tctracker.parsing import (
    BASINS,
    parse_basin_text,
    parse_tar,
    records_from_tracks,
    step_hours,
)
from eval._backends.tctracker.pipeline import TCTrackerConfig
from eval._backends.tctracker.tables import parse_and_write, parse_run_root

# Real j761 wnp output shape (verified 2026-08-18 on tar
# j761_20230826_00_m001_o320_tracks.tar).
WNP_TEXT = """00010 26/08/2023 M= 3  1 SNBR=   1
00020 2023/08/26/00*1841229  50  987*1881230*00064000600003600043*00025000000000000000*00000000000000000000*
00030 2023/08/27/00*1761240  37  989*1651239*00000000630008700000*00000000000000000000*00000000000000000000*
00040 2023/08/28/00*1781251  46  981*1711253*00093001020008000057*00000000000000000000*00000000000000000000*
00045 TS
"""

SIN_TEXT = """00010 26/08/2023 M= 2  1 SNBR=   1
00020 2023/08/26/00*1520653  40  992*1520653*00000000000000000000*
00030 2023/08/27/00*1620641  45  988*1620641*00000000000000000000*
00035 TS
"""

EMPTY_TEXT = "00000 00/00/0000 M= 0  0 SNBR=   0\n"


def test_parse_basin_text_wnp_matches_atl_grammar():
    tracks = parse_basin_text(WNP_TEXT, "wnp", "src")
    assert len(tracks) == 1
    t = tracks[0]
    assert t["basin"] == "wnp"
    assert t["classification"] == "TS"
    assert t["mslp_min_hpa"] == 981
    assert t["records"][0]["lat"] == 18.4
    assert t["records"][0]["lon_e"] == 122.9
    assert t["mslp_min_lat"] == 17.8
    assert t["genesis_lat"] == 18.4


def test_parse_basin_text_southern_hemisphere_lat_sign():
    tracks = parse_basin_text(SIN_TEXT, "sin", "src")
    assert len(tracks) == 1
    assert tracks[0]["records"][0]["lat"] == -15.2
    assert tracks[0]["mslp_min_lat"] == -16.2


def test_parse_basin_text_empty_header_yields_no_tracks():
    assert parse_basin_text(EMPTY_TEXT, "sin", "src") == []


def test_step_hours():
    assert step_hours("20230826", "00", "2023/08/26/00") == 0
    assert step_hours("20230826", "00", "2023/08/28/12") == 60


def test_records_from_tracks_step_and_track_id():
    tracks = parse_basin_text(WNP_TEXT, "wnp", "src")
    rows = records_from_tracks(tracks, init_date="20230826", time="00", member=3)
    assert len(rows) == 3
    assert rows[0]["track_id"] == "wnp-001"
    assert [r["step_h"] for r in rows] == [0, 24, 48]
    assert rows[0]["member"] == 3


def _write_multi_basin_tar(path, texts: dict[str, str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as tf:
        for basin in BASINS:
            file_path = path.parent / f"X2023082600_PF_000360_{basin}"
            file_path.write_text(texts.get(basin, EMPTY_TEXT), encoding="utf-8")
            tf.add(file_path, arcname=file_path.name)
            file_path.unlink()


def test_parse_tar_all_basins(tmp_path):
    tar = tmp_path / "t.tar"
    _write_multi_basin_tar(tar, {"wnp": WNP_TEXT, "sin": SIN_TEXT})
    tracks = parse_tar(tar, source="t")
    assert {t["basin"] for t in tracks} == {"wnp", "sin"}


def _config(tmp_path, dates=("20230826",), members=(1,)):
    return TCTrackerConfig(
        lane="o96_o320_unified_full", expver="j761", output_dir=tmp_path,
        dates=dates, members=members,
    )


def test_parse_and_write_tables_and_provenance(tmp_path):
    cfg = _config(tmp_path, dates=("20230826", "20230827"), members=(1,))
    # only the first (date, member) tar exists -> completeness 0.5
    _write_multi_basin_tar(cfg.targets[0].tar_path, {"wnp": WNP_TEXT})
    parsed = parse_and_write(cfg, role="model", source_id="j761")
    assert (parsed / "tracks.csv").exists()
    assert (parsed / "track_summary.csv").exists()
    import json
    prov = json.loads((parsed / "provenance.json").read_text())
    assert prov["role"] == "model"
    assert prov["completeness"] == 0.5
    assert prov["grid_support"] == "o320"
    import pandas as pd
    summary = pd.read_csv(parsed / "track_summary.csv")
    assert len(summary) == 1
    assert summary.iloc[0]["mslp_min_hpa"] == 981
    forecasts = pd.read_csv(parsed / "forecasts.csv")
    assert len(forecasts) == 2 and forecasts["present"].sum() == 1


def test_parse_run_root_from_filenames(tmp_path):
    tar = tmp_path / "tars" / "od_enfo_0001_20250905_00_m002_o320_tracks.tar"
    _write_multi_basin_tar(tar, {"wnp": WNP_TEXT})
    parsed = parse_run_root(tmp_path, role="target", source_id="od_enfo_0001")
    import json
    prov = json.loads((parsed / "provenance.json").read_text())
    assert prov["grid_support"] == "o320"
    assert prov["members"] == [2]
    assert prov["dates_present"] == ["20250905"]
    assert prov["completeness"] is None
