from __future__ import annotations

import tarfile
from types import SimpleNamespace

from eval._backends.tctracker.pipeline import (
    BASINS,
    KT_TO_MS,
    TCTrackerConfig,
    build_config,
    build_tctracker_command,
    parse_atlantic_tracks,
    verify_outputs,
    write_atlantic_summary,
)


def _config(tmp_path):
    return TCTrackerConfig(
        lane="o96_o320_unified_full",
        expver="j761",
        output_dir=tmp_path,
        dates=("20230826",),
        members=(1,),
    )


def _write_tar(path, atl_text=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as tf:
        for basin in BASINS:
            file_path = path.parent / f"J7612023082600_PF_000360_{basin}"
            file_path.write_text(atl_text if basin == "atl" else "", encoding="utf-8")
            tf.add(file_path, arcname=file_path.name)
            file_path.unlink()


def test_build_config_uses_lane_tctracker_defaults(tmp_path):
    args = SimpleNamespace(
        lane="o96_o320_unified_full", expver="j761", output_dir=str(tmp_path),
        dates=None, members=None, time=None, start_step=None, end_step=None,
        step_interval=None, grid=None, fdb_class=None, fdb_type=None, stream=None,
        vorticity=None, model_keyword=None, module=None, overwrite=False,
    )
    lane_config = {
        "predict": {"dates": ["bad"], "members": [99]},
        "tctracker": {"dates": ["20230826"], "members": [1], "grid": 320, "vorticity": False},
    }
    cfg = build_config(args, lane_config, {"scratch_root": str(tmp_path)}, tmp_path)
    assert cfg.dates == ("20230826",)
    assert cfg.members == (1,)
    assert cfg.grid == 320
    assert cfg.vorticity is False


def test_build_tctracker_command_matches_proven_j761_shape(tmp_path):
    cfg = _config(tmp_path)
    target = cfg.targets[0]
    cmd = build_tctracker_command(cfg, target)
    assert cmd[:3] == ["tctracker", "-v", "false"]
    assert ["-C", "rd"] == cmd[cmd.index("-C"):cmd.index("-C") + 2]
    assert ["-T", "pf"] == cmd[cmd.index("-T"):cmd.index("-T") + 2]
    assert ["-S", "enfo"] == cmd[cmd.index("-S"):cmd.index("-S") + 2]
    assert ["-E", "j761"] == cmd[cmd.index("-E"):cmd.index("-E") + 2]
    assert ["-N", "1"] == cmd[cmd.index("-N"):cmd.index("-N") + 2]
    assert ["-d", "20230826"] == cmd[cmd.index("-d"):cmd.index("-d") + 2]
    assert str(target.tar_path) in cmd


def test_verify_outputs_accepts_synthetic_complete_tar(tmp_path):
    cfg = _config(tmp_path)
    target = cfg.targets[0]
    cfg.manifests_dir.mkdir(parents=True)
    _write_tar(target.tar_path)
    target.status_path.write_text("status=done\nrc=0\n", encoding="utf-8")
    target.contents_path.write_text("\n".join(f"x_{b}" for b in BASINS) + "\n", encoding="utf-8")
    target.sha256_path.write_text("abc  tar\n", encoding="utf-8")
    (cfg.manifests_dir / "batch_manifest.csv").write_text(
        "tag,date,member,status,rc,tar,bytes,sha256\nx,20230826,1,done,0,t,1,abc\n",
        encoding="utf-8",
    )

    result = verify_outputs(cfg)

    assert result["tar_count"] == 1
    assert result["manifest_rows"] == 1
    assert result["issues"] == []


def test_parse_atlantic_tracks_extracts_extremes_and_units(tmp_path):
    cfg = _config(tmp_path)
    target = cfg.targets[0]
    atl = """00010 26/08/2023 M= 2  1 SNBR=   1
00020 2023/08/26/00*1732753  16 1008*2132759*00000000000000000000*
00030 2023/08/27/00*2122730  68  965*2272723*00000000000000000000*
00033 HR1
"""
    _write_tar(target.tar_path, atl)

    tracks = parse_atlantic_tracks(cfg)

    assert len(tracks) == 1
    track = tracks[0]
    assert track["wind_max_kt"] == 68
    assert track["wind_max_ms"] == 68 * KT_TO_MS
    assert track["mslp_min_hpa"] == 965
    assert track["classification"] == "HR1"
    assert track["member"] == 1

    md_path, json_path = write_atlantic_summary(cfg, tracks)
    assert md_path.exists()
    assert json_path.exists()
    assert "Track count: 1" in md_path.read_text()
