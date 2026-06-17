from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import xarray as xr

from eval.prepare import builder


def _lane_config() -> dict:
    return {
        "prepare": {
            "bundle_filename_tpl": "bundle_date{date}_mem{member:02d}_step{step:03d}.nc",
            "args": {
                "lres_sfc_grib": "{source_grib_root}/lres_sfc_{date}.grib",
                "lres_pl_grib": "{source_grib_root}/lres_pl_{date}.grib",
                "hres_grib": "{source_grib_root}/hres_{date}.grib",
                "target_sfc_grib": "{source_grib_root}/target_sfc_{date}.grib",
                "target_pl_grib": "{source_grib_root}/target_pl_{date}.grib",
                "lres_sfc_channels": "10u,10v",
            },
        },
        "predict": {
            "dates": ["20230826"],
            "steps": [24],
            "members": [1],
        },
    }


def _touch_required_gribs(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in [
        "lres_sfc_20230826.grib",
        "lres_pl_20230826.grib",
        "hres_20230826.grib",
        "target_sfc_20230826.grib",
        "target_pl_20230826.grib",
    ]:
        (root / name).write_bytes(b"grib")


def _write_truth_bundle(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        {"target_hres_10u": ("point_hres", [1.0])},
        attrs={"has_target_hres_fields": "yes"},
    ).to_netcdf(path)


def _write_truthless_bundle(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"in_lres_10u": ("point_lres", [1.0])}).to_netcdf(path)


def _bundle_path(bundle_dir: Path) -> Path:
    return bundle_dir / "bundle_date20230826_mem01_step024.nc"


def test_build_bundles_resolves_source_root_before_subprocess(tmp_path, monkeypatch):
    real_root = tmp_path / "real_gribs"
    _touch_required_gribs(real_root)
    link_root = tmp_path / "linked_gribs"
    link_root.symlink_to(real_root, target_is_directory=True)
    bundle_dir = tmp_path / "bundles"
    captured_cmds: list[list[str]] = []

    def fake_run(cmd, check):
        captured_cmds.append(cmd)
        out = Path(cmd[cmd.index("--out") + 1])
        assert out != _bundle_path(bundle_dir)
        assert out.name.startswith(f".{_bundle_path(bundle_dir).name}.tmp-")
        _write_truth_bundle(out)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(builder.subprocess, "run", fake_run)

    verification_path = tmp_path / "verification.json"
    builder.build_bundles(
        _lane_config(),
        bundle_dir,
        str(link_root),
        verification_path=verification_path,
    )

    assert captured_cmds
    cmd_text = "\n".join(captured_cmds[0])
    assert str(real_root.resolve()) in cmd_text
    assert str(link_root) not in cmd_text
    assert _bundle_path(bundle_dir).exists()
    payload = json.loads(verification_path.read_text())
    assert payload["source_grib_root_original"] == str(link_root)
    assert payload["source_grib_root_resolved"] == str(real_root.resolve())


def test_build_bundles_rejects_looping_source_root_before_subprocess(tmp_path, monkeypatch):
    loop_root = tmp_path / "loop"
    loop_root.symlink_to(loop_root)

    def fake_run(cmd, check):  # pragma: no cover - must not be reached
        raise AssertionError("subprocess should not run for an invalid source root")

    monkeypatch.setattr(builder.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="source_grib_root"):
        builder.build_bundles(_lane_config(), tmp_path / "bundles", str(loop_root))


def test_build_bundles_skips_existing_valid_truth_bundle(tmp_path, monkeypatch):
    grib_root = tmp_path / "gribs"
    _touch_required_gribs(grib_root)
    bundle_dir = tmp_path / "bundles"
    _write_truth_bundle(_bundle_path(bundle_dir))

    def fake_run(cmd, check):  # pragma: no cover - must not be reached
        raise AssertionError("valid truth bundle should be skipped")

    monkeypatch.setattr(builder.subprocess, "run", fake_run)

    builder.build_bundles(_lane_config(), bundle_dir, str(grib_root))


def test_build_bundles_rebuilds_existing_truthless_bundle(tmp_path, monkeypatch):
    grib_root = tmp_path / "gribs"
    _touch_required_gribs(grib_root)
    bundle_dir = tmp_path / "bundles"
    _write_truthless_bundle(_bundle_path(bundle_dir))
    calls = 0

    def fake_run(cmd, check):
        nonlocal calls
        calls += 1
        _write_truth_bundle(Path(cmd[cmd.index("--out") + 1]))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(builder.subprocess, "run", fake_run)

    builder.build_bundles(_lane_config(), bundle_dir, str(grib_root))

    assert calls == 1
    builder.validate_truth_bundle(_bundle_path(bundle_dir))


def test_stale_temp_bundle_does_not_count_as_complete(tmp_path, monkeypatch):
    grib_root = tmp_path / "gribs"
    _touch_required_gribs(grib_root)
    bundle_dir = tmp_path / "bundles"
    final_path = _bundle_path(bundle_dir)
    _write_truth_bundle(bundle_dir / f".{final_path.name}.tmp-stale")
    calls = 0

    def fake_run(cmd, check):
        nonlocal calls
        calls += 1
        _write_truth_bundle(Path(cmd[cmd.index("--out") + 1]))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(builder.subprocess, "run", fake_run)

    builder.build_bundles(_lane_config(), bundle_dir, str(grib_root))

    assert calls == 1
    assert final_path.exists()
