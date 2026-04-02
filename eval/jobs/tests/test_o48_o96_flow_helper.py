from __future__ import annotations

import os
import stat
import subprocess
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "eval/jobs/templates/submit_o48_o96_manual_eval_flow.sh"


def _write_executable(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _make_fake_profile(path: Path, *, checkpoint_path: str, host_family: str) -> Path:
    script = textwrap.dedent(
        f"""\
        #!/usr/bin/env python3
        import json

        payload = {{
            "checkpoint_path": {checkpoint_path!r},
            "stack_flavor": "new",
            "lane": "o48_o96",
            "host_family": {host_family!r},
            "recommended_venv": "/tmp/fake_venv/bin/activate",
        }}
        print(json.dumps(payload))
        """
    )
    return _write_executable(path, script)


def _make_fake_hostname(path: Path, host_short: str) -> Path:
    script = textwrap.dedent(
        f"""\
        #!/bin/sh
        if [ "$1" = "-s" ]; then
          printf '%s\\n' {host_short!r}
        else
          printf '%s\\n' {host_short!r}
        fi
        """
    )
    return _write_executable(path, script)


def _touch_required_source_gribs(root: Path, dates: list[str]) -> None:
    for date in dates:
        for name in (
            f"enfo_o48_0001_date{date}_time0000_mem1to10_step24to120_sfc.grib",
            f"enfo_o48_0001_date{date}_time0000_mem1to10_step24to120_pl.grib",
            f"enfo_o96_0001_date{date}_time0000_step24to120_sfc.grib",
            f"iekm_o96_iekm_date{date}_time0000_step24to120_sfc_y.grib",
            f"iekm_o96_iekm_date{date}_time0000_step24to120_pl_y.grib",
        ):
            (root / name).write_text("stub\n", encoding="utf-8")


def _run_helper(
    tmp_path: Path,
    *,
    phase: str,
    run_id_override: str = "",
    host_family: str = "ac",
    host_short: str = "ac6-test",
    populate_source_gribs: bool = True,
) -> subprocess.CompletedProcess[str]:
    source_root = tmp_path / "source"
    source_root.mkdir(parents=True, exist_ok=True)
    submit_root = tmp_path / "submit"
    output_root = tmp_path / "output"
    submit_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    fake_checkpoint_path = "/tmp/checkpoints/95a07500d80247dbbd4b143d78db805d/inference-anemoi-by_step-epoch_013-step_100000.ckpt"
    fake_profile = _make_fake_profile(
        bin_dir / "fake_profile.py",
        checkpoint_path=fake_checkpoint_path,
        host_family=host_family,
    )
    _make_fake_hostname(bin_dir / "hostname", host_short)

    if populate_source_gribs:
        _touch_required_source_gribs(source_root, ["20250926", "20250927", "20250928", "20250929", "20250930"])

    if run_id_override:
        (output_root / run_id_override).mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "CHECKPOINT_PATH": "/tmp/fake.ckpt",
            "SOURCE_HPC": host_family,
            "SOURCE_GRIB_ROOT": str(source_root),
            "OUTPUT_ROOT": str(output_root),
            "SUBMIT_ROOT": str(submit_root),
            "PROFILE_PYTHON": str(fake_profile),
            "NO_SUBMIT": "1",
            "ALLOW_OVERWRITE": "1",
            "RUN_DATE_UTC": "20260330",
            "RUN_SUFFIX": "manual_eval",
            "PHASE": phase,
        }
    )
    if run_id_override:
        env["RUN_ID_OVERRIDE"] = run_id_override
    env["PATH"] = f"{bin_dir}:{env['PATH']}"

    return subprocess.run(
        ["bash", str(HELPER)],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_o48_o96_helper_proxy_render_uses_truth_bundle_stage(tmp_path: Path):
    result = _run_helper(tmp_path, phase="proxy")
    assert result.returncode == 0, result.stderr

    run_id = "manual_95a07500_new_o48_o96_20260330_manual_eval"
    submit_dir = tmp_path / "submit" / "20260330"
    build_text = (submit_dir / f"{run_id}_build_truth_bundles.sbatch").read_text(encoding="utf-8")
    predict_text = (submit_dir / f"{run_id}_predict.sbatch").read_text(encoding="utf-8")
    sigma_text = (submit_dir / f"{run_id}_sigma_eval.sbatch").read_text(encoding="utf-8")
    loss_text = (submit_dir / f"{run_id}_training_loss.sbatch").read_text(encoding="utf-8")
    regional_text = (submit_dir / f"{run_id}_regions_step024.sbatch").read_text(encoding="utf-8")
    storm_text = (submit_dir / f"{run_id}_storms_step024.sbatch").read_text(encoding="utf-8")
    spectra_text = (submit_dir / f"{run_id}_spectra.sbatch").read_text(encoding="utf-8")

    assert 'BUNDLE_PAIRS="20250926:24,20250926:48,20250927:24,20250927:48,20250928:24,20250928:48,20250929:24,20250929:48,20250930:24,20250930:48"' in build_text
    assert f'INPUT_ROOT="{tmp_path / "output" / run_id / "bundles_with_y"}"' in predict_text
    assert 'ALLOW_REBUILT_BUNDLE_ROOT="1"' in predict_text
    assert 'EXPECTED_LANE="o48_o96"' in sigma_text
    assert 'SIGMAS="0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000"' in sigma_text
    assert 'CKPT_NAME="anemoi-by_step-epoch_013-step_100000.ckpt"' in sigma_text
    assert 'NAME_FILTER="95a07500,o48_o96"' in loss_text
    assert 'REGION_NAMES="amazon_forest_core,eastern_us_coast,andes_central,himalayas_central,maritime_continent,congo_basin"' in regional_text
    assert 'REGION_NAMES="eastern_us_coast,idalia_center"' in storm_text
    assert 'PREDICTIONS_DIR="' in spectra_text
    assert "phase=proxy" in result.stdout


def test_o48_o96_helper_continue_full_render_reuses_existing_run_id(tmp_path: Path):
    result = _run_helper(tmp_path, phase="continue-full", run_id_override="manual_existing")
    assert result.returncode == 0, result.stderr

    submit_dir = tmp_path / "submit" / "20260330"
    build_text = (submit_dir / "manual_existing_build_truth_bundles.sbatch").read_text(encoding="utf-8")
    predict_text = (submit_dir / "manual_existing_predict.sbatch").read_text(encoding="utf-8")

    assert f'RUN_ROOT="{tmp_path / "output" / "manual_existing"}"' in build_text
    assert 'MEMBERS="1"' in build_text
    assert 'BUNDLE_PAIRS=""' in build_text
    assert f'RUN_ID_OVERRIDE="manual_existing"' in predict_text
    assert "phase=continue-full" in result.stdout


def test_o48_o96_helper_missing_source_gribs_fails_fast(tmp_path: Path):
    result = _run_helper(tmp_path, phase="proxy", populate_source_gribs=False)
    assert result.returncode != 0
    assert "Missing required source GRIB" in result.stderr
