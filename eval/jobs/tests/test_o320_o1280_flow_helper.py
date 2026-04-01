from __future__ import annotations

import os
import stat
import subprocess
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "eval/jobs/templates/submit_o320_o1280_manual_eval_flow.sh"


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
            "lane": "o320_o1280",
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
            f"eefo_o320_0001_date{date}_time0000_mem1to10_step24to120_sfc.grib",
            f"eefo_o320_0001_date{date}_time0000_mem1to10_step24to120_pl.grib",
            f"enfo_o1280_0001_date{date}_time0000_step24to120_sfc.grib",
            f"enfo_o1280_0001_date{date}_time0000_mem1to10_step24to120_sfc_y.grib",
            f"enfo_o1280_0001_date{date}_time0000_mem1to10_step24to120_pl_y.grib",
        ):
            (root / name).write_text("stub\n", encoding="utf-8")


def _run_helper(
    tmp_path: Path,
    *,
    phase: str,
    run_id_override: str = "",
    host_family: str = "ag",
    host_short: str = "ag6-test",
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

    fake_checkpoint_path = "/tmp/checkpoints/da4d902b71084ecc884a938c4b8930d3/anemoi-step.ckpt"
    fake_profile = _make_fake_profile(
        bin_dir / "fake_profile.py",
        checkpoint_path=fake_checkpoint_path,
        host_family=host_family,
    )
    _make_fake_hostname(bin_dir / "hostname", host_short)

    if populate_source_gribs:
        if phase == "proxy":
            _touch_required_source_gribs(source_root, ["20230827", "20230828", "20230829", "20230830"])
        else:
            _touch_required_source_gribs(source_root, ["20230826", "20230827", "20230828", "20230829", "20230830"])

    if run_id_override:
        (output_root / run_id_override).mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "CHECKPOINT_PATH": "/tmp/fake.ckpt",
            "SOURCE_HPC": "ag",
            "SOURCE_GRIB_ROOT": str(source_root),
            "OUTPUT_ROOT": str(output_root),
            "SUBMIT_ROOT": str(submit_root),
            "PROFILE_PYTHON": str(fake_profile),
            "NO_SUBMIT": "1",
            "ALLOW_OVERWRITE": "1",
            "RUN_DATE_UTC": "20260327",
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


def test_o320_o1280_helper_proxy_render_uses_truth_bundle_stage(tmp_path: Path):
    result = _run_helper(tmp_path, phase="proxy")
    assert result.returncode == 0, result.stderr

    run_id = "manual_da4d902b_new_o320_o1280_20260327_manual_eval"
    submit_dir = tmp_path / "submit" / "20260327"
    build_script = submit_dir / f"{run_id}_build_truth_bundles.sbatch"
    predict_script = submit_dir / f"{run_id}_predict.sbatch"

    build_text = build_script.read_text(encoding="utf-8")
    predict_text = predict_script.read_text(encoding="utf-8")

    assert 'SOURCE_GRIB_ROOT="' in build_text
    assert 'MEMBERS="1"' in build_text
    assert 'BUNDLE_PAIRS="20230829:24,20230828:48,20230829:48,20230828:24,20230830:24,20230828:72,20230827:72,20230830:48,20230829:72,20230827:48"' in build_text
    assert f'INPUT_ROOT="{tmp_path / "output" / run_id / "bundles_with_y"}"' in predict_text
    assert 'ALLOW_REBUILT_BUNDLE_ROOT="1"' in predict_text
    assert f'RUN_ID_OVERRIDE="{run_id}"' in predict_text
    assert 'MEMBERS="1"' in predict_text
    assert "phase=proxy" in result.stdout


def test_o320_o1280_helper_continue_full_render_reuses_existing_run_id(tmp_path: Path):
    result = _run_helper(tmp_path, phase="continue-full", run_id_override="manual_existing")
    assert result.returncode == 0, result.stderr

    submit_dir = tmp_path / "submit" / "20260327"
    build_script = submit_dir / "manual_existing_build_truth_bundles.sbatch"
    predict_script = submit_dir / "manual_existing_predict.sbatch"

    build_text = build_script.read_text(encoding="utf-8")
    predict_text = predict_script.read_text(encoding="utf-8")

    assert f'RUN_ROOT="{tmp_path / "output" / "manual_existing"}"' in build_text
    assert 'MEMBERS="1,2,3,4,5,6,7,8,9,10"' in build_text
    assert 'BUNDLE_PAIRS=""' in build_text
    assert f'RUN_ID_OVERRIDE="manual_existing"' in predict_text
    assert 'MEMBERS="1,2,3,4,5,6,7,8,9,10"' in predict_text
    assert "phase=continue-full" in result.stdout


def test_o320_o1280_helper_ac_proxy_render_uses_ecmwf_and_cpu_safe_jobs(tmp_path: Path):
    result = _run_helper(
        tmp_path,
        phase="proxy",
        host_family="ac",
        host_short="ac6-test",
    )
    assert result.returncode == 0, result.stderr

    run_id = "manual_da4d902b_new_o320_o1280_20260327_manual_eval"
    submit_dir = tmp_path / "submit" / "20260327"
    build_text = (submit_dir / f"{run_id}_build_truth_bundles.sbatch").read_text(encoding="utf-8")
    local_text = (submit_dir / f"{run_id}_local_plots.sbatch").read_text(encoding="utf-8")
    spectra_text = (submit_dir / f"{run_id}_spectra.sbatch").read_text(encoding="utf-8")
    tc_text = (submit_dir / f"{run_id}_tc_eval.sbatch").read_text(encoding="utf-8")

    assert "#SBATCH --qos=nf" in build_text
    assert "#SBATCH --qos=nf" in local_text
    assert "#SBATCH --qos=nf" in tc_text
    assert "#SBATCH --gpus-per-node=" not in build_text
    assert "#SBATCH --gpus-per-node=" not in local_text
    assert "#SBATCH --gpus-per-node=" not in tc_text
    assert 'PREDICTIONS_DIR="' in spectra_text
    assert 'SUBSET_DIR="' not in spectra_text
    assert 'SUPPORT_MODE="regridded"' in tc_text


def test_o320_o1280_helper_continue_full_requires_run_id_override(tmp_path: Path):
    result = _run_helper(tmp_path, phase="continue-full")
    assert result.returncode != 0
    assert "PHASE=continue-full requires RUN_ID_OVERRIDE" in result.stderr


def test_o320_o1280_helper_missing_source_gribs_fails_fast(tmp_path: Path):
    result = _run_helper(tmp_path, phase="proxy", populate_source_gribs=False)
    assert result.returncode != 0
    assert "Missing required source GRIB" in result.stderr
