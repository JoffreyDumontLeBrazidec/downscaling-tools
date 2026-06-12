"""Tests for eval.jobs.renderer."""

from __future__ import annotations

import subprocess

import pytest

from eval.jobs.renderer import render_sbatch


def test_render_sbatch_default_resources(tmp_path):
    script = render_sbatch(
        lane="o96_o320", host="atos_ac", checkpoint="/tmp/test.ckpt",
    )
    assert "#SBATCH --qos=nf" in script
    assert "#SBATCH --time=04:00:00" in script
    assert "#SBATCH --mem=64G" in script
    assert "#SBATCH --cpus-per-task=16" in script
    # No GPU directive since qos=nf and gpus=0
    assert "--gpus-per-node" not in script

    path = tmp_path / "test.sbatch"
    path.write_text(script)
    result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, f"bash -n failed: {result.stderr}"


def test_render_sbatch_with_resource_overrides():
    script = render_sbatch(
        lane="o96_o320", host="atos_ac", checkpoint="/tmp/test.ckpt",
        resource_overrides={"gpus": 2, "time": "08:00:00", "mem": "256G"},
    )
    assert "#SBATCH --time=08:00:00" in script
    assert "#SBATCH --mem=256G" in script
    assert "#SBATCH --gpus-per-node=2" in script


def test_render_sbatch_gpu_directive():
    script = render_sbatch(
        lane="o96_o320", host="atos_ac", checkpoint="/tmp/test.ckpt",
        resource_overrides={"gpus": 1},
    )
    assert "#SBATCH --gpus-per-node=1" in script


def test_render_sbatch_ng_no_gpu_directive():
    script = render_sbatch(
        lane="o96_o320", host="atos_ac", checkpoint="/tmp/test.ckpt",
        resource_overrides={"qos": "ng", "gpus": 0},
    )
    assert "#SBATCH --qos=ng" in script
    assert "#SBATCH --gpus-per-node=0" in script


def test_render_sbatch_job_name_suffix():
    script = render_sbatch(
        lane="o96_o320", host="atos_ac", checkpoint="/tmp/test.ckpt",
        resource_overrides={"job_name_suffix": "-predict"},
    )
    assert "-predict-" in script
    assert "#SBATCH --job-name=eval-o96_o320-predict-" in script


def test_render_sbatch_cli_overrides():
    script = render_sbatch(
        lane="o96_o320", host="atos_ac", checkpoint="/tmp/test.ckpt",
        overrides={"--only": "tc"},
    )
    assert "--only" in script
    assert "tc" in script


def test_render_predict_with_source_grib_root_does_not_outer_srun_eval_cli():
    script = render_sbatch(
        lane="o320_o1280_sigma10k",
        host="atos_ag",
        checkpoint="/tmp/test.ckpt",
        mode="predict",
        overrides={
            "--source-grib-root": "/home/ecm5702/perm/reference/o320_o1280/grib/idalia",
            "--bundle-dir": "/tmp/bundles",
        },
        resource_overrides={"gpus": 4, "ntasks_per_node": 4},
    )

    assert "#SBATCH --ntasks-per-node=4" in script
    assert "#SBATCH --gpus-per-node=4" in script
    assert "# Host: atos_ag" in script
    assert "source /home/ecm5702/dev/.ds-ag/bin/activate" in script
    assert "srun python -m eval.cli predict" not in script
    assert "python -m eval.cli predict" in script
    assert "--source-grib-root /home/ecm5702/perm/reference/o320_o1280/grib/idalia" in script


def test_render_o320_o1280_predict_rejects_ac_host():
    with pytest.raises(Exception, match="stage 'predict' must be run on host"):
        render_sbatch(
            lane="o320_o1280_sigma10k",
            host="atos_ac",
            checkpoint="/tmp/test.ckpt",
            mode="predict",
            resource_overrides={"gpus": 4, "ntasks_per_node": 4},
        )


def test_render_o320_o1280_evaluate_accepts_ac_host():
    script = render_sbatch(
        lane="o320_o1280_sigma10k",
        host="atos_ac",
        checkpoint="/tmp/test.ckpt",
        mode="evaluate",
        overrides={
            "--predictions-dir": "/tmp/predictions",
            "--only": "tc",
        },
    )
    assert "# Host: atos_ac" in script
    assert "source /home/ecm5702/dev/.ds-dyn/bin/activate" in script
    assert "python -m eval.cli evaluate" in script


def test_render_o320_o1280_evaluate_rejects_ag_host():
    with pytest.raises(Exception, match="stage 'evaluate' must be run on host"):
        render_sbatch(
            lane="o320_o1280_sigma10k",
            host="atos_ag",
            checkpoint="/tmp/test.ckpt",
            mode="evaluate",
            overrides={
                "--predictions-dir": "/tmp/predictions",
                "--only": "tc",
            },
        )
