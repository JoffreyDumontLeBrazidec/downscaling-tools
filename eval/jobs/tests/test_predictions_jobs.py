from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_generate_predictions_parse_and_discover(tmp_path: Path):
    mod = _load_module(
        "gen25",
        ROOT / "eval/jobs/generate_predictions_25_files.py",
    )

    p1 = tmp_path / "eefo_o96_0001_date20230826_time0000_mem01_step024h_input_bundle.nc"
    p2 = tmp_path / "eefo_o96_0001_date20230826_time0000_mem01_step024h_input_bundle_dup.nc"
    p3 = tmp_path / "eefo_o96_0001_date20230826_time0000_mem02_step024h_input_bundle.nc"
    p_bad = tmp_path / "not_a_bundle.nc"

    # Files must exist for rglob/stat based discovery.
    p1.write_text("a", encoding="utf-8")
    p2.write_text("b", encoding="utf-8")
    p3.write_text("c", encoding="utf-8")
    p_bad.write_text("x", encoding="utf-8")

    # Keep only parseable names in discovery; p2 does not match strict regex.
    k1 = mod.parse_bundle_key(p1)
    k3 = mod.parse_bundle_key(p3)
    assert k1 is not None
    assert k1.date == "20230826"
    assert k1.step == 24
    assert k1.member == 1
    assert k3 is not None
    assert mod.parse_bundle_key(p_bad) is None

    discovered = mod.discover_bundles(tmp_path)
    assert k1 in discovered
    assert k3 in discovered
    assert discovered[k1] == p1
    assert discovered[k3] == p3


def test_generate_predictions_parse_int_list():
    mod = _load_module(
        "gen25_ints",
        ROOT / "eval/jobs/generate_predictions_25_files.py",
    )
    assert mod.parse_int_list("3,1,2,2") == [1, 2, 3]
    assert mod.parse_int_list(" 10 , 1 , 5 ") == [1, 5, 10]


def test_generate_predictions_main_binds_cuda_device_and_gpu_override(
    monkeypatch, tmp_path: Path
):
    mod = _load_module(
        "gen25_main_distributed",
        ROOT / "eval/jobs/generate_predictions_25_files.py",
    )

    input_root = tmp_path / "input"
    out_dir = tmp_path / "out"
    input_root.mkdir(parents=True, exist_ok=True)
    (
        input_root
        / "eefo_o96_0001_date20230826_time0000_mem01_step024h_input_bundle.nc"
    ).write_text("bundle", encoding="utf-8")

    set_device_calls = []
    init_group_calls = []
    load_calls = []

    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        mod.torch.cuda, "set_device", lambda idx: set_device_calls.append(idx)
    )
    monkeypatch.setattr(mod, "_get_parallel_info", lambda: (0, 1, 4))
    monkeypatch.setattr(mod, "_resolve_device", lambda requested, local: f"cuda:{local}")
    monkeypatch.setattr(
        mod,
        "_init_model_comm_group",
        lambda device, rank, world: init_group_calls.append((device, rank, world))
        or "fake_group",
    )

    def _fake_load_objects(**kwargs):
        load_calls.append(kwargs)
        return object(), object(), "/tmp/dir_exp", "exp_name"

    monkeypatch.setattr(mod, "_load_objects", _fake_load_objects)

    def _fake_predict_from_bundle(**kwargs):
        x = np.zeros((1, 1, 2, 2), dtype=np.float32)
        y = np.zeros((1, 1, 2, 2), dtype=np.float32)
        y_pred = np.zeros((1, 1, 2, 2), dtype=np.float32)
        lon_lres = np.zeros((2,), dtype=np.float32)
        lat_lres = np.zeros((2,), dtype=np.float32)
        lon_hres = np.zeros((2,), dtype=np.float32)
        lat_hres = np.zeros((2,), dtype=np.float32)
        weather_states = ["a", "b"]
        return x, y, y_pred, lon_lres, lat_lres, lon_hres, lat_hres, weather_states, None

    monkeypatch.setattr(mod, "_predict_from_bundle", _fake_predict_from_bundle)

    class _FakeDS:
        def __init__(self):
            self.attrs = {}

        def assign_coords(self, **kwargs):
            return self

        def __setitem__(self, key, value):
            return None

        def to_netcdf(self, path):
            Path(path).write_text("ok", encoding="utf-8")

    monkeypatch.setattr(mod, "build_predictions_dataset", lambda **kwargs: _FakeDS())

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_predictions_25_files.py",
            "--input-root",
            str(input_root),
            "--out-dir",
            str(out_dir),
            "--ckpt-id",
            "dummy_ckpt",
            "--ckpt-root",
            str(tmp_path / "ckpt_root"),
            "--device",
            "cuda",
            "--num-gpus-per-model",
            "4",
            "--members",
            "1",
            "--steps",
            "24",
            "--dates",
            "20230826",
        ],
    )

    mod.main()

    assert set_device_calls == [1]
    assert init_group_calls == [("cuda:1", 0, 4)]
    assert len(load_calls) == 1
    assert load_calls[0]["device"] == "cuda:1"
    assert load_calls[0]["num_gpus_per_model_override"] == 4
    assert (out_dir / "predictions_20230826_step024.nc").exists()


def test_generate_predictions_rejects_world_size_mismatch(
    monkeypatch, tmp_path: Path
):
    mod = _load_module(
        "gen25_main_world_mismatch",
        ROOT / "eval/jobs/generate_predictions_25_files.py",
    )

    input_root = tmp_path / "input"
    out_dir = tmp_path / "out"
    input_root.mkdir(parents=True, exist_ok=True)
    (
        input_root
        / "eefo_o96_0001_date20230826_time0000_mem01_step024h_input_bundle.nc"
    ).write_text("bundle", encoding="utf-8")

    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(mod, "_get_parallel_info", lambda: (0, 0, 1))
    monkeypatch.setattr(mod, "_resolve_device", lambda requested, local: "cuda:0")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_predictions_25_files.py",
            "--input-root",
            str(input_root),
            "--out-dir",
            str(out_dir),
            "--ckpt-id",
            "dummy_ckpt",
            "--device",
            "cuda",
            "--num-gpus-per-model",
            "4",
            "--members",
            "1",
            "--steps",
            "24",
            "--dates",
            "20230826",
        ],
    )

    with pytest.raises(SystemExit, match="Expected world_size=4"):
        mod.main()


def test_autopilot_submit_and_state_parsing(monkeypatch):
    mod = _load_module(
        "autopred",
        ROOT / "eval/jobs/autopilot_predictions.py",
    )

    monkeypatch.setattr(mod, "_run", lambda cmd: "Submitted batch job 123456")
    jid = mod._submit(Path("/tmp/fake.sbatch"))
    assert jid == "123456"

    # sacct parsing skips UNKNOWN and takes first meaningful state.
    monkeypatch.setattr(
        mod,
        "_run",
        lambda cmd: "UNKNOWN|\nRUNNING|\n",
    )
    assert mod._sacct_state("123456") == "RUNNING"

    # squeue parser returns None for empty output and uppercase state otherwise.
    monkeypatch.setattr(mod.subprocess, "check_output", lambda *a, **k: "")
    assert mod._squeue_state("123456") is None
    monkeypatch.setattr(mod.subprocess, "check_output", lambda *a, **k: " running \n")
    assert mod._squeue_state("123456") == "RUNNING"


def test_autopilot_write_state(tmp_path: Path):
    mod = _load_module(
        "autopred_state",
        ROOT / "eval/jobs/autopilot_predictions.py",
    )
    state_file = tmp_path / "state.json"
    jobs = {
        "predict25": mod.JobTrack(
            name="predict25",
            script=Path("/tmp/predict.sbatch"),
            job_id="111",
            state="RUNNING",
            retries=0,
            max_retries=1,
        ),
        "eval25": mod.JobTrack(
            name="eval25",
            script=Path("/tmp/eval.sbatch"),
            dependency="predict25",
            job_id="222",
            state="PENDING",
            retries=0,
            max_retries=1,
        ),
    }
    mod._write_state(state_file, "manualabcd", jobs)
    text = state_file.read_text(encoding="utf-8")
    assert "\"run_id\": \"manualabcd\"" in text
    assert "\"predict25\"" in text
    assert "\"eval25\"" in text


def test_launch_predictions_eval_suite_dry_run_generates_scripts():
    import subprocess
    import uuid

    run_id = f"manual_{uuid.uuid4().hex[:8]}"
    script = ROOT / "eval/jobs/launch_predictions_eval_suite.sh"
    run_dir = Path("/home/ecm5702/perm/eval") / run_id
    generated_dir = run_dir / "jobs"

    out = subprocess.check_output(
        [
            str(script),
            "--run-id",
            run_id,
            "--ckpt-id",
            "4a5b2f1b24b84c52872bfcec1410b00f",
            "--dry-run",
        ],
        text=True,
    )
    assert "Dry run. Scripts written" in out

    pred = generated_dir / f"predict25_{run_id}.sbatch"
    evl = generated_dir / f"eval25_{run_id}.sbatch"
    assert pred.exists()
    assert evl.exists()

    pred_text = pred.read_text(encoding="utf-8")
    evl_text = evl.read_text(encoding="utf-8")
    assert "generate_predictions_25_files.py" in pred_text
    assert "export DATA_DIR=/home/mlx/ai-ml/datasets/" in pred_text
    assert "export OUTPUT=/ec/res4/scratch/ecm5702/aifs" in pred_text
    assert f"--out-dir {run_dir}/predictions" in pred_text
    assert "python -m eval.run" in evl_text
    assert f"--eval-root {run_dir}/eval" in evl_text
    assert "Expected 25 prediction files" in evl_text
