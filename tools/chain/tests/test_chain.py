"""Chain resume method (tools/chain): state-derived segment mode, refusals, and the step guard.

The end-to-end tests run the prescribed job body (toy_job.sh) around a tiny Lightning trainer with anemoi's start
semantics (toy_train.py), including the 2026-10-09 Atos failure: a segment that would start from the donor.
"""
import io
import json
import os
import pickle
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import chain_state  # noqa: E402


def _fake_ckpt(path, step):
    """A torch-zip-format file whose data.pkl holds a Lightning-style dict (tensors as persistent ids)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    pickle.dump({"global_step": step, "epoch": 0, "state_dict": {}, "hyper_parameters": {"x": 1}}, buf, protocol=2)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("archive/data.pkl", buf.getvalue())


def _resolve(root, end):
    return chain_state._resolve(str(root), end)


def test_no_manifest_refuses(tmp_path):
    with pytest.raises(chain_state.Refuse, match="not a declared chain"):
        _resolve(tmp_path, 100)


def test_fresh_then_resume_then_done(tmp_path):
    assert chain_state.main(["init", "--root", str(tmp_path)]) == 0
    assert _resolve(tmp_path, 100)["CHAIN_ACTION"] == "fresh"
    _fake_ckpt(tmp_path / "run1" / "last.ckpt", 40)
    out = _resolve(tmp_path, 100)
    assert (out["CHAIN_ACTION"], out["CHAIN_RUN_ID"], out["CHAIN_EXPECT_STEP"]) == ("resume", "run1", "40")
    assert json.loads((tmp_path / "CHAIN.json").read_text())["run_id"] == "run1"
    _fake_ckpt(tmp_path / "run1" / "last.ckpt", 100)
    assert _resolve(tmp_path, 100)["CHAIN_ACTION"] == "done"


def test_recorded_run_missing_refuses(tmp_path):
    """The run moved or the root is wrong: never fall back to a fresh start."""
    chain_state.main(["init", "--root", str(tmp_path)])
    _fake_ckpt(tmp_path / "run1" / "last.ckpt", 40)
    _resolve(tmp_path, 100)
    (tmp_path / "run1" / "last.ckpt").rename(tmp_path / "run1" / "moved.ckpt")
    with pytest.raises(chain_state.Refuse, match="has no"):
        _resolve(tmp_path, 100)


def test_second_run_refuses(tmp_path):
    chain_state.main(["init", "--root", str(tmp_path)])
    _fake_ckpt(tmp_path / "run1" / "last.ckpt", 40)
    _resolve(tmp_path, 100)
    _fake_ckpt(tmp_path / "donor_restart" / "last.ckpt", 3)
    with pytest.raises(chain_state.Refuse, match="besides the chain's run"):
        _resolve(tmp_path, 100)


def test_init_with_parent_run_beside(tmp_path):
    _fake_ckpt(tmp_path / "parent" / "last.ckpt", 20000)
    assert chain_state.main(["init", "--root", str(tmp_path)]) == chain_state.REFUSE
    assert chain_state.main(["init", "--root", str(tmp_path), "--ignore-existing"]) == 0
    assert _resolve(tmp_path, 100)["CHAIN_ACTION"] == "fresh"
    _fake_ckpt(tmp_path / "fork" / "last.ckpt", 20500)
    assert _resolve(tmp_path, 100000)["CHAIN_RUN_ID"] == "fork"


def test_adopt_live_run(tmp_path):
    _fake_ckpt(tmp_path / "live" / "last.ckpt", 60000)
    _fake_ckpt(tmp_path / "old" / "last.ckpt", 3000)
    assert chain_state.main(["adopt", "--root", str(tmp_path), "--run-id", "live"]) == 0
    out = _resolve(tmp_path, 100000)
    assert (out["CHAIN_ACTION"], out["CHAIN_EXPECT_STEP"]) == ("resume", "60000")


def test_unreadable_last_refuses(tmp_path):
    chain_state.main(["init", "--root", str(tmp_path)])
    (tmp_path / "run1").mkdir()
    (tmp_path / "run1" / "last.ckpt").write_bytes(b"truncated")
    with pytest.raises(Exception):
        _resolve(tmp_path, 100)


# ── with Lightning ────────────────────────────────────────────────────────────────────────────────────

pl = pytest.importorskip("pytorch_lightning")
torch = pytest.importorskip("torch")

JOB = HERE / "toy_job.sh"


def _job(root, end, donor, *extra):
    env = dict(os.environ, CHAIN_PY=sys.executable)
    return subprocess.run(["bash", str(JOB), str(root), str(end), str(donor), *extra], env=env,
                          capture_output=True, text=True, timeout=600)


@pytest.fixture
def donor(tmp_path):
    root = tmp_path / "donor_root"
    chain_state.main(["init", "--root", str(root)])
    r = _job(root, 2, "null")
    assert r.returncode == 0, r.stderr
    (run,) = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    return run / "last.ckpt"


def test_real_checkpoint_step(donor):
    assert chain_state.checkpoint_step(str(donor)) == 2 == torch.load(donor, weights_only=False)["global_step"]


def test_chain_end_to_end(tmp_path, donor):
    root = tmp_path / "chain"
    # no manifest: refuse, nothing trained
    r = _job(root, 6, donor)
    assert r.returncode == 3 and "not a declared chain" in r.stderr
    assert not root.exists() or not any(p.is_dir() and not p.name.startswith(".") for p in root.iterdir())
    chain_state.main(["init", "--root", str(root), "--note", "test"])
    # segment 1: fresh from the donor, guard expects 0
    r = _job(root, 6, donor)
    assert r.returncode == 0, r.stderr
    assert "[chain-guard] ok: training starts at global_step 0" in r.stderr
    run_id = json.loads((root / "CHAIN.json").read_text())["run_id"]
    assert chain_state.checkpoint_step(str(root / run_id / "last.ckpt")) == 6
    # segment 2: identical job file, resumes at 6
    r = _job(root, 10, donor)
    assert r.returncode == 0, r.stderr
    assert "training starts at global_step 6 as expected" in r.stderr
    assert chain_state.checkpoint_step(str(root / run_id / "last.ckpt")) == 10
    # segment 3: already at its end step, exits 0 without training
    r = _job(root, 10, donor)
    assert r.returncode == 0 and "nothing to do" in r.stdout
    # a lane YAML that forces weights-only (step back to 0): the guard stops it before the first step
    r = _job(root, 14, donor, "training.load_weights_only=True")
    assert r.returncode != 0 and "REFUSED: training starts at global_step 0 but this segment expected 10" in r.stderr
    assert chain_state.checkpoint_step(str(root / run_id / "last.ckpt")) == 10
    # the Atos failure: a segment that ignores the run and starts from the donor makes a second run;
    # the next segment refuses instead of training on
    r = _job(root, 14, donor, "training.run_id=null", "training.load_weights_only=False")
    assert r.returncode != 0 and "REFUSED" in r.stderr
    assert len([p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]) == 1
