"""Tests for the prepml runner guard (eval/predict/prepml_runner_guard.py)."""
from __future__ import annotations

from pathlib import Path

import pytest

from eval.predict import prepml_runner_guard as guard

FIXED = 'RUNNER_FIX_LEVEL = "20261110-sampler-dicts-seed31"\n'


def _wrap(tmp_path: Path, name: str, runner_text: str | None, *, gl512_site: bool = False) -> Path:
    """A fake runner venv: bin/activate puts site/ (optional) and a runner dir on PYTHONPATH."""
    wrap = tmp_path / name
    (wrap / "bin").mkdir(parents=True)
    runner_dir = tmp_path / f"{name}-season-scale-tools"
    runner_dir.mkdir()
    if runner_text is not None:
        (runner_dir / guard.RUNNER_MODULE_FILE).write_text(runner_text)
    extra = str(runner_dir)
    if gl512_site:
        site = wrap / "site"
        site.mkdir()
        (site / "sitecustomize.py").write_text("import gl512_runner_patches\n")
        (site / "gl512_runner_patches.py").write_text("# patches\n")
        extra = f"{site}:{extra}"
    (wrap / "bin" / "activate").write_text(
        "#!/usr/bin/env bash\n"
        "_BASE=/nonexistent\n"
        f'_WRAP_EXTRA="{extra}"\n'
        'export PYTHONPATH="${_WRAP_EXTRA}${PYTHONPATH:+:${PYTHONPATH}}"\n'
        "unset _WRAP_EXTRA\n"
    )
    return wrap


def _cfg(venv: Path | str, base_runner: str = "downscaling_unified", parallel: bool = True) -> dict:
    runner = {"parallel": {"base_runner": base_runner}} if parallel else base_runner
    return {"runner": {"venv": str(venv)}, "model": {"runner": runner}}


def test_fixed_runner_passes(tmp_path):
    wrap = _wrap(tmp_path, "fixed", "x = 1\n" + FIXED)
    check = guard.enforce_prepml_runner(_cfg(wrap))
    assert check.ok and check.fix_level == "20261110-sampler-dicts-seed31"
    assert check.runner_file.endswith("fixed-season-scale-tools/downscaling_unified_runner.py")


def test_single_gpu_runner_stanza_is_checked(tmp_path):
    wrap = _wrap(tmp_path, "old", "x = 1\n")
    with pytest.raises(SystemExit, match="REFUSED"):
        guard.enforce_prepml_runner(_cfg(wrap, parallel=False))


def test_unfixed_runner_refused(tmp_path):
    wrap = _wrap(tmp_path, "old", "_NOISE_SCHEDULER_KEYS = set()\n")
    with pytest.raises(SystemExit, match="--allow-unfixed-runner"):
        guard.enforce_prepml_runner(_cfg(wrap))


def test_unknown_fix_level_refused(tmp_path):
    wrap = _wrap(tmp_path, "odd", 'RUNNER_FIX_LEVEL = "20250101-other"\n')
    with pytest.raises(SystemExit, match="20250101-other"):
        guard.enforce_prepml_runner(_cfg(wrap))


def test_unfixed_runner_allowed_by_flag(tmp_path):
    wrap = _wrap(tmp_path, "old", "x = 1\n")
    check = guard.enforce_prepml_runner(_cfg(wrap), allow_unfixed=True)
    assert not check.ok


def test_gl512_site_patch_counts_as_fixed(tmp_path):
    wrap = _wrap(tmp_path, "gl512", "x = 1\n", gl512_site=True)
    check = guard.enforce_prepml_runner(_cfg(wrap))
    assert check.ok and check.fix_level == "equivalent-patch"


def test_gl512_patch_not_on_pythonpath_does_not_count(tmp_path):
    wrap = _wrap(tmp_path, "gl512", "x = 1\n")
    (wrap / "site").mkdir()
    (wrap / "site" / "sitecustomize.py").write_text("")
    (wrap / "site" / "gl512_runner_patches.py").write_text("")
    with pytest.raises(SystemExit, match="REFUSED"):
        guard.enforce_prepml_runner(_cfg(wrap))


def test_missing_runner_refused(tmp_path):
    wrap = _wrap(tmp_path, "none", None)
    with pytest.raises(SystemExit, match="no downscaling_unified_runner.py"):
        guard.enforce_prepml_runner(_cfg(wrap))


def test_missing_activate_refused(tmp_path):
    with pytest.raises(SystemExit, match="no bin/activate"):
        guard.enforce_prepml_runner(_cfg(tmp_path / "nowrap"))


def test_other_base_runner_not_checked(tmp_path):
    check = guard.enforce_prepml_runner(_cfg(tmp_path / "nowrap", base_runner="downscaling"))
    assert check.ok and "not checked" in check.message


def test_prepml_predict_refuses_before_push(tmp_path, monkeypatch):
    """prepml_predict must stop at the guard, before _launch_prepml pushes anything."""
    from eval.predict import prepml

    wrap = _wrap(tmp_path, "old", "x = 1\n")
    lane = {
        "predict": {"members": [1], "steps": [24], "dates": ["20250901"], "num_gpus_per_model": 4,
                    "weather_states": ["10u"]},
        "prepml": {
            "venv": str(wrap), "runner": "anemoi-dev", "base_runner": "downscaling_unified",
            "time_step": "24h", "output_template": "t.grib", "forcings_npz": "f.npz",
            "constant_high_res_forcings": [], "high_res_input": [],
            "input": {"class": "od", "stream": "eefo", "type": "pf", "grid": "O320"},
            "output": {"class": "rd", "stream": "enfo", "type": "pf"},
            "platform": {"gpu": {"time": "0-01:00"}}, "debug_expvers": ["test"],
        },
    }
    launched = []
    monkeypatch.setattr(prepml, "_launch_prepml", lambda *a, **k: launched.append(a))
    monkeypatch.setattr(prepml, "_resolve_prepml_weather_states", lambda **k: ["10u"])
    with pytest.raises(SystemExit, match="REFUSED"):
        prepml.prepml_predict(checkpoint="/x/inference-a.ckpt", lane_config=lane, host_config={},
                              output_dir=tmp_path / "out", expver="test")
    assert launched == []
