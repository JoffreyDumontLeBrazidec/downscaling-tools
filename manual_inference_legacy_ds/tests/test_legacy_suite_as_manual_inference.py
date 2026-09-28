"""Run the legacy ds test folder with the legacy tree installed as ``manual_inference``."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

LEGACY_ROOT = Path(__file__).resolve().parents[1]


def manual_inference_is_legacy() -> bool:
    """True when ``import manual_inference`` resolves to the legacy tree (same test as conftest.py)."""
    try:
        import manual_inference
    except ImportError:
        return False
    origin = getattr(manual_inference, "__file__", None)
    return origin is not None and Path(origin).resolve().parent == LEGACY_ROOT


def test_legacy_suite_passes_when_installed_as_manual_inference(tmp_path: Path):
    if manual_inference_is_legacy():
        pytest.skip("already running with the legacy tree as manual_inference")
    alias_root = tmp_path / "alias"
    alias_root.mkdir()
    (alias_root / "manual_inference").symlink_to(LEGACY_ROOT, target_is_directory=True)

    env = {**os.environ, "PYTHONPATH": str(alias_root), "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(alias_root / "manual_inference" / "tests"),
         "-q", "-p", "no:cacheprovider", "--rootdir", str(alias_root), "-c", os.devnull],
        cwd=str(alias_root), env=env, capture_output=True, text=True, timeout=900,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]
