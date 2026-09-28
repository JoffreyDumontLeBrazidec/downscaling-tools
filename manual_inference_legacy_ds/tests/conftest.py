"""The legacy ds tree is written to be installed under the name ``manual_inference``.

Its modules import each other as ``manual_inference.*``, and the checkpoints of the
cfec83a3 lineage load it by that name. Run in a session where ``manual_inference`` is the
unified package (the normal case), these tests would exercise the wrong code and fail. So
here they are skipped, and ``test_legacy_suite_as_manual_inference.py`` runs the whole
folder once, in a subprocess where the legacy tree is aliased as ``manual_inference``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

LEGACY_ROOT = Path(__file__).resolve().parents[1]
THIS_DIR = Path(__file__).resolve().parent
WRAPPER = "test_legacy_suite_as_manual_inference.py"


def manual_inference_is_legacy() -> bool:
    """True when ``import manual_inference`` resolves to this legacy tree."""
    module = sys.modules.get("manual_inference")
    if module is None:
        try:
            import manual_inference as module  # noqa: F401
        except ImportError:
            return False
    origin = getattr(module, "__file__", None)
    return origin is not None and Path(origin).resolve().parent == LEGACY_ROOT


def pytest_collection_modifyitems(config, items):
    if manual_inference_is_legacy():
        return
    skip = pytest.mark.skip(
        reason="needs the legacy tree installed as 'manual_inference'; "
        f"{WRAPPER} runs this folder that way in a subprocess"
    )
    for item in items:
        path = Path(str(item.fspath)).resolve()
        if THIS_DIR in path.parents and path.name != WRAPPER:
            item.add_marker(skip)
