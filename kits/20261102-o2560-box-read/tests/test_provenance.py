import json
from pathlib import Path

import pytest

CHECKS = Path(__file__).resolve().parent.parent / "checks"


def load(side):
    p = CHECKS / f"provenance_{side}.json"
    if not p.exists():
        pytest.skip(f"{p} absent")
    return json.loads(p.read_text())


def test_A_and_B_trained_on_the_same_statistics_and_donor():
    a, b = load("atos_A"), load("jupiter_B")
    assert a["residual_statistics"]["sha256"] == b["residual_statistics"]["sha256"]
    assert a["warm_start"]["sha256"] == b["warm_start"]["sha256"]
    assert a["stores"] == b["stores"]   # variables and mean/stdev/minimum/maximum hashes of both box stores


def test_manifest_holds_the_training_inputs():
    p = CHECKS / "MANIFEST.json"
    if not p.exists():
        pytest.skip("manifest absent")
    m = json.loads(p.read_text())
    assert set(m.get("training_inputs", {})) == {"atos_A", "jupiter_B"}
