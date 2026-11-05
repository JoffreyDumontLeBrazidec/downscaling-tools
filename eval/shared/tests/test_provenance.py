"""The provenance block (audit 5 Nov 2026): ensemble construction from what a file says, date spacing, warnings."""
from __future__ import annotations

import numpy as np

from eval.shared.provenance import (
    date_spacing, ensemble_construction, is_single_input, members_differ, provenance, provenance_lines,
)


def test_ensemble_construction_from_file_facts():
    assert ensemble_construction({"member_mode": "pf"}, 10).startswith("one draw per input member")
    assert ensemble_construction({"competitor_row": "INPUT_PF: y_pred replaced by x_interp_members"}, 10) \
        .startswith("competitor row: INPUT_PF")
    ten = ensemble_construction({}, 10, source_bundles=[f"b{k}.nc" for k in range(10)])
    one = ensemble_construction({}, 10, source_bundles=["b.nc"] * 10)
    assert ten.startswith("one draw per input member") and is_single_input(one) and "10 generative draws" in one
    assert is_single_input(ensemble_construction({}, 10, input_members_differ=False))
    assert not is_single_input(ensemble_construction({}, 10, input_members_differ=True))
    assert ensemble_construction({"member_mode": "control"}, 10).startswith("single-input draws")
    assert ensemble_construction({}, 1) == "single draw (deterministic row)"
    unknown = ensemble_construction({}, 10)
    assert is_single_input(unknown) and "not recorded" in unknown


def test_members_differ():
    a = np.ones((3, 4))
    assert not members_differ(a)
    a[2, 1] += 1e-3
    assert members_differ(a) and not members_differ(a, tol=1e-2)
    assert not members_differ(np.ones((1, 4)))


def test_date_spacing_flags_consecutive_dates():
    d = date_spacing(["20230101", "20230111", "20230122"])
    assert d["n_dates"] == 3 and d["min_gap_days"] == 10 and not d["consecutive"]
    c = date_spacing([f"202308{k:02d}" for k in range(1, 13)])
    assert c["consecutive"] and c["median_gap_days"] == 1.0 and "optimistic" in c["note"]
    assert date_spacing(["20230101"])["n_dates"] == 1 and date_spacing([])["n_dates"] == 0


def test_provenance_block_and_warnings():
    b = provenance(verifying_field="gridded target", against_observations=False,
                   ensemble=ensemble_construction({}, 10, input_members_differ=False),
                   competitor_rows={"input": "single field (MAE)"},
                   weight_form="ring", spread_form="sd", domain_bounds={"europe": [35, 75, -25, 45], "all": None},
                   dates=[f"202308{k:02d}" for k in range(1, 6)])
    assert b["status"].startswith("diagnostic")
    w = " ".join(b["warnings"])
    assert "not observations" in w and "ONE input" in w and "single fields" in w and "consecutive" in w
    lines = provenance_lines(b)
    assert lines[0].startswith("> Provenance") and sum(l.startswith("> WARNING") for l in lines) == 4
    h = provenance(verifying_field="stations", against_observations=True, ensemble="one draw per input member (x)",
                   competitor_rows={"input": "ensemble (input members)"}, weight_form="none", spread_form="sd",
                   domain_bounds={"all": None}, dates=["20230101", "20230111"])
    assert h["status"] == "headline" and h["warnings"] == []


def test_evaluate_writes_the_run_provenance(tmp_path):
    """``eval.cli evaluate`` collects the blocks into evaluators/provenance.md and says when nothing in the run is
    against observations."""
    import json

    from eval.cli.evaluate import write_run_provenance

    root = tmp_path / "evaluators"
    (root / "probabilistic").mkdir(parents=True)
    block = provenance(verifying_field="gridded target", against_observations=False,
                       ensemble=ensemble_construction({"member_mode": "pf"}, 10), weight_form="ring",
                       spread_form="sd", domain_bounds={"europe": [35, 75, -25, 45]}, dates=["20230101", "20230111"])
    (root / "probabilistic" / "probabilistic_summary.json").write_text(json.dumps({"provenance": block}))
    out = write_run_provenance(tmp_path)
    text = out.read_text()
    assert "## probabilistic" in text and "no score against observations" in text
    (root / "obs_points").mkdir()
    (root / "obs_points" / "run_meta.json").write_text(json.dumps({"provenance": {
        **block, "against_observations": True, "status": "headline", "warnings": []}}))
    text = write_run_provenance(tmp_path).read_text()
    assert "Headline (against observations): obs_points" in text and "no score against observations" not in text
    assert write_run_provenance(tmp_path / "empty") is None
