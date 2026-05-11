"""Tests for eval.jobs.backfill_tc_extreme_percentiles.

Focus on the pure-logic pieces that don't require GRIB / metview:
- `stats_missing_fields()` detection
- `_compute_extreme()` percentile math
- Idempotency of `backfill_stats_file` short-circuit (file already complete)
- Atomic write path (tmpfile suffix)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from eval.jobs import backfill_tc_extreme_percentiles as backfill


def _make_stats(rows_per_event: dict[str, list[dict]]) -> dict:
    return {
        "events": {
            ev_name: {"extreme_tail": {"rows": rows}}
            for ev_name, rows in rows_per_event.items()
        }
    }


def test_compute_extreme_mslp_returns_p001_and_min():
    arr = np.array([1000.0, 990.0, 980.0, 970.0, 960.0, 950.0])
    out = backfill._compute_extreme(arr, "mslp")
    # p0.01 of a 6-element array → close to the smallest value
    assert out["mslp_p001"] == pytest.approx(950.0, abs=0.05)
    assert out["mslp_min"] == 950.0
    assert out["wind_p9999"] is None
    assert out["wind_max"] is None


def test_compute_extreme_wind_returns_p9999_and_max():
    arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    out = backfill._compute_extreme(arr, "wind")
    assert out["wind_p9999"] == pytest.approx(50.0, abs=0.05)
    assert out["wind_max"] == 50.0
    assert out["mslp_p001"] is None
    assert out["mslp_min"] is None


def test_compute_extreme_empty_returns_all_none():
    arr = np.array([np.nan, np.nan])
    out = backfill._compute_extreme(arr, "mslp")
    assert all(v is None for v in out.values())


def test_stats_missing_fields_detects_missing_keys(tmp_path: Path):
    stats_path = tmp_path / "tc.stats.json"
    stats_path.write_text(json.dumps(_make_stats({
        "idalia": [
            {"exp": "OPER_O320_0001", "mslp_p1": 1000, "mslp_p01": 990, "mslp_min": 980,
             "wind_p99": 20, "wind_p999": 25, "wind_max": 30},  # missing p001 and p9999
        ],
    })))
    missing = backfill.stats_missing_fields(stats_path)
    assert "idalia" in missing
    items = missing["idalia"]
    assert any("mslp_p001" in s for s in items)
    assert any("wind_p9999" in s for s in items)


def test_stats_missing_fields_empty_when_complete(tmp_path: Path):
    stats_path = tmp_path / "tc.stats.json"
    stats_path.write_text(json.dumps(_make_stats({
        "idalia": [
            {"exp": "OPER_O320_0001",
             "mslp_p001": 985, "mslp_min": 980,
             "wind_p9999": 28, "wind_max": 30},
        ],
    })))
    assert backfill.stats_missing_fields(stats_path) == {}


def test_backfill_short_circuits_when_complete(tmp_path: Path, monkeypatch):
    """If every row already has the new fields, backfill_stats_file returns False
    without touching predictions or refs (no expensive GRIB load)."""
    stats_path = tmp_path / "tc.stats.json"
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    stats_path.write_text(json.dumps(_make_stats({
        "idalia": [
            {"exp": "OPER_O320_0001",
             "mslp_p001": 985, "mslp_min": 980,
             "wind_p9999": 28, "wind_max": 30},
        ],
        "franklin": [
            {"exp": "OPER_O320_0001",
             "mslp_p001": 970, "mslp_min": 960,
             "wind_p9999": 33, "wind_max": 36},
        ],
    })))

    # Trap any attempt to call _load_event_refs — it should NOT happen for an
    # already-complete stats file.
    def _trap(*args, **kwargs):
        raise AssertionError("_load_event_refs should not be called for a complete stats file")

    monkeypatch.setattr(backfill, "_load_event_refs", _trap)
    monkeypatch.setattr(backfill, "_load_model_extreme", _trap)

    lane_config = {"tc": {"events": ["idalia", "franklin"], "analysis_expid": "OPER_O320_0001",
                          "reference_expids": [], "grib_dir": "/dummy"}}
    changed = backfill.backfill_stats_file(stats_path, pred_dir, lane_config)
    assert changed is False

    # Stats unchanged on disk
    assert stats_path.with_suffix(".json.tmp").exists() is False


def test_backfill_atomic_write_path(tmp_path: Path, monkeypatch):
    """When the file needs updating, the backfill writes via a .tmp sibling and
    then renames it atomically."""
    stats_path = tmp_path / "tc.stats.json"
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    # Existing row missing the new fields
    stats_path.write_text(json.dumps(_make_stats({
        "idalia": [
            {"exp": "OPER_O320_0001", "mslp_p1": 1000, "mslp_p01": 990, "mslp_min": 980,
             "wind_p99": 20, "wind_p999": 25, "wind_max": 30},
        ],
    })))

    # Stub out the heavy loaders to return synthetic values.
    def _stub_refs(lane_config, event_name, first_pred_files):
        refs = {
            "OPER_O320_0001": {
                "mslp_p001": 982.0, "mslp_min": 980.0,
                "wind_p9999": 28.5, "wind_max": 30.0,
            },
        }
        return refs, (np.array([0.0]), np.array([0.0]))

    def _stub_model(event_name, pred_dir, target_pts):
        return {"mslp_p001": None, "mslp_min": None, "wind_p9999": None, "wind_max": None}

    def _stub_discover(d):
        return [("dummy.nc", 20230826, 24)]

    monkeypatch.setattr(backfill, "_load_event_refs", _stub_refs)
    monkeypatch.setattr(backfill, "_load_model_extreme", _stub_model)
    monkeypatch.setattr(backfill, "discover_prediction_files", _stub_discover)

    lane_config = {"tc": {"events": ["idalia"], "analysis_expid": "OPER_O320_0001",
                          "reference_expids": [], "grib_dir": "/dummy"}}
    changed = backfill.backfill_stats_file(stats_path, pred_dir, lane_config)
    assert changed is True

    # tmp file should be gone (renamed)
    assert stats_path.with_suffix(".json.tmp").exists() is False

    # File contents updated
    data = json.loads(stats_path.read_text())
    row = data["events"]["idalia"]["extreme_tail"]["rows"][0]
    assert row["mslp_p001"] == 982.0
    assert row["wind_p9999"] == 28.5
