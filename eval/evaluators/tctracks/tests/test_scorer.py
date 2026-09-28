from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eval.evaluators.tctracks import scorer


def _source(role, n_tracks, mslp_min_base, n_forecasts=10, month="202509"):
    """Synthetic source: n_tracks tracks, one record each per 24h step 0..48."""
    records, summary, forecasts = [], [], []
    dates = [f"{month}{d:02d}" for d in range(1, 6)]
    for i in range(n_forecasts):
        forecasts.append({
            "init_date": dates[i % len(dates)], "time": "00",
            "member": i + 1, "present": True,
        })
    for t in range(n_tracks):
        init_date = dates[t % len(dates)]
        member = (t % n_forecasts) + 1
        track_id = f"atl-{t + 1:03d}"
        mslp_min = mslp_min_base + t
        day = 10 + (t % 5)
        for k, step in enumerate([0, 24, 48]):
            records.append({
                "init_date": init_date, "time": "00", "member": member,
                "basin": "atl", "track_id": track_id, "classification": "TS",
                "valid_time": f"2025/09/{day + k:02d}/00", "step_h": step,
                "lat": 20.0 + t, "lon_e": 300.0 + t,
                "wind_ms": 30.0, "mslp_hpa": mslp_min + (0 if step == 24 else 6),
            })
        summary.append({
            "init_date": init_date, "time": "00", "member": member,
            "basin": "atl", "track_id": track_id, "classification": "TS",
            "n_records": 3, "first_step_h": 0, "last_step_h": 48,
            "mslp_min_hpa": mslp_min, "wind_max_ms": 30.0,
            "mslp_min_valid_time": f"2025/09/{day + 1:02d}/00",
            "mslp_min_lat": 20.0 + t, "mslp_min_lon_e": 300.0 + t,
            "genesis_lat": 20.0 + t, "genesis_lon_e": 300.0 + t,
        })
    return {
        "records": pd.DataFrame(records),
        "summary": pd.DataFrame(summary),
        "forecasts": pd.DataFrame(forecasts),
        "provenance": {
            "role": role, "source_id": role, "grid_support": "o320",
            "steps": "0-360/24h", "vorticity": False, "time": "00",
            "members": list(range(1, n_forecasts + 1)), "completeness": 1.0,
        },
    }


def test_score_sources_counts_and_distributions():
    sources = {
        "model": _source("model", n_tracks=8, mslp_min_base=980.0),
        "target": _source("target", n_tracks=12, mslp_min_base=960.0),
    }
    metrics = scorer.score_sources(sources, months=["202509"], basins=["atl"])
    assert metrics["support"]["consistent"]
    rows = {(r["role"]): r for r in metrics["metrics"] if r["scope"] == "202509"}
    assert rows["model"]["n_tracks"] == 8
    assert rows["target"]["n_tracks"] == 12
    assert rows["model"]["tracks_per_forecast"] == pytest.approx(0.8)
    assert rows["target"]["mslp_min"] == 960.0
    assert rows["model"]["tc_days"] == 24  # 8 tracks x 3 records all >= TS wind
    steps = {s["step_h"] for s in rows["model"]["step_intensity"]}
    assert steps == {0, 24, 48}


def test_support_contract_violation_flagged():
    a = _source("model", 2, 990.0)
    b = _source("target", 2, 985.0)
    b["provenance"]["grid_support"] = "o1280"
    result = scorer.check_support_contract({"model": a, "target": b})
    assert not result["consistent"]
    assert any("grid_support" in v for v in result["violations"])


def test_scope_filters_months_and_basins():
    src = _source("model", 4, 985.0)
    other = src["summary"].copy()
    other["init_date"] = "20251001"
    src["summary"] = pd.concat([src["summary"], other], ignore_index=True)
    mask = scorer.scope_mask(src["summary"], ["202509"], ["atl"])
    assert mask.sum() == 4


def test_select_cases_matches_across_roles():
    sources = {
        "model": _source("model", n_tracks=5, mslp_min_base=980.0),
        "target": _source("target", n_tracks=5, mslp_min_base=960.0),
    }
    cases = scorer.select_cases(sources, ["202509"], "atl", top_k=2)
    assert len(cases) == 2
    # deepest target track is t=0 (960 hPa) at lat 20, lon 300; the model track
    # at the same position/time (980) must associate
    case = cases[0]
    assert case["mslp_min_hpa"] == 960.0
    assert len(case["members"]["target"]) >= 1
    assert len(case["members"]["model"]) >= 1


def test_density_grid_shape_and_mass():
    src = _source("model", 3, 985.0)
    grid = scorer.density_grid(src["records"])
    assert grid["hist"].sum() == len(src["records"])
    assert grid["hist"].shape == (60, 180)
