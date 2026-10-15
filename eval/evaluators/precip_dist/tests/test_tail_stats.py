"""precip_dist tail_stats.csv: exact tail numbers over every member (o96->o320 big eval)."""
from __future__ import annotations

import csv
from unittest.mock import patch

import netCDF4
import numpy as np
import pytest

from eval.evaluators.precip_dist import runner
from eval.evaluators.precip_dist.core import tail_stats as T
from eval.tests.synthetic_predictions import octahedral_grid, write_predictions_file


@pytest.fixture
def preds(tmp_path):
    grids = (octahedral_grid(24), octahedral_grid(8))
    d = tmp_path / "preds"
    write_predictions_file(d / "predictions_20220101_step060.nc", grids=grids,
                           states=("10u", "tp"), members=(1, 2, 3), seed=1)
    write_predictions_file(d / "predictions_20220819_step060.nc", grids=grids,
                           states=("10u", "tp"), members=(1, 2, 3), seed=2,
                           nan_truth_states=("tp",))
    return d


def _rows(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def test_tail_stats_match_numpy_on_all_members(preds, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    T.compute_tail_stats(preds, out)
    rows = _rows(out / "tail_stats.csv")
    assert list(rows[0]) == T.COLS
    pooled = {r["series"]: r for r in rows if r["scope"] == "all"}
    assert set(pooled) == {"pred", "truth", "input"}
    vals = []
    for f in sorted(preds.glob("*.nc")):
        with netCDF4.Dataset(f) as d:
            vals.append(np.asarray(d["y_pred"][0, :, :, 1], dtype=np.float32).reshape(-1) * 1000.0)
    v = np.concatenate(vals)
    r = pooled["pred"]
    assert int(r["n"]) == v.size == 2 * 3 * octahedral_grid(24)[0].size
    assert float(r["p99_9"]) == pytest.approx(np.percentile(v, 99.9), rel=1e-5)
    assert float(r["max_mm"]) == pytest.approx(v.max(), rel=1e-5)
    assert float(r["wet_frac"]) == pytest.approx(np.mean(v > 0.1), rel=1e-5)
    assert float(r["neg_share"]) == pytest.approx(np.mean(v < 0), rel=1e-5, abs=1e-9)
    assert float(r["neg_share"]) > 0          # the synthetic raw tp has negative values
    # the second date has no tp truth: truth pooled over the first date only, no truth row for it
    assert int(pooled["truth"]["n"]) == v.size // 2
    per_date = {(r["scope"], r["series"]) for r in rows if r["scope"] != "all"}
    assert ("20220819", "truth") not in per_date and ("20220819", "pred") in per_date


def test_runner_writes_tail_stats_only_when_asked(preds, tmp_path):
    with patch("eval.evaluators.precip_dist.runner.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        runner.run(preds, {}, {}, output_dir=tmp_path / "a")
        runner.run(preds, {}, {"tail_stats": True, "wet_threshold_mm": 1.0}, output_dir=tmp_path / "b")
    assert not (tmp_path / "a" / "tail_stats.csv").exists()
    rows = _rows(tmp_path / "b" / "tail_stats.csv")
    assert {r["scope"] for r in rows} == {"all", "20220101", "20220819"}
