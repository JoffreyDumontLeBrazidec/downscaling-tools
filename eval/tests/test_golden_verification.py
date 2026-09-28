"""Golden-output verification tests.

These tests verify that the new evaluator scorers produce scores identical
to the known-good values in the full 26-30 scoreboard CSV:
  /home/ecm5702/dev/docs/docs/scoreboard_o96_o320/state/source_26_30/scoreboard.csv

Row: "cfec83 k40 oldlike200k" (checkpoint_short: cfec83a3cd)

The spectra (HEALPix proxy) golden test moved to
eval/_quarantine/20260928/spectra/tests/test_golden_spectra_scorer.py when that
evaluator was retired on 2026-09-28.

Requires golden data on disk — mark with ``hpc`` since data is only
available on the HPC filesystem.  No GPU or scheduler needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Golden run root
GOLDEN_ROOT = Path(
    "/home/ecm5702/perm/eval/manual_cfec83a3_new_o96_o320_20260320_oldlike200k"
)
GOLDEN_PREDICTIONS = GOLDEN_ROOT / "predictions"
GOLDEN_TC_STATS = (
    "tc_normed_pdfs_idalia_franklin_manual_cfec83a3_new_o96_o320_20260320"
    "_oldlike200k_from_predictions.stats.json"
)
GOLDEN_SURFACE_JSON = "surface_loss_summary.json"
GOLDEN_RUN_ID = "manual_cfec83a3_new_o96_o320_20260320_oldlike200k"

# Expected values from full 26-30 scoreboard CSV
GOLDEN_TC_IDALIA = 0.760870
GOLDEN_TC_FRANKLIN = 0.834523
GOLDEN_SURFACE_MSE = 10646.679560

# Skip all tests if golden data is missing
pytestmark = pytest.mark.hpc
_golden_available = GOLDEN_ROOT.is_dir()
skip_no_golden = pytest.mark.skipif(
    not _golden_available,
    reason=f"Golden data not available at {GOLDEN_ROOT}",
)


@skip_no_golden
class TestTCScorer:
    def _score(self):
        from eval.config.loader import load_lane
        from eval.evaluators.tc.scorer import score

        lane_config = load_lane("o96_o320")
        tc_config = lane_config.get("tc", {})
        return score(
            GOLDEN_ROOT, lane_config, tc_config,
            run_id=GOLDEN_RUN_ID,
            stats_filename=GOLDEN_TC_STATS,
        )

    def test_tc_idalia_exact(self):
        records = self._score()
        by_metric = {r["metric"]: r["value"] for r in records}
        assert by_metric["tc_idalia_extreme_score"] == pytest.approx(
            GOLDEN_TC_IDALIA, abs=1e-6
        )

    def test_tc_franklin_exact(self):
        records = self._score()
        by_metric = {r["metric"]: r["value"] for r in records}
        assert by_metric["tc_franklin_extreme_score"] == pytest.approx(
            GOLDEN_TC_FRANKLIN, abs=1e-6
        )

    def test_tc_record_format(self):
        records = self._score()
        assert len(records) >= 3  # idalia, franklin, mean at minimum
        for r in records:
            assert set(r.keys()) == {"metric", "value", "unit"}
            assert isinstance(r["value"], float)


@skip_no_golden
class TestSurfaceScorer:
    def _score(self):
        from eval.config.loader import load_lane
        from eval.evaluators.surface.scorer import score

        lane_config = load_lane("o96_o320")
        surface_config = lane_config.get("surface", {})
        return score(
            GOLDEN_ROOT, lane_config, surface_config,
            surface_json=GOLDEN_SURFACE_JSON,
        )

    def test_surface_mse_exact(self):
        by_metric = {r["metric"]: r["value"] for r in self._score()}
        assert by_metric["surface_weighted_mse"] == pytest.approx(
            GOLDEN_SURFACE_MSE, abs=1e-3
        )

    def test_surface_record_format(self):
        records = self._score()
        assert len(records) >= 1
        for r in records:
            assert set(r.keys()) == {"metric", "value", "unit"}


@skip_no_golden
class TestDiscoveryGolden:
    def test_find_predictions_count(self):
        from eval.discovery.predictions import find_predictions

        preds = find_predictions(GOLDEN_PREDICTIONS)
        assert len(preds) == 25  # 5 dates x 5 steps

    def test_find_predictions_dates(self):
        from eval.discovery.predictions import find_predictions

        preds = find_predictions(GOLDEN_PREDICTIONS)
        dates = sorted({p.date for p in preds})
        assert dates == ["20230826", "20230827", "20230828", "20230829", "20230830"]

    def test_find_predictions_steps(self):
        from eval.discovery.predictions import find_predictions

        preds = find_predictions(GOLDEN_PREDICTIONS)
        steps = sorted({p.step for p in preds})
        assert steps == [24, 48, 72, 96, 120]

    def test_find_predictions_member_zero(self):
        from eval.discovery.predictions import find_predictions

        preds = find_predictions(GOLDEN_PREDICTIONS)
        assert all(p.member == 0 for p in preds)


@skip_no_golden
class TestEffectiveConfigDryRun:
    def test_dry_run_effective_config_structure(self):
        import json
        import os
        import subprocess
        import sys

        env = os.environ.copy()
        from pathlib import Path

        code_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = code_root + ":" + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [
                sys.executable, "-m", "eval.cli", "run",
                "--dry-run",
                "--lane", "o96_o320",
                "--checkpoint", "/tmp/golden_test.ckpt",
            ],
            capture_output=True, text=True, env=env,
            cwd=code_root,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

        config = json.loads(result.stdout)
        # Verify required fields
        for key in (
            "lane", "host", "checkpoint", "resolved", "overrides",
            "cli_args", "timestamp_utc", "git_commit", "code_root",
            "config_file_paths", "output_dir", "evaluators",
        ):
            assert key in config, f"Missing key in effective config: {key}"

        assert config["lane"] == "o96_o320"
        assert config["host"] == "atos_ac"
        assert isinstance(config["evaluators"], list)
        assert len(config["evaluators"]) > 0
