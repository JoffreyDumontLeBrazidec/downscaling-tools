"""Golden-output test of the retired spectra (HEALPix proxy) scorer.

Moved here from eval/tests/test_golden_verification.py when the evaluator was
retired and quarantined on 2026-09-28. It imports eval.evaluators.spectra, which
no longer exists, so it cannot run from this location; it is kept as the record
of the proxy's golden values.
"""

from __future__ import annotations

from pathlib import Path

import pytest

GOLDEN_ROOT = Path(
    "/home/ecm5702/perm/eval/manual_cfec83a3_new_o96_o320_20260320_oldlike200k"
)
GOLDEN_SPECTRA = GOLDEN_ROOT / "spectra_step120_5dates_m10_ecmwf"
GOLDEN_SPECTRA_10U_SCORE = 0.978893
GOLDEN_SPECTRA_10V_SCORE = 0.977061
GOLDEN_SPECTRA_2T_SCORE = 0.984530
GOLDEN_SPECTRA_MEAN_SCORE = 0.980161

pytestmark = pytest.mark.hpc
skip_no_golden = pytest.mark.skipif(
    not GOLDEN_ROOT.is_dir(), reason=f"Golden data not available at {GOLDEN_ROOT}",
)


@skip_no_golden
class TestSpectraScorer:
    def _score(self):
        from eval.config.loader import load_lane
        from eval.evaluators.spectra.scorer import score

        lane_config = load_lane("o96_o320")
        spectra_config = lane_config.get("spectra", {})
        return score(GOLDEN_SPECTRA, lane_config, spectra_config)

    def test_spectra_10u_score_exact(self):
        by_metric = {r["metric"]: r["value"] for r in self._score()}
        assert by_metric["spectra_10u_score"] == pytest.approx(
            GOLDEN_SPECTRA_10U_SCORE, abs=1e-6
        )

    def test_spectra_10v_score_exact(self):
        by_metric = {r["metric"]: r["value"] for r in self._score()}
        assert by_metric["spectra_10v_score"] == pytest.approx(
            GOLDEN_SPECTRA_10V_SCORE, abs=1e-6
        )

    def test_spectra_2t_score_exact(self):
        by_metric = {r["metric"]: r["value"] for r in self._score()}
        assert by_metric["spectra_2t_score"] == pytest.approx(
            GOLDEN_SPECTRA_2T_SCORE, abs=1e-6
        )

    def test_spectra_mean_score_exact(self):
        by_metric = {r["metric"]: r["value"] for r in self._score()}
        assert by_metric["spectra_mean_score"] == pytest.approx(
            GOLDEN_SPECTRA_MEAN_SCORE, abs=1e-6
        )

    def test_spectra_record_pairs(self):
        records = self._score()
        metrics = {r["metric"] for r in records}
        # Each field should have both relative_l2 and score entries
        for field in ("10u", "10v", "2t"):
            assert f"spectra_{field}_relative_l2" in metrics
            assert f"spectra_{field}_score" in metrics
        assert "spectra_mean_relative_l2" in metrics
        assert "spectra_mean_score" in metrics
