"""Unit tests of the aggregator: paired test, fair CRPS fallback, precipitation tail, rows file."""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
for p in (str(HERE.parent), str(HERE.parents[2])):
    if p not in sys.path:
        sys.path.insert(0, p)

from aggregate_lib import fcrps as F  # noqa: E402
from aggregate_lib import synthetic as syn  # noqa: E402
from aggregate_lib.common import load_rows, sanitize_spec  # noqa: E402
from aggregate_lib.extremes import _Tail  # noqa: E402
from aggregate_lib.scores import paired_test  # noqa: E402
from aggregate_lib.spectra import compare_bin  # noqa: E402

DATES = [f"2022{m:02d}01" for m in range(1, 7)]


def _vals(xs):
    return dict(zip(DATES, xs))


# ----------------------------------------------------------------------------- paired test

def test_paired_test_hand_example_worse():
    # p_d = [2, 1, 3, -1, 2, 1] %; mean 8/6; sd^2 = 9.3333/5; se = sd/sqrt(6); t = 2.3905;
    # reference better on 5 of 6 dates (>= 4 = 2/3 of 6) -> established, |mean| > 1 % -> worse
    t = paired_test(_vals([102, 101, 103, 99, 102, 101]), _vals([100] * 6))
    assert t["n"] == 6
    assert t["mean_pct"] == pytest.approx(4 / 3)
    sd = math.sqrt(28 / 3 / 5)
    assert t["sd"] == pytest.approx(sd)
    assert t["se"] == pytest.approx(sd / math.sqrt(6))
    assert t["t"] == pytest.approx((4 / 3) / (sd / math.sqrt(6)))
    assert t["t"] == pytest.approx(2.3905, abs=1e-4)
    assert (t["k_ref_better"], t["k_row_better"]) == (5, 1)
    assert t["established"] and t["status"] == "worse"


def test_paired_test_better_equal_tie():
    better = paired_test(_vals([98, 99, 97, 101, 98, 99]), _vals([100] * 6))
    assert better["status"] == "better" and better["k_ref_better"] == 1
    assert better["t"] == pytest.approx(-2.3905, abs=1e-4)
    # same t, mean 0.667 % -> established but within 1 %: equal in practice
    eq = paired_test(_vals([101, 100.5, 101.5, 99.5, 101, 100.5]), _vals([100] * 6))
    assert eq["established"] and eq["status"] == "equal in practice"
    assert eq["mean_pct"] == pytest.approx(2 / 3)
    # |t| = 2.09 >= 2.03 but the reference wins only 3 of 6 dates (< 4) -> tie
    tie = paired_test(_vals([103, 103, 103, 99.9, 99.9, 99.9]), _vals([100] * 6))
    assert abs(tie["t"]) >= 2.03 and tie["k_ref_better"] == 3
    assert not tie["established"] and tie["status"] == "tie"
    # small |t| -> tie
    assert paired_test(_vals([101, 99, 101, 99, 101, 99]), _vals([100] * 6))["status"] == "tie"


def test_paired_test_uses_common_dates_only_and_handles_short_input():
    row = _vals([102, 101, 103, 99, 102, 101])
    row["20230101"] = 500.0  # date the reference lacks: ignored
    t = paired_test(row, _vals([100] * 6))
    assert t["n"] == 6 and t["mean_pct"] == pytest.approx(4 / 3)
    short = paired_test({"a": 1.0}, {"a": 2.0})
    assert short["status"].startswith("Not available")


# ----------------------------------------------------------------------------- fair CRPS

def test_fair_crps_points_matches_double_sum():
    rng = np.random.default_rng(3)
    m, p = 7, 50
    x = rng.normal(size=(m, p))
    y = rng.normal(size=p)
    brute = np.array([np.mean(np.abs(x[:, k] - y[k]))
                      - np.abs(x[:, k][:, None] - x[:, k][None, :]).sum() / (2 * m * (m - 1))
                      for k in range(p)])
    np.testing.assert_allclose(F.fair_crps_points(x, y), brute, rtol=1e-12, atol=1e-12)
    x[2, 5] = np.nan
    assert np.isnan(F.fair_crps_points(x, y)[5])


def test_fair_crps_matches_probabilistic_function():
    from eval.evaluators.probabilistic.core.scoring import crps_ensemble_components
    rng = np.random.default_rng(4)
    x = rng.normal(size=(10, 300))
    y = rng.normal(size=300)
    np.testing.assert_allclose(F.fair_crps_points(x, y),
                               crps_ensemble_components(x, y)["fcrps"], rtol=1e-12)


def test_fallback_per_date_agrees_with_probabilistic_evaluator(tmp_path):
    """The fallback reproduces probabilistic's per-date fCRPS (scores_by_lead.csv) exactly
    for 2t and wind speed, and its tp value equals the evaluator's own functions applied to
    tp clipped at 0 in members and truth."""
    from eval.evaluators.probabilistic.core import compute_probabilistic_scores
    from eval.evaluators.probabilistic.core.scoring import (_domain_mask, _weighted_mean,
                                                            crps_ensemble_components)
    from eval.evaluators.surface.core.compute import _area_weights
    import xarray as xr

    lat, lon = syn.grid(800, seed=1)
    pred = tmp_path / "pred"
    for i, d in enumerate(DATES[:3]):
        syn.write_prediction(pred / f"predictions_{d}_step060.nc", lat=lat, lon=lon, members=6,
                             seed=i, truth_seed=20 + i, bias=0.1 * i)
    domains = ["global", "europe", "n.hem", "tropics", "s.hem"]
    compute_probabilistic_scores(pred, tmp_path / "prob", weather_states=["2t", "10ff", "msl"],
                                 domains=domains)
    ev = {(r["date"], r["weather_state"], r["domain"]): float(r["value"])
          for r in csv.DictReader(open(tmp_path / "prob" / "scores_by_lead.csv")) if r["metric"] == "fcrps"}
    mine = F.per_date_fcrps(pred, ["2t", "ws", "msl"], domains, clip_vars=())
    assert len(mine) == 3 * 3 * len(domains)
    for r in mine:
        k = (r["date"], "10ff" if r["variable"] == "ws" else r["variable"], r["domain"])
        assert r["value"] == pytest.approx(ev[k], rel=1e-12)

    # tp: clipped fallback against the evaluator's functions on clipped inputs
    tp = {(r["date"], r["domain"]): r["value"] for r in F.per_date_fcrps(pred, ["tp"], domains)}
    for path in sorted(pred.glob("*.nc")):
        d = path.name.split("_")[1]
        with xr.open_dataset(path) as ds:
            i = list(ds["weather_state"].values).index("tp")
            f = np.maximum(ds["y_pred"].values[0, :, :, i].astype(float), 0.0)
            y = np.maximum(ds["y"].values[0, 0, :, i].astype(float), 0.0)
            assert (ds["y_pred"].values[0, :, :, i] < 0).any()   # the clip matters here
            w = _area_weights(ds, f.shape[1])
            comp = crps_ensemble_components(f, y)["fcrps"]
            for dom in domains:
                m = _domain_mask(dom, ds["lat_hres"].values, ds["lon_hres"].values)
                assert tp[(d, dom)] == pytest.approx(_weighted_mean(comp, np.where(m, w, 0.0)), rel=1e-12)


# ----------------------------------------------------------------------------- precip tail

def test_tail_percentiles_exact_against_numpy():
    rng = np.random.default_rng(5)
    chunks = [rng.gamma(0.3, 2.0, 5000) - 0.01 for _ in range(8)]
    t = _Tail(keep_frac=0.03)
    for c in chunks:
        t.update(c)
    allv = np.maximum(np.concatenate(chunks), 0)
    st = t.stats()
    assert st["exact"]
    for k, q in (("p99", 99), ("p99.9", 99.9), ("p99.99", 99.99)):
        assert st[k] == pytest.approx(np.percentile(allv, q), rel=1e-12)
    assert st["wet_fraction"] == pytest.approx(np.mean(allv > 0.1))
    assert st["negative_share"] == pytest.approx(np.mean(np.concatenate(chunks) < 0))
    assert st["max"] == pytest.approx(allv.max())


# ----------------------------------------------------------------------------- spectra bin

def test_compare_bin_per_date_rule():
    dates = DATES
    consistent = {"pooled": {("2t", "b"): {"ratio_full": 1.05}},
                  "dated": {("2t", "b"): {d: {"ratio_full": 1.05 + 0.001 * i} for i, d in enumerate(dates)}}}
    noisy = {"pooled": {("2t", "b"): {"ratio_full": 1.05}},
             "dated": {("2t", "b"): {d: {"ratio_full": 1.0 + (0.3 if i % 2 else -0.2)} for i, d in enumerate(dates)}}}
    ref = {"pooled": {("2t", "b"): {"ratio_full": 1.0}},
           "dated": {("2t", "b"): {d: {"ratio_full": 1.0} for d in dates}}}
    c = compare_bin(consistent, ref, "2t", "b", "ratio_full")
    assert c["off"] and c["n"] == 6 and c["mean"] == pytest.approx(0.0525)
    c2 = compare_bin(noisy, ref, "2t", "b", "ratio_full")
    assert c2["mean"] == pytest.approx(0.05) and abs(c2["t"]) < 2.03 and not c2["off"]
    # no per-date values: the 0.03 rule alone on the pooled difference
    c3 = compare_bin({"pooled": consistent["pooled"], "dated": {}}, {"pooled": ref["pooled"], "dated": {}},
                     "2t", "b", "ratio_full")
    assert c3["off"] and c3["rule"].startswith("0.03 rule alone")


# ----------------------------------------------------------------------------- rows file

def test_load_rows_variants(tmp_path):
    p = tmp_path / "rows.yaml"
    p.write_text("rows:\n"
                 "  - {label: edm_pw30c0, arm: edm, sampler: 'edm_piecewise30@S_churn=0', calls: 59, sets: [A, B], step: 150000}\n"
                 "  - {label: mf_j1, arm: meanflow/step_150000, sampler: jumps1, calls: 1, sets: 'A,B'}\n"
                 "  - {label: x, arm: a, sampler: s, calls: 2, sets: {A: /some/dir}}\n")
    rows = load_rows(p)
    assert [r.label for r in rows] == ["edm_pw30c0", "mf_j1", "x"]
    assert rows[0].sampler_dir == "edm_piecewise30_S_churn_0" and rows[0].step == 150000
    assert rows[1].arm == "meanflow" and rows[1].step == 150000 and rows[1].sets == ["A", "B"]
    assert rows[2].pred_dirs == {"A": "/some/dir"} and rows[2].calls == 2
    assert sanitize_spec("edm_piecewise30@S_churn=0") == "edm_piecewise30_S_churn_0"
