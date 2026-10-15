"""End-to-end test of aggregate.py on a synthetic EVAL_ROOT and prediction tree.

Three rows: the reference ``edm_pw30c0`` (59 calls), ``mf_j1`` (1 call) with every output, and
``sf_j2`` (2 calls) with almost nothing (and one corrupt file), which must give "Not available"
cells without stopping the run. probabilistic outputs are made by the real evaluator on
synthetic prediction files; the other evaluators' outputs are written with their real columns.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
for p in (str(HERE.parent), str(HERE.parents[2])):
    if p not in sys.path:
        sys.path.insert(0, p)

import aggregate  # noqa: E402
from aggregate_lib import fcrps as F  # noqa: E402
from aggregate_lib import synthetic as syn  # noqa: E402
from aggregate_lib.scores import paired_test  # noqa: E402

REF, MF, SF = "edm_pw30c0", "mf_j1", "sf_j2"
DATES = ["20220115", "20220415", "20220715", "20220819", "20221015"]
ROWS = {REF: dict(arm="edm", sampler="edm_piecewise30@S_churn=0", calls=59),
        MF: dict(arm="meanflow", sampler="jumps1", calls=1),
        SF: dict(arm="soflow", sampler="jumps2", calls=2)}


def _read(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bigeval")
    eval_root, pred_root, out = tmp / "eval", tmp / "pred", tmp / "out"
    lat, lon = syn.grid(1200, seed=2)
    rows_yaml = tmp / "rows.yaml"
    rows_yaml.write_text("rows:\n" + "".join(
        f"  - {{label: {lab}, arm: {d['arm']}, sampler: '{d['sampler']}', calls: {d['calls']}, "
        f"step: 150000, sets: [A, B]}}\n" for lab, d in ROWS.items()))

    from eval.evaluators.probabilistic.core import compute_probabilistic_scores
    pdirs = {}
    for k, (lab, d) in enumerate(ROWS.items()):
        if lab == SF:
            continue
        stepdir = pred_root / "A" / d["arm"] / "step_150000"
        pdir = stepdir / d["sampler"].replace("@", "_").replace("=", "_") / "predictions"
        for i, date in enumerate(DATES):
            syn.write_prediction(pdir / f"predictions_{date}_step060.nc", lat=lat, lon=lon, members=4,
                                 seed=100 * k + i, truth_seed=10 + i,
                                 bias=0.0 if lab == REF else 0.15 + 0.01 * i)
        pdirs[lab] = pdir
        with open(stepdir / "costs.jsonl", "w") as fh:
            for m in range(4):
                fh.write(json.dumps({"spec": d["sampler"], "sample": 0, "member": m, "seed": 756001 + m,
                                     "calls": d["calls"], "seconds": 2.0 * d["calls"] + m,
                                     **({"peak_mem_gb": 30.5} if lab == REF else {})}) + "\n")
                fh.write(json.dumps({"spec": "other_spec", "calls": 999, "seconds": 999}) + "\n")
        ev = eval_root / "A" / lab / "evaluators"
        compute_probabilistic_scores(pdir, ev / "probabilistic",
                                     weather_states=["2t", "10ff", "msl"], domains=["global", "europe"])
        (ev / "precip_dist" / "plots").mkdir(parents=True)

    # ---- box spectra (two views) and global spectra
    def ratio_mf(v, km, scope):
        if v == "2t" and km == 100.0:          # consistent +0.05 on every date -> off the reference
            return 1.05, 1.0
        if v == "10u" and km == 60.0:          # resid: +0.05 pooled, noisy per date -> not off
            if scope == "all":
                return 1.0, 1.05
            i = DATES.index(scope)
            return 1.0, 1.0 + (0.3 if i % 2 else -0.2)
        if v == "10u" and km == 400.0:         # outside the truth band, same as the reference
            return 0.95, 1.0
        return 1.0, 1.0

    def ratio_ref(v, km, scope):
        return (0.95, 1.0) if (v == "10u" and km == 400.0) else (1.0, 1.0)

    for lab, fn in ((REF, ratio_ref), (MF, ratio_mf)):
        ev = eval_root / "A" / lab / "evaluators"
        syn.box_bins(ev / "box_spectra", dates=DATES, ratio=fn)
        syn.box_bins(ev / "box_spectra__natl", dates=DATES, box=(35, 55, -45, -20))
        syn.coherence_json(ev / "spectra_coherence", amp=1.0 if lab == REF else 1.1)

    # ---- shape rows.csv (real columns)
    from eval.evaluators.shape.fullrung import CSV_COLS
    for lab, (fr, nc) in ((REF, (0.20, 20)), (MF, (0.30, 20))):
        rows = []
        for i, date in enumerate(DATES):
            for m in range(3):
                for field, (f_, n_) in (("draw", (fr, nc)), ("truth", (0.28, 20)), ("driver", (0.1, 10))):
                    rows.append(dict(arm=lab, dirname="x", date=date, step="060", var="10u", member=m,
                                     field=field, band="mid", elong_frac_gt3=f_ + 0.01 * ((i + m) % 3),
                                     n_components=n_, ell_ratio_median=1.5))
        syn.write_csv(eval_root / "A" / lab / "evaluators" / "shape" / "rows.csv", CSV_COLS, rows)

    # ---- texture and wind_extremes
    for lab, nn in ((REF, 0.60), (MF, 0.50)):
        agg = [{"state": "10u", "stratum": "europe", "n_samples": 50,
                "truth": {"fine_nn_corr": {"mean": 0.62, "sd": 0.05, "n": 50}, "fine_var": {"mean": 1.0, "sd": 0.1, "n": 50}},
                "model": {"fine_nn_corr": {"mean": nn, "sd": 0.05, "n": 50}, "fine_var": {"mean": 0.9, "sd": 0.1, "n": 50}}}]
        syn.write_json(eval_root / "A" / lab / "evaluators" / "texture" / "texture.json",
                       {"strata": {"europe": {"kind": "region"}}, "aggregate": agg})
        syn.write_json(eval_root / "A" / lab / "evaluators" / "wind_extremes" / "wind_extremes.json",
                       {"aggregate": [{"box": "north_sea", "n_samples": 40, "model_minus_truth": {
                           "retention": {"30": {"mean": -0.05 if lab == MF else -0.01, "sd": 0.02, "n": 40}}}}]})
    # corrupt texture for sf_j2 (one row unreadable must not blank the others)
    (eval_root / "A" / SF / "evaluators" / "texture").mkdir(parents=True)
    (eval_root / "A" / SF / "evaluators" / "texture" / "texture.json").write_text("{not json")

    # ---- set B: tc_structure cases.csv and tc stats.json
    from eval.evaluators.tc_structure.runner import CASE_COLS
    for lab in (REF, MF):
        cases = []
        for date in ("20230826", "20230827"):
            for lead in (24, 48):
                for m in range(5):
                    base = 990.0 + 0.1 * m + (0.2 if date.endswith("7") else 0.0)
                    off = (5.0 + 0.1 * (m - 2)) if lab == MF else 0.0
                    for field, p in (("model", base - off), ("truth", 982.0), ("input", 995.0)):
                        cases.append(dict(arm=lab, event="idalia", date=date, lead=lead, field=field, member=m,
                                          storm_ok=1, found=1, pmin_hpa=p, maxwind300_ms=40.0,
                                          zeta100_s=2e-4, zeta200_s=1e-4, zeta50_s=3e-4, rmw_km=60.0))
        syn.write_csv(eval_root / "B" / lab / "evaluators" / "tc_structure" / "cases.csv", CASE_COLS, cases)
        syn.write_json(eval_root / "B" / lab / "evaluators" / "tc" / "stats.json", {"events": {"idalia": {
            "analysis_key": "OPER_O320_0001",
            "extreme_tail": {"rows": [
                {"exp": "OPER_O320_0001", "mslp_min": 980.0, "mslp_p001": 982.0, "mslp_p01": 985.0,
                 "wind_p9999": 40.0, "wind_max": 45.0},
                {"exp": "input", "mslp_min": 995.0, "wind_max": 30.0},
                {"exp": lab, "mslp_min": 990.0 if lab == MF else 985.0, "mslp_p001": 990.0, "mslp_p01": 991.0,
                 "wind_p9999": 35.0, "wind_max": 40.0 if lab == MF else 42.0}]}}}})

    aggregate.main(["--eval-root", str(eval_root), "--rows", str(rows_yaml), "--reference", REF,
                    "--pred-root", str(pred_root), "--out", str(out)])
    return {"out": out, "pdirs": pdirs, "eval_root": eval_root}


def test_all_outputs_written(built):
    out = built["out"]
    for name in ("1_structure.md", "2_spectra.md", "3_extremes.md", "4_scores.md", "5_costs.md", "summary.md"):
        assert (out / name).exists(), name
        assert (out / name).read_text().strip()
    for name in ("structure_shape", "structure_tc_structure", "structure_texture", "structure_wind_extremes",
                 "extremes_tc", "extremes_precip", "scores_by_variable_domain", "scores_paired_test", "costs"):
        assert (out / "csv" / f"{name}.csv").exists(), name
    # nothing crashed into a section-level failure
    for name in ("1_structure.md", "2_spectra.md", "3_extremes.md", "4_scores.md", "5_costs.md"):
        assert "Traceback" not in (out / name).read_text()


def test_paired_test_values_from_evaluator(built):
    rows = _read(built["out"] / "csv" / "scores_paired_test.csv")
    r = next(x for x in rows if x["row"] == MF and x["variable"] == "2t" and x["domain"] == "global")
    assert r["source"] == "probabilistic scores_by_lead.csv"
    ev = built["eval_root"] / "A"
    get = lambda lab: {(x["date"], x["step"]): float(x["value"])  # noqa: E731
                       for x in _read(ev / lab / "evaluators" / "probabilistic" / "scores_by_lead.csv")
                       if x["metric"] == "fcrps" and x["weather_state"] == "2t" and x["domain"] == "global"}
    a, b = get(MF), get(REF)
    p = [100 * (a[k] - b[k]) / b[k] for k in sorted(a)]
    mean = sum(p) / len(p)
    sd = math.sqrt(sum((x - mean) ** 2 for x in p) / (len(p) - 1))
    assert int(r["n"]) == len(DATES)
    assert float(r["mean % (row-ref)/ref"]) == pytest.approx(mean, abs=0.006)
    assert float(r["t"]) == pytest.approx(mean / (sd / math.sqrt(len(p))), abs=0.006)
    assert r["ref better (k of n)"] == f"{sum(1 for k in a if b[k] < a[k])} of {len(DATES)}"
    # a biased row is clearly worse on every date
    assert r["established"] == "yes" and r["status"] == "worse"
    # ws is the evaluator's 10ff
    assert any(x["row"] == MF and x["variable"] == "ws" and x["source"].startswith("probabilistic") for x in rows)


def test_paired_test_fallback_for_tp(built):
    rows = _read(built["out"] / "csv" / "scores_paired_test.csv")
    r = next(x for x in rows if x["row"] == MF and x["variable"] == "tp" and x["domain"] == "global")
    assert r["source"] == "fallback: prediction files"
    a = {(x["date"], str(x["step"])): x["value"] for x in F.per_date_fcrps(built["pdirs"][MF], ["tp"], ["global"])}
    b = {(x["date"], str(x["step"])): x["value"] for x in F.per_date_fcrps(built["pdirs"][REF], ["tp"], ["global"])}
    t = paired_test(a, b)
    assert float(r["mean % (row-ref)/ref"]) == pytest.approx(t["mean_pct"], abs=0.006)
    # a row without predictions or probabilistic output: Not available, no crash
    s = [x for x in rows if x["row"] == SF]
    assert s and all(x["n"].startswith("Not available") for x in s)


def test_scores_table(built):
    rows = _read(built["out"] / "csv" / "scores_by_variable_domain.csv")
    r = next(x for x in rows if x["row"] == MF and x["variable"] == "2t" and x["domain"] == "global")
    assert float(r["fCRPS %"]) > 0
    assert float(r["fCRPS %"]) == pytest.approx(100 * (float(r["fCRPS"]) / float(r["fCRPS ref"]) - 1), abs=0.05)
    assert any(x["row"] == SF and x["fCRPS"].startswith("Not available") for x in rows)


def test_spectra_box_off_reference_and_truth_band(built):
    out = built["out"] / "csv"
    long = _read(out / "spectra_box_A_2t_ratio_full_vs_ref_long.csv")
    r = next(x for x in long if x["row"] == MF and float(x["bin_centre_km"]) == 100.0)
    assert r["off_ref"] == "1" and float(r["diff_perdate_mean"]) == pytest.approx(0.05)
    long_u = _read(out / "spectra_box_A_10u_ratio_resid_vs_ref_long.csv")
    r2 = next(x for x in long_u if x["row"] == MF and float(x["bin_centre_km"]) == 60.0)
    assert r2["off_ref"] == "0" and float(r2["diff_pooled"]) == pytest.approx(0.05)
    outside = _read(out / "spectra_box_A_outside_truth.csv")
    assert {(x["row"], x["variable"], x["km"]) for x in outside if x["quantity"] == "ratio_full"} >= {
        (REF, "10u", "400"), (MF, "10u", "400"), (MF, "2t", "100")}
    wide = _read(out / "spectra_box_A_2t_ratio_full.csv")
    assert [x["row"] for x in wide] == [REF, MF]
    assert float(next(x for x in wide if x["row"] == MF)["100 km"]) == pytest.approx(1.05)
    # the second view is found and tabulated separately
    assert (out / "spectra_box_A_natl_2t_ratio_full.csv").exists()
    summ = (built["out"] / "summary.md").read_text()
    assert "box A: fail (1 gating bins off ref" in summ
    assert "box A/natl: pass" in summ


def test_spectra_global(built):
    out = built["out"] / "csv"
    wide = _read(out / "spectra_global_A_2t_ratio_full.csv")
    mf = next(x for x in wide if x["row"] == MF)
    vals = [float(v) for k, v in mf.items() if k != "row" and v]
    assert vals and all(v == pytest.approx(1.1) for v in vals)
    long = _read(out / "spectra_global_A_2t_ratio_full_vs_ref_long.csv")
    assert all(x["off_ref"] == "1" and x["rule"].startswith("0.03 rule alone") for x in long if x["row"] == MF)
    al = _read(out / "spectra_global_A_2t_aligned.csv")
    assert float([v for k, v in next(x for x in al if x["row"] == MF).items() if k != "row"][0]) == pytest.approx(0.99)
    assert (out / "spectra_global_A_per_degree.csv").exists()


def test_structure_tables(built):
    out = built["out"] / "csv"
    shape = _read(out / "structure_shape.csv")
    ef = next(x for x in shape if x["row"] == MF and x["statistic"] == "elong_frac_gt3")
    assert float(ef["diff row-ref"]) == pytest.approx(0.10, abs=1e-6)
    assert float(ef["truth"]) == pytest.approx(0.28 + 0.01 * np.mean([(i + m) % 3 for i in range(5) for m in range(3)]))
    assert ef["differs (2 SE)"] == "yes" and ef["direction"].startswith("towards truth; gain")
    ne = next(x for x in shape if x["row"] == MF and x["statistic"] == "n_elong")
    assert float(ne["diff row-ref"]) == pytest.approx(2.0, abs=1e-6)     # 0.1 x 20 features
    tcs = _read(out / "structure_tc_structure.csv")
    pm = next(x for x in tcs if x["row"] == MF and x["lead"] == "all" and x["quantity"].startswith("Pmin"))
    assert float(pm["diff row-ref"]) == pytest.approx(-5.0, abs=0.01)
    assert pm["n members"] == "5" and pm["status"] == "gain"           # deeper, towards the truth 982
    mw = next(x for x in tcs if x["row"] == MF and x["lead"] == "all" and x["quantity"].startswith("max wind"))
    assert mw["status"] == "no difference"
    tex = _read(out / "structure_texture.csv")
    t = next(x for x in tex if x["row"] == MF and x["statistic"] == "fine_nn_corr")
    assert float(t["row-ref"]) == pytest.approx(-0.10) and float(t["truth value"]) == pytest.approx(0.62)
    assert any(x["row"] == SF and "unreadable input" in x["state"] for x in tex)
    we = _read(out / "structure_wind_extremes.csv")
    w = next(x for x in we if x["row"] == MF)
    assert float(w["row - ref"]) == pytest.approx(-0.04) and float(w["row SE"]) == pytest.approx(0.02 / math.sqrt(40), abs=1e-4)


def test_extremes_tables(built):
    out = built["out"] / "csv"
    tc = _read(out / "extremes_tc.csv")
    r = next(x for x in tc if x["row"] == MF and x["quantity"].startswith("min msl"))
    assert float(r["row - ref"]) == pytest.approx(5.0) and float(r["row - analysis"]) == pytest.approx(10.0)
    assert r["status"].startswith("beyond threshold, away from analysis")
    w = next(x for x in tc if x["row"] == MF and x["quantity"].startswith("wind max"))
    assert float(w["row - ref %"]) == pytest.approx(100 * (40 / 42 - 1), abs=0.01)
    pr = _read(out / "extremes_precip.csv")
    wet = next(x for x in pr if x["row"] == MF and x["quantity"].startswith("wet"))
    import xarray as xr
    vals, truth = [], []
    for f in sorted(built["pdirs"][MF].glob("*.nc")):
        with xr.open_dataset(f) as ds:
            i = list(ds["weather_state"].values).index("tp")
            vals.append(ds["y_pred"].values[0, :, :, i].ravel() * 1000.0)
            truth.append(ds["y"].values[0, :, :, i].ravel() * 1000.0)   # every member, as tail_stats
    v = np.concatenate(vals).astype(float)
    assert float(wet["row value"]) == pytest.approx(np.mean(np.maximum(v, 0) > 0.1), abs=1e-4)
    p999 = next(x for x in pr if x["row"] == MF and x["quantity"].startswith("p99.9 "))
    assert float(p999["row value"]) == pytest.approx(np.percentile(np.maximum(v, 0), 99.9), rel=1e-3)
    assert float(p999["truth value"]) == pytest.approx(np.percentile(np.maximum(np.concatenate(truth).astype(float), 0), 99.9), rel=1e-3)
    neg = next(x for x in pr if x["row"] == MF and x["quantity"].startswith("negative"))
    assert float(neg["row value"]) == pytest.approx(np.mean(v < 0), abs=1e-4)
    assert wet["source"] == "fallback: prediction files"


def test_costs_and_summary(built):
    costs = _read(built["out"] / "csv" / "costs.csv")
    ref = next(x for x in costs if x["row"] == REF)
    assert float(ref["calls (logged, mean)"]) == 59
    assert float(ref["seconds/member mean"]) == pytest.approx(2 * 59 + 1.5)
    assert float(ref["peak memory GB"]) == pytest.approx(30.5)
    mf = next(x for x in costs if x["row"] == MF)
    assert mf["peak memory GB"].startswith("Not available")
    assert next(x for x in costs if x["row"] == SF)["calls (logged, mean)"].startswith("Not available")
    s = (built["out"] / "summary.md").read_text()
    pos = [s.index(h) for h in ("## 1 call per member", "## 2 calls per member", "## 59 calls per member")]
    assert pos == sorted(pos)
    block = s[s.index(f"### `{MF}`"):s.index("## 2 calls")]
    for part in ("Structure", "Spectra", "Extremes", "Scores", "Costs"):
        assert f"- **{part}**:" in block
    assert "global: 10u +" in block and "; 2t +" in block and "[W]" in block
    sf_block = s[s.index(f"### `{SF}`"):s.index("## 59 calls")]
    assert "Not available" in sf_block


def test_empty_inputs_never_crash(tmp_path):
    rows = tmp_path / "rows.yaml"
    rows.write_text("- {label: edm_pw30c0, arm: edm, sampler: x, calls: 59, sets: [A]}\n"
                    "- {label: r1, arm: a, sampler: y, calls: 1, sets: [A]}\n")
    out = tmp_path / "out"
    assert aggregate.main(["--eval-root", str(tmp_path / "nothing"), "--rows", str(rows),
                           "--reference", "edm_pw30c0", "--out", str(out)]) == 0
    for name in ("1_structure.md", "2_spectra.md", "3_extremes.md", "4_scores.md", "5_costs.md", "summary.md"):
        assert "Not available" in (out / name).read_text(), name
