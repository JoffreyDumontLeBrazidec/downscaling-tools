"""The aggregator on outputs of the REAL writers: spectra_coherence (curves.csv, per date, with
the residual) and precip_dist tail_stats.csv, on harness-schema files on octahedral grids."""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
for p in (str(HERE.parent), str(HERE.parents[2])):
    if p not in sys.path:
        sys.path.insert(0, p)

pytest.importorskip("healpy")

import aggregate  # noqa: E402
from aggregate_lib.common import Context, Found, load_rows  # noqa: E402
from aggregate_lib.extremes import precip_from_evaluator, precip_stats  # noqa: E402
from aggregate_lib.spectra import read_global  # noqa: E402

REF, ROW = "edm_pw30c0", "mf_j1"
DATES = ("20220115", "20220415", "20220819")


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    from eval.evaluators.precip_dist.core.tail_stats import compute_tail_stats
    from eval.evaluators.spectra_coherence import runner as SC
    from eval.tests.synthetic_predictions import octahedral_grid, write_predictions_file

    tmp = tmp_path_factory.mktemp("realwriters")
    grids = (octahedral_grid(48), octahedral_grid(16))
    rows = tmp / "rows.yaml"
    rows.write_text(f"- {{label: {REF}, arm: edm, sampler: s30, calls: 59, step: 1, sets: [A]}}\n"
                    f"- {{label: {ROW}, arm: mf, sampler: jumps1, calls: 1, step: 1, sets: [A]}}\n")
    pdirs = {}
    for k, (lab, arm, samp) in enumerate(((REF, "edm", "s30"), (ROW, "mf", "jumps1"))):
        pdir = tmp / "pred" / "A" / arm / "step_000001" / samp / "predictions"
        for i, d in enumerate(DATES):
            write_predictions_file(pdir / f"predictions_{d}_step060.nc", grids=grids,
                                   states=("10u", "10v", "2t", "tp"), members=(1, 2, 3),
                                   seed=10 * k + i)
        pdirs[lab] = pdir
        ev = tmp / "eval" / "A" / lab / "evaluators"
        SC.run(pdir, {}, {"weather_states": ["10u", "2t"], "nside": "auto", "steps": [60],
                          "per_date": True}, output_dir=ev / "spectra_coherence")
        (ev / "precip_dist" / "plots").mkdir(parents=True)
        if lab == ROW:
            compute_tail_stats(pdir, ev / "precip_dist")
    out = tmp / "out"
    aggregate.main(["--eval-root", str(tmp / "eval"), "--rows", str(rows), "--reference", REF,
                    "--pred-root", str(tmp / "pred"), "--out", str(out), "--only", "spectra,extremes",
                    "--global-lmin", "2", "--global-nbins", "12"])
    return {"tmp": tmp, "out": out, "pdirs": pdirs, "eval": tmp / "eval"}


def test_global_bins_from_curves_csv(tree):
    found = Found("A", "", tree["eval"] / "A" / ROW / "evaluators" / "spectra_coherence")
    g = read_global(found, lmin=2, nbins=12)
    assert g["source"] == "curves.csv" and g["has_resid"] and g["dated"]
    with open(found.path / "curves.csv") as fh:
        rows = [r for r in csv.DictReader(fh) if r["scope"] == "all" and r["state"] == "2t"]
    for (state, b), rec in g["pooled"].items():
        if state != "2t":
            continue
        lo, hi = (int(x) for x in b[1:].split("-"))
        sel = [r for r in rows if lo <= int(r["ell"]) <= hi]
        pt = sum(float(r["P_true"]) for r in sel)
        pp = sum(float(r["P_pred"]) for r in sel)
        rp = sum(float(r["P_resid_pred"]) for r in sel)
        rt = sum(float(r["P_resid_true"]) for r in sel)
        x = sum(float(r["coherence"]) * math.sqrt(float(r["P_pred"]) * float(r["P_true"])) for r in sel)
        assert rec["ratio_full"] == pytest.approx(math.sqrt(pp / pt), rel=1e-9)
        assert rec["ratio_resid"] == pytest.approx(math.sqrt(rp / rt), rel=1e-9)
        assert rec["coherence"] == pytest.approx(x / math.sqrt(pp * pt), rel=1e-9)
        assert rec["aligned"] == pytest.approx(rec["coherence"] * rec["ratio_full"], rel=1e-12)
        if lo == hi:   # a one-degree bin equals the tool's own per-degree ratio
            assert rec["ratio_full"] == pytest.approx(float(sel[0]["amplitude_ratio"]), rel=1e-5)
    # per-date rows exist for every date
    assert {d for v in g["dated"].values() for d in v} == set(DATES)


def test_global_tables_use_per_date_rule_and_residual(tree):
    csvd = tree["out"] / "csv"
    long = list(csv.DictReader(open(csvd / "spectra_global_A_2t_ratio_resid_vs_ref_long.csv")))
    assert long and all(r["rule"] == "per-date" for r in long if r["row"] == ROW)
    assert all(int(r["n_dates"]) == len(DATES) for r in long if r["row"] == ROW)
    assert (csvd / "spectra_global_A_2t_ratio_resid.csv").exists()
    md = (tree["out"] / "2_spectra.md").read_text()
    assert "Read from curves.csv" in md and "per-date rule applies" in md


def test_precip_tail_stats_csv_agrees_with_fallback(tree):
    ev = precip_from_evaluator(Found("A", "", tree["eval"] / "A" / ROW / "evaluators" / "precip_dist"))
    fb = precip_stats(tree["pdirs"][ROW])
    for src in ("model", "truth", "input"):
        for k in ("wet_fraction", "p99", "p99.9", "p99.99", "max", "negative_share"):
            a, b = ev[src][k], fb[src][k]
            assert a == pytest.approx(b, rel=2e-5, abs=1e-9), (src, k)
    rows = list(csv.DictReader(open(tree["out"] / "csv" / "extremes_precip.csv")))
    srcs = {r["row"]: r["source"] for r in rows}
    assert srcs[ROW] == "precip_dist tail_stats.csv"
    assert srcs[REF] == "fallback: prediction files"


def test_context_finds_views_and_costs(tmp_path):
    rows = tmp_path / "rows.yaml"
    rows.write_text("- {label: r, arm: a, sampler: 'x@k=1', calls: 1, step: 5, sets: [A]}\n")
    ev = tmp_path / "eval" / "A" / "r"
    for p in ("evaluators/box_spectra", "evaluators/box_spectra__natl", "alps/evaluators/box_spectra",
              "evaluators/tc_structure", "evaluators/tc"):
        (ev / p).mkdir(parents=True)
    for p in ("evaluators/box_spectra", "evaluators/box_spectra__natl", "alps/evaluators/box_spectra"):
        (ev / p / "bins.csv").write_text("x\n")
    (ev / "evaluators" / "tc" / "stats.json").write_text("{}")
    step = tmp_path / "pred" / "A" / "a" / "step_000005"
    (step / "x_k_1" / "predictions").mkdir(parents=True)
    (step / "costs.jsonl").write_text("")
    ctx = Context(eval_root=tmp_path / "eval", rows=load_rows(rows), reference="r",
                  out_dir=tmp_path / "out", pred_roots={"": tmp_path / "pred"})
    keys = sorted(f.key for f in ctx.results("r", "box_spectra", "bins.csv"))
    assert keys == ["A", "A/alps", "A/natl"]
    # "tc" never matches "tc_structure"
    assert [f.path.name for f in ctx.results("r", "tc", "stats.json")] == ["tc"]
    assert ctx.costs_file(ctx.by_label["r"], "A") == step / "costs.jsonl"
    assert ctx.pred_dir(ctx.by_label["r"], "A") is None      # no prediction file in it
