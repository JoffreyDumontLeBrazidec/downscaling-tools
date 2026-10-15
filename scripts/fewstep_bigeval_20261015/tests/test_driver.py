"""CPU tests of the scoring driver: slim_predictions, score_row (plan, paths, skip) and rows.yaml.

Run: python -m pytest scripts/fewstep_bigeval_20261015/tests/test_driver.py
The synthetic files are built in the tests (eval/tests/synthetic_predictions.py); nothing from
Atos is read. Tests marked ``atos`` need the real Atos files and are skipped elsewhere.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
DRV = HERE.parent
REPO = HERE.parents[2]
for p in (str(DRV), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

netCDF4 = pytest.importorskip("netCDF4")
yaml = pytest.importorskip("yaml")

import score_row as SR  # noqa: E402
import slim_predictions as SP  # noqa: E402
from eval.tests.synthetic_predictions import octahedral_grid, write_predictions_file  # noqa: E402

STATES = ("10u", "10v", "2t", "2d", "msl", "sp", "tp", "cp", "t_850", "z_500", "q_850")
ATOS_PRED = os.environ.get("BIGEVAL_PRED_ROOT", "/home/ecm5702/scratch/agent-work/20261015-fewstep-bigeval/pred")


def _add_member_views(path: Path):
    """Add x_<i>, y_pred_<i>, ... as build_predictions_dataset does."""
    with netCDF4.Dataset(str(path), "a") as d:
        for base in ("x", "y", "x_interp", "y_pred"):
            v = d[base]
            for i in range(len(d.dimensions["ensemble_member"])):
                dims = tuple(x for x in v.dimensions if x != "ensemble_member")
                w = d.createVariable(f"{base}_{i}", "f4", dims, zlib=True)
                w[:] = v[:, i]
                w.setncatts({k: v.getncattr(k) for k in v.ncattrs()})


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("pred")
    grids = (octahedral_grid(24), octahedral_grid(8))
    pdir = root / "setB" / "edm" / "step_150000" / "edm_piecewise30_S_churn_0" / "predictions"
    for i, (date, step) in enumerate((("20230826", 24), ("20230828", 48))):
        f = write_predictions_file(pdir / f"predictions_{date}_step{step:03d}.nc", n_hres=24, n_lres=8,
                                   grids=grids, states=STATES, members=(1, 2, 3),
                                   per_member_truth=True, nan_truth_states=("tp", "cp"), seed=i)
        _add_member_views(f)
    (pdir / "predictions_manifest.csv").write_text("date,step,file\n20230826,24,a\n20230828,48,b\n")
    (pdir.parent.parent / "costs.jsonl").write_text('{"spec": "edm", "calls": 59}\n')
    return root


def test_slim_keeps_states_schema_attrs_and_sidecars(tree, tmp_path):
    dst = tmp_path / "slim"
    assert SP.main(["--src", str(tree), "--dst", str(dst)]) == 0
    rel = Path("setB/edm/step_150000/edm_piecewise30_S_churn_0/predictions")
    src_f = tree / rel / "predictions_20230828_step048.nc"
    dst_f = dst / rel / "predictions_20230828_step048.nc"
    assert (dst / rel / "predictions_manifest.csv").read_text() == (tree / rel / "predictions_manifest.csv").read_text()
    assert (dst / "setB/edm/step_150000/costs.jsonl").exists()
    with netCDF4.Dataset(src_f) as s, netCDF4.Dataset(dst_f) as d:
        s_states = list(s["weather_state"][:])
        d_states = list(d["weather_state"][:])
        assert d_states == [x for x in STATES if x in SP.KEEP]          # source order, q_850 gone
        assert len(d.dimensions["weather_state"]) == len(SP.KEEP)
        for k in s.ncattrs():
            assert d.getncattr(k) == s.getncattr(k)
        assert d.getncattr("slimmed_weather_states") == ",".join(d_states)
        # member views dropped by default, everything else kept with dims, dtype and attrs
        assert not any(SP.MEMBER_VIEW_RE.match(n) for n in d.variables)
        for name, v in s.variables.items():
            if SP.MEMBER_VIEW_RE.match(name):
                continue
            w = d[name]
            assert w.dimensions == v.dimensions and w.dtype == v.dtype, name
            assert {k: str(v.getncattr(k)) for k in v.ncattrs()} == {k: str(w.getncattr(k)) for k in w.ncattrs()}
            if v.dtype != str and v.ndim == 4:
                assert w.filters().get("zlib") == v.filters().get("zlib")
        idx = [s_states.index(x) for x in d_states]
        for name in ("x", "y", "x_interp", "y_pred"):
            a = np.asarray(s[name][:])[..., idx]
            b = np.asarray(d[name][:])
            np.testing.assert_array_equal(a, b)              # NaN tp truth survives as NaN
        assert np.isnan(np.asarray(d["y"][:])[..., d_states.index("tp")]).all()
        np.testing.assert_array_equal(d["lat_hres"][:], s["lat_hres"][:])
        np.testing.assert_array_equal(d["ensemble_member"][:], s["ensemble_member"][:])


def test_slim_refuses_to_overwrite_and_resumes(tree, tmp_path):
    dst = tmp_path / "slim"
    assert SP.main(["--src", str(tree), "--dst", str(dst)]) == 0
    one = next(dst.rglob("predictions_20230826_step024.nc"))
    mtime = one.stat().st_mtime_ns
    assert SP.main(["--src", str(tree), "--dst", str(dst)]) == 3          # refused
    assert one.stat().st_mtime_ns == mtime
    assert SP.main(["--src", str(tree), "--dst", str(dst), "--skip-existing"]) == 0
    assert one.stat().st_mtime_ns == mtime
    assert not list(dst.rglob("*.part"))
    with pytest.raises(FileExistsError):
        SP.slim_file(next(tree.rglob("predictions_20230826_step024.nc")), one, list(SP.KEEP))
    assert SP.main(["--src", str(tree), "--dst", str(tree / "inside")]) == 2


def test_slim_keep_member_views_and_extra(tree, tmp_path):
    dst = tmp_path / "slim"
    assert SP.main(["--src", str(tree), "--dst", str(dst), "--keep-member-views",
                    "--extra", "q_850"]) == 0
    f = next(dst.rglob("predictions_20230826_step024.nc"))
    with netCDF4.Dataset(f) as d:
        assert list(d["weather_state"][:]) == list(STATES)
        assert d["y_pred_2"].shape[-1] == len(STATES)
        np.testing.assert_array_equal(d["y_pred_2"][:], d["y_pred"][:, 2])


def test_slim_output_reads_with_the_evaluator_loader(tree, tmp_path):
    """precip_dist's tail_stats (netCDF4) and xarray both read the slim file."""
    xr = pytest.importorskip("xarray")
    from eval.evaluators.precip_dist.core.tail_stats import compute_tail_stats

    dst = tmp_path / "slim"
    SP.main(["--src", str(tree), "--dst", str(dst)])
    pdir = next(dst.rglob("predictions_manifest.csv")).parent
    out = compute_tail_stats(pdir, tmp_path, steps=[48])
    rows = list(csv.DictReader(open(out)))
    assert {r["series"] for r in rows} == {"pred", "input"}               # truth tp is all NaN
    with xr.open_dataset(next(pdir.glob("*.nc"))) as ds:
        assert ds["y_pred"].sizes["weather_state"] == len(SP.KEEP)


# --------------------------------------------------------------------------------------
# score_row and rows.yaml
# --------------------------------------------------------------------------------------

def test_rows_file_matches_the_preregistered_rows():
    data = yaml.safe_load((DRV / "rows.yaml").read_text())["rows"]
    labels = [r["label"] for r in data]
    assert len(labels) == 20 and len(set(labels)) == 20
    assert labels[0] == "edm_pw30c0"
    calls = {r["label"]: r["calls"] for r in data}
    assert calls["edm_pw30c0"] == 59 and calls["sf_rn4"] == 4 and calls["drift_anneal"] == 1
    assert all(r["group"] == r["calls"] for r in data)
    assert all(set(r["sets"]) == {"A", "B"} for r in data)
    assert SR.sampler_dir("edm_piecewise30@S_churn=0") == "edm_piecewise30_S_churn_0"
    # the aggregator reads the same file
    from aggregate_lib.common import load_rows
    rows = load_rows(DRV / "rows.yaml")
    assert rows[0].sampler_dir == "edm_piecewise30_S_churn_0" and rows[0].step == 150000


def test_plan_has_the_options_the_aggregator_needs(tmp_path):
    a = {(t.evaluator, t.view): t for t in SR.plan("A", eval_root=tmp_path, static_root="/s",
                                                     dates=["20220819"], with_ecmwf_spectra=True)}
    assert a[("probabilistic", "")].opts["domains"][0] == "global"
    assert "10ff" in a[("probabilistic", "")].opts["weather_states"]
    assert a[("spectra_coherence", "")].opts["per_date"] is True
    assert a[("precip_dist", "")].opts["tail_stats"] is True
    assert a[("box_spectra", "")].opts["box"] == [33, 70, -15, 40]
    assert a[("box_spectra", "natl")].opts["box"] == "north_atlantic"
    assert a[("spectra_ecmwf_v2", "")].opts["dates"] == ["20220819"]
    assert all(t.opts.get("steps") == [60] for t in a.values())
    b = [t.evaluator for t in SR.plan("B", eval_root=tmp_path, static_root="/s", dates=[],
                                      with_ecmwf_spectra=False)]
    assert b == ["tc", "tc_structure"]
    # every option round-trips through eval.cli's --opt reader
    from eval.cli._common import parse_opts
    for t in a.values():
        cmd = SR.evaluate_cmd(["python"], tmp_path, tmp_path, t, "r")
        opts = [cmd[i + 1] for i, c in enumerate(cmd) if c == "--opt"]
        got = {k: v for _, k, v in parse_opts(opts)}
        assert got == t.opts, t.evaluator


def test_score_row_paths_dry_run_and_complete_skip(tree, tmp_path, capsys):
    ev = tmp_path / "eval"
    done = ev / "B" / "edm_pw30c0" / "evaluators" / "tc" / ".complete"
    done.parent.mkdir(parents=True)
    done.write_text("")
    rc = SR.main(["--row", "edm_pw30c0", "--set", "B", "--pred-root", str(tree),
                  "--eval-root", str(ev), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "2 files in" in out and "edm/step_150000/edm_piecewise30_S_churn_0/predictions" in out
    assert "[skip] tc:" in out
    assert "--only tc_structure" in out and f"--output-dir {ev}/B/edm_pw30c0" in out
    # a set the row is not on, and an unknown row
    assert SR.main(["--row", "edm_pw30c0", "--set", "A", "--pred-root", str(tree),
                    "--eval-root", str(ev), "--dry-run"]) == 0
    with pytest.raises(SystemExit):
        SR.main(["--row", "nope", "--set", "A", "--pred-root", str(tree), "--eval-root", str(ev)])


def test_submit_all_dry_run_orders_the_reference_first(tree, tmp_path):
    env = dict(os.environ, PRED_ROOT=str(tree), EVAL_ROOT=str(tmp_path / "eval"), DT_ROOT=str(REPO),
               PY=sys.executable)
    r = subprocess.run(["bash", str(DRV / "submit_all.sh"), "--sets", "B"], env=env,
                       capture_output=True, text=True, check=True)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("[dry-run]")]
    assert len(lines) == 1 and "ROW=edm_pw30c0" in lines[0] and "sbatch" in lines[0]
    assert "fm_heun30/B" in r.stdout                                    # listed as missing
    assert "Dry run" in r.stdout


@pytest.mark.skipif(not Path(ATOS_PRED).is_dir(), reason="Atos only: real slimmed prediction tree")
def test_atos_real_tree_has_every_row():  # pragma: no cover - Atos only
    data = yaml.safe_load((DRV / "rows.yaml").read_text())["rows"]
    missing = []
    for r in data:
        for s in r["sets"]:
            d = SR.predictions_dir(r, s, Path(ATOS_PRED), f"set{s}")
            n = len(SR.prediction_files(d))
            if n != (36 if s == "A" else 25):
                missing.append(f"{r['label']}/{s}: {n} files in {d}")
    assert not missing, "\n".join(missing)
