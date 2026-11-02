"""End-to-end run of the scoring scripts on small synthetic prediction files (3 rows, 3 dates, +24 h, 10 members, a 1-degree
grid over the box): paired_scores_o2560.py, summarize.py, zoom_maps_o2560.py and spectra_table.py must finish and print
their DONE lines, and the input row must score as a one-member forecast (spread 0, fair CRPS = mean absolute error)."""
import csv
import json
import subprocess
import sys

import numpy as np
import pytest
from conftest import KIT_DIR
from o2560_common import OUTPUT_VARS, case_seeds

pytest.importorskip("netCDF4")
from o2560_box_predict import write_nc  # noqa: E402

DATES = ["20241101", "20241102", "20241103"]
ROWS = ("A100", "B24", "B50")


def make_rows(root):
    la, lo = np.meshgrid(np.arange(10.0, 70.01, 1.0), np.arange(-80.0, 40.01, 1.0), indexing="ij")
    lat, lon = la.ravel(), lo.ravel(); P = lat.size; S = len(OUTPUT_VARS)
    rng = np.random.default_rng(3)
    base = {"10u": 5, "10v": -3, "2d": 280, "2t": 285, "msl": 101000, "skt": 286, "tcw": 20, "cp": 0.0005, "tp": 0.001,
            "z_500": 55000, "t_850": 275}
    dirs = {}
    for d in DATES:
        y = np.stack([base[v] + rng.normal(scale=abs(base[v]) * 0.005 + 1e-4, size=P) for v in OUTPUT_VARS], -1)
        y[:, OUTPUT_VARS.index("tp")] = np.abs(y[:, OUTPUT_VARS.index("tp")]); y[:, OUTPUT_VARS.index("cp")] = np.abs(y[:, OUTPUT_VARS.index("cp")])
        y[np.argmin((lat - 55) ** 2 + (lon + 20) ** 2), OUTPUT_VARS.index("msl")] = 97000.0   # a deep low
        xi = y + rng.normal(scale=0.01, size=y.shape) * np.abs(y).mean(0)
        xi[:, [OUTPUT_VARS.index("tp"), OUTPUT_VARS.index("cp")]] = np.abs(xi[:, [OUTPUT_VARS.index("tp"), OUTPUT_VARS.index("cp")]])
        for i, r in enumerate(ROWS):
            yp = y[None] + rng.normal(scale=0.02 * (i + 1), size=(10, P, S)) * np.abs(y).mean(0)
            yp[..., [OUTPUT_VARS.index("tp"), OUTPUT_VARS.index("cp")]] = np.abs(yp[..., [OUTPUT_VARS.index("tp"), OUTPUT_VARS.index("cp")]])
            dd = root / "pred" / f"{r}_L024"; dd.mkdir(parents=True, exist_ok=True); dirs[r] = dd
            write_nc(dd / f"predictions_{d}_step024.nc", lat, lon, list(OUTPUT_VARS), y.astype(np.float32), xi.astype(np.float32),
                     yp.astype(np.float32), 0, 24, dict(model=r, ckpt_step=1, sampler="s", seeds=json.dumps(case_seeds(d, 24, 10)),
                                                        weights_sha256="0" * 64, code_pin="x" * 40, complete="yes"))
    return dirs


def run(args):
    r = subprocess.run([sys.executable] + [str(a) for a in args], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    return r.stdout


def test_scoring_scripts_end_to_end(tmp_path):
    dirs = make_rows(tmp_path)
    P = tmp_path / "pairs" / "L024"
    out = run([KIT_DIR / "scripts" / "paired_scores_o2560.py", 24, P] + [f"{r}={d}" for r, d in dirs.items()])
    assert "PAIRED_SCORES_O2560_DONE" in out
    rows = list(csv.DictReader(open(P / "per_date.csv")))
    sp_in = [float(r["value"]) for r in rows if r["row"] == "IN" and r["metric"] == "spread"]
    assert sp_in and max(sp_in) == 0.0
    assert {r["row"] for r in rows} == {"A100", "B24", "B50", "IN", "truth"}
    assert "A100-B24" in (P / "paired.md").read_text() and "p99.99" in (P / "tail.md").read_text()
    out = run([KIT_DIR / "scripts" / "summarize.py", tmp_path, "24"])
    assert "SUMMARY_DONE" in out and (tmp_path / "summary" / "summary_box.md").exists()
    pytest.importorskip("matplotlib")
    out = run([KIT_DIR / "scripts" / "zoom_maps_o2560.py", 24, tmp_path / "maps"] + [f"{r}={d}" for r, d in dirs.items()])
    assert "ZOOM_MAPS_DONE" in out
    cases = json.loads((tmp_path / "maps" / "cases.json").read_text())
    assert cases["storm"]["is_storm"] and abs(cases["storm"]["truth_msl_min_hpa"] - 970.0) < 1e-6
    assert len(list((tmp_path / "maps").glob("*.png"))) == 4


def test_spectra_table(tmp_path):
    cols = ["scope", "variable", "bin_lo_km", "bin_hi_km", "bin_centre_km", "amp_resid_pred", "amp_resid_true", "ratio_resid",
            "amp_pred_full", "amp_truth_full", "ratio_full", "amp_interp", "ratio_interp_to_truth", "coherence", "aligned", "n_draws"]
    args = []
    for r in ROWS:
        f = tmp_path / f"{r}.csv"
        with open(f, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
            for v in ("ws", "tp"):
                for km in (100.0, 20.0, 10.0):
                    w.writerow({**{c: 1.0 for c in cols}, "scope": "all", "variable": v, "bin_centre_km": km, "n_draws": 120})
        args.append(f"{r}={f}")
    out = run([KIT_DIR / "scripts" / "spectra_table.py", 24, tmp_path / "o"] + args)
    assert "SPECTRA_TABLE_DONE" in out and "(model only)" in (tmp_path / "o" / "spectra.md").read_text()


def test_box_spectra_runner_reads_the_kit_layout(tmp_path):
    pytest.importorskip("eval.evaluators.box_spectra.runner")
    from eval.evaluators.box_spectra.runner import run as bs_run
    dirs = make_rows(tmp_path)
    o = bs_run(str(dirs["A100"]), {}, {"steps": [24], "variables": ["ws", "tp", "2t"], "res_deg": 0.5}, output_dir=str(tmp_path / "bs"),
               overwrite=True, run_label="A100_L024")
    rows = list(csv.DictReader(open(o / "bins.csv")))
    pooled = [r for r in rows if r["scope"] == "all"]
    assert {r["variable"] for r in pooled} == {"ws", "tp", "2t"}
    assert all(int(r["n_draws"]) == 30 for r in pooled)   # 3 dates x 10 members
