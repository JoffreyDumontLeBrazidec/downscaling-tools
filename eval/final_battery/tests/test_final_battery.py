"""Tests of the final-model battery: the opt-in gate, the plan, the summary against the 17 Nov reference tables,
and (slow) a full run on a synthetic copy of the Jupiter stores."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from eval.final_battery import references, runner, summary

REPO = Path(__file__).resolve().parents[3]


def _docs_support() -> Path | None:
    """The docs repo's support folder of the 17 Nov realism reads (DOCS_ROOT, ~/dev/docs, or a sibling checkout)."""
    rel = Path("epics/fine-scale-o320-o1280/in-progress/support")
    for root in (os.environ.get("DOCS_ROOT"), Path.home() / "dev" / "docs", REPO.parent / "docs"):
        if root and (Path(root) / rel / "20261117_realism_t2m" / "tables").exists():
            return Path(root) / rel
    return None


def _ref_tree(tmp_path: Path, s: Path) -> Path:
    """A battery output tree made of links onto the committed GLB450e2_m reference tables."""
    out = tmp_path / "ref"
    t = s / "20261117_realism_t2m" / "tables"
    for sub, target in (("terrain/tables", t), ("terrain/z/tables", t / "z"), ("terrain/z6/tables", t / "z"), ("terrain/d2/tables", t / "d2"),
                        ("consistency/edge1", s / "20261117_realism_consistency" / "results"),
                        ("consistency/edge5", s / "20261117_realism_consistency" / "results_edge5")):
        p = out / sub
        p.parent.mkdir(parents=True, exist_ok=True)
        p.symlink_to(target)
    for lead in (24, 72):
        w = out / "wind" / f"out{lead:03d}"
        w.mkdir(parents=True)
        (w / "small").symlink_to(s / "20261117_realism_wind" / f"out{lead}" / "small")
        (w / "cases.csv.gz").symlink_to(s / "20261117_realism_wind" / f"out{lead}" / "cases.csv.gz")
    return out


def test_refuses_without_final_model(tmp_path):
    r = subprocess.run([sys.executable, "-m", "eval.cli", "final_battery", "run", "--row", "X_m", "--pred-root", str(tmp_path),
                        "--out", str(tmp_path / "o")], cwd=REPO, capture_output=True, text=True)
    assert r.returncode != 0
    assert "--final-model is required" in r.stderr + r.stdout


def test_not_in_judge():
    """The battery is never part of a development read."""
    from eval.evaluators import registry
    assert "final_battery" not in registry.runnable_names()
    judge = (REPO / "eval" / "cli" / "judge.py").read_text()
    assert "final_battery" not in judge


def test_plan_lists_every_part(tmp_path):
    cfg = runner.Config(final_model="T", row="X_m", pred_root=tmp_path / "missing", out=tmp_path / "o").resolved()
    steps = runner.build_steps(cfg, {"has_10_outputs": True})
    names = [s.name for s in steps]
    for need in ("terrain.statics", "terrain.t", "terrain.z", "terrain.z6", "terrain.d2", "consistency.edge1", "consistency.edge5",
                 "wind.orog", "wind.extract.24", "wind.accumulate.72", "wind.spread.24"):
        assert need in names, need
    assert "terrain.d2" not in [s.name for s in runner.build_steps(cfg, {"has_10_outputs": False})]
    env = runner.env_for(cfg)
    assert env["FB_ROW"] == "X_m" and env["FB_TERRAIN_OUT"].endswith("o/terrain")


def test_summary_reproduces_reference(tmp_path):
    s = _docs_support()
    if s is None:
        pytest.skip("docs repo with the 17 Nov reference tables not found (set DOCS_ROOT)")
    out = _ref_tree(tmp_path, s)
    sec = summary.build(out)
    n = {ln.id: ln.numbers for lines in sec.values() for ln in lines}
    assert n["T1"]["flagged"] == 15 and set(n["T1"]["by_var"]) == {"sp"}
    assert n["T1b"]["bias"] == pytest.approx(21.1, abs=0.05)
    assert n["T2"]["corr"] == pytest.approx(0.983, abs=0.0005) and n["T2"]["n"] == 1440
    assert n["T3"]["ratio"] == pytest.approx(1.11, abs=0.005)
    assert n["T4"]["by_lead"][12] == pytest.approx(0.064, abs=0.0005)
    assert n["T6"]["mean"] == pytest.approx(-0.14, abs=0.005)
    assert n["Z1a"]["by_lead"][0] == pytest.approx(1.26, abs=0.005) and n["Z1a"]["by_lead"][-1] == pytest.approx(1.01, abs=0.005)
    assert n["Z4"]["coast"] == pytest.approx(1.38, abs=0.005)
    assert [round(x, 2) for x in n["Z6"]["ratios"]] == [1.47, 1.67, 1.69]
    assert [round(x, 2) for x in n["C1.tp"]["by_lead"]][:1] == [1.85]
    assert [round(x, 2) for x in n["C1.msl"]["by_lead"]] == [1.23, 1.22, 1.06, 1.08, 1.03, 1.03]
    assert n["C3"]["ratio"] == pytest.approx(0.41, abs=0.005)
    assert n["C6"]["neg_pct"] == pytest.approx(2.5, abs=0.05)
    assert n["W7.24"]["ws"] == pytest.approx(1.23, abs=0.005) and n["W7.24"]["msl"] == pytest.approx(1.30, abs=0.01)
    assert n["W8.24"]["xt"] == pytest.approx(1.55, abs=0.005) and n["W8.72"]["xt"] == pytest.approx(1.10, abs=0.005)
    page = summary.render(sec, {"final_model": "GL512 450k", "row": "GLB450e2_m"})
    assert "| T1 |" in page and "| C1.tp |" in page and "| W8.24 |" in page


def test_every_line_has_a_reference():
    ids = {"T1", "T2", "T3", "T4", "T7", "Z1a", "Z6", "C1.tp", "C3", "C4", "W7.24", "W8.24"}
    assert ids <= set(references.REF)


@pytest.mark.skipif(not os.environ.get("FB_SYNTHETIC"), reason="about 5 min and 10 GB of memory; set FB_SYNTHETIC=1")
def test_synthetic_run(tmp_path):
    """Every compute step of the three kits on a small synthetic copy of the stores (2 dates, leads 12 and 24)."""
    pytest.importorskip("zarr"); pytest.importorskip("netCDF4"); pytest.importorskip("eccodes"); pytest.importorskip("scipy")
    from eval.final_battery.tests import synthetic
    p = synthetic.build(tmp_path / "syn")
    cfg = runner.Config(final_model="synthetic", row=p["row"], pred_root=Path(p["pred_root"]), out=tmp_path / "out",
                        user_root=p["user_root"], member_stores=p["member_stores"], leads=(12, 24), wind_leads=(24,),
                        workers=2, consistency_workers=2, maps=False, edge5=False)
    rc = runner.run(cfg)
    out = tmp_path / "out"
    failed = sorted(x.name for x in (out / "logs").glob("*.log") if not (out / ".done" / x.stem).exists())
    required = [f for f in failed if not any(f.startswith(o) for o in ("terrain.figs", "consistency.figs", "terrain.print",
                                                                        "terrain.inversions", "terrain.d2.print"))]
    assert required == [], f"failed compute steps: {required} (rc {rc})"
    assert (out / "terrain" / "tables" / "T1_bias_rmse_binned.csv").exists()
    assert (out / "terrain" / "d2" / "tables" / "D4_2d_vs_2t.csv").exists()
    assert (out / "consistency" / "edge1" / "ratios.csv").exists()
    assert (out / "wind" / "out024" / "cases.csv").exists()
    assert (out / "summary.md").exists()
