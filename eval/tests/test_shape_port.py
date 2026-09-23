"""Tests of the shape port and of the date-clustered bootstrap used by the realism scores."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from eval.shared.date_bootstrap import boot_mean, boot_mean_diff, boot_slope

ORIG = Path("/home/ecm5702/agent-work/20260917-shape-probes/scripts")
PKG = Path(__file__).resolve().parents[1] / "evaluators" / "shape"


@pytest.mark.skipif(not ORIG.exists(), reason="original instrument not on this host")
@pytest.mark.parametrize("port,orig", [("instrument.py", "shape_stats.py"),
                                       ("fullrung.py", "shape_fullrung.py")])
def test_port_lines_are_verbatim(port, orig):
    """Every kept line of the port occurs in the original, in the same order,
    except the lines marked PORT."""
    o = (ORIG / orig).read_text().split("\n")
    pos = 0
    lines = (PKG / port).read_text().split("\n")
    while lines and (lines[0].startswith("# PORT") or not lines[0].strip()):
        lines.pop(0)                      # the port header and the blank line after it
    for line in lines:
        if "PORT" in line:
            continue
        while pos < len(o) and o[pos] != line:
            pos += 1
        assert pos < len(o), f"{port}: line not found in order in {orig}: {line!r}"
        pos += 1


def test_boot_mean_constant_per_date_has_between_date_error_only():
    dates = np.repeat(["d1", "d2", "d3", "d4", "d5"], 10)
    vals = np.repeat([1.0, 2.0, 3.0, 4.0, 5.0], 10)
    b = boot_mean(dates, vals, n_boot=4000)
    assert b["value"] == pytest.approx(3.0)
    # the between-date error of 5 dates with sd sqrt(2.5): about sqrt(2/5) = 0.63
    assert 0.5 < b["se"] < 0.75
    naive = vals.std(ddof=1) / np.sqrt(vals.size)
    assert b["se"] > 2.5 * naive
    assert b["n"] == 50 and b["n_dates"] == 5


def test_boot_mean_ignores_nan_and_diff_shares_dates():
    dates = np.repeat(["a", "b", "c"], 4)
    x = np.arange(12, dtype=float)
    x[3] = np.nan
    assert boot_mean(dates, x)["n"] == 11
    d = boot_mean_diff(dates, x + 1.0, dates, x)
    assert d["value"] == pytest.approx(1.0)
    assert d["se"] == pytest.approx(0.0, abs=1e-12)


def test_boot_slope_recovers_line():
    rng = np.random.default_rng(0)
    dates = np.repeat(np.arange(5).astype(str), 20)
    x = rng.uniform(0, 50, 100)
    y = 0.8 * x + 5 + rng.normal(0, 1, 100)
    s = boot_slope(dates, x, y)
    assert s["slope"]["value"] == pytest.approx(0.8, abs=0.05)
    assert s["scatter"]["value"] == pytest.approx(1.0, abs=0.3)


def test_paired_and_scorer_on_synthetic_rows(tmp_path):
    from eval.evaluators.shape.paired import paired
    from eval.evaluators.shape.runner import summarise

    def rows(offset):
        out = []
        for d in ("20250926", "20250927", "20230826", "20230827"):
            for m in range(3):
                out.append({"step": "024", "var": "10u", "date": d, "member": str(m),
                            "field": "draw", "band": "mid",
                            "elong_frac_gt3": str(0.15 + offset + 0.001 * m),
                            "ell_ratio_median": str(1.6 + offset), "A_open_ocean": "0.8",
                            "A_all_interior": "0.8", "elong_median": "2", "ell_major_km_median": "20",
                            "ell_minor_km_median": "12", "n_components": "300"})
        return out
    pr = paired(rows(0.01), rows(0.0), n_boot=500)
    both = [r for r in pr if r["window"] == "both" and r["statistic"] == "elong_frac_gt3"][0]
    assert both["paired_diff"] == pytest.approx(0.01)
    assert both["n_pairs"] == 12 and both["n_dates"] == 4
    summ = summarise(rows(0.0), {"202509": "humberto", "202308": "idalia"}, 500, 1)
    assert {r["window"] for r in summ} == {"humberto", "idalia", "both"}
