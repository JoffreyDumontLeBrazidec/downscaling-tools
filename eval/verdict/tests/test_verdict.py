"""eval.cli verdict: statistics, every gate on synthetic tables, the merge rule, and the command line end to end."""
from __future__ import annotations

import json

import numpy as np
import pytest

from eval.verdict import merge, preflight, reader_a, reader_b, reader_c
from eval.verdict import stats as S
from eval.verdict.gates import HARD_GATES, GateLine, read_csv, write_csv
from eval.verdict.tests import synthetic as syn


# ------------------------------------------------------------------------------------------------ statistics
def test_blocks_fixed5_gives_six_blocks_on_thirty_dates():
    b = S.blocks(syn.DATES, "fixed5")
    assert len(set(b.values())) == 6
    assert b["20250901"] == b["20250905"] != b["20250906"]


def test_blocks_chain_joins_dates_five_days_apart():
    b = S.blocks(["20240801", "20240806", "20240812", "20240901"], "chain")
    assert b["20240801"] == b["20240806"] and b["20240812"] != b["20240806"] and len(set(b.values())) == 3


def test_jackknife_matches_the_textbook_se_of_a_mean():
    x = np.array([1.0, 2.0, 4.0, 7.0, 11.0])
    v, se = S.jackknife(lambda k: x[k].mean(), len(x))
    assert v == pytest.approx(x.mean())
    assert se == pytest.approx(x.std(ddof=1) / np.sqrt(len(x)))


def test_t_table_and_lead_bands():
    assert S.t975(5) == pytest.approx(2.571) and S.t975(100) == pytest.approx(1.96)
    assert S.band_of(12) == "+12..+72 h" and S.band_of(240) == "+192..+240 h" and S.band_of(78) is None


def test_states():
    assert S.state_not_worse(-0.5, 0.5, 1.0) == S.PASS
    assert S.state_not_worse(0.1, 0.9, 1.0) == S.PASS            # worse beyond noise but inside the margin
    assert S.state_not_worse(1.2, 2.0, 1.0) == S.FAIL            # worse than the margin beyond noise
    assert S.state_not_worse(-0.5, 1.5, 1.0) == S.INCONCLUSIVE
    assert S.state_better(-3, -1, False) == S.PASS
    assert S.state_better(-1, 1, True) == S.INCONCLUSIVE and S.state_better(-1, 1, False) == S.FAIL
    assert S.combine([S.PASS, S.INCONCLUSIVE]) == S.INCONCLUSIVE and S.combine([]) == S.MISSING


def test_gates_csv_round_trip(tmp_path):
    ln = [GateLine("G-A1", "10ff +12..+72 h", True, "PASS", value=0.5, se=0.01, n_dates=30, n_blocks=6, note="x, y")]
    back = read_csv(write_csv(ln, tmp_path / "g.csv"))
    assert repr(back[0]) == repr(ln[0])


# ------------------------------------------------------------------------------------------------ reader A
def _a(tmp_path, factor, input_factor=1.2, spread=1.0):
    run = syn.scores(tmp_path / "run.csv", factor=factor, seed=1, spread=spread)
    ref = syn.scores(tmp_path / "ref.csv", factor=1.0, seed=2)
    ens = syn.scores(tmp_path / "ens.csv", factor=1.0, seed=3)
    inp = syn.scores(tmp_path / "inp.csv", rows=("input",), input_factor=input_factor, seed=4)
    return reader_a.run(str(run), str(ref), str(ens), str(inp))


def _state(lines, gate):
    return S.combine([ln.state for ln in lines if ln.gate == gate and ln.hard])


def test_reader_a_equal_run_passes_every_gate(tmp_path):
    lines = _a(tmp_path, 1.0)
    for g in ("G-A1", "G-A2", "G-A3", "G-A4", "G-A5"):
        assert _state(lines, g) == S.PASS, [ln for ln in lines if ln.gate == g and ln.state != S.PASS][:2]
    a1 = [ln for ln in lines if ln.gate == "G-A1"]
    assert len(a1) == 8 and all(ln.n_dates == 30 and ln.n_blocks == 6 for ln in a1)   # 2 vars x 4 bands


def test_reader_a_worse_run_fails_against_450k_and_ens9(tmp_path):
    lines = _a(tmp_path, 1.05)
    assert _state(lines, "G-A1") == S.FAIL and _state(lines, "G-A2") == S.FAIL and _state(lines, "G-A3") == S.FAIL
    d = [ln.diff for ln in lines if ln.gate == "G-A1"]
    assert np.allclose(d, 5.0, atol=0.3)


def test_reader_a_no_gain_over_input_fails_except_at_the_last_band(tmp_path):
    lines = _a(tmp_path, 1.0, input_factor=1.0)
    a4 = {ln.sub: ln.state for ln in lines if ln.gate == "G-A4"}
    assert a4["10ff +12..+72 h"] == S.FAIL
    assert a4["10ff +192..+240 h"] == S.INCONCLUSIVE


def test_reader_a_overdispersed_run_fails_spread_gate(tmp_path):
    lines = _a(tmp_path, 1.0, spread=1.3)
    assert _state(lines, "G-A5") == S.FAIL


# ------------------------------------------------------------------------------------------------ reader B
def test_spectra_equal_passes_and_overshoot_fails(tmp_path):
    ref = syn.bins(tmp_path / "ref", "REF", seed=1)
    same = syn.bins(tmp_path / "same", "RUN", seed=2)
    lines = reader_b.spectra(same, ref)
    assert _state(lines, "G-B1") == S.PASS and _state(lines, "G-B2") == S.PASS
    bad = syn.bins(tmp_path / "bad", "RUN", ratio={62.0: 1.3, 50.0: 1.3}, seed=3)
    lines = reader_b.spectra(bad, ref)
    assert _state(lines, "G-B1") == S.FAIL and _state(lines, "G-B2") == S.FAIL
    one = syn.bins(tmp_path / "one", "RUN", ratio={62.0: 1.12}, seed=4)        # one failing bin is not two adjacent
    assert _state(reader_b.spectra(one, ref), "G-B1") == S.PASS


def test_spectra_missing_lead_is_missing(tmp_path):
    ref = syn.bins(tmp_path / "ref", "REF", leads=(24,))
    run = syn.bins(tmp_path / "run", "RUN", leads=(24,))
    lines = reader_b.spectra(run, ref)
    assert sum(ln.state == S.MISSING for ln in lines if ln.gate == "G-B1") == 6     # 3 leads x 2 variables
    assert _state(lines, "G-B1") == S.INCONCLUSIVE                                   # present leads pass, absent ones block


def test_battery_out_lines(tmp_path):
    ref = syn.battery(tmp_path / "ref", ["C1", "T3", "Z4"])
    ok = syn.battery(tmp_path / "ok", ["C1", "T3"])
    new = syn.battery(tmp_path / "new", ["C1", "W2"])
    lines = reader_b.battery(ok, ref)
    assert [ln.state for ln in lines if ln.gate == "G-B3"] == [S.PASS]
    assert [ln.state for ln in lines if ln.gate == "G-B5" and "C11" in ln.sub] == [S.PASS]
    assert _state(lines, "G-B5") == S.INCONCLUSIVE        # +84..+240 h has no diagnostic yet
    assert _state(reader_b.battery(new, ref), "G-B3") == S.FAIL
    far = syn.battery(tmp_path / "far", [], c11="0.80-1.02; 0.5")
    assert [ln.state for ln in reader_b.battery(far, ref) if "C11" in ln.sub] == [S.FAIL]


def test_shape_and_texture(tmp_path):
    import pandas as pd
    def shape_rows(p, off):
        rows = [dict(date=d, field=f, band="b1", elong_frac_gt3=0.2 + (off if f == "draw" else 0) + 0.001 * i)
                for i, d in enumerate(syn.DATES) for f in ("draw", "truth")]
        pd.DataFrame(rows).to_csv(p, index=False)
        return p
    ref = shape_rows(tmp_path / "ref.csv", 0.01)
    assert _state(reader_b.shape(shape_rows(tmp_path / "a.csv", 0.01), ref), "G-B4") == S.PASS
    assert _state(reader_b.shape(shape_rows(tmp_path / "b.csv", 0.05), ref), "G-B4") == S.FAIL

    def tex(p, g):
        rng = np.random.default_rng(0)
        s = [dict(date=d, grain_index={"fine_nn_corr": g + 0.01 * rng.standard_normal()}) for d in syn.DATES for _ in range(3)]
        p.write_text(json.dumps({"samples": s}))
        return p
    tref = tex(tmp_path / "tr.json", 0.1)
    assert _state(reader_b.texture(tex(tmp_path / "ta.json", 0.1), tref), "G-B4") == S.PASS
    assert _state(reader_b.texture(tex(tmp_path / "tb.json", 0.3), tref), "G-B4") == S.FAIL


# ------------------------------------------------------------------------------------------------ reader C
def _c(tmp_path, factor=1.0, shift=0.0, tail=1.0):
    ens = syn.xset(tmp_path / "ENS9", "ENS9", refs=True, seed=1)
    ref = syn.xset(tmp_path / "REF", "GL512_450k", seed=2)
    run = syn.xset(tmp_path / "RUN", "RUN", factor=factor, shift_km=shift, seed=3)
    h = tmp_path / "hist"
    syn.hist(h, "ENS9", 1.0, seed=1), syn.hist(h, "GL512_450k", 1.3, seed=2), syn.hist(h, "RUN", tail, seed=3)
    lines = reader_c.cyclones(str(run), str(ref), str(ens)) + reader_c.rain_tails(str(run), str(ref), str(ens))
    return lines + reader_c.tails(h, "RUN", "GL512_450k", "ENS9")


def test_reader_c_equal_run_passes(tmp_path):
    lines = _c(tmp_path, tail=1.1)
    for g in ("G-C1", "G-C2", "G-C3", "G-C4"):
        assert _state(lines, g) == S.PASS, [ln for ln in lines if ln.gate == g]
    c1 = [ln for ln in lines if ln.gate == "G-C1"]
    assert all(ln.n_blocks == 12 for ln in c1)                 # storms are the jackknife units


def test_reader_c_faults_fail(tmp_path):
    lines = _c(tmp_path, factor=1.10, shift=50.0, tail=3.0)
    for g in ("G-C1", "G-C2", "G-C3", "G-C4"):
        assert _state(lines, g) == S.FAIL, g


# ------------------------------------------------------------------------------------------------ preflight, merge
def test_preflight(tmp_path):
    m = syn.manifest(tmp_path / "M.json", tmp_path / "pred", tmp_path / "log.txt")
    r = syn.manifest(tmp_path / "R.json", tmp_path / "pred", tmp_path / "log.txt")
    lines = preflight.run(m, r)
    st = {ln.gate: ln.state for ln in lines}
    assert st["P1"] == st["P2"] == st["P3"] == st["P4"] == S.PASS and st["P5"] == S.FAIL   # P5: no station files
    bad = syn.manifest(tmp_path / "B.json", tmp_path / "pred", tmp_path / "log2.txt", seed_base=756)
    (tmp_path / "log2.txt").write_text("GUARD FAILED\n")
    st = {ln.gate: ln.state for ln in preflight.run(bad, r)}
    assert st["P1"] == S.FAIL and st["P3"] == S.FAIL


def _pass_lines():
    out = [GateLine(p, "", True, S.PASS) for p in ("P1", "P2", "P3", "P4", "P5")]
    return out + [GateLine(g, "x", True, S.PASS) for g in HARD_GATES]


def test_merge_rules():
    lines = _pass_lines()
    assert merge.decide(lines)["verdict"] == "PASS"
    lines[-1].state = S.INCONCLUSIVE
    assert merge.decide(lines)["verdict"] == "INCONCLUSIVE"
    lines[-2].state = S.FAIL
    d = merge.decide(lines)
    assert d["verdict"] == "FAIL" and merge.verdict_line(d).startswith("VERDICT: FAIL (gates G-C3")
    lines[0].state = S.FAIL
    assert merge.decide(lines)["verdict"] == "INVALID"
    assert merge.decide(_pass_lines()[5:])["verdict"] == "INVALID"            # preflight not run
    assert merge.decide(_pass_lines()[:-1])["verdict"] == "INCONCLUSIVE"     # a hard gate absent


def test_cli_end_to_end(tmp_path, capsys):
    from eval.cli import main
    run = syn.scores(tmp_path / "run.csv", seed=1)
    ref = syn.scores(tmp_path / "ref.csv", seed=2)
    ens = syn.scores(tmp_path / "ens.csv", seed=3)
    inp = syn.scores(tmp_path / "inp.csv", rows=("input",), seed=4)
    m = syn.manifest(tmp_path / "M.json", tmp_path / "pred", tmp_path / "log.txt")
    out = tmp_path / "v"
    sc = ["--run", str(run), "--ref", str(ref), "--ens9", str(ens), "--input", str(inp)]
    main(["verdict", "--reader", "P", "--out", str(out), "--manifest", str(m), "--ref-manifest", str(m)] + sc)
    main(["verdict", "--reader", "A", "--out", str(out)] + sc)
    sb = syn.bins(tmp_path / "sr", "RUN", seed=1), syn.bins(tmp_path / "sf", "REF", seed=2)
    b = syn.battery(tmp_path / "br", ["C1"]), syn.battery(tmp_path / "bf", ["C1"])
    main(["verdict", "--reader", "B", "--out", str(out), "--spectra-run", str(sb[0]), "--spectra-ref", str(sb[1]),
          "--battery-run", str(b[0]), "--battery-ref", str(b[1])])
    ens9 = syn.xset(tmp_path / "ENS9", "ENS9", refs=True, seed=1)
    xr, xf = syn.xset(tmp_path / "XR", "RUN", seed=3), syn.xset(tmp_path / "XF", "GL512_450k", seed=2)
    main(["verdict", "--reader", "C", "--out", str(out), "--xset-run", str(xr), "--xset-ref", str(xf), "--xset-ens9", str(ens9)])
    main(["verdict", "--out", str(out), "--row", "RUN"])
    text = capsys.readouterr().out
    # G-B4 (no shape/texture), G-B5 (+84..+240 h) and G-C4 (no histograms) are MISSING: INCONCLUSIVE, never PASS
    assert "VERDICT: INCONCLUSIVE (gates not passed: G-B4, G-B5, G-C4)" in text
    assert (out / "verdict.md").exists() and json.loads((out / "verdict.json").read_text())["verdict"] == "INCONCLUSIVE"
