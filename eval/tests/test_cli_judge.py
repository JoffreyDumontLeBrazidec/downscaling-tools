"""eval.cli judge: the read skeleton, the shape-box choice, and the small CLI helpers of the
run-judging protocol (--opt, zoom_maps --region)."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest

from eval.cli import judge
from eval.cli._common import parse_opts
from eval.cli.zoom_maps import apply_region

# the part headings and banned phrases of docs scripts/check_run_read.py
PART_RE = re.compile(r"^### (\d)\.", re.M)
BANNED = (r"40\s*-\s*400\s*km distance", r"band[- ]averag")


def _arm(tmp_path: Path, label: str, n_points=1000) -> dict:
    # seed-locked arms (rule 31): both carry the same per-draw seeds, so no seed warning by default
    return {"label": label, "dir": tmp_path / label, "eval": tmp_path / label / "evaluators",
            "status": {"shape": "failed: no stats file"},
            "facts": {"n_members": 10, "n_points": n_points, "checkpoint_id": f"ck_{label}",
                      "sampling_config_json": '{"num_steps": 30, "S_churn": 0}',
                      "seeds_by_date": {"20230226": "11,12"}, "seed_rule": "per_draw", "seed_base": "1234",
                      "seed_formula": "numpy SeedSequence([base, yyyymmdd, lead_h, member]).generate_state(1, uint32)"}}


def test_read_without_outputs_has_four_parts_and_says_why(tmp_path):
    arms = {"member": 1, "run": _arm(tmp_path, "h9dm"), "reference": _arm(tmp_path, "donor")}
    text = judge.compose_read(arms, step=24, dates=["20230226"], shape_box="north_atlantic",
                              maps={"Alps": ([], "no file")}, domain="europe", weights=None,
                              out_dir=tmp_path, commit="abc")
    assert text.startswith("## Run read: h9dm")
    assert PART_RE.findall(text) == ["1", "2", "3", "4"]
    for part in re.split(r"^### \d\..*$", text, flags=re.M)[1:]:
        assert "Not available:" in part
    assert "Reference: donor" in text and "S_churn" in text and "Instrument:" in text
    assert not any(re.search(b, text, re.I) for b in BANNED)
    assert "WARNING" not in text


def test_read_warns_on_unequal_instruments(tmp_path):
    arms = {"member": 1, "run": _arm(tmp_path, "a", 6_599_680), "reference": _arm(tmp_path, "b", 1_200_000)}
    arms["reference"]["facts"]["sampling_config_json"] = '{"num_steps": 40}'
    text = judge.compose_read(arms, step=24, dates=["20230226"], shape_box="e2", maps={},
                              domain="europe", weights="EMA-10000 / raw", out_dir=tmp_path, commit="abc")
    assert "different point sets" in text and "sampler settings" in text


@pytest.mark.parametrize("lat,lon,box", [
    ([10.0, 70.0], [280.0, 40.0], "north_atlantic"),     # E2 file, 0..360 longitudes
    ([-89.9, 89.9], [0.0, 359.9], "north_atlantic"),     # global file
    ([10.0, 40.0], [260.0, 302.0], "tc_atlantic"),       # regional TC file
])
def test_shape_box_auto(lat, lon, box):
    assert judge.choose_shape_box("auto", {"lat_range": lat, "lon_range": lon}) == box
    assert judge.choose_shape_box("e2", {}) == "e2"


def test_parse_label_and_opts():
    assert judge.parse_label("h9dm=/x/y", "--run") == ("h9dm", Path("/x/y"))
    with pytest.raises(SystemExit):
        judge.parse_label("/x/y", "--run")
    assert parse_opts(["shape.box=north_atlantic", "zoom_maps.steps=[24]"]) == [
        ("shape", "box", "north_atlantic"), ("zoom_maps", "steps", [24])]
    with pytest.raises(SystemExit):
        parse_opts(["box=north_atlantic"])


def test_zoom_maps_region_sets_the_protocol_extent():
    args = argparse.Namespace(region="alps", extent=[0, 1, 2, 3], region_tag="europe-cutout")
    apply_region(args)
    assert args.extent == [4.0, 16.0, 43.0, 49.0] and args.region_tag == "alps"


# --------------------------------------------------------------- part 4: the owner's rule on the scores

def _calibration_csv(path: Path, rows: list[tuple]) -> None:
    """(var, band, fcrps, rmse, corrected ratio, input_share) rows of calibration_by_lead.csv at +24 h, europe."""
    path.mkdir(parents=True, exist_ok=True)
    head = ("step,weather_state,domain,band,n_members,n_dates,n_points_total,fcrps_mean,fcrps_stderr,crps_mean,"
            "spread_rms,rmse_ens_mean,spread_skill_ratio,spread_skill_ratio_corrected,correction_factor,"
            "spread_input_rms,input_n_members,spread_input_share\n")
    body = "".join(f"24,{v},europe,{b},10,3,300,{f},0.01,{f * 1.05},{r * c / 1.0488},{r},{c / 1.0488},{c},1.0488,"
                   f"{'' if s is None else s},{0 if s is None else 10},{'' if s is None else s}\n"
                   for v, b, f, r, c, s in rows)
    (path / "calibration_by_lead.csv").write_text(head + body)


def _band_scores_csv(path: Path, values: dict) -> None:
    """{(var, band): [fcrps per date]} -> band_scores_by_lead.csv rows."""
    lines = ["date,step,weather_state,domain,band,metric,value,n_points,n_members,source_path"]
    for (var, band), per_date in values.items():
        for i, v in enumerate(per_date):
            lines.append(f"2023010{i + 1},24,{var},europe,{band},fcrps,{v},100,10,x.nc")
    (path / "band_scores_by_lead.csv").write_text("\n".join(lines) + "\n")


def _rank_csv(path: Path, outer: float) -> None:
    (path / "rank_histograms.csv").write_text(
        "step,weather_state,domain,n_members,n_bins,n_dates,n_points_total,outer_two_fraction,outer_two_expected,"
        "flatness_rms,rank_mean_norm,chi2_per_bin,shape\n"
        f"24,2t,europe,10,11,3,300,{outer},0.1818,0.1,0.0,1.0,flat\n")


def test_part_scores_labels_every_variable_and_band(tmp_path):
    arms = {"member": 1, "run": _arm(tmp_path, "h9dm"), "reference": _arm(tmp_path, "donor")}
    a, b = arms["run"]["eval"] / "probabilistic", arms["reference"]["eval"] / "probabilistic"
    _calibration_csv(a, [("2t", "full", 0.90, 1.9, 1.00, None), ("2t", "large", 0.80, 1.5, 1.20, 0.6),
                         ("2t", "fine", 0.95, 1.1, 0.98, 0.0), ("tp", "full", 1.20, 2.1, 1.0, None)])
    _calibration_csv(b, [("2t", "full", 1.00, 2.0, 0.90, None), ("2t", "large", 1.00, 1.6, 0.90, 0.5),
                         ("2t", "fine", 0.96, 1.0, 0.98, 0.0), ("tp", "full", 1.00, 2.0, 1.0, None)])
    _band_scores_csv(a, {("2t", "full"): [0.9, 0.9, 0.9], ("2t", "large"): [0.8, 0.8, 0.8],
                         ("2t", "fine"): [0.95, 0.89, 1.02], ("tp", "full"): [1.2, 1.2, 1.2]})
    _band_scores_csv(b, {("2t", "full"): [1.0, 1.0, 1.0], ("2t", "large"): [1.0, 1.0, 1.0],
                         ("2t", "fine"): [0.96, 0.91, 1.01], ("tp", "full"): [1.0, 1.0, 1.0]})
    _rank_csv(a, 0.25)
    _rank_csv(b, 0.18)
    lines = judge.part_scores(arms, 24, "europe")
    text = "\n".join(lines)
    assert text.startswith("### 4. Scores")
    rows = {(ln.split("|")[1].strip(), band): ln.split("|")[-2].strip()
            for band in ("full field", "large part", "fine part")
            for ln in text.split(f"{band}", 1)[1].split("\n\n")[1].splitlines()[2:] if ln.startswith("| ")}
    assert rows[("2t", "full field")] == "better model"
    assert rows[("2t", "large part")] == "better centre, worse calibration"     # ratio 1.20
    assert rows[("2t", "fine part")] == "tie"           # -0.01 within 2 paired se (diffs -0.01, -0.02, +0.01)
    assert rows[("tp", "full field")] == "worse CRPS"
    assert "1 better model, 1 better centre, worse calibration, 1 worse CRPS, 1 tie" in text
    assert "Large-scale spread of the input members" in text and "2t 0.60 / 0.50" in text
    assert "Rank histograms (11 bins" in text and "0.250 (flat)" in text
    assert "Realism and spectra (parts 1-2) come first" in text
    assert lines[-1].startswith("- Verdict on scores (never outranks a loss in parts 1 or 2): TODO")
    assert not any(re.search(b, text, re.I) for b in BANNED)


def test_part_scores_falls_back_to_the_old_table_without_calibration(tmp_path):
    arms = {"member": 1, "run": _arm(tmp_path, "h9dm"), "reference": _arm(tmp_path, "donor")}
    for arm in ("run", "reference"):
        d = arms[arm]["eval"] / "probabilistic"
        d.mkdir(parents=True)
        (d / "summary_by_lead.csv").write_text(
            "step,weather_state,domain,metric,mean,std,stderr,n_dates,n_points_total\n"
            "24,2t,europe,fcrps,1.0,0,0,3,300\n24,2t,europe,spread,0.9,0,0,3,300\n"
            "24,2t,europe,rmse_ens_mean,1.2,0,0,3,300\n")
    text = "\n".join(judge.part_scores(arms, 24, "europe"))
    assert "| 2t | fair CRPS |" in text and "predates calibration_by_lead.csv" in text
    assert "better model" not in text


# ------------------------------------------------------------------------ rule 31: seeds

def _seeded(facts: dict, seeds: dict, base: int = 1234) -> dict:
    return {**facts, "seeds_by_date": seeds, "seed_rule": "per_draw", "seed_base": str(base),
            "seed_formula": "numpy SeedSequence([base, yyyymmdd, lead_h, member]).generate_state(1, uint32)"}


def _unseeded(facts: dict) -> dict:
    return {k: v for k, v in facts.items() if k not in ("seeds_by_date", "seed_rule", "seed_base", "seed_formula")}


def test_seed_warnings_rule_31(tmp_path):
    plain = _unseeded(_arm(tmp_path, "a")["facts"])
    same = {"20230226": "1,2,3", "20230829": "4,5,6"}
    other = {"20230226": "1,2,3", "20230829": "7,8,9"}
    assert judge.seed_warnings(_seeded(plain, same), _seeded(plain, same), "run", "ref") == []
    w = judge.seed_warnings(_seeded(plain, same), _seeded(plain, other), "run", "ref")
    assert len(w) == 1 and "differ on 1 date(s)" in w[0] and "20230829" in w[0]
    w = judge.seed_warnings(plain, _seeded(plain, same), "run", "ref")
    assert len(w) == 1 and "old seeding: not comparable with seed-locked arms" in w[0] and w[0].startswith("RULE 31")
    w = judge.seed_warnings(plain, dict(plain), "run", "ref")
    assert "run and ref carry no per-draw seeds" in w[0]
    w = judge.seed_warnings(_seeded(plain, same), {**_seeded(plain, same), "seed_rule": "per_process"}, "run", "ref")
    assert any("different rules" in x for x in w)
    assert judge.seed_line(plain) == "no per-draw seeds in the files (old seeding)"
    assert "per_draw" in judge.seed_line(_seeded(plain, same)) and "2 date(s)" in judge.seed_line(_seeded(plain, same))


def test_read_header_carries_the_seed_warning(tmp_path):
    arms = {"member": 1, "run": _arm(tmp_path, "h9dm"), "reference": _arm(tmp_path, "donor")}
    arms["reference"]["facts"] = _unseeded(arms["reference"]["facts"])
    text = judge.compose_read(arms, step=24, dates=["20230226"], shape_box="e2", maps={}, domain="europe",
                              weights=None, out_dir=tmp_path, commit="abc")
    assert "WARNING: RULE 31, SEEDS: donor carries no per-draw seeds" in text
    assert "Seeds (rule 31): run seed_rule per_draw, base 1234" in text
