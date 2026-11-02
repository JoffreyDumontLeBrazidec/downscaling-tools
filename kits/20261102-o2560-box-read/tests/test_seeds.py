import json
from pathlib import Path

import pytest
from conftest import J, need
from o2560_common import DATES12, LEADS_ALL, MEMBERS, SEED_BASE, case_seeds, derive_draw_seed


def test_check_values():
    # the values every rule-31 kit logs (E2 seed-fix kit, GL512 kit dry run 4097385291 / 2762430685)
    assert derive_draw_seed(1234, 20230101, 12, 0) == 4097385291
    assert derive_draw_seed(1234, 20230101, 12, 1) == 2762430685


def test_one_seed_per_draw_no_reuse():
    seeds = [s for d in DATES12 for L in LEADS_ALL for s in case_seeds(d, L, MEMBERS)]
    assert len(seeds) == 12 * 12 * 10
    assert len(set(seeds)) == len(seeds), "a seed is reused across (date, lead, member)"
    assert SEED_BASE == 1234


def test_seeds_do_not_depend_on_the_row(monkeypatch):
    # the seed is a function of (date, lead, member) only; changing every row variable leaves it unchanged
    ref = {(d, L): case_seeds(d, L, MEMBERS) for d in DATES12 for L in LEADS_ALL}
    for row in ("A100", "B24", "B50", "B24X"):
        monkeypatch.setenv("ROW", row)
        assert {(d, L): case_seeds(d, L, MEMBERS) for d in DATES12 for L in LEADS_ALL} == ref


def test_dry_run_records_hold_the_same_seeds_for_every_row():
    out = Path("/e/data1/jureap-data/ecmwf/users/jdumont/outputs/o2560_box_read_20261102/dry")
    files = {r: out / r / "dry_records.json" for r in ("A100", "B24", "B50")}
    if not all(f.exists() for f in files.values()):
        pytest.skip("dry runs not present")
    recs = {r: {(x["hdate"], x["lead"]): x["seeds"] for x in json.loads(f.read_text())} for r, f in files.items()}
    shared = set(recs["A100"]) & set(recs["B24"]) | set(recs["A100"]) & set(recs["B50"])
    assert shared, "no case shared between the dry runs"
    for r in ("B24", "B50"):
        for k in set(recs["A100"]) & set(recs[r]):
            assert recs["A100"][k] == recs[r][k] == case_seeds(k[0], k[1], MEMBERS)
    assert len(recs["A100"]) == 144, "A100 dry run covers 12 dates x 12 leads"
