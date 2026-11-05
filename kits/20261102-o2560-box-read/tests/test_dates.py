import json
import zipfile

import pytest
import yaml
from conftest import CFG_A, CFG_READ, CKPTS, STORE, need
from o2560_common import DATES12


def training_end(cfg):
    ff = cfg["dataloader"]["training"]["datasets"]["out_hres"]["dataset"]["dataset"]["fake_forecasts"]
    assert "start" not in ff
    return str(ff["end"])


@pytest.mark.parametrize("cfg_path", [CFG_A, CFG_READ])
def test_validation_dates_outside_training(cfg_path):
    cfg = yaml.safe_load(open(need(cfg_path)))
    d = cfg["dataloader"]
    end = d["training_date"] if training_end(cfg).startswith("${") else training_end(cfg)
    assert str(end) == "2024-10"
    assert str(d["valid_date"]) == str(d["test_date"]) == "2024-11"
    for dd in DATES12:
        assert dd[:6] == "202411" and dd[:6] > "202410"


@pytest.mark.parametrize("row", sorted(CKPTS))
def test_checkpoint_metadata_training_range(row):
    with zipfile.ZipFile(need(CKPTS[row])) as z:
        n = [x for x in z.namelist() if x.endswith("anemoi-metadata/anemoi.json")][0]
        md = json.loads(z.read(n))
    s = json.dumps(md["config"]["dataloader"]["training"])
    assert '"end": "2024-10"' in s or "'end': '2024-10'" in s or '"training_date": "2024-10"' in json.dumps(md["config"]["dataloader"])
    assert str(md["config"]["dataloader"]["training_date"]) == "2024-10"


def test_store_ends_on_the_last_validation_date():
    import zarr
    r = dict(zarr.open(need(f"{STORE}/hres_o2560.zarr"), "r").attrs)["recipe"]["dates"]
    assert r["end"].startswith("2024-11-12") and DATES12[-1] == "20241112"
