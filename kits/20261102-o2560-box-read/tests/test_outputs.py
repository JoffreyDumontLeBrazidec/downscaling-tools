import json
import zipfile

import pytest
from conftest import CKPTS, need
from o2560_common import OUTPUT_VARS


def metadata(path):
    with zipfile.ZipFile(path) as z:
        n = [x for x in z.namelist() if x.endswith("anemoi-metadata/anemoi.json")]
        assert len(n) == 1
        return json.loads(z.read(n[0]))


@pytest.mark.parametrize("row", sorted(CKPTS))
def test_eleven_outputs_in_order_no_sp(row):
    md = metadata(need(CKPTS[row]))
    n2i = md["data_indices"]["out_hres"]["model"]["output"]["name_to_index"]
    names = tuple(sorted(n2i, key=n2i.get))
    assert names == OUTPUT_VARS, names
    assert "sp" not in n2i and len(names) == 11


@pytest.mark.parametrize("row", sorted(CKPTS))
def test_file_is_the_ema10000_copy(row):
    md = metadata(need(CKPTS[row]))
    assert "inference-ema10000-" in CKPTS[row]
    hl = md.get("ema_halflife_steps")
    assert hl is not None and 10000 in (hl if isinstance(hl, (list, tuple)) else [hl]), hl
