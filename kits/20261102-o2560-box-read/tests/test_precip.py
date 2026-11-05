import numpy as np
import pytest
from o2560_common import OUTPUT_VARS, clip_precip, deaccumulate


def test_clip_only_precip_and_report():
    a = np.zeros((2, 5, len(OUTPUT_VARS)))
    a[..., OUTPUT_VARS.index("tp")] = [[-1e-4, 0, 2e-3, -5e-5, 1e-3]] * 2
    a[..., OUTPUT_VARS.index("cp")] = -2e-5
    a[..., OUTPUT_VARS.index("2t")] = -3.0          # not precipitation: never clipped
    out, info = clip_precip(a, OUTPUT_VARS)
    assert (out[..., OUTPUT_VARS.index("tp")] >= 0).all() and (out[..., OUTPUT_VARS.index("cp")] >= 0).all()
    assert (out[..., OUTPUT_VARS.index("2t")] == -3.0).all()
    assert info["tp"] == (4, -1e-4) and info["cp"][0] == 10
    assert out[0, 2, OUTPUT_VARS.index("tp")] == 2e-3
    assert (a[..., OUTPUT_VARS.index("tp")] < 0).any(), "input array must not be modified"


def test_deaccumulate_from_start_totals():
    rate = np.array([[1.0, 0.0], [2.0, 1.0], [0.5, 0.5], [3.0, 0.0]])   # mm per 6 h window
    totals = {6 * (i + 1): rate[: i + 1].sum(0) for i in range(4)}
    win = deaccumulate(totals)
    for i, L in enumerate((6, 12, 18, 24)):
        np.testing.assert_allclose(win[L], rate[i])


def test_deaccumulate_needs_previous_lead():
    with pytest.raises(KeyError):
        deaccumulate({6: np.ones(2), 18: np.ones(2)})
