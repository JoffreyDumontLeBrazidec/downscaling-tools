"""The B2L statistics pass (tools/b2l): accumulator, file format, and the eval-predict base hook."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import base_residual_statistics as brs  # noqa: E402


def test_accumulator_matches_numpy_on_a_nan_holed_field():
    rng = np.random.default_rng(0)
    r = rng.normal(size=(3, 1, 1, 50, 4)) * np.array([1.0, 2.0, 0.5, 10.0]) + np.array([0.0, 1.0, -3.0, 100.0])
    r[1, 0, 0, 7, 2] = np.nan
    acc = brs.ResidualStatsAccumulator(["a", "b", "c", "d"])
    for i in range(3):
        acc.update(r[i : i + 1])
    out = acc.result()
    flat = r.reshape(-1, 4)
    for j, name in enumerate("abcd"):
        col = flat[:, j]
        col = col[np.isfinite(col)]
        assert out["mean"][name] == pytest.approx(col.mean(), rel=1e-9, abs=1e-9)
        assert out["stdev"][name] == pytest.approx(col.std(), rel=1e-9, abs=1e-9)
        assert out["minimum"][name] == pytest.approx(col.min())
        assert out["maximum"][name] == pytest.approx(col.max())
    assert acc.n_samples == 150
    assert acc.count[2] == 149 and acc.count[0] == 150


def test_accumulator_takes_torch_tensors():
    torch = pytest.importorskip("torch")
    acc = brs.ResidualStatsAccumulator(["x"])
    acc.update(torch.tensor([[[1.0], [3.0]]]))
    out = acc.result()
    assert out["mean"]["x"] == 2.0 and out["stdev"]["x"] == 1.0


def test_written_file_is_what_the_datamodule_loader_reads(tmp_path):
    acc = brs.ResidualStatsAccumulator(["10u", "tp"])
    acc.update(np.array([[1.0, 0.0], [3.0, 2.0]]))
    out = tmp_path / "s.npy"
    brs.write_stats(out, acc.result(), {"note": "test"})
    # DownscalingAnemoiDatasetsDataModule._load_residual_statistics: np.load(path, allow_pickle=True).item()
    loaded = np.load(out, allow_pickle=True).item()
    assert set(loaded) == {"mean", "stdev", "minimum", "maximum"}
    assert loaded["mean"]["10u"] == 2.0 and loaded["stdev"]["tp"] == 1.0
    assert (tmp_path / "s.npy.json").exists()


def test_compare_with_old_reports_the_ratio(tmp_path):
    old = tmp_path / "old.npy"
    np.save(old, {"mean": {"10u": 0.0}, "stdev": {"10u": 4.0}, "minimum": {}, "maximum": {}}, allow_pickle=True)
    cmp = brs.compare_with_old(str(old), {"stdev": {"10u": 2.0, "tp": 1.0}})
    assert cmp["per_variable"]["10u"]["ratio_new_over_old"] == 0.5
    assert cmp["per_variable"]["tp"]["old_stdev"] is None


def test_output_variable_names_follow_the_model_output_order():
    class _Idx:
        class model:
            class output:
                name_to_index = {"tp": 1, "10u": 0}

    class _Model:
        _decoder_datasets = ["out_hres"]
        data_indices = {"out_hres": _Idx}

    class _Iface:
        model = _Model()

    assert brs.output_variable_names(_Iface()) == ["10u", "tp"]


def test_predict_hook_attaches_the_base_and_strips_its_keys(tmp_path):
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(HERE.parents[2]))
    from eval.predict import model_loader as ml

    class _Inner:
        attached = None

        def attach_regression_base(self, base, *, sigma, noise_seed, source):
            self.attached = (base, sigma, noise_seed, source)

    class _Model:
        def __init__(self):
            self.model = _Inner()

    ckpt = tmp_path / "anemoi-by_step-epoch_000-step_000100.ckpt"
    torch.save(torch.nn.Linear(2, 2), tmp_path / ("inference-" + ckpt.name))  # 6 parameters
    cfg = type("Cfg", (), {"local_scope_json": ""})()
    extra = {"base_checkpoint": str(ckpt), "base_sigma": 5e5, "base_noise_seed": 3, "num_steps": 30}
    model = _Model()
    info = ml._activate_regression_base(model, extra, cfg, "cpu")
    assert extra == {"num_steps": 30}  # the base keys never reach predict_step
    base, sigma, seed, source = model.model.attached
    assert isinstance(base, torch.nn.Linear) and sigma == 5e5 and seed == 3 and source.endswith("inference-" + ckpt.name)
    assert info["base_parameters"] == 6 and model._regression_base_info is info
    # absent key: exact current behaviour, nothing touched
    model2 = _Model()
    assert ml._activate_regression_base(model2, {"num_steps": 30}, cfg, "cpu") is None
    assert model2.model.attached is None
