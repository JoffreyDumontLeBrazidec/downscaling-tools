from __future__ import annotations

import csv
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

im = importlib.import_module("eval.tools.sigma_evaluator.input_modes")


# ------------------------------------------------------------------ pure helpers
def test_parse_input_modes():
    assert im.parse_input_modes("truth,noise_only,other_date") == ["truth", "noise_only", "other_date"]
    assert im.parse_input_modes(" noise_only ") == ["noise_only"]
    for bad in ("", "pure", "truth,truth"):
        with pytest.raises(ValueError):
            im.parse_input_modes(bad)


def test_resolve_other_dates_default_and_explicit():
    dates = ["20230826", "20230827", "20230828", "20230829", "20230830"]
    assert im.resolve_other_dates(dates) == {
        "20230826": "20230828", "20230827": "20230829", "20230828": "20230830",
        "20230829": "20230826", "20230830": "20230827"}
    assert im.resolve_other_dates(["20230827", "20230826"]) == {"20230826": "20230827", "20230827": "20230826"}
    explicit = im.resolve_other_dates(dates[:2], "20230826:20230827, 20230827:20230826")
    assert explicit == {"20230826": "20230827", "20230827": "20230826"}
    with pytest.raises(ValueError):
        im.resolve_other_dates(dates[:2], "20230826:20230826,20230827:20230826")
    with pytest.raises(ValueError):
        im.resolve_other_dates(dates[:2], "20230826:20230827")
    with pytest.raises(ValueError):
        im.resolve_other_dates(dates[:2], "20230826:20230901,20230827:20230826")
    with pytest.raises(ValueError):
        im.resolve_other_dates(["20230826"])


def test_eps_is_reproducible_and_shard_independent():
    seed = im.eps_seed(756, "20230826", 3)
    assert seed == im.eps_seed(756, "20230826", 3)
    assert seed != im.eps_seed(756, "20230826", 4) != im.eps_seed(756, "20230827", 3)
    full = im.draw_eps(10, 3, seed)
    assert torch.equal(full, im.draw_eps(10, 3, seed))
    sizes = [3, 3, 2, 2]
    parts = [full[slice(*im.shard_bounds(sizes, r, 10))] for r in range(4)]
    assert torch.equal(torch.cat(parts), full)
    assert im.shard_bounds(None, 0, 10) == (0, 10)
    with pytest.raises(ValueError):
        im.shard_bounds([3, 3], 0, 10)


def test_noisy_state_modes():
    eps = torch.randn(5, 2, dtype=torch.float64)
    y0, y1 = torch.randn(5, 2, dtype=torch.float64), torch.randn(5, 2, dtype=torch.float64)
    s = 300.0
    assert torch.equal(im.noisy_state("truth", s, eps, y0, y1), y0 + s * eps)
    assert torch.equal(im.noisy_state("noise_only", s, eps, y0, y1), s * eps)
    assert torch.equal(im.noisy_state("other_date", s, eps, y0, y1), y1 + s * eps)
    with pytest.raises(ValueError):
        im.noisy_state("pure", s, eps, y0, y1)


# ------------------------------------------------------------------ end to end on a fake model
N, C = 12, 2
NAMES = ["10u", "msl"]
SCALE = torch.tensor([2.0, 50.0], dtype=torch.float64)      # b of the output map
DATES = ["20230826", "20230827", "20230828"]


def _inputs(date: str):
    rng = np.random.default_rng(int(date))
    x = rng.normal(size=(N, C)).astype(np.float32)           # in_lres on the hres grid (identity upsample)
    truth = (x + rng.normal(size=(N, C)) * np.array([2.0, 50.0])).astype(np.float32)
    return x, truth


class _FakeInner:
    def __init__(self):
        self.calls = []

    def _before_sampling(self, batch, pre, n_step_input, mcg, **kw):
        return (batch["in_lres"].float(), batch["in_hres"].float()), None

    def fwd_with_preconditioning(self, x, y, sigma, mcg, gss):
        self.calls.append((y["out_hres"].detach().clone(), sigma["out_hres"].detach().clone()))
        s = sigma["out_hres"]
        return {"out_hres": y["out_hres"] / (s ** 2 + 1) + 0.5 * x["in_lres"]}

    def _after_sampling(self, out, post, before, mcg, gss, gather_out, **kw):
        return before[0].to(out.dtype) + out * SCALE.to(out.dtype)


class _FakeIface(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _FakeInner()
        self.pre_processors = None
        self.post_processors = None
        self.n_step_input = 1


def _datamodule():
    n2i = {n: i for i, n in enumerate(NAMES)}
    return SimpleNamespace(data_indices={
        "out_hres": SimpleNamespace(model=SimpleNamespace(output=SimpleNamespace(name_to_index=n2i))),
        "in_lres": SimpleNamespace(data=SimpleNamespace(input=SimpleNamespace(name_to_index=n2i))),
        "in_hres": SimpleNamespace(data=SimpleNamespace(input=SimpleNamespace(name_to_index={"lsm": 0}))),
    })


def _args(tmp_path, root, **kw):
    base = dict(name_exp="exp", name_ckpt="ckpt.ckpt", ckpt_root=str(tmp_path / "ck"), out_file="x.csv",
                out_csv=str(tmp_path / "out" / "u2.csv"), device="cpu", num_gpus_per_model=0, n_samples=0,
                validation_frequency="50h", sigmas="300,100000", bundle_root=str(root),
                checkpoint_compat_profile="", noisy_input_mode="truth,noise_only,other_date", other_date_map="",
                save_scope_json=json.dumps({"mode": "bbox", "lat_min": 0, "lat_max": 90, "lon_min": -180,
                                            "lon_max": 180}),
                save_weather_states="")
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    from manual_inference.input_data_construction import bundle as mib
    from manual_inference.prediction import predict as mip

    root = tmp_path / "bundles"
    root.mkdir()
    for d in DATES:
        (root / f"eefo_o320_0001_date{d}_time0000_mem01_step024h_input_bundle.nc").write_bytes(b"")
    iface = _FakeIface()
    lat = np.linspace(-80, 80, N)                              # half the points in the northern box
    lon = np.linspace(0, 330, N)

    def _load_inputs(ds, n2i_l, n2i_h, prev_bundle=None):
        x, _ = _inputs(ds)
        return x, np.zeros((N, 1), np.float32), lon, lat, lon, lat

    monkeypatch.setattr(mip, "_get_parallel_info", lambda: (0, 0, 1))
    monkeypatch.setattr(mip, "_load_objects", lambda **kw: (iface, _datamodule(), "", ""))
    monkeypatch.setattr(mib, "open_bundle_dataset", lambda p: im.bundle_date(p))
    monkeypatch.setattr(mib, "previous_step_bundle_path", lambda p: None)
    monkeypatch.setattr(mib, "check_input_distribution", lambda *a, **k: None)
    monkeypatch.setattr(mib, "load_inputs_from_bundle_numpy", _load_inputs)
    monkeypatch.setattr(mib, "extract_target_from_bundle_dataset", lambda ds, names: (_inputs(ds)[1], len(names)))
    monkeypatch.delenv("ANEMOI_BASE_SEED", raising=False)
    return root, iface, lat


def test_probe_builds_the_right_inputs_and_saves(tmp_path, fake_env):
    root, iface, lat = fake_env
    out_csv = im.run_input_mode_probe(_args(tmp_path, root))
    calls = iface.model.calls
    sigmas, modes = [300.0, 100000.0], ["truth", "noise_only", "other_date"]
    assert len(calls) == len(DATES) * len(sigmas) * len(modes)
    other = im.resolve_other_dates(DATES)
    y0 = {}
    for d in DATES:
        x, truth = _inputs(d)
        y0[d] = (torch.from_numpy(truth).double() - torch.from_numpy(x).double()) / SCALE
    box = lat >= 0
    k = 0
    for d in DATES:
        for si, s in enumerate(sigmas):
            eps = im.draw_eps(N, C, im.eps_seed(756, d, si))
            expect = {"truth": y0[d] + s * eps, "noise_only": s * eps, "other_date": y0[other[d]] + s * eps}
            for m in modes:
                y_in, sig = calls[k]
                k += 1
                # same rounding as the denoiser input: y0 kept float32, state built in float64, cast to float32
                y0f = {kk: v.float().double() for kk, v in y0.items()}
                exp32 = {"truth": y0f[d] + s * eps, "noise_only": s * eps,
                         "other_date": y0f[other[d]] + s * eps}[m].float()
                assert torch.equal(y_in[0, 0, 0], exp32), (d, s, m)
                assert torch.allclose(y_in[0, 0, 0].double(), expect[m], rtol=1e-6, atol=1e-5 * max(s, 1.0))
                assert float(sig.reshape(-1)[0]) == pytest.approx(s)
                f = np.load(tmp_path / "out" / "denoised" / f"D_{d}_s{s:g}_{m}.npz")
                x, _ = _inputs(d)
                D = exp32.double() / (s ** 2 + 1) + 0.5 * torch.from_numpy(x).double()
                post = (torch.from_numpy(x).double() + D * SCALE).numpy()[box]
                np.testing.assert_allclose(f["y_pred"], post, rtol=1e-5, atol=1e-3)
    ref = np.load(tmp_path / "out" / "denoised" / f"ref_{DATES[0]}.npz")
    np.testing.assert_array_equal(ref["y"], _inputs(DATES[0])[1][box])
    np.testing.assert_allclose(ref["b"], np.broadcast_to(SCALE.numpy(), (int(box.sum()), C)))
    rows = list(csv.DictReader(open(out_csv)))
    assert len(rows) == len(DATES) * len(sigmas) * len(modes) * C
    truth_rows = [r for r in rows if r["mode"] == "truth"]
    assert all(float(r["rms_D_minus_D_truth"]) == 0.0 for r in truth_rows)
    man = json.loads((tmp_path / "out" / "denoised" / "manifest.json").read_text())
    assert man["save_points"] == int(box.sum()) and man["other_date"] == other
    assert all(c["recon_truth_maxrel_to_std"] < 1e-6 for c in man["checks"].values())


def test_probe_refuses_without_bundles_or_sigmas(tmp_path, fake_env):
    root, _iface, _lat = fake_env
    with pytest.raises(ValueError):
        im.run_input_mode_probe(_args(tmp_path, root, bundle_root=""))
    with pytest.raises(ValueError):
        im.run_input_mode_probe(_args(tmp_path, root, sigmas=""))


def test_run_sigma_evaluator_dispatches_only_when_mode_set(tmp_path, monkeypatch):
    mod = importlib.import_module("eval.tools.sigma_evaluator.run_sigma_evaluator")
    parsed = mod._build_parser().parse_args(["--name_exp", "e", "--name_ckpt", "c"])
    assert parsed.noisy_input_mode == "" and parsed.other_date_map == ""
    assert parsed.save_scope_json == "" and parsed.save_weather_states == ""
    seen = []
    monkeypatch.setattr(im, "run_input_mode_probe", lambda args: seen.append(args) or Path("probe.csv"))
    args = mod._build_parser().parse_args(["--name_exp", "e", "--name_ckpt", "c", "--noisy-input-mode", "truth"])
    assert mod.run_sigma_evaluator(args) == Path("probe.csv") and seen == [args]
