"""The zarr_split input route and the EMA weights key of eval.predict (2026-11-03).

A stub datamodule (three readers with a fake-hindcast mapping) and a stub model stand in for
an E2 box checkpoint: the route must write files that pass validate_predictions_dataset, read
the row of (date, lead) for every member, seed each draw by rule 31, and hand a flow_map block
to predict_step as the two dicts; predict.ema must load inference-<ema>-<name>.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import xarray as xr
from omegaconf import OmegaConf

from eval.predict.dataset_builder import validate_predictions_dataset
from eval.predict.seeding import derive_draw_seed

FLOW_MAP = {"schedule_type": "flow_map", "num_steps": 2, "sigma_max": 80.0, "sigma_min": 0.0, "sampler": "jumps"}
ZARR_TEST = json.dumps({"type": "zarr_split", "split": "test"})
N_LRES, N_HRES = 6, 12
LRES_VARS = {"2t": 0, "10u": 1, "lsm": 2}          # in_lres data order (no tp)
HRES_VARS = {"z": 0, "cos_latitude": 1}            # in_hres forcings
OUT_DATA = {"2t": 0, "tp": 1, "10u": 2}            # out_hres store order (truth)
OUT_MODEL = {"2t": 0, "tp": 1}                     # model outputs
LEADS = (12, 24)
DATES = ("20230101", "20230829")


class _FakeHindcasts:
    """The Subset that owns the fake-hindcast mapping (under a Select, as in the E2 readers)."""

    def __init__(self, table):
        self.fake_hindcasts = True
        self._table = table

    def fake_date_to_date_step(self, fake):
        init, step = self._table[np.datetime64(fake, "s")]
        return fake, init, step


class _Reader:
    def __init__(self, n_points, table, fakes, *, store_offset):
        self.data = SimpleNamespace(
            shape=(len(fakes), 3, 1, n_points),
            latitudes=np.linspace(10.0, 70.0, n_points) + store_offset,
            longitudes=np.linspace(-80.0, 40.0, n_points),
            forward=_FakeHindcasts(table),
        )
        self.dates = np.array(fakes, dtype="datetime64[s]")


class _SplitDataset:
    """MultiDataset stand-in: row i holds value (i + 1) * 100 + variable index everywhere."""

    def __init__(self):
        table, fakes = {}, []
        for d, date in enumerate(DATES):
            for lead in LEADS:
                fake = np.datetime64("1900-01-01T00:00:00", "s") + np.timedelta64(len(fakes), "h")
                table[fake] = (np.datetime64(f"{date[:4]}-{date[4:6]}-{date[6:]}T00:00:00", "s"), lead)
                fakes.append(fake)
        self.datasets = {
            "in_lres": _Reader(N_LRES, table, fakes, store_offset=0.0),
            "in_hres": _Reader(N_HRES, table, fakes, store_offset=0.5),
            "out_hres": _Reader(N_HRES, table, fakes, store_offset=0.5),
        }
        self.name_to_index = {"in_lres": dict(LRES_VARS), "in_hres": dict(HRES_VARS), "out_hres": dict(OUT_DATA)}
        self.calls = []

    def set_comm_group_info(self, *args):
        self.calls.append(("set_comm_group_info", args))

    def per_worker_init(self, n_workers, worker_id):
        self.calls.append(("per_worker_init", (n_workers, worker_id)))

    def get_sample(self, index):
        self.calls.append(("get_sample", index))
        out = {}
        for name, n_points in (("in_lres", N_LRES), ("in_hres", N_HRES), ("out_hres", N_HRES)):
            n_vars = len(self.name_to_index[name])
            row = (index + 1) * 100.0 + np.arange(n_vars, dtype=np.float32)
            out[name] = torch.from_numpy(np.broadcast_to(row, (1, 1, n_points, n_vars)).copy())
        return out


def _indices(data_input, model_output=None):
    return SimpleNamespace(
        data=SimpleNamespace(input=SimpleNamespace(name_to_index=dict(data_input))),
        model=SimpleNamespace(output=SimpleNamespace(name_to_index=dict(model_output or data_input))),
    )


class _DataModule:
    def __init__(self):
        self.ds_test = _SplitDataset()
        self.data_indices = {
            "in_lres": _indices(LRES_VARS),
            "in_hres": _indices(HRES_VARS),
            "out_hres": _indices(OUT_DATA, OUT_MODEL),
        }
        self.config = OmegaConf.create({"dataloader": {"test": {"datasets": {
            "in_lres": {"dataset": "/stores/in_lres_o320_halo1p5.zarr", "drop": ["tp"]},
            "in_hres": {"dataset": "/stores/in_hres_o1280_forcings.zarr"},
            "out_hres": {"dataset": "/stores/out_hres_o1280.zarr"},
        }}}})


class _FlowMapNet:
    """Inner model: takes the two dicts, as the flow-map adapter does; draws its noise with torch."""

    def __init__(self):
        self.calls = []

    def predict_step(self, batch, model_comm_group=None, noise_scheduler_params=None, sampler_params=None, **kwargs):
        self.calls.append({
            "noise_scheduler_params": noise_scheduler_params,
            "sampler_params": sampler_params,
            "kwargs": kwargs,
            "seed": torch.initial_seed(),
            "in_lres": batch["in_lres"].clone(),
        })
        noise = torch.randn(1, 1, 1, N_HRES, len(OUT_MODEL))
        return {"out_hres": noise + batch["in_lres"][..., :1].mean()}


class _Interface:
    """AnemoiModelInterface stand-in: predict_step takes only **kwargs (see _predict_step_accepts)."""

    def __init__(self):
        self.model = _FlowMapNet()

    def predict_step(self, batch, model_comm_group=None, gather_out=True, **kwargs):
        return self.model.predict_step(batch, model_comm_group=model_comm_group, **kwargs)

    def interpolate_down(self, member_x, grad_checkpoint=False):
        # (batch, grid_lres, vars) -> (batch, grid_hres, vars): each hres point takes lres point i // 2
        return member_x[:, torch.arange(N_HRES) // 2, :]


def _main_module():
    # eval.predict re-exports the function main, which shadows the submodule as an attribute
    import importlib

    return importlib.import_module("eval.predict.main")


def _run_main(monkeypatch, tmp_path, *extra, members="1,2", dates="20230829,20230101", steps="24"):
    main_mod = _main_module()
    datamodule, model = _DataModule(), _Interface()
    seen = {}

    def _fake_load(config):
        seen["config"] = config
        return model, datamodule, json.loads(config.extra_args_json), "cpu", None, 0, 0, 1

    monkeypatch.setattr(main_mod, "load_inference_model", _fake_load)
    monkeypatch.setenv("ANEMOI_BASE_SEED", "1234")
    out_dir = tmp_path / "predictions"
    argv = [
        "--out-dir", str(out_dir),
        "--name-ckpt", str(tmp_path / "run" / "anemoi-by_step-epoch_029-step_030000.ckpt"),
        "--device", "cpu",
        "--members", members, "--dates", dates, "--steps", steps,
        "--extra-args-json", json.dumps(FLOW_MAP),
        "--output-weather-state-mode", "all",
        *extra,
    ]
    assert main_mod.main(argv) == 0
    return out_dir, datamodule, model, seen["config"]


def _row(date, lead):
    return DATES.index(date) * len(LEADS) + LEADS.index(lead)


def test_zarr_split_route_writes_valid_files(monkeypatch, tmp_path):
    out_dir, datamodule, model, config = _run_main(
        monkeypatch, tmp_path, "--input-source-json", ZARR_TEST, "--ema", "ema2000"
    )
    assert config.input_source == {"type": "zarr_split", "split": "test"} and config.ema == "ema2000"
    for date in ("20230829", "20230101"):
        path = out_dir / f"predictions_{date}_step024.nc"
        with xr.open_dataset(path) as ds:
            assert validate_predictions_dataset(ds) == []
            assert list(ds["weather_state"].values) == ["2t", "tp"]
            idx = _row(date, 24)
            # truth from out_hres of the row, in the saved-state order
            y = ds["y"].values[0]
            assert np.all(y[..., 0] == (idx + 1) * 100.0 + OUT_DATA["2t"])
            assert np.all(y[..., 1] == (idx + 1) * 100.0 + OUT_DATA["tp"])
            # x: the in_lres row for 2t, NaN for tp (absent from in_lres); x_interp the same
            x = ds["x"].values[0]
            assert x.shape == (2, N_LRES, 2)
            assert np.all(x[..., 0] == (idx + 1) * 100.0 + LRES_VARS["2t"]) and np.all(np.isnan(x[..., 1]))
            xi = ds["x_interp"].values[0]
            assert np.all(xi[..., 0] == (idx + 1) * 100.0) and np.all(np.isnan(xi[..., 1]))
            assert np.allclose(ds["lat_hres"].values, np.linspace(10.0, 70.0, N_HRES) + 0.5)
            # provenance in place of bundle paths; EMA weights recorded
            assert "source_bundle" not in ds
            assert ds.attrs["input_source"] == "zarr_split" and ds.attrs["input_split"] == "test"
            assert int(ds.attrs["input_row_index"]) == idx
            assert json.loads(ds.attrs["input_stores"])["out_hres"] == "/stores/out_hres_o1280.zarr"
            assert ds.attrs["checkpoint_weights"] == "ema2000"
            assert ds.attrs["checkpoint_id"] == "run:ema2000"
            assert ds.attrs["weights_path"].endswith("/run/inference-ema2000-anemoi-by_step-epoch_029-step_030000.ckpt")
            assert ds.attrs["sampler_summary"] == "jumps 2 from sigma 80"
            # rule 31: one seed per (base, date, lead, member)
            expected = [derive_draw_seed(1234, int(date), 24, m) for m in (1, 2)]
            assert ds.attrs["seeds"] == ",".join(str(s) for s in expected)
            assert int(ds.attrs["seed_base"]) == 1234
            # two members, same row, different noise
            yp = ds["y_pred"].values[0]
            assert not np.allclose(yp[0], yp[1])
    # the datamodule split was set up for one reader on the full grid, before any draw
    assert datamodule.ds_test.calls[0][0] == "set_comm_group_info"
    assert datamodule.ds_test.calls[1] == ("per_worker_init", (1, 0))
    # one read per case, shared by the members
    assert [c for c in datamodule.ds_test.calls if c[0] == "get_sample"] == [
        ("get_sample", _row("20230829", 24)), ("get_sample", _row("20230101", 24))]
    manifest = (out_dir / "predictions_manifest.csv").read_text().splitlines()
    assert manifest[1].startswith(f"20230829,24,1,zarr_split:test:row={_row('20230829', 24)}:fake_date=")


def test_flow_map_block_and_seeds_reach_predict_step(monkeypatch, tmp_path):
    _, _, model, _ = _run_main(monkeypatch, tmp_path, "--input-source-json", ZARR_TEST,
                               members="1,2,3", dates="20230829", steps="24")
    calls = model.model.calls
    assert len(calls) == 3
    for member, call in zip((1, 2, 3), calls):
        assert call["noise_scheduler_params"] == {"schedule_type": "flow_map", "num_steps": 2,
                                                  "sigma_max": 80.0, "sigma_min": 0.0}
        assert call["sampler_params"] == {"sampler": "jumps"}
        assert call["kwargs"] == {}
        assert call["seed"] == derive_draw_seed(1234, 20230829, 24, member)
        # every member draws from the same row
        assert torch.equal(call["in_lres"], calls[0]["in_lres"])
        assert tuple(call["in_lres"].shape) == (1, 1, 1, N_LRES, len(LRES_VARS))


def test_zarr_split_refuses_a_missing_row(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="no test row for 20230830"):
        _run_main(monkeypatch, tmp_path, "--input-source-json", ZARR_TEST, dates="20230830")


def test_bundle_route_still_needs_input_root(tmp_path):
    main_mod = _main_module()
    with pytest.raises(SystemExit, match="--input-root is required"):
        main_mod.main(["--out-dir", str(tmp_path / "p"), "--name-ckpt", "/x/run/a.ckpt"])


# --- EMA weights: load_objects picks inference-<ema>-<name> ------------------------------------


class _LoadedModel:
    graph_data = "graph"

    def to(self, device):
        return self

    def modules(self):
        return []


def _patch_load_objects(monkeypatch):
    import manual_inference.prediction.predict as mi_predict

    loaded, checkpoints = [], []

    def _get_checkpoint(dir_exp, name_exp, name_ckpt):
        checkpoints.append(name_ckpt)
        return {}, OmegaConf.create({"dataloader": {"validation": {}}})

    monkeypatch.setattr(mi_predict, "get_checkpoint", _get_checkpoint)
    monkeypatch.setattr(mi_predict, "adapt_config_hpc", lambda cfg: cfg)
    monkeypatch.setattr(mi_predict, "get_datamodule", lambda cfg, graph: "datamodule")
    monkeypatch.setattr(mi_predict.torch, "load", lambda path, **kw: loaded.append(os.path.basename(path)) or _LoadedModel())
    return mi_predict, loaded, checkpoints


@pytest.mark.parametrize("ema, expected", [
    (None, "inference-anemoi-step_030000.ckpt"),
    ("ema2000", "inference-ema2000-anemoi-step_030000.ckpt"),
    ("ema10000", "inference-ema10000-anemoi-step_030000.ckpt"),
])
def test_ema_key_picks_the_weights_file(monkeypatch, tmp_path, ema, expected):
    run = tmp_path / "run"
    run.mkdir()
    for name in ("anemoi-step_030000.ckpt", "inference-anemoi-step_030000.ckpt",
                 "inference-ema2000-anemoi-step_030000.ckpt", "inference-ema10000-anemoi-step_030000.ckpt"):
        (run / name).write_bytes(b"")
    mi_predict, loaded, checkpoints = _patch_load_objects(monkeypatch)
    mi_predict.load_objects(ckpt_path=str(run / "anemoi-step_030000.ckpt"), device="cpu",
                            validation_frequency="50h", precision="fp32", ema=ema)
    assert loaded == [expected]
    assert checkpoints == ["anemoi-step_030000.ckpt"]  # the config still comes from the base checkpoint


def test_missing_ema_file_is_refused_before_loading(monkeypatch, tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "anemoi-step_030000.ckpt").write_bytes(b"")
    (run / "inference-anemoi-step_030000.ckpt").write_bytes(b"")
    mi_predict, loaded, checkpoints = _patch_load_objects(monkeypatch)
    with pytest.raises(FileNotFoundError, match="inference-ema10000-anemoi-step_030000.ckpt"):
        mi_predict.load_objects(ckpt_path=str(run / "anemoi-step_030000.ckpt"), device="cpu",
                                validation_frequency="50h", precision="fp32", ema="ema10000")
    assert loaded == [] and checkpoints == []


# --- eval.cli predict: the lane keys become eval.predict.main flags ------------------------------


def _cli_cmd(monkeypatch, tmp_path, opts, **arg_overrides):
    import argparse

    import eval.cli.predict as cli_predict
    from eval.cli._common import _build_lane_overrides
    from eval.config.loader import load_lane

    args = argparse.Namespace(mode="manual", checkpoint=str(tmp_path / "run" / "a.ckpt"), bundle_dir=None,
                              source_grib_root=None, opt=opts, members="1", steps="24", dates="20230829",
                              weather_states=None, num_gpus_per_model=None)
    for key, value in arg_overrides.items():
        setattr(args, key, value)
    lane = load_lane("o320_o1280_e2_flowmap", overrides=_build_lane_overrides(args) or None)
    captured = []
    monkeypatch.setattr(cli_predict.subprocess, "run", lambda cmd, check: captured.append(cmd))
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.delenv("SLURM_STEP_ID", raising=False)
    cli_predict.cmd_predict(args, lane, {}, tmp_path / "out")
    return captured[0]


def _flag(cmd, name):
    return cmd[cmd.index(name) + 1]


def test_cli_zarr_split_lane_passes_route_and_ema(monkeypatch, tmp_path):
    cmd = _cli_cmd(monkeypatch, tmp_path, ["predict.ema=ema2000", "sampler.num_steps=1"])
    assert cmd[:3] == [sys.executable, "-m", "eval.predict.main"]  # one GPU per model: no srun
    assert json.loads(_flag(cmd, "--input-source-json")) == {"type": "zarr_split", "split": "test"}
    assert "--input-root" not in cmd
    assert _flag(cmd, "--ema") == "ema2000"
    assert _flag(cmd, "--num-gpus-per-model") == "1"
    assert json.loads(_flag(cmd, "--extra-args-json"))["num_steps"] == 1


def test_cli_zarr_split_refuses_bundle_dir(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="zarr_split, which reads no bundle"):
        _cli_cmd(monkeypatch, tmp_path, [], bundle_dir=str(tmp_path / "bundles"))


def test_cli_raw_weights_by_default(monkeypatch, tmp_path):
    cmd = _cli_cmd(monkeypatch, tmp_path, [])
    assert "--ema" not in cmd


@pytest.mark.parametrize("opt, message", [
    ("predict.ema=ema_2000", "EMA tag"),
    ("predict.input_source={type: zarr_split, split: test, store: /x}", "takes only 'split'"),
    ("predict.input_source={type: zarr_split, split: valid}", "split must be one of"),
])
def test_lane_refuses_bad_values(monkeypatch, tmp_path, opt, message):
    from eval.config.loader import ConfigValidationError

    with pytest.raises(ConfigValidationError, match=message):
        _cli_cmd(monkeypatch, tmp_path, [opt])


def test_zarr_split_refuses_a_split_whose_variables_differ_from_the_model():
    from eval.predict.zarr_split import ZarrSplitSource

    datamodule = _DataModule()
    datamodule.data_indices["in_lres"] = _indices({"2t": 1, "10u": 0, "lsm": 2})
    with pytest.raises(SystemExit, match="differ from the model's data indices"):
        ZarrSplitSource(datamodule, "test")
