import numpy as np
import torch

from manual_inference.prediction import predict


class _DummyData:
    def __init__(self, x_in, x_in_hres, y, lon_lres, lat_lres, lon_hres, lat_hres, dates):
        self._x_in = x_in
        self._x_in_hres = x_in_hres
        self._y = y
        self.longitudes = [lon_lres, None, lon_hres]
        self.latitudes = [lat_lres, None, lat_hres]
        self.dates = dates

    def __getitem__(self, idx):
        return self._x_in[idx], self._x_in_hres[idx], self._y[idx]


class _DummyDataModule:
    def __init__(self, data, name_to_idx_in, name_to_idx_out):
        class _DataIndices:
            def __init__(self, name_to_idx_in, name_to_idx_out):
                class _Input:
                    def __init__(self, name_to_idx):
                        self.name_to_index = name_to_idx

                class _Model:
                    def __init__(self, name_to_idx):
                        self.output = _Input(name_to_idx)

                self.data = type("_Data", (), {"input": [_Input(name_to_idx_in)]})
                self.model = _Model(name_to_idx_out)

        self.ds_valid = type("_DS", (), {"data": data})
        self.data_indices = _DataIndices(name_to_idx_in, name_to_idx_out)


class _DummyModel:
    def __init__(self, grid_hres, n_states):
        self.grid_hres = grid_hres
        self.n_states = n_states

    def predict_step(self, x_l, x_h, extra_args=None):
        shape = (1, 1, 1, self.grid_hres, self.n_states)
        return torch.ones(shape, dtype=torch.float32)


def test_predict_from_dataloader_shapes_and_members():
    n_samples = 2
    n_vars_in = 3
    n_vars_out = 2
    n_ens = 2
    grid_lres = 4
    grid_hres = 5

    x_in = np.random.rand(n_samples, n_vars_in, n_ens, grid_lres).astype(np.float32)
    x_in_hres = np.random.rand(n_samples, n_vars_in, n_ens, grid_hres).astype(
        np.float32
    )
    y = np.random.rand(n_samples, n_vars_out, n_ens, grid_hres).astype(np.float32)

    lon_lres = np.linspace(0, 3, grid_lres).astype(np.float32)
    lat_lres = np.linspace(10, 13, grid_lres).astype(np.float32)
    lon_hres = np.linspace(0, 4, grid_hres).astype(np.float32)
    lat_hres = np.linspace(20, 24, grid_hres).astype(np.float32)
    dates = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[D]")

    data = _DummyData(
        x_in=x_in,
        x_in_hres=x_in_hres,
        y=y,
        lon_lres=lon_lres,
        lat_lres=lat_lres,
        lon_hres=lon_hres,
        lat_hres=lat_hres,
        dates=dates,
    )

    name_to_idx_in = {"a": 0, "b": 1, "c": 2}
    name_to_idx_out = {"b": 0, "c": 1}
    datamodule = _DummyDataModule(data, name_to_idx_in, name_to_idx_out)
    model = _DummyModel(grid_hres=grid_hres, n_states=len(name_to_idx_out))

    out = predict._predict_from_dataloader(
        inference_model=model,
        datamodule=datamodule,
        device="cpu",
        idx=0,
        n_samples=n_samples,
        members=[0, 1],
        extra_args={},
    )

    x_out, y_out, y_pred, lon_l, lat_l, lon_h, lat_h, states, out_dates = out
    assert x_out.shape == (n_samples, 2, grid_lres, len(name_to_idx_out))
    assert y_out.shape == (n_samples, 2, grid_hres, n_vars_out)
    assert y_pred.shape == (n_samples, 2, grid_hres, len(name_to_idx_out))
    assert np.allclose(lon_l, lon_lres)
    assert np.allclose(lat_l, lat_lres)
    assert np.allclose(lon_h, lon_hres)
    assert np.allclose(lat_h, lat_hres)
    assert states == ["b", "c"]
    assert np.all(out_dates == dates)


def test_predict_from_dataloader_no_members():
    x_in = np.zeros((1, 1, 1, 2), dtype=np.float32)
    x_in_hres = np.zeros((1, 1, 1, 3), dtype=np.float32)
    y = np.zeros((1, 1, 1, 3), dtype=np.float32)
    data = _DummyData(
        x_in=x_in,
        x_in_hres=x_in_hres,
        y=y,
        lon_lres=np.zeros(2, dtype=np.float32),
        lat_lres=np.zeros(2, dtype=np.float32),
        lon_hres=np.zeros(3, dtype=np.float32),
        lat_hres=np.zeros(3, dtype=np.float32),
        dates=np.array(["2024-01-01"], dtype="datetime64[D]"),
    )
    datamodule = _DummyDataModule(data, {"a": 0}, {"a": 0})
    model = _DummyModel(grid_hres=3, n_states=1)

    try:
        predict._predict_from_dataloader(
            inference_model=model,
            datamodule=datamodule,
            device="cpu",
            idx=0,
            n_samples=1,
            members=[],
            extra_args={},
        )
    except ValueError as exc:
        assert "No members selected" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty members")


def test_predict_from_bundle_minimal(monkeypatch):
    point_lres = 3
    point_hres = 4
    n_lres_features = 2

    x_template = torch.zeros((1, 1, 1, point_lres, n_lres_features), dtype=torch.float32)
    x_hres_template = torch.zeros((1, 1, 1, point_hres, 1), dtype=torch.float32)

    def _fake_template(datamodule, batch_index, device):
        return x_template.clone(), x_hres_template.clone(), None

    def _fake_fill(bundle_nc, x_lres, x_hres, name_to_idx_lres, name_to_idx_hres, device, **kwargs):
        x_lres[0, 0, 0, :, 0] = 1.0
        x_lres[0, 0, 0, :, 1] = 2.0

    monkeypatch.setattr(predict, "_get_template_batch", _fake_template)
    monkeypatch.setattr(predict, "fill_inputs_from_bundle", _fake_fill)

    class _Indices:
        def __init__(self):
            self.data = type(
                "_Data",
                (),
                {
                    "input": [
                        type("_In", (), {"name_to_index": {"b": 0, "c": 1}}),
                        type("_In", (), {"name_to_index": {"z": 0}}),
                    ]
                },
            )
            self.model = type("_Model", (), {"output": type("_Out", (), {"name_to_index": {"b": 0, "c": 1}})})

    class _DummyDM:
        def __init__(self):
            self.data_indices = _Indices()
            self.ds_valid = type(
                "_DS",
                (),
                {
                    "data": type(
                        "_Data",
                        (),
                        {
                            "longitudes": [np.arange(point_lres), None, np.arange(point_hres)],
                            "latitudes": [np.arange(point_lres) + 10, None, np.arange(point_hres) + 20],
                        },
                    )
                },
            )

    model = _DummyModel(grid_hres=point_hres, n_states=2)
    dm = _DummyDM()

    out = predict._predict_from_bundle(
        inference_model=model,
        datamodule=dm,
        device="cpu",
        bundle_nc="/tmp/fake.nc",
        batch_index=0,
        extra_args={},
    )

    x_out, y_out, y_pred, lon_l, lat_l, lon_h, lat_h, states, dates = out
    assert y_out is None
    assert x_out.shape == (1, point_lres, 2)
    assert y_pred.shape == (1, 1, point_hres, 2)
    assert states == ["b", "c"]
    assert dates is None
    assert np.allclose(lon_l, np.arange(point_lres))
    assert np.allclose(lat_h, np.arange(point_hres) + 20)
