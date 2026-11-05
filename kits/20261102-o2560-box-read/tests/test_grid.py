import numpy as np
import pytest
from conftest import GRAPH_OUT, MAT, STORE, need


def test_store_matrix_and_graph_point_counts():
    import scipy.sparse as sp
    import zarr
    out = zarr.open(need(f"{STORE}/hres_o2560.zarr"), "r")["data"].shape[-1]
    inn = zarr.open(need(f"{STORE}/in_lres_o1280.zarr"), "r")["data"].shape[-1]
    M = sp.load_npz(need(MAT))
    assert out == 3_248_042 and inn == 876_159
    assert M.shape == (out, inn)
    torch = pytest.importorskip("torch")
    g = torch.load(need(GRAPH_OUT), map_location="cpu", weights_only=False)
    n_data = g["data"].num_nodes if hasattr(g["data"], "num_nodes") else g["data"]["x"].shape[0]
    n_hidden = g["hidden"].num_nodes if hasattr(g["hidden"], "num_nodes") else g["hidden"]["x"].shape[0]
    assert n_data == out, (n_data, out)
    assert n_hidden == 434_914, n_hidden   # r9 box mesh (note section 2: 434,914 hidden nodes)
