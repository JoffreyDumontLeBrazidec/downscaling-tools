from __future__ import annotations

import numpy as np
import torch

from eval.predict.graph_cut import activate_local_graph_cut


class _Provider(torch.nn.Module):
    def __init__(self, edge_index, edge_attr, src_size, dst_size):
        super().__init__()
        self.register_buffer("edge_index_base", torch.as_tensor(edge_index, dtype=torch.long))
        self.register_buffer("edge_attr", torch.as_tensor(edge_attr, dtype=torch.float32))
        self.register_buffer("edge_inc", torch.tensor([[src_size], [dst_size]], dtype=torch.long))
        self.trainable = torch.nn.Module()
        self.trainable.trainable = None
        self._edges_sorted_by_dst = True


class _Node:
    def __init__(self, lat_lon_deg):
        radians = np.radians(np.asarray(lat_lon_deg, dtype=np.float32))
        self.x = torch.as_tensor(radians)
        self.num_nodes = self.x.shape[0]


class _Edge:
    def __init__(self, edge_index):
        self.edge_index = torch.as_tensor(edge_index, dtype=torch.long)


class _Graph(dict):
    @property
    def node_types(self):
        return ["data", "hidden"]

    def node_items(self):
        return [(name, self[name]) for name in self.node_types]


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        graph = _Graph()
        graph["data"] = _Node([(0, 0), (0, 1), (50, 50)])
        graph["hidden"] = _Node([(0, 0), (0, 1), (50, 50)])
        graph[("data", "to", "hidden")] = _Edge([[0, 1, 2], [0, 1, 2]])
        graph[("hidden", "to", "data")] = _Edge([[0, 1, 2], [0, 1, 2]])
        graph[("hidden", "to", "hidden")] = _Edge([[0, 1, 2], [1, 0, 2]])
        self._graph_data = {"out_hres": graph}
        self._graph_name_data = "data"
        self._graph_name_hidden = "hidden"
        from anemoi.models.layers.graph import NamedNodesAttributes
        self.node_attributes = torch.nn.ModuleDict({"out_hres": NamedNodesAttributes(0, graph)})
        self.encoder_graph_provider = torch.nn.ModuleDict({
            "out_hres": _Provider([[0, 1, 2], [0, 1, 2]], [[1.0], [1.0], [1.0]], 3, 3)
        })
        self.decoder_graph_provider = torch.nn.ModuleDict({
            "out_hres": _Provider([[0, 1, 2], [0, 1, 2]], [[1.0], [1.0], [1.0]], 3, 3)
        })
        self.processor_graph_provider = _Provider([[0, 1, 2], [1, 0, 2]], [[1.0], [1.0], [1.0]], 3, 3)

    def _before_sampling(self, batch, pre_processors, n_step_input, model_comm_group=None, **kwargs):
        x = torch.zeros(1, 1, 1, 3, 2)
        return (x, x.clone()), None

    def apply_interpolate_to_high_res(self, x, grid_shard_sizes=None, model_comm_group=None):
        return torch.zeros(1, 1, 1, 3, 2)


class _Interface(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Model()


def test_activate_local_graph_cut_slices_providers_and_hooks():
    interface = _Interface()
    stats = activate_local_graph_cut(
        interface,
        {"mode": "bbox", "lat_min": -1, "lat_max": 2, "lon_min": -1, "lon_max": 2, "cut_graph": True},
    )

    assert stats["data_nodes_after"] == 2
    assert interface.model.encoder_graph_provider["out_hres"].edge_inc[:, 0].tolist() == [2, 2]
    before, _ = interface.model._before_sampling({}, {}, 1)
    assert before[0].shape[-2] == 2
    assert interface.model.apply_interpolate_to_high_res(torch.zeros(1, 1, 3, 2)).shape[-2] == 2


def test_activate_local_graph_cut_row_cuts_interpolation_matrix():
    """When the model carries the InterpolationConnection layout, the sparse
    lres->hres matrix is row-cut to the box and the hooks stop cropping
    x_interp (it is already box-sized)."""
    interface = _Interface()
    model = interface.model

    # 3 hres rows x 2 lres cols, one entry per row.
    provider = torch.nn.Module()
    provider.projection_matrix = torch.sparse_coo_tensor(
        torch.tensor([[0, 1, 2], [0, 1, 0]]),
        torch.tensor([1.0, 2.0, 3.0]),
        size=(3, 2),
    )
    conn = torch.nn.Module()
    conn.provider = provider
    model.residual = torch.nn.ModuleDict({"in_lres": conn})

    # These now return BOX-sized tensors, mimicking the pre-cut matrix output.
    def _before(self, batch, pre_processors, n_step_input, model_comm_group=None, **kwargs):
        x_interp = torch.zeros(1, 1, 1, 2, 2)  # already box-sized (2 nodes)
        x_hres = torch.zeros(1, 1, 1, 3, 2)  # still full-grid (3 nodes)
        return (x_interp, x_hres), None

    def _interp(self, x, grid_shard_sizes=None, model_comm_group=None):
        return torch.zeros(1, 1, 1, 2, 2)  # already box-sized

    from types import MethodType

    model._before_sampling = MethodType(_before, model)
    model.apply_interpolate_to_high_res = MethodType(_interp, model)

    stats = activate_local_graph_cut(
        interface,
        {"mode": "bbox", "lat_min": -1, "lat_max": 2, "lon_min": -1, "lon_max": 2, "cut_graph": True},
    )

    assert stats["interp_precut"] is True
    assert stats["interp_nnz_before"] == 3
    assert stats["interp_nnz_after"] == 2
    matrix = model.residual["in_lres"].provider.projection_matrix.coalesce()
    assert matrix.shape == (2, 2)
    assert matrix.values().tolist() == [1.0, 2.0]

    before, _ = model._before_sampling({}, {}, 1)
    assert before[0].shape[-2] == 2  # x_interp untouched (pre-cut)
    assert before[1].shape[-2] == 2  # x_hres cropped by the hook
    assert model.apply_interpolate_to_high_res(torch.zeros(1, 1, 2, 2)).shape[-2] == 2


def test_activate_local_graph_cut_interp_shape_mismatch_falls_back():
    """A projection matrix whose rows are not the data grid is left alone and
    the crop-after-interp behavior is kept."""
    interface = _Interface()
    model = interface.model

    provider = torch.nn.Module()
    provider.projection_matrix = torch.sparse_coo_tensor(
        torch.tensor([[0, 4], [0, 1]]),
        torch.tensor([1.0, 2.0]),
        size=(5, 2),  # 5 rows != 3 data nodes
    )
    conn = torch.nn.Module()
    conn.provider = provider
    model.residual = torch.nn.ModuleDict({"in_lres": conn})

    stats = activate_local_graph_cut(
        interface,
        {"mode": "bbox", "lat_min": -1, "lat_max": 2, "lon_min": -1, "lon_max": 2, "cut_graph": True},
    )

    assert stats["interp_precut"] is False
    assert model.residual["in_lres"].provider.projection_matrix.shape == (5, 2)
    before, _ = model._before_sampling({}, {}, 1)
    assert before[0].shape[-2] == 2  # full-grid x_interp cropped by the hook
