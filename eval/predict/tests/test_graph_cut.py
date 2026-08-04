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
