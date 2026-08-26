"""Experimental local graph cutting for bundle inference."""
from __future__ import annotations

import json
from types import MethodType
from typing import Any

import numpy as np
import torch

from .local_scope import hres_mask_for_scope, load_local_scope


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _coords_deg(node_store) -> tuple[np.ndarray, np.ndarray]:
    xy = node_store.x.detach().cpu().numpy()
    lat = np.degrees(xy[:, 0])
    lon = (np.degrees(xy[:, 1]) + 180.0) % 360.0 - 180.0
    return lat, lon


def _mask_to_index(mask: torch.Tensor) -> torch.Tensor:
    out = torch.full((int(mask.numel()),), -1, dtype=torch.long, device=mask.device)
    out[mask] = torch.arange(int(mask.sum()), dtype=torch.long, device=mask.device)
    return out


def _expand_hidden_halo(edge_index: torch.Tensor, hidden_mask: torch.Tensor, hops: int) -> torch.Tensor:
    if hops <= 0:
        return hidden_mask
    src, dst = edge_index[0], edge_index[1]
    mask = hidden_mask.clone()
    for _ in range(hops):
        connected = mask[src] | mask[dst]
        if not bool(connected.any()):
            break
        new_mask = mask.clone()
        new_mask[src[connected]] = True
        new_mask[dst[connected]] = True
        if bool(torch.equal(new_mask, mask)):
            break
        mask = new_mask
    return mask


def _subset_provider(provider, src_mask: torch.Tensor, dst_mask: torch.Tensor) -> None:
    """Subset a StaticGraphProvider in-place and relabel its edge indices."""

    edge_index = provider.edge_index_base.detach().cpu()
    edge_attr = provider.edge_attr.detach().cpu()
    src_keep = src_mask[edge_index[0]]
    dst_keep = dst_mask[edge_index[1]]
    edge_keep = src_keep & dst_keep
    if not bool(edge_keep.any()):
        raise ValueError("local graph cut removed all edges for a graph provider")

    src_relabel = _mask_to_index(src_mask)
    dst_relabel = _mask_to_index(dst_mask)
    new_edge_index = torch.stack(
        (src_relabel[edge_index[0, edge_keep]], dst_relabel[edge_index[1, edge_keep]]),
        dim=0,
    )
    device = provider.edge_index_base.device
    provider.edge_index_base = new_edge_index.to(device=device)
    provider.edge_attr = edge_attr[edge_keep].to(device=device)
    provider.edge_inc = torch.tensor(
        [[int(src_mask.sum())], [int(dst_mask.sum())]],
        dtype=provider.edge_inc.dtype,
        device=device,
    )
    trainable_param = getattr(getattr(provider, "trainable", None), "trainable", None)
    if trainable_param is not None:
        provider.trainable.trainable = torch.nn.Parameter(trainable_param.detach().cpu()[edge_keep].to(device=device))
    provider._edges_sorted_by_dst = False


def _subset_interpolation(model, data_mask: torch.Tensor) -> dict[str, int] | None:
    """Row-cut the lres->hres interpolation projection matrix to the data mask.

    The residual connection's ProjectionGraphProvider bakes a sparse matrix of
    shape (n_hres_full, n_lres_full). At o2560 the full-grid product tensor of
    torch.sparse.mm is ~9 GiB (26.3M nodes x 91 channels) — too large a
    transient on a 40 GB card. Keeping only the masked rows makes the
    interpolation produce box-sized output directly; the crop-after-interp hook
    is then skipped for x_interp. Falls back to None (crop-after behavior kept)
    whenever the module layout or shapes do not match — behavior for models
    without this residual layout is unchanged.
    """

    residual = getattr(model, "residual", None)
    if residual is None:
        return None
    try:
        conn = residual["in_lres"]
    except (KeyError, TypeError):
        return None
    provider = getattr(conn, "provider", None)
    matrix = getattr(provider, "projection_matrix", None)
    if matrix is None or not matrix.is_sparse:
        return None
    if int(matrix.shape[0]) != int(data_mask.numel()):
        return None

    matrix = matrix.coalesce()
    indices = matrix.indices()
    values = matrix.values()
    mask_dev = data_mask.to(indices.device)
    keep = mask_dev[indices[0]]
    if not bool(keep.any()):
        raise ValueError("local graph cut removed all interpolation matrix rows")
    relabel = _mask_to_index(mask_dev)
    new_indices = torch.stack((relabel[indices[0, keep]], indices[1, keep]), dim=0)
    new_matrix = torch.sparse_coo_tensor(
        new_indices,
        values[keep],
        size=(int(mask_dev.sum()), int(matrix.shape[1])),
        device=matrix.device,
        dtype=values.dtype,
    ).coalesce()
    stats = {
        "interp_nnz_before": int(values.numel()),
        "interp_nnz_after": int(new_matrix._nnz()),
    }
    provider.projection_matrix = new_matrix
    return stats


def _replace_node_attributes(model, dataset: str, graph, data_mask: torch.Tensor, hidden_mask: torch.Tensor) -> None:
    from anemoi.models.layers.graph import NamedNodesAttributes

    data_name = model._graph_name_data
    hidden_name = model._graph_name_hidden
    old_attrs = model.node_attributes[dataset]
    trainable_size = 0
    old_hidden_trainable = getattr(old_attrs.trainable_tensors[hidden_name], "trainable", None)
    if old_hidden_trainable is not None:
        trainable_size = int(old_hidden_trainable.shape[1])

    new_attrs = NamedNodesAttributes(trainable_size, graph)
    for name, mask in ((data_name, data_mask), (hidden_name, hidden_mask)):
        old_param = getattr(old_attrs.trainable_tensors[name], "trainable", None)
        new_param = getattr(new_attrs.trainable_tensors[name], "trainable", None)
        if old_param is not None and new_param is not None:
            with torch.no_grad():
                new_param.copy_(old_param.detach().cpu()[mask])
    model.node_attributes[dataset] = new_attrs.to(_model_device(model))


def _cut_graph_data(model, dataset: str, scope: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    graph = model._graph_data[dataset]
    data_name = model._graph_name_data
    hidden_name = model._graph_name_hidden

    data_lat, data_lon = _coords_deg(graph[data_name])
    data_mask_np = hres_mask_for_scope(data_lon, data_lat, scope)
    data_mask = torch.as_tensor(data_mask_np, dtype=torch.bool)
    if int(data_mask.sum()) == 0:
        raise ValueError(f"local graph cut selected zero {data_name!r} nodes")
    if int(data_mask.sum()) == int(data_mask.numel()):
        raise ValueError("local graph cut requested, but scope selected the full data graph")

    enc = graph[(data_name, "to", hidden_name)].edge_index.detach().cpu()
    hidden_count = int(graph[hidden_name].num_nodes)
    hidden_mask = torch.zeros(hidden_count, dtype=torch.bool)
    hidden_mask[enc[1, data_mask[enc[0]]]] = True
    proc_edge_index = graph[(hidden_name, "to", hidden_name)].edge_index.detach().cpu()
    hidden_mask = _expand_hidden_halo(proc_edge_index, hidden_mask, int(scope.get("hidden_halo_hops", 1)))
    if int(hidden_mask.sum()) == 0:
        raise ValueError("local graph cut selected zero hidden nodes")

    # Keep graph_data consistent for diagnostics and any code that reads it after activation.
    graph[data_name].x = graph[data_name].x[data_mask]
    graph[hidden_name].x = graph[hidden_name].x[hidden_mask]
    graph[data_name].num_nodes = int(data_mask.sum())
    graph[hidden_name].num_nodes = int(hidden_mask.sum())

    _replace_node_attributes(model, dataset, graph, data_mask, hidden_mask)
    return data_mask, hidden_mask, {
        "data_nodes_before": int(data_mask.numel()),
        "data_nodes_after": int(data_mask.sum()),
        "hidden_nodes_before": int(hidden_mask.numel()),
        "hidden_nodes_after": int(hidden_mask.sum()),
    }


def _patch_sampling_hooks(model, data_mask: torch.Tensor, interp_precut: bool) -> None:
    mask_device_cache: dict[torch.device, torch.Tensor] = {}

    def mask_for(device: torch.device) -> torch.Tensor:
        if device not in mask_device_cache:
            mask_device_cache[device] = data_mask.to(device=device)
        return mask_device_cache[device]

    original_before = model._before_sampling
    original_interp = model.apply_interpolate_to_high_res

    def _before_sampling_local(self, batch, pre_processors, n_step_input, model_comm_group=None, **kwargs):
        if model_comm_group is not None and getattr(model_comm_group, "size", lambda: 1)() != 1:
            raise ValueError("local cut-graph inference currently supports only one model rank")
        before, grid_shard_sizes = original_before(batch, pre_processors, n_step_input, model_comm_group, **kwargs)
        x_interp, x_hres = before
        m = mask_for(x_interp.device)
        if not interp_precut:
            x_interp = x_interp[..., m, :]
        x_hres = x_hres[..., m, :]
        return (x_interp, x_hres), grid_shard_sizes

    def _apply_interpolate_local(self, x, grid_shard_sizes=None, model_comm_group=None):
        out = original_interp(x, grid_shard_sizes=grid_shard_sizes, model_comm_group=model_comm_group)
        if interp_precut:
            return out
        return out[..., mask_for(out.device), :]

    model._before_sampling = MethodType(_before_sampling_local, model)
    model.apply_interpolate_to_high_res = MethodType(_apply_interpolate_local, model)


def activate_local_graph_cut(inference_model, raw_scope: str | dict[str, Any] | None) -> dict[str, int | str]:
    """Activate an experimental local cut graph on an Anemoi inference interface.

    The cut is opt-in via ``local_scope.cut_graph: true``. It reduces the
    ``out_hres`` data graph to the requested local support, keeps hidden nodes
    connected to that support plus an optional processor halo, slices graph
    providers/node attributes, row-cuts the baked lres->hres interpolation
    matrix where the layout allows it (falling back to cropping the full-grid
    interpolation output), and crops sampling tensors to the same hres mask.
    """

    scope = load_local_scope(raw_scope)
    if not bool(scope.get("cut_graph", False)):
        return {"mode": "disabled"}
    if scope["mode"] == "global":
        raise ValueError("local graph cut requires a non-global local_scope mode")

    model = getattr(inference_model, "model", inference_model)
    dataset = str(scope.get("target_dataset", "out_hres"))
    if not hasattr(model, "_graph_data") or dataset not in model._graph_data:
        raise ValueError(f"model does not expose graph_data for dataset {dataset!r}")
    if not hasattr(model, "encoder_graph_provider") or dataset not in model.encoder_graph_provider:
        raise ValueError(f"model does not expose encoder graph provider for dataset {dataset!r}")

    data_mask, hidden_mask, stats = _cut_graph_data(model, dataset, scope)
    _subset_provider(model.encoder_graph_provider[dataset], data_mask, hidden_mask)
    _subset_provider(model.decoder_graph_provider[dataset], hidden_mask, data_mask)
    _subset_provider(model.processor_graph_provider, hidden_mask, hidden_mask)
    interp_stats = _subset_interpolation(model, data_mask)
    if interp_stats:
        stats.update(interp_stats)
    _patch_sampling_hooks(model, data_mask, interp_precut=interp_stats is not None)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    inference_model._local_graph_cut_scope_json = json.dumps(scope, sort_keys=True)
    inference_model._local_graph_cut_data_mask = data_mask
    inference_model._local_graph_cut_stats = stats
    stats["mode"] = "cut_graph"
    stats["interp_precut"] = bool(interp_stats)
    return stats
