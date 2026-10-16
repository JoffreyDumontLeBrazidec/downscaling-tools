#!/usr/bin/env python3
"""Convert our per-dataset downscaling graph files into the one fused graph Vera's branch loads.

Our single-time models (AnemoiD2ModelEncProcDec, e.g. the do0 recipe) read one graph file per
dataset, <stem>_in_lres.pt, <stem>_in_hres.pt, <stem>_out_hres.pt, each a HeteroData with node
sets "data" and "hidden". The network uses:
  encoder   (data, to, hidden) of the out_hres file (encoder_datasets: [out_hres]);
  decoder   (hidden, to, data) of the out_hres file;
  processor (hidden, to, hidden) of the FIRST dataset's file (in_lres: AnemoiD2ModelEncProcDec
            takes next(iter(graph_data)), and the datasets come in dataloader order);
  node attributes of the out_hres file's data and hidden nodes.

Vera Gahlen's AnemoiTransportSpatialDownscalerModelEncProcDec reads ONE HeteroData with node sets
in_lres, in_hres, out_hres, hidden (all three data sets on the target grid), encoder edges
(in_lres, to, hidden), processor edges (hidden, to, hidden) and decoder edges (hidden, to, out_hres).

This script writes that graph from our files: the out_hres file's "data" node store becomes
in_lres, in_hres and out_hres (same coordinates, area weights, ring-mask weights), its hidden node
store becomes hidden, its encoder and decoder edge stores are renamed, and the processor edges are
taken from the in_lres file (and compared with the out_hres file's). Every tensor is copied
unchanged; the script prints what it checked.

Usage: python make_vera_graph.py --stem <dir>/<stem> --out <file.pt>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def store_tensors(store):
    import torch

    return {k: v for k, v in store.items() if torch.is_tensor(v)}


def main() -> int:
    import torch
    from torch_geometric.data import HeteroData

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stem", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if Path(args.out).exists():
        print(f"refusing: {args.out} exists")
        return 2
    files = {ds: torch.load(f"{args.stem}_{ds}.pt", map_location="cpu", weights_only=False)
             for ds in ("in_lres", "in_hres", "out_hres")}
    for ds, g in files.items():
        print(f"{ds}: nodes {[(nt, g[nt].num_nodes, sorted(store_tensors(g[nt]))) for nt in g.node_types]}")
        print(f"{ds}: edges {[(et, g[et].edge_index.shape[1], sorted(store_tensors(g[et]))) for et in g.edge_types]}")
    go, gl = files["out_hres"], files["in_lres"]

    def same(a, b):
        ta, tb = store_tensors(a), store_tensors(b)
        return set(ta) == set(tb) and all(ta[k].shape == tb[k].shape and torch.equal(ta[k], tb[k]) for k in ta)

    hh = ("hidden", "to", "hidden")
    print(f"processor edges of the in_lres file equal those of the out_hres file: {same(gl[hh], go[hh])}")
    print(f"hidden nodes of the in_lres file equal those of the out_hres file: {same(gl['hidden'], go['hidden'])}")

    g = HeteroData()
    for name in ("in_lres", "in_hres", "out_hres"):
        for k, v in store_tensors(go["data"]).items():
            g[name][k] = v.clone()
    for k, v in store_tensors(go["hidden"]).items():
        g["hidden"][k] = v.clone()
    for src_et, dst_et, src_graph in ((("data", "to", "hidden"), ("in_lres", "to", "hidden"), go),
                                      (hh, hh, gl),
                                      (("hidden", "to", "data"), ("hidden", "to", "out_hres"), go)):
        for k, v in store_tensors(src_graph[src_et]).items():
            g[dst_et][k] = v.clone()
    print(f"fused graph: {g}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(g, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
