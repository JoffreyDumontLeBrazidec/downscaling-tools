import sys, time, json
import numpy as np
t0 = time.time()
import scipy.sparse as sp
for p in ["/ec/res4/hpcperm/ecm5702/data/inter_mat/interpol_o320_to_o1280_grid-box-average.mat.npz",
          "/home/ecm5702/hpcperm/lanes/o320_o1280_europe_e2/inter_mat/interpol_O320_to_O1280_linear-europe_e2_10_70N_80W_40E_halo1p5.mat.npz"]:
    z = np.load(p, allow_pickle=True)
    print(p, "keys", list(z.keys()))
    try:
        m = sp.load_npz(p).tocsr()
        rs = np.asarray(m.sum(axis=1)).ravel()
        print("  shape", m.shape, "nnz", m.nnz, "row sums min %.9f max %.9f" % (rs.min(), rs.max()), "max|rs-1| %.3e" % np.abs(rs - 1).max())
    except Exception as e:
        print("  load_npz failed", e)
print("t", time.time() - t0, flush=True)
import torch
print("torch", time.time() - t0, flush=True)
G = "/home/ecm5702/hpcperm/lanes/o320_o1280_europe_e2/graph_m600_ringmask_knn13/europe_e2_m600rmknn13_graph"
for ds in ["out_hres", "in_lres", "in_hres"]:
    g = torch.load(f"{G}_{ds}.pt", weights_only=False, map_location="cpu")
    print(ds, type(g).__name__)
    for nt in g.node_types:
        print("  node", nt, {k: (tuple(v.shape), str(v.dtype)) for k, v in g[nt].items() if torch.is_tensor(v)})
    for et in g.edge_types:
        print("  edge", et, {k: (tuple(v.shape), str(v.dtype)) for k, v in g[et].items() if torch.is_tensor(v)})
    print("  other", {k: v for k, v in g.items() if not torch.is_tensor(v)} if hasattr(g, "items") else None)
print("t", time.time() - t0, flush=True)
D = "/home/ecm5702/perm/checkpoints/o320_o1280/12dcefeafa92457aab286d4931ffe63f/anemoi-by_epoch-epoch_172-step_400000.ckpt"
ck = torch.load(D, weights_only=False, map_location="cpu", mmap=True)
print("donor keys", list(ck.keys()))
sd = ck["state_dict"]
roots = {}
for k in sd:
    r = ".".join(k.split(".")[:3]); roots[r] = roots.get(r, 0) + 1
print("donor roots", roots)
net = [k for k in sd if k.split(".")[2] in ("encoder", "decoder", "processor", "noise_cond_mlp", "noise_embedder")]
print("n network tensors", len(net))
for k in net:
    if "processor.proc" in k and not ".proc.0." in k:
        continue
    print("  ", k, tuple(sd[k].shape))
print("non-network:", [(k, tuple(sd[k].shape)) for k in sd if k not in net][:60])
di = ck["hyper_parameters"]["data_indices"]
for ds, ix in di.items():
    print("donor data_indices", ds, "input", len(ix.model.input), "output", len(ix.model.output), list(ix.name_to_index)[:5], "...", list(ix.name_to_index)[-3:])
print("callbacks", list(ck.get("callbacks", {}).keys()))
print("global_step", ck.get("global_step"))
print("t", time.time() - t0, flush=True)
