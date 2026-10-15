"""Linear interpolation matrices between two point sets on the sphere (any grids).

``linear_matrix(src_lat, src_lon, dst_lat, dst_lon)`` returns the sparse (n_dst, n_src) matrix
that interpolates a field given at the source points linearly to the target points: the
source points are triangulated on the sphere (the convex hull of their unit vectors is the
spherical Delaunay triangulation), each target point is located in its triangle, and its three
weights are the barycentric coordinates of the target's direction in the cone of that triangle
(normalised to sum to one). Every row sums to one; a target that coincides with a source point
gets weight one there.

Use: the texture evaluator defines the fine part of a residual as ``r - up @ (down @ r)``. Its
default matrices are the O320<->O1280 files on Atos; for o96->o320 a matching O96<->O320 pair is
needed, and this module builds one from the coordinates of a prediction file::

    python -m eval.shared.inter_mat --predictions-file <predictions_YYYYMMDD_stepNNN.nc> \\
        --out-dir <dir>          # writes up_lres_to_hres_linear.mat.npz and down_hres_to_lres_linear.mat.npz

The pair is linear, like the lane's own ``interpol_*_linear`` matrices, but it is not the same
operator (those come from the ECMWF interpolation library); use one pair for every arm.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

LOG = logging.getLogger(__name__)


def unit_vectors(lat_deg, lon_deg) -> np.ndarray:
    la = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    lo = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    c = np.cos(la)
    return np.column_stack([c * np.cos(lo), c * np.sin(lo), np.sin(la)])


def _incident(tris: np.ndarray, n_vertices: int) -> np.ndarray:
    """(n_vertices, max_degree) triangle indices touching each vertex, -1 padded."""
    flat = tris.ravel()
    order = np.argsort(flat, kind="stable")
    counts = np.bincount(flat, minlength=n_vertices)
    deg = int(counts.max())
    out = np.full((n_vertices, deg), -1, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    pos = np.arange(flat.size) - np.repeat(starts, counts)
    out[flat[order], pos] = order // 3
    return out


def _locate(P, tris, inc, Q, cand_vertex, eps=1e-10):
    """For each target, the first incident triangle of its candidate vertex holding it:
    returns (found mask, triangle index, normalised weights (n, 3))."""
    cand = inc[cand_vertex]                                   # (n, deg)
    n, deg = cand.shape
    found = np.zeros(n, dtype=bool)
    tri_of = np.full(n, -1, dtype=np.int64)
    w_of = np.zeros((n, 3))
    for j in range(deg):
        t = cand[:, j]
        todo = (~found) & (t >= 0)
        if not todo.any():
            continue
        idx = np.flatnonzero(todo)
        T = tris[t[idx]]                                      # (m, 3)
        M = np.stack([P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]], axis=2)   # columns A, B, C
        w = np.linalg.solve(M, Q[idx][:, :, None])[:, :, 0]
        ok = np.all(w >= -eps, axis=1) & (w.sum(axis=1) > 0)
        sel = idx[ok]
        found[sel] = True
        tri_of[sel] = t[idx[ok]]
        w_of[sel] = np.clip(w[ok], 0.0, None) / np.clip(w[ok], 0.0, None).sum(axis=1, keepdims=True)
    return found, tri_of, w_of


def linear_matrix(src_lat, src_lon, dst_lat, dst_lon, *, chunk: int = 200_000):
    """Sparse (n_dst, n_src) linear interpolation matrix (rows sum to one)."""
    import scipy.sparse as sps
    from scipy.spatial import ConvexHull, cKDTree

    P = unit_vectors(src_lat, src_lon)
    Q = unit_vectors(dst_lat, dst_lon)
    hull = ConvexHull(P)
    tris = hull.simplices.astype(np.int64)
    inc = _incident(tris, P.shape[0])
    tree = cKDTree(P)
    rows, cols, vals = [], [], []
    n_fallback = 0
    for a in range(0, Q.shape[0], chunk):
        q = Q[a:a + chunk]
        _, nn = tree.query(q, k=8, workers=-1)
        found = np.zeros(q.shape[0], dtype=bool)
        tri = np.full(q.shape[0], -1, dtype=np.int64)
        w = np.zeros((q.shape[0], 3))
        for k in range(nn.shape[1]):
            todo = np.flatnonzero(~found)
            if todo.size == 0:
                break
            f, t, ww = _locate(P, tris, inc, q[todo], nn[todo, k])
            sel = todo[f]
            found[sel], tri[sel], w[sel] = True, t[f], ww[f]
        r = np.arange(a, a + q.shape[0])
        ok = np.flatnonzero(found)
        rows.append(np.repeat(r[ok], 3))
        cols.append(tris[tri[ok]].ravel())
        vals.append(w[ok].ravel())
        miss = np.flatnonzero(~found)
        if miss.size:                     # never seen on octahedral grids; nearest point
            n_fallback += miss.size
            rows.append(r[miss])
            cols.append(nn[miss, 0])
            vals.append(np.ones(miss.size))
    m = sps.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(Q.shape[0], P.shape[0]))
    m.eliminate_zeros()
    if n_fallback:
        LOG.warning("inter_mat: %d target points fell back to the nearest source point", n_fallback)
    return m


def build_pair_from_predictions(pred_file, out_dir) -> dict[str, Path]:
    """up (lres -> hres) and down (hres -> lres) matrices from a prediction file's coordinates."""
    import netCDF4
    import scipy.sparse as sps

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(str(pred_file)) as d:
        lat_h, lon_h = (np.asarray(d[v][:], dtype=np.float64).reshape(-1) for v in ("lat_hres", "lon_hres"))
        lat_l, lon_l = (np.asarray(d[v][:], dtype=np.float64).reshape(-1) for v in ("lat_lres", "lon_lres"))
    paths = {"up_matrix": out_dir / "up_lres_to_hres_linear.mat.npz",
             "down_matrix": out_dir / "down_hres_to_lres_linear.mat.npz"}
    for key, (sl, so, dl, do) in (("up_matrix", (lat_l, lon_l, lat_h, lon_h)),
                                  ("down_matrix", (lat_h, lon_h, lat_l, lon_l))):
        if paths[key].exists():
            LOG.info("inter_mat: %s exists, kept", paths[key])
            continue
        m = linear_matrix(sl, so, dl, do)
        tmp = paths[key].with_name(paths[key].name + ".tmp.npz")
        sps.save_npz(tmp, m)
        tmp.replace(paths[key])
        LOG.info("inter_mat: wrote %s %s", paths[key], m.shape)
    return paths


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--predictions-file", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for k, v in build_pair_from_predictions(args.predictions_file, args.out_dir).items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
