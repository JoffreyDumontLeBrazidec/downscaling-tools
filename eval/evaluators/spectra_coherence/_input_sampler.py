"""Linear sampling of the native input on the sphere, for the input curves of spectra_coherence.

Rule 41 (owner, 9 Nov 2026): the input is scored on its own grid, never interpolated to
the output grid first. The analysis map here is HEALPix; binning the coarse native points
into its pixels, as the truth and the model are binned, would leave most pixels empty (an
O320 field on nside 512 fills about one pixel in seven), and the shared hole pattern would
manufacture coherence (see runner.py on nside 742). So the native input is SAMPLED at the
pixel centres, with the operator box_spectra uses for its native input: linear
(barycentric) interpolation over the triangulation of the native points. On the sphere the
triangulation is the convex hull of the unit vectors (the spherical Delaunay
triangulation); each target takes the three vertices of the facet its direction crosses,
with gnomonic barycentric weights. The weights are built once per grid.
"""
from __future__ import annotations

import logging

import numpy as np

LOG = logging.getLogger(__name__)


def _xyz(lat, lon) -> np.ndarray:
    la = np.radians(np.asarray(lat, dtype=np.float64).reshape(-1))
    lo = np.radians(np.asarray(lon, dtype=np.float64).reshape(-1))
    c = np.cos(la)
    return np.column_stack((c * np.cos(lo), c * np.sin(lo), np.sin(la)))


class SphereLinearSampler:
    """Barycentric sampling of scattered points (src) at target points (dst) on the unit sphere."""

    def __init__(self, src_lat, src_lon, dst_lat, dst_lon, *, k: int = 16, chunk: int = 50_000):
        from scipy.spatial import ConvexHull, cKDTree

        P = _xyz(src_lat, src_lon)
        Q = _xyz(dst_lat, dst_lon)
        hull = ConvexHull(P)
        # Facets whose outward normal points away from the centre are the spherical triangles.
        # A regional point set (a box) also has flat facets closing the cap; they face the
        # centre (offset > 0) and are dropped, so targets outside the box find no facet.
        keep = hull.equations[:, 3] < 0.0
        tri = hull.simplices[keep]
        A = P[tri]                                            # (F, 3 vertices, xyz)
        Minv = np.linalg.inv(np.transpose(A, (0, 2, 1)))      # q = M w  ->  w = M^-1 q
        cen = A.mean(axis=1)
        cen /= np.linalg.norm(cen, axis=1, keepdims=True)
        tree = cKDTree(cen)
        n = Q.shape[0]
        self.verts = np.empty((n, 3), dtype=np.int64)
        self.w = np.empty((n, 3), dtype=np.float64)
        found = np.zeros(n, dtype=bool)
        kk = min(k, len(tri))
        for s in range(0, n, chunk):
            q = Q[s:s + chunk]
            _, cand = tree.query(q, k=kk, workers=-1)
            cand = cand.reshape(q.shape[0], kk)
            W = np.einsum("nkij,nj->nki", Minv[cand], q)      # (n, k, 3)
            ok = (W >= -1e-9).all(axis=-1)
            has = ok.any(axis=1)
            first = ok.argmax(axis=1)
            rows = np.arange(q.shape[0])
            wsel = W[rows, first]
            wsel /= wsel.sum(axis=1, keepdims=True)
            self.w[s:s + chunk] = wsel
            self.verts[s:s + chunk] = tri[cand[rows, first]]
            found[s:s + chunk] = has
        self.n_nearest = int((~found).sum())
        if self.n_nearest:
            # outside the triangulation (beyond a regional input) or a facet-search miss: nearest point
            _, nn = cKDTree(P).query(Q[~found], k=1, workers=-1)
            self.verts[~found] = nn[:, None]
            self.w[~found] = np.array([1.0, 0.0, 0.0])
            LOG.info("native input sampler: %d of %d targets outside the triangulation take the nearest point",
                     self.n_nearest, n)

    def __call__(self, values: np.ndarray) -> np.ndarray:
        v = np.asarray(values, dtype=np.float64).reshape(-1)
        return np.sum(v[self.verts] * self.w, axis=1)
