"""The box-edge strip of a regional prediction, and the scan for the slot-head edge artefact.

A slot-head model on a regional box (the lossless class of anemoi-core with ``write: slot_head``) writes the
points of the half cells at the box edge from tokens it never sees inside the box. On XCR100 (exact_only read)
this gives 800-840 mm of tp in the first 0.25 degrees on 30/30 dates, 2t error 8x the inner box (docs ablation
note s21, 6 Nov 2026). Every score of a regional box therefore leaves out ``DEFAULT_BOX_EDGE_DEG`` (1 degree)
at the box edge; a global point set has no edge and keeps every point.

``scan_edge_artefact`` reads a few prediction files and says whether the artefact is present, so that
``eval.cli`` refuses the evaluators that do not leave the strip out.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_BOX_EDGE_DEG = 1.0
# scan: a point within SCAN_EDGE_DEG of the edge whose tp (mm, as stored x 1000) exceeds HOT_MM in some member
SCAN_EDGE_DEG = 0.25
HOT_MM = 300.0
# artefact = such a point on at least this share of the scanned files, while the inner box stays below HOT_MM there
PERSIST_SHARE = 0.5


def wrap_lon(lon) -> np.ndarray:
    return ((np.asarray(lon, dtype=np.float64) + 180.0) % 360.0) - 180.0


def is_global(lat, lon) -> bool:
    """True when the point set covers the sphere (no box edges)."""
    return bool(np.ptp(np.asarray(lat, dtype=np.float64)) > 170.0 and np.ptp(wrap_lon(lon)) > 350.0)


def edge_distance_deg(lat, lon) -> np.ndarray:
    """Distance in degrees (latitude or longitude, the smaller) from each point to the edge of its box.

    The longitude span is measured on the -180..180 wrap unless the box crosses the dateline, when the
    0..360 wrap is tighter. A global set gets +inf everywhere."""
    lat = np.asarray(lat, dtype=np.float64)
    if is_global(lat, lon):
        return np.full(lat.shape, np.inf)
    lon180 = wrap_lon(lon)
    lon360 = lon180 % 360.0
    lo = lon180 if np.ptp(lon180) <= np.ptp(lon360) else lon360
    return np.minimum.reduce([lat - lat.min(), lat.max() - lat, lo - lo.min(), lo.max() - lo])


def box_edge_mask(lat, lon, edge_deg: float = DEFAULT_BOX_EDGE_DEG) -> np.ndarray:
    """Points at least ``edge_deg`` degrees inside a regional box (True = keep). Global sets keep every point."""
    lat = np.asarray(lat, dtype=np.float64)
    if edge_deg <= 0.0:
        return np.ones(lat.shape, dtype=bool)
    return edge_distance_deg(lat, lon) >= float(edge_deg)


def _tp_members(ds) -> np.ndarray | None:
    names = [str(s) for s in ds["weather_state"][:]] if "weather_state" in ds.variables else []
    if "tp" not in names or "y_pred" not in ds.variables:
        return None
    y = np.asarray(ds["y_pred"][:], dtype=np.float64)
    y = y.reshape(-1, y.shape[-2], y.shape[-1])  # (members, points, vars)
    return np.clip(y[:, :, names.index("tp")], 0.0, None) * 1000.0


def scan_edge_artefact(predictions_dir: Path, *, max_files: int = 6) -> dict:
    """Look for the slot-head edge artefact in up to ``max_files`` prediction files of a directory.

    Returns ``{"checked": n_files, "regional": bool, "artefact": bool, "hot_share": float, "edge_max_mm": [...],
    "inner_max_mm": [...], "detail": str}``. Files without tp or with a global grid are not counted."""
    import netCDF4

    files = sorted(Path(predictions_dir).glob("predictions_*.nc"))
    if len(files) > max_files:
        files = [files[i] for i in np.linspace(0, len(files) - 1, max_files).round().astype(int)]
    out = {"checked": 0, "regional": False, "artefact": False, "hot_share": 0.0,
           "edge_max_mm": [], "inner_max_mm": [], "detail": ""}
    hot = 0
    for f in files:
        try:
            with netCDF4.Dataset(f) as ds:
                ds.set_auto_mask(False)
                if "lat_hres" not in ds.variables:
                    continue
                lat = np.asarray(ds["lat_hres"][:], dtype=np.float64)
                lon = np.asarray(ds["lon_hres"][:], dtype=np.float64)
                tp = _tp_members(ds)
        except (OSError, KeyError, IndexError):
            continue
        if tp is None or is_global(lat, lon):
            continue
        out["regional"] = True
        de = edge_distance_deg(lat, lon)
        e_max = float(tp[:, de < SCAN_EDGE_DEG].max()) if (de < SCAN_EDGE_DEG).any() else 0.0
        i_max = float(tp[:, de >= DEFAULT_BOX_EDGE_DEG].max()) if (de >= DEFAULT_BOX_EDGE_DEG).any() else 0.0
        out["edge_max_mm"].append(round(e_max, 1))
        out["inner_max_mm"].append(round(i_max, 1))
        out["checked"] += 1
        hot += int(e_max > HOT_MM)
    if out["checked"]:
        out["hot_share"] = hot / out["checked"]
        out["artefact"] = out["hot_share"] >= PERSIST_SHARE
        out["detail"] = (f"tp above {HOT_MM:g} mm within {SCAN_EDGE_DEG:g} deg of the box edge on {hot}/{out['checked']} "
                         f"scanned files (edge max {out['edge_max_mm']} mm, inner max {out['inner_max_mm']} mm)")
    return out
