"""Wind-extreme evaluator: is the strongest 10 m wind a real feature or grain?

Why this exists
---------------
The o320->o1280 diffusion downscaler produces stronger 10 m wind maxima than
its O320 driver. The open question is whether a maximum is carried by a
coherent meteorological structure -- a cyclone core, a jet, a squall line --
or by grid-scale grain, that is by a handful of isolated points that happen to
be large. A single number cannot answer that, but the way a maximum survives
spatial averaging can: average a real feature over a disk of 30 km and most of
its amplitude remains, because its neighbours are strong too; average grain over
the same disk and it collapses towards the local mean.

Everything is measured on the native O1280 grid, with no regridding, so the
model output, the truth and the driver receive identical treatment. The
comparison that carries the verdict is model against truth on the same case,
never an absolute threshold, and never pooled across cases.

Definitions
-----------
For one prediction file, one ensemble member and one geographical box, three
wind-speed fields are built from the 10 m wind components::

    W_model = hypot(y_pred[10u], y_pred[10v])
    W_truth = hypot(y     [10u], y     [10v])
    W_input = hypot(up @ x[10u], up @ x[10v])

``up`` is the linear O320 -> O1280 interpolation matrix the lane itself uses, so
``W_input`` is the driver seen on the output grid.

For each field and each averaging radius R (kilometres) the evaluator forms the
disk average ``S_R``, the mean of the field over every grid point within R km,
and reports:

    peak                the maximum of the field over the box
    peak_lat, peak_lon  where that maximum sits
    smoothed_peak[R]    the maximum of S_R over the box
    retention[R]        smoothed_peak[R] / peak
    local_retention[R]  S_R at the location of the unsmoothed peak, over peak
    n_above_90pct       how many grid points exceed 0.9 * peak
    n_above_95pct       how many exceed 0.95 * peak
    patch_points_90pct  size of the connected patch of points above 0.9 * peak
                        that contains the peak (adjacency: within adj_km)
    patch_area_90pct_km2   that patch in square kilometres
    peak_excess_over_30km  (peak - S_30 at the peak) / peak, the share of the
                        maximum that a 30 km disk average does not keep

``retention`` is the headline. A coherent feature keeps most of its amplitude
under averaging, grain does not, and the truth on the same case fixes what
"most" means for that weather situation.

Displacement of the maximum. The evaluator also reports the great-circle
distance between the model's peak location and the truth's, and between the
model's and the driver's. That is the feature-based half of the displacement
question; the field-based half lives in the ``displacement`` evaluator.

Boxes. The evaluator works inside geographical boxes because "the maximum wind"
is a question about a storm, not about the globe. Each box is padded by more
than the largest radius before the disk averages are built, so no point in the
box sees a truncated disk, and the statistics are taken over the box itself.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

LOG = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0088
SPHERE_AREA_KM2 = 4.0 * np.pi * EARTH_RADIUS_KM ** 2

DEFAULT_RADII_KM = [10.0, 20.0, 30.0, 50.0, 75.0, 100.0]
DEFAULT_ADJ_KM = 15.0
DEFAULT_COMPONENTS = ("10u", "10v")

# [lat_min, lat_max, lon_min, lon_max], longitudes in -180..180.
DEFAULT_BOXES: dict[str, list[float]] = {
    "west_tropical_atlantic": [10.0, 32.0, -85.0, -55.0],
    "open_north_atlantic": [35.0, 60.0, -45.0, -10.0],
    "europe": [35.0, 62.0, -12.0, 25.0],
}

SOURCES = ("model", "truth", "input")

_FILE_RE = re.compile(r"predictions_(\d{8})_step(\d+)\.nc$")


def _default_paths() -> dict[str, str]:
    inter = os.environ.get("INTER_MAT_DIR", "/home/ecm5702/hpcperm/data/inter_mat")
    return {"up_matrix": os.path.join(inter, "interpol_O320_to_O1280_linear.mat.npz")}


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _unit_vectors(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    c = np.cos(lat)
    return np.column_stack([c * np.cos(lon), c * np.sin(lon), np.sin(lat)])


def _chord(radius_km: float) -> float:
    """Chord length on the unit sphere subtending a great-circle arc of radius_km."""
    return 2.0 * np.sin(0.5 * radius_km / EARTH_RADIUS_KM)


def _great_circle_km(lat1, lon1, lat2, lon2) -> float:
    a = _unit_vectors(np.atleast_1d(lat1), np.atleast_1d(lon1))[0]
    b = _unit_vectors(np.atleast_1d(lat2), np.atleast_1d(lon2))[0]
    chord = float(np.linalg.norm(a - b))
    return float(EARTH_RADIUS_KM * 2.0 * np.arcsin(np.clip(0.5 * chord, 0.0, 1.0)))


def _box_mask(lat: np.ndarray, lon180: np.ndarray, box, pad_deg: float = 0.0) -> np.ndarray:
    lat_min, lat_max, lon_min, lon_max = [float(v) for v in box]
    lat_ok = (lat >= lat_min - pad_deg) & (lat <= lat_max + pad_deg)
    # Longitude padding grows with latitude, so a padded disk is never clipped
    # in the zonal direction near the poles.
    scale = float(np.cos(np.deg2rad(min(abs(lat_min), abs(lat_max), 80.0))))
    lon_pad = pad_deg / max(scale, 0.2)
    lo, hi = lon_min - lon_pad, lon_max + lon_pad
    if lo < -180.0 or hi > 180.0:      # box straddles the date line after padding
        lon360 = np.mod(lon180, 360.0)
        lo360, hi360 = np.mod(lo, 360.0), np.mod(hi, 360.0)
        lon_ok = (lon360 >= lo360) | (lon360 <= hi360) if lo360 > hi360 else (
            (lon360 >= lo360) & (lon360 <= hi360))
    else:
        lon_ok = (lon180 >= lo) & (lon180 <= hi)
    return lat_ok & lon_ok


def _build_box(lat: np.ndarray, lon180: np.ndarray, box, radii_km, adj_km: float) -> dict[str, Any]:
    """Disk-average matrices and the adjacency graph for one box.

    The padded set supplies the neighbours; the core set is where statistics are
    taken. Both index into the global grid.
    """
    from scipy.spatial import cKDTree
    import scipy.sparse as sps

    pad_deg = float(max(radii_km)) / 111.0 + 0.5
    padded_idx = np.flatnonzero(_box_mask(lat, lon180, box, pad_deg))
    core_idx = np.flatnonzero(_box_mask(lat, lon180, box, 0.0))
    if core_idx.size == 0:
        raise ValueError(f"box {box} contains no grid points")
    # Position of every core point inside the padded set.
    order = np.argsort(padded_idx)
    core_in_padded = order[np.searchsorted(padded_idx, core_idx, sorter=order)]

    xyz_padded = _unit_vectors(lat[padded_idx], lon180[padded_idx])
    tree = cKDTree(xyz_padded)
    xyz_core = xyz_padded[core_in_padded]

    smoothers: dict[float, Any] = {}
    for r in radii_km:
        neighbours = tree.query_ball_point(xyz_core, _chord(float(r)), workers=-1)
        indptr = np.zeros(core_idx.size + 1, dtype=np.int64)
        indptr[1:] = np.cumsum([len(n) for n in neighbours])
        indices = np.fromiter((j for n in neighbours for j in n), dtype=np.int32,
                              count=int(indptr[-1]))
        counts = np.diff(indptr)
        data = np.repeat(1.0 / counts, counts)
        smoothers[float(r)] = sps.csr_matrix(
            (data, indices, indptr), shape=(core_idx.size, padded_idx.size))
        LOG.info("wind_extremes: box radius %.0f km -> %.1f points per disk (median)",
                 r, float(np.median(counts)))

    adj = tree.query_ball_point(xyz_core, _chord(float(adj_km)), workers=-1)
    # Adjacency between core points only, expressed in core indexing.
    padded_to_core = np.full(padded_idx.size, -1, dtype=np.int64)
    padded_to_core[core_in_padded] = np.arange(core_idx.size)
    adj_core = [padded_to_core[np.asarray(n, dtype=np.int64)] for n in adj]
    adj_core = [n[n >= 0] for n in adj_core]

    return {
        "core_idx": core_idx,
        "padded_idx": padded_idx,
        "core_in_padded": core_in_padded,
        "smoothers": smoothers,
        "adjacency": adj_core,
        "lat": lat[core_idx],
        "lon": lon180[core_idx],
        "n_core": int(core_idx.size),
        "n_padded": int(padded_idx.size),
    }


def _connected_patch(values: np.ndarray, adjacency, seed: int, threshold: float) -> int:
    """Size of the connected set of points above threshold that contains seed."""
    if values[seed] < threshold:
        return 0
    above = values >= threshold
    seen = np.zeros(values.size, dtype=bool)
    stack = [int(seed)]
    seen[seed] = True
    count = 0
    while stack:
        i = stack.pop()
        count += 1
        for j in adjacency[i]:
            j = int(j)
            if above[j] and not seen[j]:
                seen[j] = True
                stack.append(j)
    return count


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _field_stats(w_core: np.ndarray, w_padded: np.ndarray, box_static: dict,
                 radii_km, point_area_km2: float) -> dict[str, Any]:
    peak_i = int(np.argmax(w_core))
    peak = float(w_core[peak_i])
    out: dict[str, Any] = {
        "peak": peak,
        "peak_lat": float(box_static["lat"][peak_i]),
        "peak_lon": float(box_static["lon"][peak_i]),
        "n_above_90pct": int((w_core >= 0.90 * peak).sum()),
        "n_above_95pct": int((w_core >= 0.95 * peak).sum()),
    }
    patch = _connected_patch(w_core, box_static["adjacency"], peak_i, 0.90 * peak)
    out["patch_points_90pct"] = int(patch)
    out["patch_area_90pct_km2"] = float(patch * point_area_km2)

    smoothed_peak: dict[str, float] = {}
    retention: dict[str, float] = {}
    local_retention: dict[str, float] = {}
    for r in radii_km:
        s = box_static["smoothers"][float(r)] @ w_padded
        key = f"{float(r):g}"
        smoothed_peak[key] = float(s.max())
        retention[key] = float(s.max() / peak) if peak > 0 else float("nan")
        local_retention[key] = float(s[peak_i] / peak) if peak > 0 else float("nan")
    out["smoothed_peak"] = smoothed_peak
    out["retention"] = retention
    out["local_retention"] = local_retention
    ref = f"{float(min(radii_km, key=lambda r: abs(float(r) - 30.0))):g}"
    out["peak_excess_over_30km"] = float(1.0 - local_retention[ref]) if peak > 0 else float("nan")
    out["reference_radius_km"] = float(ref)
    return out


def _mean_sd(values) -> dict[str, Any]:
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if a.size == 0:
        return {"mean": None, "sd": None, "n": 0}
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "n": int(a.size)}


def _aggregate(samples: list[dict], boxes: list[str], radii_km) -> list[dict]:
    rows = []
    radius_keys = [f"{float(r):g}" for r in radii_km]
    for box in boxes:
        sel = [s for s in samples if s["box"] == box]
        if not sel:
            continue
        row: dict[str, Any] = {"box": box, "n_samples": len(sel)}
        for source in SOURCES:
            entry: dict[str, Any] = {}
            for stat in ("peak", "n_above_90pct", "n_above_95pct", "patch_points_90pct",
                         "patch_area_90pct_km2", "peak_excess_over_30km"):
                entry[stat] = _mean_sd([s[source][stat] for s in sel])
            for name in ("retention", "local_retention", "smoothed_peak"):
                entry[name] = {k: _mean_sd([s[source][name][k] for s in sel])
                               for k in radius_keys}
            row[source] = entry
        row["model_minus_truth"] = {
            "retention": {k: _mean_sd([s["model"]["retention"][k] - s["truth"]["retention"][k]
                                       for s in sel]) for k in radius_keys},
            "peak": _mean_sd([s["model"]["peak"] - s["truth"]["peak"] for s in sel]),
            "patch_points_90pct": _mean_sd(
                [s["model"]["patch_points_90pct"] - s["truth"]["patch_points_90pct"] for s in sel]),
        }
        row["peak_displacement_km"] = {
            "model_vs_truth": _mean_sd([s["peak_displacement_km"]["model_vs_truth"] for s in sel]),
            "model_vs_input": _mean_sd([s["peak_displacement_km"]["model_vs_input"] for s in sel]),
            "truth_vs_input": _mean_sd([s["peak_displacement_km"]["truth_vs_input"] for s in sel]),
        }
        rows.append(row)
    return rows


def _as_list(value, cast=str) -> list | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [cast(v) for v in value]
    return [cast(v) for v in str(value).split(",") if v != ""]


def _read_component(var, member_index: int, state_index: int) -> np.ndarray:
    """One weather state of one member, as a 1-D array over grid points.

    Reading a single column rather than the whole member keeps the working set
    to one field instead of ten, which matters at 6.6 million points.
    """
    return np.asarray(var[0, member_index, :, state_index], dtype=np.float64)


def _select_members(ds, members, max_members):
    labels = [int(v) for v in np.asarray(ds.variables["ensemble_member"][:]).reshape(-1)]
    pairs = list(enumerate(labels))
    if members:
        wanted = set(int(m) for m in members)
        pairs = [(i, lab) for i, lab in pairs if lab in wanted]
    if max_members:
        pairs = pairs[: int(max_members)]
    return [(lab, i) for i, lab in pairs]


def _write_summary(path: Path, payload: dict) -> None:
    radius_keys = [f"{float(r):g}" for r in payload["config"]["radii_km"]]
    lines = [
        f"# Wind extremes — {payload['run_label']}",
        "",
        f"Files: {payload['n_files']}. Samples per box: "
        f"{max((r['n_samples'] for r in payload['aggregate']), default=0)} "
        "(one per file and member).",
        "",
        "`retention[R]` is the maximum of the R-km disk average divided by the raw maximum.",
        "A coherent wind maximum keeps most of its amplitude under averaging; grain does not.",
        "Read the model column against the truth column of the same box.",
        "",
    ]
    for row in payload["aggregate"]:
        lines.append(f"## {row['box']} ({row['n_samples']} samples)")
        lines.append("")
        header = "| quantity | " + " | ".join(SOURCES) + " |"
        lines.append(header)
        lines.append("|---|" + "---|" * len(SOURCES))

        def _fmt(v, nd=3):
            return "n/a" if v is None else f"{v:.{nd}f}"

        lines.append("| peak wind (m/s) | " + " | ".join(
            _fmt(row[s]["peak"]["mean"], 2) for s in SOURCES) + " |")
        for k in radius_keys:
            lines.append(f"| retention at {k} km | " + " | ".join(
                _fmt(row[s]["retention"][k]["mean"]) for s in SOURCES) + " |")
        for k in radius_keys:
            lines.append(f"| the peak's own {k} km disk average / peak | " + " | ".join(
                _fmt(row[s]["local_retention"][k]["mean"]) for s in SOURCES) + " |")
        lines.append("| points above 90% of peak | " + " | ".join(
            _fmt(row[s]["n_above_90pct"]["mean"], 1) for s in SOURCES) + " |")
        lines.append("| connected patch above 90% (points) | " + " | ".join(
            _fmt(row[s]["patch_points_90pct"]["mean"], 1) for s in SOURCES) + " |")
        lines.append("")
        d = row["peak_displacement_km"]
        lines.append(
            "Peak displacement in km: model against truth "
            f"{_fmt(d['model_vs_truth']['mean'], 1)}, model against driver "
            f"{_fmt(d['model_vs_input']['mean'], 1)}, truth against driver "
            f"{_fmt(d['truth_vs_input']['mean'], 1)}."
        )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    return obj


def run(
    predictions_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    run_label: str = "",
    **kwargs,
) -> Path:
    import netCDF4
    import scipy.sparse as sps

    from eval.evaluators.spectra.proxy_runner import valid_prediction_files

    predictions_dir = Path(predictions_dir).expanduser().resolve()
    output_dir = Path(output_dir) if output_dir else predictions_dir / "evaluators" / "wind_extremes"
    output_dir.mkdir(parents=True, exist_ok=True)

    radii_km = [float(r) for r in (eval_config.get("radii_km") or DEFAULT_RADII_KM)]
    adj_km = float(eval_config.get("adjacency_km", DEFAULT_ADJ_KM))
    boxes = eval_config.get("boxes") or dict(DEFAULT_BOXES)
    for name, box in boxes.items():
        if len(box) != 4:
            raise ValueError(
                f"wind_extremes.boxes[{name!r}] must be [lat_min, lat_max, lon_min, lon_max]")
    components = tuple(eval_config.get("components") or DEFAULT_COMPONENTS)
    steps = _as_list(eval_config.get("steps"), int)
    dates = _as_list(eval_config.get("dates"), str)
    members = _as_list(eval_config.get("members"), int)
    max_members = eval_config.get("max_members")
    paths = {**_default_paths(), **(eval_config.get("paths") or {})}

    files = valid_prediction_files(predictions_dir, steps=steps)
    if dates:
        wanted = set(dates)
        files = [f for f in files
                 if (_FILE_RE.search(f.name) and _FILE_RE.search(f.name).group(1) in wanted)]
    if not files:
        raise RuntimeError(
            f"No prediction files under {predictions_dir} (steps={steps}, dates={dates})")

    LOG.info("wind_extremes: %d file(s), boxes=%s radii=%s members=%s",
             len(files), list(boxes), radii_km, members)

    t0 = time.time()
    up = sps.load_npz(paths["up_matrix"]).tocsr()

    box_static: dict[str, dict] = {}
    up_box: dict[str, Any] = {}
    point_area_km2 = None
    samples: list[dict[str, Any]] = []

    for file_path in files:
        m = _FILE_RE.search(file_path.name)
        date, step = (m.group(1), int(m.group(2))) if m else ("", -1)
        with netCDF4.Dataset(file_path) as ds:
            ds.set_auto_mask(False)
            for required in ("x", "y", "y_pred", "lat_hres", "lon_hres"):
                if required not in ds.variables:
                    raise RuntimeError(f"{file_path} is missing {required!r}")
            ws = [str(v) for v in np.asarray(ds.variables["weather_state"][:]).reshape(-1)]
            idx_of = {s: i for i, s in enumerate(ws)}
            missing = [c for c in components if c not in idx_of]
            if missing:
                raise RuntimeError(f"{file_path.name} lacks wind components {missing}")
            lat = np.asarray(ds.variables["lat_hres"][:]).reshape(-1).astype(np.float64)
            lon = np.asarray(ds.variables["lon_hres"][:]).reshape(-1).astype(np.float64)
            lon180 = np.where(np.mod(lon, 360.0) > 180.0, np.mod(lon, 360.0) - 360.0,
                              np.mod(lon, 360.0))
            if point_area_km2 is None:
                point_area_km2 = float(SPHERE_AREA_KM2 / lat.size)
            if not box_static:
                for name, box in boxes.items():
                    t_box = time.time()
                    box_static[name] = _build_box(lat, lon180, box, radii_km, adj_km)
                    up_box[name] = up[box_static[name]["padded_idx"]].tocsr()
                    LOG.info("wind_extremes: box %s built in %.1fs (%d core, %d padded points)",
                             name, time.time() - t_box, box_static[name]["n_core"],
                             box_static[name]["n_padded"])
            if up.shape[0] != lat.size:
                raise RuntimeError(
                    f"grid mismatch: file has {lat.size} points, up matrix {up.shape}")
            lead = ds.getncattr("lead_step_hours") if "lead_step_hours" in ds.ncattrs() else step

            for member_label, member_index in _select_members(ds, members, max_members):
                t_mem = time.time()
                cu, cv = idx_of[components[0]], idx_of[components[1]]
                yp_u = _read_component(ds.variables["y_pred"], member_index, cu)
                yp_v = _read_component(ds.variables["y_pred"], member_index, cv)
                yt_u = _read_component(ds.variables["y"], member_index, cu)
                yt_v = _read_component(ds.variables["y"], member_index, cv)
                xx_u = _read_component(ds.variables["x"], member_index, cu)
                xx_v = _read_component(ds.variables["x"], member_index, cv)

                for name, st in box_static.items():
                    padded = st["padded_idx"]
                    core_in_padded = st["core_in_padded"]
                    fields_padded = {
                        "model": np.hypot(yp_u[padded], yp_v[padded]),
                        "truth": np.hypot(yt_u[padded], yt_v[padded]),
                        "input": np.hypot(up_box[name] @ xx_u, up_box[name] @ xx_v),
                    }
                    entry: dict[str, Any] = {
                        "file": file_path.name, "date": date, "step": int(lead),
                        "member": int(member_label), "box": name,
                    }
                    for source, w_padded in fields_padded.items():
                        w_core = w_padded[core_in_padded]
                        if not np.isfinite(w_core).all():
                            raise RuntimeError(
                                f"{file_path.name} member {member_label} box {name} source "
                                f"{source}: non-finite wind speeds")
                        entry[source] = _field_stats(w_core, w_padded, st, radii_km,
                                                     point_area_km2)
                    entry["peak_displacement_km"] = {
                        "model_vs_truth": _great_circle_km(
                            entry["model"]["peak_lat"], entry["model"]["peak_lon"],
                            entry["truth"]["peak_lat"], entry["truth"]["peak_lon"]),
                        "model_vs_input": _great_circle_km(
                            entry["model"]["peak_lat"], entry["model"]["peak_lon"],
                            entry["input"]["peak_lat"], entry["input"]["peak_lon"]),
                        "truth_vs_input": _great_circle_km(
                            entry["truth"]["peak_lat"], entry["truth"]["peak_lon"],
                            entry["input"]["peak_lat"], entry["input"]["peak_lon"]),
                    }
                    samples.append(entry)
                LOG.info("wind_extremes: %s member %s done in %.1fs",
                         file_path.name, member_label, time.time() - t_mem)

    aggregate = _aggregate(samples, list(boxes), radii_km)
    payload = {
        "run_label": run_label or predictions_dir.name,
        "predictions_dir": str(predictions_dir),
        "n_files": len(files),
        "files": [f.name for f in files],
        "config": {
            "radii_km": radii_km, "adjacency_km": adj_km, "boxes": boxes,
            "components": list(components), "steps": steps, "dates": dates,
            "members": members, "max_members": max_members, "paths": paths,
            "point_area_km2": point_area_km2,
        },
        "boxes": {name: {"box": [float(v) for v in boxes[name]],
                         "n_core_points": st["n_core"],
                         "n_padded_points": st["n_padded"]}
                  for name, st in box_static.items()},
        "sources": list(SOURCES),
        "aggregate": aggregate,
        "samples": samples,
        "elapsed_s": time.time() - t0,
    }
    payload = _clean(payload)
    (output_dir / "wind_extremes.json").write_text(json.dumps(payload, indent=1) + "\n")
    _write_summary(output_dir / "wind_extremes_summary.md", payload)
    LOG.info("wind_extremes: wrote %s (%d samples) in %.1fs",
             output_dir / "wind_extremes.json", len(samples), time.time() - t0)
    return output_dir
