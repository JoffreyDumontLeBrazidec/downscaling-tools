"""Displacement evaluator: does the model move features away from its driver?

Why this exists
---------------
The o320->o1280 downscaler is conditioned on a coarse driver. Whatever detail it
adds, the large-scale features it inherits -- the centre of a depression, the
axis of a wind maximum -- should stay where the driver put them. If they move,
the added detail is placed against the wrong background, and every local score
is then measured at the wrong point. This evaluator measures that movement in
kilometres, in two independent ways.

The shift that best aligns the fields. Inside a geographical box, the model
output and the driver interpolated onto the output grid are both sampled onto a
regular longitude-latitude mesh, smoothed to keep only scales the driver can
carry, and then compared under every whole-cell shift within a search window.
The shift with the highest correlation is reported in kilometres, split into an
eastward and a northward component, and refined to a fraction of a cell by
fitting a parabola through the correlation peak and its two neighbours. The sign
convention is that a positive eastward number means the second field's feature
sits that far east of the first field's; for the model against the driver, a
positive number therefore places the driver's feature east of the model's. A
displacement is real when the median shift over members and cases is away from
zero by more than its own scatter; a cloud of shifts scattered around zero means
the model leaves features where it found them.

The position of the pressure minimum. Inside the same box, the minimum of mean
sea level pressure is located in the model, in the driver and in the truth, and
the great-circle distances between those positions are reported. In a box that
contains one dominant depression this is the position of that depression, which
is the interpretable version of the same question.

One caution belongs with every number that involves the truth. The coarse driver
comes from the extended-range ensemble and the truth from the medium-range
ensemble, and the two are not paired: the truth is a genuine atmosphere, but not
the same realisation the driver describes. Model against driver is therefore the
measurement that carries the verdict; model against truth is context, and its
scatter is the weather's, not the model's.
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
KM_PER_DEG_LAT = np.pi * EARTH_RADIUS_KM / 180.0

DEFAULT_GRID_RES_DEG = 0.25
DEFAULT_MAX_SHIFT_DEG = 2.0
DEFAULT_SMOOTH_DEG = 0.5
DEFAULT_FIELDS = ("msl", "wind10m")

# [lat_min, lat_max, lon_min, lon_max], longitudes in -180..180.
DEFAULT_BOXES: dict[str, list[float]] = {
    "west_tropical_atlantic": [10.0, 32.0, -85.0, -55.0],
    "open_north_atlantic": [35.0, 60.0, -45.0, -10.0],
}

# Which weather states each renderable field needs, and how they combine.
FIELD_STATES = {
    "msl": (("msl",), "single"),
    "wind10m": (("10u", "10v"), "hypot"),
    "2t": (("2t",), "single"),
    "t_850": (("t_850",), "single"),
    "z_500": (("z_500",), "single"),
}

PAIRS = ("model_vs_input", "model_vs_truth", "truth_vs_input")
SOURCE_OF = {"model": "model", "input": "input", "truth": "truth"}

_FILE_RE = re.compile(r"predictions_(\d{8})_step(\d+)\.nc$")


def _default_paths() -> dict[str, str]:
    inter = os.environ.get("INTER_MAT_DIR", "/home/ecm5702/hpcperm/data/inter_mat")
    return {"up_matrix": os.path.join(inter, "interpol_O320_to_O1280_linear.mat.npz")}


def _as_list(value, cast=str) -> list | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [cast(v) for v in value]
    return [cast(v) for v in str(value).split(",") if v != ""]


# ---------------------------------------------------------------------------
# Box geometry and resampling
# ---------------------------------------------------------------------------

def _build_box(lat: np.ndarray, lon180: np.ndarray, box, res_deg: float,
               max_shift_deg: float) -> dict[str, Any]:
    """A regular mesh over the box plus the nearest native point for each cell.

    The mesh is padded by the search window so that a shifted comparison never
    runs off the sampled area.
    """
    from scipy.spatial import cKDTree

    lat_min, lat_max, lon_min, lon_max = [float(v) for v in box]
    pad = float(max_shift_deg) + 2.0 * float(res_deg)
    gy = np.arange(lat_min - pad, lat_max + pad + 0.5 * res_deg, res_deg)
    gx = np.arange(lon_min - pad, lon_max + pad + 0.5 * res_deg, res_deg)
    mesh_lon, mesh_lat = np.meshgrid(gx, gy)

    margin = pad + 1.0
    sel = ((lat >= lat_min - margin) & (lat <= lat_max + margin)
           & (lon180 >= lon_min - margin) & (lon180 <= lon_max + margin))
    native_idx = np.flatnonzero(sel)
    if native_idx.size == 0:
        raise ValueError(f"box {box} contains no grid points")
    # A latitude-weighted lookup keeps the nearest-neighbour search roughly
    # isotropic in grid spacing on a reduced Gaussian grid.
    scale = 1.4
    tree = cKDTree(np.column_stack([lon180[native_idx], lat[native_idx] * scale]))
    _, nearest = tree.query(
        np.column_stack([mesh_lon.ravel(), mesh_lat.ravel() * scale]), workers=-1)

    core = ((mesh_lat >= lat_min) & (mesh_lat <= lat_max)
            & (mesh_lon >= lon_min) & (mesh_lon <= lon_max))
    return {
        "gx": gx, "gy": gy, "shape": mesh_lat.shape,
        "mesh_lat": mesh_lat, "mesh_lon": mesh_lon,
        "native_idx": native_idx, "nearest": nearest.reshape(mesh_lat.shape),
        "core": core,
        "centre_lat": 0.5 * (lat_min + lat_max),
        "n_native": int(native_idx.size),
        "n_cells": int(core.sum()),
    }


def _resample(values_native: np.ndarray, box_static: dict) -> np.ndarray:
    """Nearest-neighbour sample of a native-grid field onto the box mesh."""
    return values_native[box_static["nearest"]]


def _smooth(grid: np.ndarray, sigma_cells: float) -> np.ndarray:
    if sigma_cells <= 0:
        return grid
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(grid, sigma_cells, mode="nearest")


# ---------------------------------------------------------------------------
# The shift that best aligns two fields
# ---------------------------------------------------------------------------

def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else float("nan")


def _parabolic_offset(left: float, centre: float, right: float) -> float:
    """Sub-cell position of a peak sampled at -1, 0, +1, clipped to that interval."""
    den = left - 2.0 * centre + right
    if den == 0.0:
        return 0.0
    return float(np.clip(0.5 * (left - right) / den, -1.0, 1.0))


def _best_shift(a_core: np.ndarray, b_full: np.ndarray, core_slice, max_cells: int
                ) -> dict[str, Any]:
    """Shift of b that best matches a, searched over whole cells then refined.

    ``a_core`` is the reference over the core window; ``b_full`` is the other
    field over the padded mesh, from which the same window is cut at each trial
    offset. The best offset is where b sampled that many cells further east and
    north reproduces a, which means the feature b carries sits that far east and
    north of the one a carries.
    """
    r0, r1, c0, c1 = core_slice
    scores = np.full((2 * max_cells + 1, 2 * max_cells + 1), np.nan)
    for di in range(-max_cells, max_cells + 1):
        for dj in range(-max_cells, max_cells + 1):
            window = b_full[r0 + di:r1 + di, c0 + dj:c1 + dj]
            scores[di + max_cells, dj + max_cells] = _correlation(a_core, window)
    flat = int(np.nanargmax(scores))
    bi, bj = np.unravel_index(flat, scores.shape)
    best = float(scores[bi, bj])
    di, dj = bi - max_cells, bj - max_cells
    # Sub-cell refinement, only where the peak has neighbours on both sides.
    ref_i = ref_j = 0.0
    if 0 < bi < scores.shape[0] - 1:
        ref_i = _parabolic_offset(scores[bi - 1, bj], best, scores[bi + 1, bj])
    if 0 < bj < scores.shape[1] - 1:
        ref_j = _parabolic_offset(scores[bi, bj - 1], best, scores[bi, bj + 1])
    return {
        "shift_rows": float(di + ref_i),
        "shift_cols": float(dj + ref_j),
        "corr_best": best,
        "corr_zero": float(scores[max_cells, max_cells]),
        "at_search_edge": bool(abs(di) == max_cells or abs(dj) == max_cells),
    }


def _extremum(grid_core: np.ndarray, lat_core: np.ndarray, lon_core: np.ndarray,
              kind: str) -> tuple[float, float, float]:
    flat = int(np.argmin(grid_core) if kind == "min" else np.argmax(grid_core))
    i, j = np.unravel_index(flat, grid_core.shape)
    return float(lat_core[i, j]), float(lon_core[i, j]), float(grid_core[i, j])


def _great_circle_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = np.deg2rad([lat1, lat2]), np.deg2rad([lon1, lon2])
    d = 2.0 * np.arcsin(np.sqrt(
        np.sin(0.5 * (p1[1] - p1[0])) ** 2
        + np.cos(p1[0]) * np.cos(p1[1]) * np.sin(0.5 * (p2[1] - p2[0])) ** 2))
    return float(EARTH_RADIUS_KM * d)


def _mean_sd(values) -> dict[str, Any]:
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if a.size == 0:
        return {"mean": None, "sd": None, "median": None, "n": 0}
    return {"mean": float(a.mean()),
            "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "median": float(np.median(a)), "n": int(a.size)}


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    return obj


def _aggregate(samples: list[dict], boxes: list[str], fields: list[str]) -> list[dict]:
    rows = []
    for box in boxes:
        for field in fields:
            sel = [s for s in samples if s["box"] == box and s["field"] == field]
            if not sel:
                continue
            row: dict[str, Any] = {"box": box, "field": field, "n_samples": len(sel)}
            for pair in PAIRS:
                entries = [s["shift"][pair] for s in sel if pair in s["shift"]]
                if not entries:
                    continue
                row[pair] = {
                    "east_km": _mean_sd([e["east_km"] for e in entries]),
                    "north_km": _mean_sd([e["north_km"] for e in entries]),
                    "distance_km": _mean_sd([e["distance_km"] for e in entries]),
                    "corr_zero": _mean_sd([e["corr_zero"] for e in entries]),
                    "corr_best": _mean_sd([e["corr_best"] for e in entries]),
                    "n_at_search_edge": int(sum(bool(e["at_search_edge"]) for e in entries)),
                }
            if field == "msl":
                row["minimum_distance_km"] = {
                    pair: _mean_sd([s["minimum"]["distance_km"][pair] for s in sel
                                    if s.get("minimum")])
                    for pair in PAIRS
                }
                row["minimum_value_hpa"] = {
                    src: _mean_sd([s["minimum"]["value"][src] / 100.0 for s in sel
                                   if s.get("minimum")])
                    for src in SOURCE_OF
                }
            rows.append(row)
    return rows


def _write_summary(path: Path, payload: dict) -> None:
    lines = [
        f"# Feature displacement — {payload['run_label']}",
        "",
        "A positive eastward or northward number means the second field's feature sits that",
        "far east or north of the first field's. The verdict rests",
        "on model against driver: the model is conditioned on the driver, so a shift there is",
        "the model moving what it was given. Model against truth is context only, because the",
        "extended-range driver and the medium-range truth are different realisations of the",
        "weather.",
        "",
    ]
    for row in payload["aggregate"]:
        lines.append(f"## {row['box']} — {row['field']} ({row['n_samples']} samples)")
        lines.append("")
        lines.append("| pair | east (km) | north (km) | distance (km) | correlation at zero shift | at best shift |")
        lines.append("|---|---|---|---|---|---|")
        for pair in PAIRS:
            e = row.get(pair)
            if not e:
                continue

            def _f(cell, nd=1):
                return "n/a" if cell["mean"] is None else f"{cell['mean']:.{nd}f} ± {cell['sd']:.{nd}f}"

            lines.append(
                f"| {pair.replace('_', ' ')} | {_f(e['east_km'])} | {_f(e['north_km'])} | "
                f"{_f(e['distance_km'])} | {_f(e['corr_zero'], 3)} | {_f(e['corr_best'], 3)} |"
            )
        lines.append("")
        if row.get("minimum_distance_km"):
            parts = []
            for pair, cell in row["minimum_distance_km"].items():
                if cell["mean"] is not None:
                    parts.append(f"{pair.replace('_', ' ')} {cell['mean']:.1f} km")
            if parts:
                lines.append("Distance between the pressure minima: " + ", ".join(parts) + ".")
                lines.append("")
    path.write_text("\n".join(lines) + "\n")


def _read_component(var, member_index: int, state_index: int) -> np.ndarray:
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
    output_dir = Path(output_dir) if output_dir else predictions_dir / "evaluators" / "displacement"
    output_dir.mkdir(parents=True, exist_ok=True)

    boxes = eval_config.get("boxes") or dict(DEFAULT_BOXES)
    fields = list(eval_config.get("fields") or DEFAULT_FIELDS)
    unknown = [f for f in fields if f not in FIELD_STATES]
    if unknown:
        raise ValueError(f"displacement.fields: unknown field(s) {unknown}")
    res_deg = float(eval_config.get("grid_res_deg", DEFAULT_GRID_RES_DEG))
    max_shift_deg = float(eval_config.get("max_shift_deg", DEFAULT_MAX_SHIFT_DEG))
    smooth_deg = float(eval_config.get("smooth_deg", DEFAULT_SMOOTH_DEG))
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

    max_cells = int(round(max_shift_deg / res_deg))
    sigma_cells = smooth_deg / res_deg / 2.0
    LOG.info("displacement: %d file(s), boxes=%s fields=%s search=+/-%d cells of %.2f deg",
             len(files), list(boxes), fields, max_cells, res_deg)

    t0 = time.time()
    up = sps.load_npz(paths["up_matrix"]).tocsr()
    box_static: dict[str, dict] = {}
    up_box: dict[str, Any] = {}
    samples: list[dict[str, Any]] = []

    for file_path in files:
        m = _FILE_RE.search(file_path.name)
        date, step = (m.group(1), int(m.group(2))) if m else ("", -1)
        with netCDF4.Dataset(file_path) as ds:
            ds.set_auto_mask(False)
            ws = [str(v) for v in np.asarray(ds.variables["weather_state"][:]).reshape(-1)]
            idx_of = {s: i for i, s in enumerate(ws)}
            usable = [f for f in fields if all(st in idx_of for st in FIELD_STATES[f][0])]
            skipped = [f for f in fields if f not in usable]
            if skipped:
                LOG.warning("displacement: %s lacks the states for %s; skipped",
                            file_path.name, skipped)
            lat = np.asarray(ds.variables["lat_hres"][:]).reshape(-1).astype(np.float64)
            lon = np.asarray(ds.variables["lon_hres"][:]).reshape(-1).astype(np.float64)
            lon360 = np.mod(lon, 360.0)
            lon180 = np.where(lon360 > 180.0, lon360 - 360.0, lon360)
            if not box_static:
                for name, box in boxes.items():
                    t_box = time.time()
                    box_static[name] = _build_box(lat, lon180, box, res_deg, max_shift_deg)
                    up_box[name] = up[box_static[name]["native_idx"]].tocsr()
                    LOG.info("displacement: box %s ready in %.1fs (%d cells, %d native points)",
                             name, time.time() - t_box, box_static[name]["n_cells"],
                             box_static[name]["n_native"])
            lead = ds.getncattr("lead_step_hours") if "lead_step_hours" in ds.ncattrs() else step

            for member_label, member_index in _select_members(ds, members, max_members):
                t_mem = time.time()
                for field in usable:
                    states, combine = FIELD_STATES[field]
                    cols = [idx_of[st] for st in states]
                    yp = [_read_component(ds.variables["y_pred"], member_index, c) for c in cols]
                    yt = [_read_component(ds.variables["y"], member_index, c) for c in cols]
                    xx = [_read_component(ds.variables["x"], member_index, c) for c in cols]

                    for name, st in box_static.items():
                        native = st["native_idx"]
                        if combine == "hypot":
                            model_n = np.hypot(yp[0][native], yp[1][native])
                            truth_n = np.hypot(yt[0][native], yt[1][native])
                            input_n = np.hypot(up_box[name] @ xx[0], up_box[name] @ xx[1])
                        else:
                            model_n = yp[0][native]
                            truth_n = yt[0][native]
                            input_n = up_box[name] @ xx[0]

                        grids = {}
                        for src, values in (("model", model_n), ("truth", truth_n),
                                            ("input", input_n)):
                            grids[src] = _smooth(_resample(values, st), sigma_cells)

                        rows, cols_n = st["shape"]
                        r0, r1 = max_cells + 2, rows - max_cells - 2
                        c0, c1 = max_cells + 2, cols_n - max_cells - 2
                        window = (r0, r1, c0, c1)
                        km_per_col = res_deg * KM_PER_DEG_LAT * float(
                            np.cos(np.deg2rad(st["centre_lat"])))
                        km_per_row = res_deg * KM_PER_DEG_LAT

                        entry: dict[str, Any] = {
                            "file": file_path.name, "date": date, "step": int(lead),
                            "member": int(member_label), "box": name, "field": field,
                            "shift": {},
                        }
                        for pair, (a_src, b_src) in {
                            "model_vs_input": ("model", "input"),
                            "model_vs_truth": ("model", "truth"),
                            "truth_vs_input": ("truth", "input"),
                        }.items():
                            a_core = grids[a_src][r0:r1, c0:c1]
                            res = _best_shift(a_core, grids[b_src], window, max_cells)
                            east = res["shift_cols"] * km_per_col
                            north = res["shift_rows"] * km_per_row
                            entry["shift"][pair] = {
                                "east_km": east, "north_km": north,
                                "distance_km": float(np.hypot(east, north)),
                                "corr_zero": res["corr_zero"], "corr_best": res["corr_best"],
                                "at_search_edge": res["at_search_edge"],
                            }

                        if field == "msl":
                            core_slice = (slice(r0, r1), slice(c0, c1))
                            pos, val = {}, {}
                            for src in ("model", "truth", "input"):
                                la, lo, v = _extremum(
                                    grids[src][core_slice],
                                    st["mesh_lat"][core_slice], st["mesh_lon"][core_slice],
                                    "min")
                                pos[src], val[src] = (la, lo), v
                            entry["minimum"] = {
                                "position": {k: list(v) for k, v in pos.items()},
                                "value": val,
                                "distance_km": {
                                    "model_vs_input": _great_circle_km(*pos["model"], *pos["input"]),
                                    "model_vs_truth": _great_circle_km(*pos["model"], *pos["truth"]),
                                    "truth_vs_input": _great_circle_km(*pos["truth"], *pos["input"]),
                                },
                            }
                        samples.append(entry)
                LOG.info("displacement: %s member %s done in %.1fs",
                         file_path.name, member_label, time.time() - t_mem)

    aggregate = _aggregate(samples, list(boxes), fields)
    payload = {
        "run_label": run_label or predictions_dir.name,
        "predictions_dir": str(predictions_dir),
        "n_files": len(files),
        "files": [f.name for f in files],
        "config": {
            "boxes": boxes, "fields": fields, "grid_res_deg": res_deg,
            "max_shift_deg": max_shift_deg, "smooth_deg": smooth_deg,
            "steps": steps, "dates": dates, "members": members,
            "max_members": max_members, "paths": paths,
            "search_cells": max_cells, "smooth_sigma_cells": sigma_cells,
        },
        "boxes": {name: {"box": [float(v) for v in boxes[name]],
                         "n_cells": st["n_cells"], "n_native_points": st["n_native"]}
                  for name, st in box_static.items()},
        "pairs": list(PAIRS),
        "aggregate": aggregate,
        "samples": samples,
        "elapsed_s": time.time() - t0,
    }
    payload = _clean(payload)
    (output_dir / "displacement.json").write_text(json.dumps(payload, indent=1) + "\n")
    _write_summary(output_dir / "displacement_summary.md", payload)
    LOG.info("displacement: wrote %s (%d samples) in %.1fs",
             output_dir / "displacement.json", len(samples), time.time() - t0)
    return output_dir
