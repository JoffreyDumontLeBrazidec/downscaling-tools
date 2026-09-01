"""Field-touching measurements behind the o1280->o2560 lane diagnostic figures.

Everything in this module reads prediction NetCDF files and reduces them to small
JSON summaries.  It exists so that the figures downstream never have to open a
full O2560 field: the reductions run once, on a compute node, and the plotting
stage is then cheap enough to re-run anywhere.

Four reductions live here:

``box_wind``
    The maximum 10 m wind speed inside a fixed box, per (date, lead step, member),
    for the target, for the interpolated driver and for the model.  This is the
    wind counterpart of the box-minimum pressure table, and it is recorded per
    case rather than pooled, because a pooled extreme on this lane is set by the
    single best case and hides the typical one.

``loss_budget``
    How the squared error of the precipitation field is distributed over rain
    intensity: what share of the total squared error comes from points where the
    target exceeds a threshold, and what share of the grid those points are.

``pair_coherence``
    How close the interpolated low-resolution input already is to the
    high-resolution target, per lead time, for one channel and one member.  Run
    on two lanes it answers whether the o1280->o2560 pair is looser than the
    o320->o1280 pair.

``sampler_peaks``
    The per-(slice, member) precipitation maximum for each sampler arm, so the
    arms can be compared on one axis.
"""
from __future__ import annotations

import glob
import json
import logging
import os
from pathlib import Path

import numpy as np

LOG = logging.getLogger(__name__)

MM = 1000.0            # metres -> millimetres
PA_TO_HPA = 100.0
CHANNEL_BLOCK = 2_000_000   # grid points per read, keeps peak memory modest


def _channels(ds) -> list[str]:
    return [str(s) for s in ds["weather_state"].values]


def _parse_name(path: str) -> tuple[str, int]:
    base = os.path.basename(path)
    date = base.split("_")[1]
    step = int(base.split("step")[1].split(".")[0])
    return date, step


def _box_index(lat: np.ndarray, lon: np.ndarray, box: dict):
    """Return (i0, i1, keep, lat_kept, lon_kept) for a latitude-banded box.

    The O2560 grid is stored in latitude order, so restricting to a latitude band
    first turns a 26-million-point mask into a contiguous slice.
    """
    inb = (lat >= box["south"]) & (lat <= box["north"])
    i0 = int(np.argmax(inb))
    i1 = int(len(inb) - np.argmax(inb[::-1]))
    lo = np.where(lon[i0:i1] > 180, lon[i0:i1] - 360, lon[i0:i1])
    keep = (lo >= box["west"]) & (lo <= box["east"]) & inb[i0:i1]
    return i0, i1, keep, lat[i0:i1][keep], lo[keep]


# ---------------------------------------------------------------------------
# 1. box maximum 10 m wind speed, per (date, step, member)
# ---------------------------------------------------------------------------

def box_wind(predictions_dir: str | Path, box: dict) -> list[dict]:
    """Maximum 10 m wind speed inside ``box`` for target, driver and model."""
    import netCDF4 as nc

    files = sorted(glob.glob(os.path.join(str(predictions_dir), "predictions_*_step*.nc")))
    if not files:
        raise FileNotFoundError(f"no prediction files under {predictions_dir}")

    rows: list[dict] = []
    band = None
    for f in files:
        d = nc.Dataset(f)
        chans = [str(c) for c in d.variables["weather_state"][:]]
        iu, iv = chans.index("10u"), chans.index("10v")
        if band is None:
            band = _box_index(np.asarray(d.variables["lat_hres"][:]),
                              np.asarray(d.variables["lon_hres"][:]), box)
        i0, i1, keep, blat, blon = band
        date, step = _parse_name(f)
        got = {}
        for name, key in (("model", "y_pred"), ("interp", "x_interp"), ("truth", "y")):
            u = np.asarray(d.variables[key][0, :, i0:i1, iu], dtype=np.float32)[:, keep]
            v = np.asarray(d.variables[key][0, :, i0:i1, iv], dtype=np.float32)[:, keep]
            spd = np.sqrt(u * u + v * v)
            j = np.nanargmax(spd, axis=1)
            got[name] = (np.take_along_axis(spd, j[:, None], 1)[:, 0], blat[j], blon[j])
        for m in range(got["model"][0].shape[0]):
            rows.append({
                "date": date, "step": step, "member": m,
                "truth": float(got["truth"][0][m]),
                "truth_lat": float(got["truth"][1][m]), "truth_lon": float(got["truth"][2][m]),
                "interp": float(got["interp"][0][m]),
                "interp_lat": float(got["interp"][1][m]), "interp_lon": float(got["interp"][2][m]),
                "model": float(got["model"][0][m]),
                "model_lat": float(got["model"][1][m]), "model_lon": float(got["model"][2][m]),
            })
        d.close()
        LOG.info("box_wind: %s", os.path.basename(f))
    return rows


# ---------------------------------------------------------------------------
# 2. how the precipitation squared-error budget is spread over rain intensity
# ---------------------------------------------------------------------------

def loss_budget(
    predictions_dir: str | Path,
    precip_cfg: dict,
    thresholds_mm: list[float],
    *,
    stride: int = 5,
    member: int = 0,
) -> dict:
    """Share of the tp squared-error budget contributed above each threshold.

    For every sampled slice this accumulates the total squared error between the
    model's precipitation and the target's, and the part of it that comes from
    points where the TARGET exceeds each threshold, together with how large a
    share of the grid those points are.  The point of the figure is to show that
    the heavy tail is a large part of what the loss is made of, so "the objective
    cannot see the tail" is not an available explanation.
    """
    import xarray as xr
    from eval._backends.precip.sources import PrecipTruthSource

    files = sorted(glob.glob(os.path.join(str(predictions_dir), "predictions_*_step*.nc")))
    files = files[::stride]
    if not files:
        raise FileNotFoundError(f"no prediction files under {predictions_dir}")

    tpl = precip_cfg.get("truth_grib_tpl")
    if not tpl:
        raise RuntimeError("lane precip block has no truth_grib_tpl; cannot load tp truth")
    truth_src = PrecipTruthSource(tpl, var="tp")

    thr = np.asarray(sorted(thresholds_mm), dtype=np.float64)
    sse_above = np.zeros(thr.size)
    pts_above = np.zeros(thr.size)
    sse_total = 0.0
    pts_total = 0.0
    loaded_date = None

    for f in files:
        date, step = _parse_name(f)
        if date != loaded_date:
            truth_src.preload(date)
            loaded_date = date
        with xr.open_dataset(f) as ds:
            ti = _channels(ds).index("tp")
            truth_mm = truth_src.load(date, step).astype(np.float64) * MM
            pred_mm = ds["y_pred"][0, member].values[:, ti].astype(np.float64) * MM
        err2 = (pred_mm - truth_mm) ** 2
        good = np.isfinite(err2) & np.isfinite(truth_mm)
        err2, truth_mm = err2[good], truth_mm[good]
        sse_total += float(err2.sum())
        pts_total += float(err2.size)
        for k, t in enumerate(thr):
            sel = truth_mm > t
            sse_above[k] += float(err2[sel].sum())
            pts_above[k] += float(sel.sum())
        LOG.info("loss_budget: %s", os.path.basename(f))

    return {
        "n_slices": len(files),
        "member_index": member,
        "n_points_total": pts_total,
        "thresholds_mm": thr.tolist(),
        "sse_share_above": (sse_above / sse_total).tolist(),
        "point_share_above": (pts_above / pts_total).tolist(),
        "sse_total": sse_total,
    }


# ---------------------------------------------------------------------------
# 3. how close the interpolated input already is to the target, by lead
# ---------------------------------------------------------------------------

def pair_coherence(
    predictions_dir: str | Path,
    channel: str = "msl",
    member: int = 0,
    steps: list[int] | None = None,
) -> list[dict]:
    """Root-mean-square error of interpolated input, and of the model, against truth.

    Accumulated in blocks over grid points so a full high-resolution field is
    never held in memory at once.
    """
    import netCDF4 as nc

    files = sorted(glob.glob(os.path.join(str(predictions_dir), "predictions_*_step*.nc")))
    if steps is not None:
        wanted = set(int(v) for v in steps)
        files = [f for f in files if _parse_name(f)[1] in wanted]
    if not files:
        raise FileNotFoundError(f"no prediction files under {predictions_dir}")

    rows: list[dict] = []
    for f in files:
        d = nc.Dataset(f)
        chans = [str(c) for c in d.variables["weather_state"][:]]
        ci = chans.index(channel)
        n = d.dimensions["grid_point_hres"].size
        acc = dict.fromkeys(("sy", "syy", "d2x", "d2p", "cnt"), 0.0)
        for s in range(0, n, CHANNEL_BLOCK):
            e = min(s + CHANNEL_BLOCK, n)
            xi = np.asarray(d.variables["x_interp"][0, member, s:e, ci], dtype=np.float64)
            y = np.asarray(d.variables["y"][0, member, s:e, ci], dtype=np.float64)
            yp = np.asarray(d.variables["y_pred"][0, member, s:e, ci], dtype=np.float64)
            m = np.isfinite(xi) & np.isfinite(y) & np.isfinite(yp)
            xi, y, yp = xi[m], y[m], yp[m]
            acc["cnt"] += xi.size
            acc["sy"] += y.sum()
            acc["syy"] += (y * y).sum()
            acc["d2x"] += ((xi - y) ** 2).sum()
            acc["d2p"] += ((yp - y) ** 2).sum()
        d.close()
        date, step = _parse_name(f)
        n_ok = acc["cnt"]
        rows.append({
            "date": date, "step": step, "channel": channel, "member": member,
            "n": int(n_ok),
            "rmse_interp": float(np.sqrt(acc["d2x"] / n_ok)),
            "rmse_model": float(np.sqrt(acc["d2p"] / n_ok)),
            "std_truth": float(np.sqrt(acc["syy"] / n_ok - (acc["sy"] / n_ok) ** 2)),
        })
        LOG.info("pair_coherence: %s", os.path.basename(f))
    return rows


# ---------------------------------------------------------------------------
# 4. per-slice precipitation peak, one entry per sampler arm
# ---------------------------------------------------------------------------

def sampler_peaks(arm_roots: dict) -> dict:
    """Per (slice, member) 6 h precipitation maximum for each sampler arm.

    Arms are read only.  Each arm reports its own sampler configuration, taken
    from the prediction files themselves rather than from a directory name, and
    the number of member-slices actually found, so an arm that is still filling
    is visible as a smaller sample rather than silently averaged in.
    """
    import netCDF4 as nc

    out: dict = {}
    for label, root in arm_roots.items():
        files = sorted(glob.glob(os.path.join(str(root), "date_*", "predictions", "*.nc")))
        if not files:
            files = sorted(glob.glob(os.path.join(str(root), "predictions", "*.nc")))
        if not files:
            out[label] = {"n_files": 0, "peaks_mm": [], "truth_peaks_mm": [],
                          "sampler": {}, "note": "no prediction files found"}
            LOG.warning("sampler_peaks: arm %s has no prediction files (%s)", label, root)
            continue
        peaks: list[float] = []
        truth_peaks: list[float] = []
        cfg: dict = {}
        for f in files:
            d = nc.Dataset(f)
            if not cfg:
                try:
                    cfg = json.loads(getattr(d, "sampling_config_json"))
                except Exception:
                    cfg = {}
            chans = [str(c) for c in d.variables["weather_state"][:]]
            ti = chans.index("tp")
            a = np.asarray(d.variables["y_pred"][0, :, :, ti], dtype=np.float32) * MM
            for m in range(a.shape[0]):
                v = a[m][np.isfinite(a[m])]
                if v.size:
                    peaks.append(float(v.max()))
            y = np.asarray(d.variables["y"][0, 0, :, ti], dtype=np.float32) * MM
            if np.isfinite(y).any():
                truth_peaks.append(float(np.nanmax(y)))
            d.close()
        out[label] = {
            "n_files": len(files),
            "n_member_slices": len(peaks),
            "peaks_mm": peaks,
            "truth_peaks_mm": truth_peaks,
            "sampler": {k: cfg.get(k) for k in
                        ("num_steps", "sigma_max", "S_churn", "S_noise", "schedule_type")},
        }
        LOG.info("sampler_peaks: %s -> %d member-slices from %d files",
                 label, len(peaks), len(files))
    return out
