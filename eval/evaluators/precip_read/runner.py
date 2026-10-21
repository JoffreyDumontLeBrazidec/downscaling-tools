"""precip_read run(): one pass over the prediction files, raw results only (no figures)."""
from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval.discovery.predictions import find_predictions
from eval.evaluators.precip_read.core.metrics import EDGES, score_sums

LOG = logging.getLogger(__name__)

MM = 1000.0
DEFAULT_REGIONS = {
    "all": None,
    "10-25N": [10.0, 25.0, -180.0, 180.0],
    "25-40N": [25.0, 40.0, -180.0, 180.0],
    "40-70N": [40.0, 70.01, -180.0, 180.0],
    "europe": [35.0, 70.01, -12.0, 40.0],
}
DEFAULT_THRESHOLDS = (20, 50, 100, 150, 200)


def read_tp(path, var: str = "tp"):
    """lat, lon (-180..180), truth, input, members (M, points) in mm, clipped at 0.

    Truth and input may or may not carry an ensemble_member axis; member 0 is used when they do.
    """
    import netCDF4

    with netCDF4.Dataset(str(path)) as ds:
        ds.set_auto_mask(False)
        ws = [str(s) for s in ds["weather_state"][:]]
        k = ws.index(var)
        lat = np.asarray(ds["lat_hres"][:], dtype=np.float32)
        lon = np.asarray(ds["lon_hres"][:], dtype=np.float32)
        get = lambda n: np.clip(np.asarray(ds[n][0][..., k], dtype=np.float64), 0, None) * MM
        yp, y, xi = get("y_pred"), get("y"), get("x_interp")
    lon = np.where(lon > 180, lon - 360, lon)
    return lat, lon, y.reshape(-1, lat.size)[0], xi.reshape(-1, lat.size)[0], yp.reshape(-1, lat.size)


def region_masks(lat, lon, regions: dict) -> dict:
    out = {}
    for name, b in regions.items():
        out[name] = (np.ones(lat.size, bool) if b is None else
                     (lat >= b[0]) & (lat < b[1]) & (lon >= b[2]) & (lon < b[3]))
    return out


def run(predictions_dir, lane_config: dict, eval_config: dict, *, output_dir=None, overwrite: bool = False,
        checkpoint=None, **kwargs) -> Path:
    predictions_dir = Path(predictions_dir).expanduser().resolve()
    output_dir = Path(output_dir) if output_dir else predictions_dir / "evaluators" / "precip_read"
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"precip_read output exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cases").mkdir(exist_ok=True)

    var = str(eval_config.get("var", "tp"))
    regions = dict(eval_config.get("regions") or DEFAULT_REGIONS)
    thresholds = tuple(float(t) for t in eval_config.get("thresholds", DEFAULT_THRESHOLDS))
    only_dates = {str(d) for d in eval_config.get("dates", [])} or None
    only_steps = {int(s) for s in eval_config.get("steps", [])} or None
    cases = list(eval_config.get("cases", []))

    preds = [p for p in find_predictions(predictions_dir)
             if (only_dates is None or p.date in only_dates) and (only_steps is None or p.step in only_steps)]
    if not preds:
        raise FileNotFoundError(f"No prediction files to read in {predictions_dir}")
    by_step = defaultdict(list)
    for p in preds:
        by_step[p.step].append(p)

    rows, files, members = [], [], set()
    for step in sorted(by_step):
        H = {"truth": [], "input": [], "model": []}
        dates, npts = [], None
        for p in sorted(by_step[step], key=lambda q: q.date):
            lat, lon, y, xi, yp = read_tp(p.path, var)
            members.add(int(yp.shape[0]))
            masks = region_masks(lat, lon, regions)
            npts = [int(m.sum()) for m in masks.values()]
            H["truth"].append([np.histogram(y[m], EDGES)[0] for m in masks.values()])
            H["input"].append([np.histogram(xi[m], EDGES)[0] for m in masks.values()])
            H["model"].append([[np.histogram(yp[j][m], EDGES)[0] for m in masks.values()] for j in range(yp.shape[0])])
            for name, m in masks.items():
                if m.any():
                    rows.append({"lead": step, "date": p.date, "region": name,
                                 **score_sums(yp[:, m], xi[m], y[m], thresholds)})
            for c in cases:
                if str(c["date"]) == p.date and int(c["step"]) == step:
                    b = c["box"]
                    m = (lat >= b[0]) & (lat <= b[1]) & (lon >= b[2]) & (lon <= b[3])
                    np.savez_compressed(output_dir / "cases" / f"{c['name']}.npz", lat=lat[m], lon=lon[m], truth=y[m],
                                        input=xi[m], model=yp[:, m], box=np.asarray(b, float), date=p.date, step=step,
                                        label=str(c.get("label", c["name"])), source=str(p.path))
            dates.append(p.date)
            files.append(str(p.path))
            LOG.info("precip_read: step %03d %s read (%d members)", step, p.date, yp.shape[0])
        if len({len(m) for m in H["model"]}) > 1:
            raise ValueError(f"step {step}: files carry different member counts")
        np.savez_compressed(output_dir / f"hist_step{step:03d}.npz", edges=EDGES, regions=np.array(list(regions)),
                            dates=np.array(dates), npts=np.array(npts), truth=np.array(H["truth"], np.int64),
                            input=np.array(H["input"], np.int64), model=np.array(H["model"], np.int64))

    with open(output_dir / "scores_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    missing = [c["name"] for c in cases if not (output_dir / "cases" / f"{c['name']}.npz").exists()]
    meta = {"predictions_dir": str(predictions_dir), "var": var, "unit": "mm per accumulation window, clipped at 0",
            "accum_hours": eval_config.get("accum_hours", 6), "members": sorted(members), "regions": regions,
            "thresholds": list(thresholds), "steps": sorted(by_step), "files": files, "cases_missing": missing,
            "run_label": str(eval_config.get("run_label") or kwargs.get("run_label") or "")}
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    if missing:
        LOG.warning("precip_read: no file for cases %s", missing)
    return output_dir
