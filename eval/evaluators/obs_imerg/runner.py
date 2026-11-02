"""obs_imerg run(): the run's 12 h precipitation against IMERG on the same O1280 points, tables only."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from eval.discovery.predictions import find_predictions
from eval.evaluators.obs_points.core.scores import ensemble_scores, rank_counts
from eval.evaluators.obs_points.core.stations import resolve_cache, wrap_lon
from eval.evaluators.obs_points.core.summary import paired, summarise, write_summary_md
from eval.evaluators.obs_points.runner import rank_seed
from eval.evaluators.precip_read.core.metrics import EDGES
from eval.evaluators.precip_read.runner import read_tp

LOG = logging.getLogger(__name__)

DEFAULT_REGIONS = {"25-60N": [25.0, 60.0, -80.0, 40.0], "europe": [35.0, 60.0, -12.5, 40.0],
                   "25-40N": [25.0, 40.0, -80.0, 40.0], "40-60N": [40.0, 60.0, -80.0, 40.0]}
DEFAULT_THRESHOLDS = (5.0, 10.0, 20.0)
ROW_KIND = {"model": "ensemble", "input": "single field", "truth": "single field"}
WHERE = "against IMERG"


def resolve_file(spec) -> Path:
    """The IMERG cut: a file path, or a list or mapping of candidate paths of which the first existing is used."""
    if spec is None:
        raise ValueError("obs_imerg: the lane block needs 'imerg' (the file written by `eval.cli obs fetch-imerg`)")
    cands = list(spec.values()) if isinstance(spec, dict) else ([spec] if isinstance(spec, (str, Path)) else list(spec))
    for c in cands:
        if Path(str(c)).expanduser().is_file():
            return Path(str(c)).expanduser()
    raise FileNotFoundError(f"obs_imerg: no IMERG file found among {cands}")


class Imerg:
    """The IMERG cut file: one 12 h field per valid time on the E2 points."""

    def __init__(self, path):
        import netCDF4

        self.ds = netCDF4.Dataset(str(path))
        self.ds.set_auto_mask(False)
        self.lat = np.asarray(self.ds["lat"][:], np.float64)
        self.lon = wrap_lon(self.ds["lon"][:])
        self.valid = {str(v): i for i, v in enumerate(self.ds["valid"][:])}
        self.accum_h = int(getattr(self.ds, "accum_hours", 12))

    def at(self, valid):
        i = self.valid.get(valid)
        return None if i is None else np.asarray(self.ds["tp"][i, :], np.float64)

    def check_grid(self, lat, lon):
        if lat.size != self.lat.size or not (np.allclose(lat, self.lat, atol=1e-5)
                                             and np.allclose(wrap_lon(lon), self.lon, atol=1e-5)):
            raise ValueError(f"obs_imerg: the prediction grid ({lat.size} points) is not the IMERG cut's "
                             f"({self.lat.size} points, same order expected)")


def score_file(fields, ref, lat, lon, *, regions, thresholds, date, lead, min_points=1000):
    """Rows, rank rows and histogram counts of one file against one IMERG field. A point is used only where
    IMERG and every field (all members, input, truth) are finite, so every row is scored on the same points."""
    ok = np.isfinite(ref)
    for x in fields.values():
        ok &= np.isfinite(x).all(0) if x.ndim == 2 else np.isfinite(x)
    rows, ranks, hist = [], [], {}
    for reg, b in regions.items():
        m = ok & (lat >= b[0]) & (lat <= b[1]) & (lon >= b[2]) & (lon <= b[3])
        if m.sum() < min_points:
            continue
        r = np.clip(ref[m], 0.0, None)
        base = dict(var="tp", variant="imerg", radius_km=0.0, region=reg, lead=int(lead), date=str(date))
        hist[reg] = {"imerg": np.histogram(r, EDGES)[0]}
        for k, x in fields.items():
            ens = x[:, m] if x.ndim == 2 else x[m][None]
            rows.append({**base, "row": k, "kind": ROW_KIND[k], **ensemble_scores(ens, r, thresholds)})
            hist[reg][k] = sum(np.histogram(e, EDGES)[0] for e in ens)
            if ens.shape[0] > 1:
                ranks.append({**base, "row": k, "members": int(ens.shape[0]), "n": int(m.sum()),
                              **{f"r{i}": int(c) for i, c in enumerate(rank_counts(ens, r, seed=rank_seed(base, k)))}})
    return rows, ranks, hist


def run(predictions_dir, lane_config: dict, eval_config: dict, *, output_dir=None, overwrite: bool = False,
        checkpoint=None, run_label: str = "", **kwargs) -> Path:
    import pandas as pd

    predictions_dir = Path(predictions_dir).expanduser().resolve()
    output_dir = Path(output_dir) if output_dir else predictions_dir / "evaluators" / "obs_imerg"
    output_dir.mkdir(parents=True, exist_ok=True)
    imerg_path = resolve_file(eval_config.get("imerg"))
    imerg = Imerg(imerg_path)
    regions = dict(eval_config.get("regions") or DEFAULT_REGIONS)
    thresholds = tuple(float(t) for t in eval_config.get("thresholds", DEFAULT_THRESHOLDS))
    dates = {str(d) for d in eval_config.get("dates", [])}
    steps = {int(s) for s in eval_config.get("steps", [])}
    preds = [p for p in find_predictions(predictions_dir)
             if (not dates or p.date in dates) and (not steps or p.step in steps)]
    if not preds:
        raise FileNotFoundError(f"No prediction files to read in {predictions_dir}")
    rows, ranks, log, files, members = [], [], [], [], set()
    H = {}
    for p in preds:
        valid = (pd.Timestamp(p.date) + pd.Timedelta(hours=p.step)).strftime("%Y%m%d%H")
        ref = imerg.at(valid)
        if ref is None:
            log.append(f"WARN no IMERG field for {valid} ({p.date} +{p.step} h)")
            continue
        lat, lon, y, xi, yp = read_tp(p.path, str(eval_config.get("var", "tp")))
        imerg.check_grid(lat, lon)
        members.add(int(yp.shape[0]))
        r, k, h = score_file({"model": yp, "input": xi, "truth": y}, ref, lat, lon, regions=regions,
                             thresholds=thresholds, date=p.date, lead=p.step,
                             min_points=int(eval_config.get("min_points", 1000)))
        rows += r
        ranks += k
        for reg, d in h.items():
            for row, c in d.items():
                H.setdefault((p.step, reg, row), np.zeros(EDGES.size - 1, np.int64))
                H[(p.step, reg, row)] += c
        files.append(str(p.path))
        LOG.info("obs_imerg: scored %s +%03d h", p.date, p.step)
    if not rows:
        raise RuntimeError("obs_imerg: nothing scored (no IMERG field for any file?)")
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "scores_per_date.csv", index=False)
    pd.DataFrame(ranks).to_csv(output_dir / "rank_hist.csv", index=False)
    summ = summarise(df)
    summ.to_csv(output_dir / "summary.csv", index=False)
    pr = paired(df, pairs=(("model", "input"), ("model", "truth"), ("truth", "input")))
    pr.to_csv(output_dir / "paired.csv", index=False)
    keys = sorted(H)
    np.savez_compressed(output_dir / "hist.npz", edges=EDGES, leads=np.array([k[0] for k in keys]),
                        regions=np.array([k[1] for k in keys]), rows=np.array([k[2] for k in keys]),
                        counts=np.array([H[k] for k in keys]))
    meta = {"evaluator": "obs_imerg", "reference": f"IMERG Final V07B ({imerg.accum_h} h)", "where": WHERE,
            "sample_unit": "grid points", "predictions_dir": str(predictions_dir),
            "run_label": str(run_label or eval_config.get("run_label") or predictions_dir.name),
            "imerg_file": str(imerg_path), "imerg_attrs": {k: str(imerg.ds.getncattr(k)) for k in imerg.ds.ncattrs()},
            "regions": regions, "thresholds": list(thresholds), "members": sorted(members),
            "dates": sorted({p.date for p in preds}), "steps": sorted({p.step for p in preds}), "files": files,
            "config": {"weighting": "none (O1280 points, nearly equal area)"}, "warnings": log}
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    write_summary_md(summ, pr, meta, output_dir / "summary.md", region=next(iter(regions)))
    for line in log:
        LOG.warning("obs_imerg: %s", line)
    return output_dir
