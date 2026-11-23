"""Synthetic inputs in the formats the verdict reads (obs_points, box_spectra, final_battery, XSET-1 kit)."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATES = [(dt.date(2025, 9, 1) + dt.timedelta(days=i)).strftime("%Y%m%d") for i in range(30)]
LEADS = list(range(12, 241, 12))
VARS = {"2t": "lapse", "10ff": "raw", "msl": "raw", "tp": "raw"}


def scores(path: Path, factor: float = 1.0, spread: float = 1.0, seed: int = 0, rows=("model",), input_factor=1.2,
           leads=LEADS, dates=DATES) -> Path:
    """scores_per_date.csv: base CRPS depends on date and lead (shared by every row); the row's CRPS = base x factor
    x (1 + 0.2 % noise)."""
    rng = np.random.default_rng(seed)
    base_rng = np.random.default_rng(42)
    out = []
    for var, variant in VARS.items():
        for lead in leads:
            for d in dates:
                b = (1.0 + lead / 240) * (1 + 0.3 * base_rng.random())
                for row in rows:
                    f = input_factor if row == "input" else factor
                    c = b * f * (1 + 0.002 * rng.standard_normal())
                    rmse = b * 1.3
                    out.append(dict(var=var, variant=variant, radius_km=0.0, region="all", lead=lead, date=d, row=row,
                                    n=1000, fcrps=c, spread=rmse * spread, spread_rms=rmse * spread * np.sqrt(10 / 11),
                                    rmse_mean=rmse, twcrps20=0.1 * c, twcrps50=0.05 * c, twcrps100=0.01 * c))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(path, index=False)
    return path


KMS = [125.0, 100.0, 80.0, 62.0, 50.0, 40.0, 31.0, 25.0, 17.6]


def bins(root: Path, label: str, ratio: dict[float, float] | None = None, leads=(24, 72, 120, 240), seed=0) -> Path:
    """<root>/<label>_L<LLL>/bins.csv with per-date and pooled scopes; ratio = full-field amplitude ratio per bin."""
    rng = np.random.default_rng(seed)
    ratio = ratio or {}
    for L in leads:
        rows = []
        for var in ("ws", "tp"):
            acc = {}
            for d in DATES:
                for km in KMS:
                    t = 10.0 * km / 125 * (1 + 0.05 * rng.standard_normal())
                    r = ratio.get(km, 1.0) * (1 + 0.003 * rng.standard_normal())
                    rows.append(dict(scope=d, variable=var, bin_centre_km=km, amp_pred_full=t * r, amp_truth_full=t,
                                     amp_resid_pred=0.5 * t * r, amp_resid_true=0.5 * t, ratio_full=r, n_draws=10))
                    a = acc.setdefault(km, [0.0, 0.0])
                    a[0] += (t * r) ** 2
                    a[1] += t ** 2
            for km, (p, q) in acc.items():
                rf = np.sqrt(p / q)
                rows.append(dict(scope="all", variable=var, bin_centre_km=km, amp_pred_full=np.sqrt(p / 30),
                                 amp_truth_full=np.sqrt(q / 30), amp_resid_pred=0.5 * np.sqrt(p / 30),
                                 amp_resid_true=0.5 * np.sqrt(q / 30), ratio_full=rf, n_draws=300))
        d = Path(root) / f"{label}_L{L:03d}"
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(d / "bins.csv", index=False)
    return Path(root)


def battery(root: Path, out_ids, c11="0.97-1.02; 0.50") -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lines = [dict(id=i, what="", value="", status="OUT") for i in out_ids]
    lines.append(dict(id="C11", what="lead-to-lead", value=c11, status="ok"))
    (root / "summary.json").write_text(json.dumps({"consistency": lines}))
    return root


XDATES = ["20240801", "20240803", "20240815", "20240901", "20240920", "20241007", "20241031", "20250115"]


def xset(root: Path, label: str, factor=1.0, shift_km=0.0, refs=False, seed=0) -> Path:
    """tc_cases_scored.csv, tc_members.csv and obs_points/scores_per_date.csv of one XSET row folder."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    (root / "obs_points").mkdir(parents=True, exist_ok=True)
    labs = [label] + (["INPUT", "CF9"] if refs else [])
    cases, mem = [], []
    for s in range(12):
        sid = f"S{s:02d}"
        for d in XDATES[s % 4: s % 4 + 2]:
            for lead in (12, 24, 48, 72):
                b = 2 + s * 0.1 + lead / 72
                for lab in labs:
                    f = 1.3 if lab == "INPUT" else factor
                    cases.append(dict(sid=sid, date=d, lead=lead, row=lab, basin="NA", bt_wind_kt=80,
                                      pmin_crps_all=b * f * (1 + 0.003 * rng.standard_normal()),
                                      wind_crps_all=b * f * (1 + 0.003 * rng.standard_normal())))
                    for k in range(1, 11):
                        base_lat = 20 + s + 0.01 * k
                        dlat = 0.0 if lab == "INPUT" else (0.2 + shift_km / 111.0)
                        mem.append(dict(sid=sid, date=d, lead=lead, row=lab, member=k, found=1,
                                        clat=base_lat + dlat, clon=-60.0))
    pd.DataFrame(cases).to_csv(root / "tc_cases_scored.csv", index=False)
    pd.DataFrame(mem).to_csv(root / "tc_members.csv", index=False)
    scores(root / "obs_points" / "scores_per_date.csv", factor=factor, seed=seed, rows=("model",),
           leads=[12, 24, 36, 48, 60, 72], dates=XDATES)
    return root


def hist(root: Path, label: str, tail: float, seed=0) -> Path:
    """<root>/<label>_<date>_step<LLL>.npz with tp and ws counts (bands, members, bins)."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tpe = np.array([0, 10, 50, 100, 200, 1000.0])
    wse = np.array([0, 10, 25, 32.5, 50, 100.0])
    for d in XDATES:
        for L in (24, 48):
            base = np.array([1e6, 1e4, 1e3, 1e2, 10])
            tp = np.tile(base * np.r_[1, 1, tail, tail, tail] * (1 + 0.01 * rng.standard_normal(5)), (3, 10, 1))
            ws = np.tile(base * np.r_[1, 1, tail, tail, tail] * (1 + 0.01 * rng.standard_normal(5)), (3, 10, 1))
            np.savez(root / f"{label}_{d}_step{L:03d}.npz", tp_counts=tp, ws_counts=ws, tp_edges_mm=tpe, ws_edges_ms=wse,
                     tp_max_mm=np.ones((3, 10)), ws_max_ms=np.ones((3, 10)), bands=np.array(["a", "b", "c"]))
    return root


def manifest(path: Path, pred_dir: Path, log: Path, **over) -> Path:
    pred_dir.mkdir(parents=True, exist_ok=True)
    for d in DATES[:2]:
        for L in (12, 24):
            (pred_dir / f"predictions_{d}_step{L:03d}.nc").write_text("x")
    log.write_text("... ALL_PASS\n")
    m = {"row": "RUN", "checkpoint": {"path": "", "md5": "abc", "ema": "EMA-2000"}, "sampler": {"steps": 30},
         "precision": "bf16", "seed_base": 1234, "code_sha": "deadbeef", "input_stream_check": "PASS",
         "instruments": {"MI-S25": {"pred_dir": str(pred_dir), "dates": DATES[:2], "leads": [12, 24], "members": 10}},
         "logs": {str(log): "ALL_PASS"}}
    m.update(over)
    path.write_text(json.dumps(m))
    return path
