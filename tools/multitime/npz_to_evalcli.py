#!/usr/bin/env python3
"""Turn the per-member .npz trajectories of predict_sequence.py into eval.cli prediction files.

The E2 box kit (``predict_e2.py`` of the dropout-pair and TD-227 read kits) writes one
``predictions_YYYYMMDD_stepSSS.nc`` per (hindcast date, lead) with ten sampler-seed members:
``y_pred``, ``y`` (truth) and ``x_interp`` (the input on the output grid) of shape
(sample, ensemble_member, grid_point_hres, weather_state), ``x`` on the input grid, the
per-member views ``y_pred_<i>``/``y_<i>``/``x_<i>``/``x_interp_<i>``, coordinates and the eval
metadata attributes. ``eval.cli`` and ``eval.tools.temporal_consistency.read`` read that format.

This converter does not rebuild the format: it copies a TEMPLATE file of the same date and lead
written by the kit (any checkpoint: ``x``, ``x_interp`` and ``y`` depend only on the case, not on
the network, because every E2 run reads the same stores through the same E2-box projection
matrix), then overwrites ``y_pred`` and every ``y_pred_<i>`` with the multi-time prediction and
rewrites the provenance attributes. Before writing it checks, and refuses on any mismatch:
the weather states (names and order) of the template against the npz ``variables``; the
template's ``lat_hres``/``lon_hres`` against the npz grid; the template's sampler seeds against
the npz member seeds; the template truth ``y`` of every member against the npz truth of that
lead (identical float32 values expected, since both read the same store row); the npz
``hdate`` and lead against the file name. The member order follows the template's
``ensemble_member`` (1..10) = npz member ids.

Usage:
  python npz_to_evalcli.py --npz-prefix <dir>/mt2_20230226 --hdate 20230226 \
      --template-dir <kit run>/predictions --out-dir <run>/predictions --label mt2_e2_20k_stride2 \
      [--members 10] [--leads 12,24,36,48,60,72]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz-prefix", required=True, help="<dir>/<name>: reads <name>_m01.npz ... and <name>_truth.npz")
    ap.add_argument("--hdate", required=True, help="YYYYMMDD")
    ap.add_argument("--template-dir", required=True, help="kit predictions folder with predictions_<hdate>_stepSSS.nc")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label", required=True, help="run label written into the attributes")
    ap.add_argument("--members", type=int, default=10)
    ap.add_argument("--leads", default=None, help="default: the npz leads")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--check-only", action="store_true",
                    help="run every check against the template (read-only, members may be fewer) and write nothing")
    args = ap.parse_args()

    import netCDF4

    prefix = Path(args.npz_prefix)
    zs = []
    for m in range(1, args.members + 1):
        p = prefix.parent / f"{prefix.name}_m{m:02d}.npz"
        if not p.is_file():
            raise SystemExit(f"missing member file {p}")
        zs.append(np.load(p, allow_pickle=False))
    z0 = zs[0]
    variables = [str(v) for v in z0["variables"]]
    leads = [int(v) for v in z0["leads_h"]]
    if str(z0["hdate"]) != args.hdate:
        raise SystemExit(f"npz hdate {z0['hdate']} differs from --hdate {args.hdate}")
    if "truth" in z0.files:
        truth = z0["truth"]
    else:
        truth = np.load(str(z0["truth_file"]), allow_pickle=False)["truth"]
    seeds = [int(z["seed"]) for z in zs]
    for z in zs[1:]:
        if [str(v) for v in z["variables"]] != variables or [int(v) for v in z["leads_h"]] != leads:
            raise SystemExit("member files disagree on variables or leads")
    wanted = [int(v) for v in args.leads.split(",")] if args.leads else leads
    sampling = json.loads(str(z0["sampling_json"]))
    sampling["seed_members"] = seeds
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for lead in wanted:
        t = leads.index(lead)
        name = f"predictions_{args.hdate}_step{lead:03d}.nc"
        dst = out_dir / name
        if dst.exists() and not args.overwrite:
            print(f"{dst} exists, skipped")
            continue
        src = Path(args.template_dir) / name
        if not src.is_file():
            raise SystemExit(f"missing template {src}")
        tmp = out_dir / (name + ".partial")
        if args.check_only:
            tmp = src
        else:
            shutil.copyfile(src, tmp)
        with netCDF4.Dataset(tmp, "r" if args.check_only else "r+") as ds:
            ws = [str(v) for v in ds.variables["weather_state"][:]]
            if ws != variables:
                raise SystemExit(f"{name}: template weather states differ from the npz variables")
            for key, zkey in (("lat_hres", "lat"), ("lon_hres", "lon")):
                if not np.allclose(np.asarray(ds.variables[key][:]), z0[zkey], rtol=0, atol=1e-6):
                    raise SystemExit(f"{name}: template {key} differs from the npz grid")
            tseeds = [int(v) for v in ds.variables["sampler_seed"][:]]
            n_mem = ds.dimensions["ensemble_member"].size
            if args.check_only:
                tseeds, n_mem = tseeds[: len(zs)], min(n_mem, len(zs))
            if tseeds != seeds:
                raise SystemExit(f"{name}: template seeds {tseeds} differ from npz seeds {seeds}")
            if n_mem != len(zs):
                raise SystemExit(f"{name}: template has {n_mem} members, npz {len(zs)}")
            if int(ds.getncattr("lead_step_hours_int")) != lead or str(ds.getncattr("hdate")).replace("-", "") != args.hdate:
                raise SystemExit(f"{name}: template attributes name another case")
            ymax = 0.0
            for m in range(n_mem):
                y_t = np.asarray(ds.variables["y"][0, m], dtype=np.float32)
                d = float(np.nanmax(np.abs(y_t - truth[t])))
                ymax = max(ymax, d)
            scale = float(np.nanmax(np.abs(truth[t])))
            if ymax > 1e-6 * max(scale, 1.0):
                raise SystemExit(f"{name}: template truth differs from the npz truth (max abs {ymax:.3e})")
            pred = np.stack([z["pred"][t] for z in zs], axis=0).astype(np.float32)  # (member, grid, vars)
            if not np.isfinite(pred).all():
                raise SystemExit(f"{name}: non-finite prediction values")
            if args.check_only:
                print(f"check-only {name}: weather states, grid, seeds {seeds}, truth (max abs {ymax:.2e}) OK; "
                      f"pred std {pred.std():.4g}")
                continue
            ds.variables["y_pred"][0] = pred
            for m in range(n_mem):
                ds.variables[f"y_pred_{m}"][0] = pred[m]
            template_ckpt = ds.getncattr("checkpoint_path")
            ckpt = sampling.get("ckpt", "")
            ds.setncattr("checkpoint_path", ckpt)
            ds.setncattr("training_checkpoint", ckpt)
            ds.setncattr("sampling_config_json", json.dumps(sampling, sort_keys=True))
            ds.setncattr("weights", "ema2000" if "ema2000" in ckpt else ("raw" if "inference-anemoi" in ckpt else "see checkpoint_path"))
            ds.setncattr("run_label", args.label)
            ds.setncattr("multitime_converter", "tools/multitime/npz_to_evalcli.py")
            ds.setncattr("x_interp_y_x_from_template", f"{src} (kit checkpoint {template_ckpt})")
            ds.setncattr("truth_check_max_abs", ymax)
        os.replace(tmp, dst)
        print(f"wrote {dst}: {len(zs)} members, truth check max abs {ymax:.2e}, pred std {pred.std():.4g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
