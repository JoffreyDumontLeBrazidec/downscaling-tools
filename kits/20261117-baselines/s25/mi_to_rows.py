#!/usr/bin/env python3
"""S25 GL512_450k row from the month-inference (MI-S25) predictions instead of a second XSET prediction run (Jupiter CPU).

MI-S25 (thread "Build a month inference dataset to bypass prepml", kit 2defaf4) writes GL512_450k on all 30 S25 dates
with the XSET predictor's own settings: derive_draw_seed(1234, date, lead, member-1), GlobalNoise, PW_SCHED / PW_SAMPLER,
EMA-2000 450k; its xcheck gate checks bit identity against the XSET 500k row. Files:
    MI/<date>/pred_<date>_step<LLL>.npz  q (10 members, 10 outputs, 6,599,680 points) uint16, lo, hi (per member, output),
                                         states, seeds;  MI/grid_o1280.npz  the out_hres point order (latitudes)
Unpacked value = lo + q / 65535 * (hi - lo): max error span / 131070 (tp ~0.002 mm), so the row is NOT bit-equal to an
XSET float32 row; the row attrs say so.
Writes, like xset_predict_global.py: OUT/rows/<label>/predictions_<date>_step<LLL>.nc (y_pred at the template's
point_index; y, native x, imerg_tp12, why, coordinates from the ENS9 template) and OUT/hist/<label>_<date>_step<LLL>.npz.

    python3 mi_to_rows.py --mi DIR --templates XOUT/rows/ENS9 --out XOUT --date 20250901 [--date ...] [--label GL512_450k]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "qm"))
sys.path.insert(0, str(HERE.parents[1] / "20261116-extremes-set"))
from qm_apply import read_template  # noqa: E402
from xset_common import LEADS, SAVE_VARS, tail_histograms, write_row_file  # noqa: E402

SEED_BASE = 1234


def seed(date, lead, member0):
    return int(np.random.SeedSequence([SEED_BASE, int(date), int(lead), int(member0)]).generate_state(1, dtype=np.uint32)[0])


def model_attr(a, z) -> str:
    """Row attr 'model': --model-attr, else the MI npz attrs (model label, step, sampler), else the 21 Nov fixed text."""
    tail = " (MI month row, uint16-packed fields, max error span/131070)"
    if a.model_attr:
        return f"{a.label}: {a.model_attr}{tail}"
    if "attrs" in z.files:
        try:
            m = json.loads(str(z["attrs"]))
            keys = [k for k in ("model", "model_label", "ckpt_step", "step", "ema", "sampler") if k in m]
            if keys:
                return f"{a.label}: " + "; ".join(f"{k}={json.dumps(m[k], sort_keys=True)}" for k in keys) + tail
        except (ValueError, TypeError):
            pass
    return f"{a.label}: db_global_512_300k step 450000 EMA-2000, from the MI-S25 month row{tail}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mi", required=True)
    ap.add_argument("--templates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--date", action="append", required=True)
    ap.add_argument("--label", default="GL512_450k")
    ap.add_argument("--model-attr", default=None, help="row attr 'model' (default: built from the MI npz attrs json; "
                    "the old fixed GL512 450k text only when the npz has none)")
    a = ap.parse_args()
    mi, O = Path(a.mi), Path(a.out)
    g = np.load(mi / "grid_o1280.npz")
    hlat = np.asarray(g["latitudes"] if "latitudes" in g else g["lat"], np.float64)
    rows, hist = O / "rows" / a.label, O / "hist"
    rows.mkdir(parents=True, exist_ok=True)
    hist.mkdir(parents=True, exist_ok=True)
    for d in a.date:
        for L in LEADS:
            name = f"predictions_{d}_step{L:03d}.nc"
            if (rows / name).exists():
                continue
            z = np.load(mi / d / f"pred_{d}_step{L:03d}.npz")
            states = [str(s) for s in z["states"]]
            want = [seed(d, L, k) for k in range(z["q"].shape[0])]
            if [int(s) for s in z["seeds"]] != want:
                raise SystemExit(f"{d} +{L}: MI seeds {list(z['seeds'])} are not rule 31's {want}")
            q, lo, hi = z["q"], np.asarray(z["lo"], np.float64), np.asarray(z["hi"], np.float64)
            assert q.shape[2] == hlat.size, (q.shape, hlat.size)

            def full(v):   # (members, points) float32
                i = states.index(v)
                return (lo[:, i, None] + q[:, i, :].astype(np.float64) / 65535.0 * (hi[:, i, None] - lo[:, i, None])).astype(np.float32)

            tpl = read_template(Path(a.templates) / name)
            pidx = tpl["point_index"].astype(np.int64)
            f = {v: full(v) for v in set(SAVE_VARS) | {"tp", "10u", "10v"}}
            np.savez(hist / f"{a.label}_{d}_step{L:03d}.npz", **tail_histograms(hlat, f["tp"], f["10u"], f["10v"]))
            y_pred = np.stack([f[v][:, pidx] for v in SAVE_VARS], axis=-1)
            assert np.isfinite(y_pred).all(), f"non-finite {d} +{L}"
            tmp = rows / (name + ".part")
            write_row_file(tmp, lat=tpl["lat_hres"], lon=tpl["lon_hres"], point_index=pidx, states=list(SAVE_VARS), y=tpl["y"],
                           y_pred=y_pred, x=tpl["x"], lat_lres=tpl["lat_lres"], lon_lres=tpl["lon_lres"],
                           lres_index=tpl["lres_index"], extra=tpl["extra"],
                           attrs=dict(model=model_attr(a, z),
                                      truth="y = od enfo cf", date=d, lead=L, member_rule="y_pred member k = draw k, seed "
                                      "derive_draw_seed(1234, date, lead, k-1), GlobalNoise", source=str(mi / d), tp_units="m per 12 h"))
            tmp.rename(rows / name)
        print("ROWS", d, flush=True)
    print("MI_ROWS_DONE", flush=True)


if __name__ == "__main__":
    main()
