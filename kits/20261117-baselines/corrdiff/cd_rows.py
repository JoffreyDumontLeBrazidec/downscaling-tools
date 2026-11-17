#!/usr/bin/env python3
"""CorrDiff baseline, row files (Atos CPU): cdout_<date>_step<LLL>.npz of cd_predict.py -> A/rows/CORRDIFF/ in the XSET
layout (y, native x, imerg_tp12, why and coordinates from the ENS9 template of the same date and lead). y_pred is NaN
outside the E2 box: obs_points drops those stations by itself; read CorrDiff against the other rows on the region
e2_box_in of xset_score_stations.py and on the cyclone / IMERG lines where CorrDiff is finite (cd_box_table.py).
No global tail histograms (a regional model).

    python3 cd_rows.py --work A --cdout DIR --date 20240804 [--date ...] [--label CORRDIFF]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "qm"))
sys.path.insert(0, str(HERE.parents[1] / "20261116-extremes-set"))
from qm_apply import read_template  # noqa: E402
from xset_common import LEADS, SAVE_VARS, write_row_file  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--cdout", required=True)
    ap.add_argument("--date", action="append", required=True)
    ap.add_argument("--label", default="CORRDIFF")
    a = ap.parse_args()
    A = Path(a.work)
    rows = A / "rows" / a.label
    rows.mkdir(parents=True, exist_ok=True)
    for d in a.date:
        for L in LEADS:
            name = f"predictions_{d}_step{L:03d}.nc"
            if (rows / name).exists():
                continue
            z = np.load(Path(a.cdout) / f"cdout_{d}_step{L:03d}.npz")
            tpl = read_template(A / "rows" / "ENS9" / name)
            states = [str(s) for s in z["states"]]
            y_pred = np.stack([z["y_pred"][..., states.index(v)] for v in SAVE_VARS], axis=-1)
            assert y_pred.shape[1] == tpl["point_index"].size, (y_pred.shape, tpl["point_index"].size)
            tmp = rows / (name + ".part")
            write_row_file(tmp, lat=tpl["lat_hres"], lon=tpl["lon_hres"], point_index=tpl["point_index"], states=list(SAVE_VARS),
                           y=tpl["y"], y_pred=y_pred, x=tpl["x"], lat_lres=tpl["lat_lres"], lon_lres=tpl["lon_lres"],
                           lres_index=tpl["lres_index"], extra=tpl["extra"],
                           attrs=dict(model=f"{a.label}: NVIDIA PhysicsNeMo CorrDiff v2.2.2 (regression U-Net + patched residual "
                                            "diffusion), trained on the E2-v4 box 2003-2021, published defaults",
                                      competitor_row=f"{a.label}: external generative baseline, E2 box only (NaN outside)",
                                      truth="y = od enfo cf", date=d, lead=L, member_rule="y_pred member k = CorrDiff(input member k), "
                                      "one draw, seed derive_draw_seed(1234, date, lead, k-1)", tp_units="m per 12 h"))
            tmp.rename(rows / name)
        print("ROWS", d, flush=True)
    print("CD_ROWS_DONE", flush=True)


if __name__ == "__main__":
    main()
