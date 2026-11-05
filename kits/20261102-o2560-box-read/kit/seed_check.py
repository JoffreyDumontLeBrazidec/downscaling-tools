"""o2560 box read smoke: check that the per-draw seed actually drives the sampler (review fix 1, 2 Nov 2026).
Inputs (all row B24, case 20241101, model on 4 ranks unless said): <dir2> holds the 2-member files at +24 h and +48 h, <dir1> the 1-member file at +24 h
from a separate process, <dirc8> the 1-member file at +24 h with ANEMOI_INFERENCE_NUM_CHUNKS=8 (the smoke's other rows use 16; the read uses 8 since the smoke showed them bit-identical).
Checks, per variable, in units of the truth's standard deviation:
  1. member 1 of the 1-member run equals member 1 of the 2-member run: bit-identical, or at most 1e-4 (float noise of GPU
     kernels between processes); a different seed would differ by order 1. Else SEED_CHECK_FAIL.
  2. member 1 differs from member 2 (mean |difference| at least 0.01 for 10u, 10v and 2t). Else SEED_CHECK_FAIL.
  3. member 1 at +24 h differs from member 1 at +48 h, and their seeds differ. Else SEED_CHECK_FAIL.
  4. chunks 8 against 16: max |difference| reported; above 1e-2 a WARN line (information, not a failure).
  5. the model on 2 ranks (<dirg2>) against 4 ranks: the noise is one box-wide field either way, so member 1 must agree to
     float level: mean |difference| at most 1e-3 of the truth's std (a noise mismatch differs by order 0.1, the spread
     between members); the maximum is reported but not judged (smoke 2645931: tp and cp maxima 1.75e-2 in the heavy tail
     at mean 5e-5, reduction order amplified over 59 sampler calls). Else SEED_CHECK_FAIL.
  The seeds written in the files equal case_seeds(date, lead, M). Prints SEED_CHECK_PASS or SEED_CHECK_FAIL lines."""
import json
import sys
from pathlib import Path

import netCDF4
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from o2560_common import case_seeds  # noqa: E402

d2, d1, dc8, dg2 = (Path(x) for x in sys.argv[1:5])


def load(p):
    with netCDF4.Dataset(str(p)) as d:
        d.set_auto_mask(False)
        return ([str(s) for s in d["weather_state"][:]], np.asarray(d["y_pred"][0], np.float64), np.asarray(d["y"][0], np.float64),
                json.loads(d.getncattr("seeds")))


fails = []
st, p24, y, s24 = load(d2 / "predictions_20241101_step024.nc")
_, p48, _, s48 = load(d2 / "predictions_20241101_step048.nc")
_, q24, _, t24 = load(d1 / "predictions_20241101_step024.nc")
_, c24, _, u24 = load(dc8 / "predictions_20241101_step024.nc")
_, g24, _, v24 = load(dg2 / "predictions_20241101_step024.nc")
sd = y.std(axis=0)
for name, got, want in (("2-member +24", s24, case_seeds("20241101", 24, 2)), ("2-member +48", s48, case_seeds("20241101", 48, 2)),
                        ("1-member +24", t24, case_seeds("20241101", 24, 1)), ("chunks-8 +24", u24, case_seeds("20241101", 24, 1)), ("2-rank +24", v24, case_seeds("20241101", 24, 1))):
    print(f"seeds {name}: {got} expected {want}")
    if got != want:
        fails.append(f"seeds of {name} {got} != {want}")
if t24[0] != s24[0]:
    fails.append("member-1 seed differs between the 1- and 2-member runs")
for j, v in enumerate(st):
    r1 = float(np.abs(q24[0, :, j] - p24[0, :, j]).max() / sd[j])
    r2 = float(np.abs(p24[0, :, j] - p24[1, :, j]).mean() / sd[j])
    r3 = float(np.abs(p24[0, :, j] - p48[0, :, j]).mean() / sd[j])
    r4 = float(np.abs(c24[0, :, j] - p24[0, :, j]).max() / sd[j])
    r5 = float(np.abs(g24[0, :, j] - p24[0, :, j]).max() / sd[j])
    r5m = float(np.abs(g24[0, :, j] - p24[0, :, j]).mean() / sd[j])
    same = "bit-identical" if r1 == 0.0 else ("float-level" if r1 <= 1e-4 else "DIFFERENT")
    print(f"{v:6s} rerun m1: max|d|/sd {r1:.2e} ({same}) | m1 vs m2 mean|d|/sd {r2:.3e} | +24 vs +48 m1 mean|d|/sd {r3:.3e} | "
          f"chunks 8 vs 16 max|d|/sd {r4:.2e} | 2 vs 4 ranks mean|d|/sd {r5m:.2e} max {r5:.2e}")
    if r1 > 1e-4:
        fails.append(f"{v}: member 1 not reproduced by the 1-member run (max|d|/sd {r1:.2e})")
    if v in ("10u", "10v", "2t") and r2 < 1e-2:
        fails.append(f"{v}: members 1 and 2 nearly equal (mean|d|/sd {r2:.2e}): the seed does not reach the sampler")
    if v in ("10u", "10v", "2t") and r3 < 1e-2:
        fails.append(f"{v}: +24 and +48 member 1 nearly equal")
    if r5m > 1e-3:
        fails.append(f"{v}: 2 ranks and 4 ranks differ (mean|d|/sd {r5m:.2e}): the noise is not one box-wide field")
    if r4 > 1e-2:
        print(f"WARN {v}: chunks 8 vs 16 differ by max|d|/sd {r4:.2e} (information only)")
for f in fails:
    print("SEED_CHECK_FAIL", f)
print("SEED_CHECK_PASS" if not fails else f"SEED_CHECK_FAIL {len(fails)} checks")
sys.exit(0 if not fails else 1)
