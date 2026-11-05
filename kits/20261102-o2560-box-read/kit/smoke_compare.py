"""o2560 box read smoke: max |difference| per variable between two prediction files of the same case and seeds
(B24 on code 20ba15241 against B24 on code 4ad32838c, crop off). Information only: prints the numbers, SMOKE_COMPARE_PASS
when the seeds and truth agree and every variable differs by at most 1e-2 of the truth's standard deviation, otherwise
SMOKE_COMPARE_WARN (TF32 and Triton kernels can make two code pins differ at float level). Exits 1 only if the seeds or
the truth differ (then the two runs are not the same case)."""
import json
import sys

import netCDF4
import numpy as np

a, b = sys.argv[1], sys.argv[2]
with netCDF4.Dataset(a) as A, netCDF4.Dataset(b) as B:
    A.set_auto_mask(False); B.set_auto_mask(False)
    st = [str(s) for s in A["weather_state"][:]]
    if st != [str(s) for s in B["weather_state"][:]] or json.loads(A.seeds) != json.loads(B.seeds) or not np.array_equal(A["y"][:], B["y"][:]):
        print("SMOKE_COMPARE_FAIL states, seeds or truth differ: not the same case"); sys.exit(1)
    ya, yb, y = np.asarray(A["y_pred"][:]), np.asarray(B["y_pred"][:]), np.asarray(A["y"][:])
worst = 0.0
for j, s in enumerate(st):
    d = float(np.abs(ya[..., j] - yb[..., j]).max()); sd = float(y[..., j].std()); worst = max(worst, d / sd)
    print(f"{s:6s} max|diff| {d:.3e}  truth std {sd:.3e}  ratio {d / sd:.2e}")
print(f"SMOKE_COMPARE_PASS worst ratio {worst:.2e}" if worst <= 1e-2 else f"SMOKE_COMPARE_WARN worst ratio {worst:.2e} (> 1e-2; information only)")
