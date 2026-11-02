"""o2560 box read smoke: max |difference| per variable between two prediction files of the same case and seeds
(B24 on code 20ba15241 against B24 on code 4ad32838c). Prints SMOKE_COMPARE_PASS when the seeds and truth agree and the
predictions differ by at most 1e-3 of the truth's standard deviation (float noise of chunked kernels), else FAIL."""
import json, sys
import netCDF4
import numpy as np

a, b = sys.argv[1], sys.argv[2]
with netCDF4.Dataset(a) as A, netCDF4.Dataset(b) as B:
    A.set_auto_mask(False); B.set_auto_mask(False)
    st = [str(s) for s in A["weather_state"][:]]
    assert st == [str(s) for s in B["weather_state"][:]]
    assert json.loads(A.seeds) == json.loads(B.seeds), "seeds differ"
    assert np.array_equal(A["y"][:], B["y"][:]), "truth differs"
    ya, yb, y = np.asarray(A["y_pred"][:]), np.asarray(B["y_pred"][:]), np.asarray(A["y"][:])
ok = True
for j, s in enumerate(st):
    d = float(np.abs(ya[..., j] - yb[..., j]).max()); sd = float(y[..., j].std())
    ok &= d <= 1e-3 * sd
    print(f"{s:6s} max|diff| {d:.3e}  truth std {sd:.3e}  ratio {d / sd:.2e}")
print("SMOKE_COMPARE_PASS" if ok else "SMOKE_COMPARE_FAIL")
sys.exit(0 if ok else 1)
