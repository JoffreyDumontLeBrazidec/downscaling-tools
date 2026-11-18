import sys, numpy as np, netCDF4 as nc
sys.path.insert(0, "/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261117-realism-t2m/kit")
import realism_t2m as R
st = R.Stat()
d = nc.Dataset(f"{R.X}/pred/{R.ROW}_L024/predictions_20230226_step024.nc"); d.set_auto_mask(False)
im = R.W.index("msl"); N = 814810
for k in (0, 1):
    r = np.asarray(d["y_pred"][0, k, :, im], np.float64) - np.asarray(d["x_interp_members"][0, k, :, im], np.float64)
    h = r - st.lowpass(r); ok = np.isfinite(h)
    h = np.where(ok, h, 0.0)
    out = []
    for parts in (2, 4):
        L = N // parts
        seg = [h[i * L:(i + 1) * L] for i in range(parts)]
        out += [f"{parts}parts {i}-{j} r={np.corrcoef(seg[i], seg[j])[0,1]:+.4f}" for i in range(parts) for j in range(i + 1, parts)]
    # control: same-length shifted segments of the raw (not high-passed) field and the noise of two different draws
    print(f"draw {k+1}: hp std {h.std():.2f} Pa; " + "; ".join(out), flush=True)
a = np.asarray(d["y_pred"][0, 0, :, im], np.float64) - np.asarray(d["x_interp_members"][0, 0, :, im], np.float64)
b = np.asarray(d["y_pred"][0, 1, :, im], np.float64) - np.asarray(d["x_interp_members"][0, 1, :, im], np.float64)
ha, hb = a - st.lowpass(a), b - st.lowpass(b)
print(f"draw1 vs draw2 same points hp r={np.corrcoef(ha, hb)[0,1]:+.4f}")
print(f"lat ranges of halves: {st.lat[:N//2].min():.1f}-{st.lat[:N//2].max():.1f} | {st.lat[N//2:].min():.1f}-{st.lat[N//2:].max():.1f}")
