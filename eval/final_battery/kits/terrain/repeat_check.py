import os, sys, numpy as np, netCDF4 as nc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import realism_t2m as R
st = R.Stat()
# usage: repeat_check.py [DATE] [LEAD]  (default 20230226 24, the 17 Nov case)
DATE = sys.argv[1] if len(sys.argv) > 1 else "20230226"; LEAD = int(sys.argv[2]) if len(sys.argv) > 2 else 24
d = nc.Dataset(f"{R.X}/pred/{R.ROW}_L{LEAD:03d}/predictions_{DATE}_step{LEAD:03d}.nc"); d.set_auto_mask(False)
im = R.states(d).index("msl"); N = st.lat.size
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
