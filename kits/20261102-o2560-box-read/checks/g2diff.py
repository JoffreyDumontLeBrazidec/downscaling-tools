import netCDF4, numpy as np
S="/e/data1/jureap-data/ecmwf/users/jdumont/outputs/o2560_box_read_20261102/smoke_2645931"
def L(p):
    d=netCDF4.Dataset(p); d.set_auto_mask(False); r=([str(s) for s in d["weather_state"][:]], np.asarray(d["y_pred"][0],np.float64), np.asarray(d["y"][0],np.float64)); d.close(); return r
st,a,y=L(f"{S}/B24/predictions_20241101_step024.nc"); _,g,_=L(f"{S}/B24_g2/predictions_20241101_step024.nc")
for j,v in enumerate(st):
    d=np.abs(g[0,:,j]-a[0,:,j]); sd=y[:,j].std(); m12=np.abs(a[0,:,j]-a[1,:,j])
    print(f"{v:6s} 2v4 ranks: mean|d|/sd {d.mean()/sd:.2e} p99.99|d|/sd {np.percentile(d,99.99)/sd:.2e} max|d|/sd {d.max()/sd:.2e} points |d|>1e-3sd {int((d>1e-3*sd).sum())} of {d.size} | m1 vs m2 mean|d|/sd {m12.mean()/sd:.2e}")
