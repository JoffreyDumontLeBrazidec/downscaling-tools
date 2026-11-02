import json, numpy as np, zarr
D = "/e/data1/jureap-data/ecmwf/users/jdumont/datasets/o1280_o2560_europe_e2"
z = zarr.open(f"{D}/hres_o2560.zarr", "r"); a = dict(z.attrs)
ff = a.get("fake_forecasts")
s = json.dumps(ff)
print("fake_forecasts len", len(s)); print(s[:1500])
print("recipe", json.dumps(a.get("recipe"))[:2500])
print("description", a.get("description"))
print("history", json.dumps(a.get("history"))[:1500])
print("missing_dates", a.get("missing_dates")[:5] if a.get("missing_dates") else None)
