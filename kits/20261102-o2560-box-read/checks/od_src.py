import json, numpy as np, zarr
P = "/e/data1/jureap-data/ecmwf/ai-ml/datasets/downscaling-od-fc-oper-0001-mars-o1280-2023-2024-24h-v1.zarr"
z = zarr.open(P, "r"); a = dict(z.attrs)
print(z["data"].shape, z["data"].chunks, z["data"].dtype)
v = a["variables"]; print(len(v), v[:12], "tp" in v, "cp" in v, v.index("tp") if "tp" in v else None)
print({k: a[k] for k in ("start_date", "end_date", "frequency") if k in a})
ff = a.get("fake_forecasts"); print("fake_forecasts", type(ff), (list(ff.items())[:3] if isinstance(ff, dict) else str(ff)[:300]))
d = np.asarray(z["dates"][:]); print("dates", d[:3], d[-3:], len(d))
print("tp meta", json.dumps(a.get("variables_metadata", {}).get("tp"))[:400])
print("keys", list(a))
