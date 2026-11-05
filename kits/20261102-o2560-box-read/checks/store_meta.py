import json, sys, numpy as np, zarr
D = "/e/data1/jureap-data/ecmwf/users/jdumont/datasets/o1280_o2560_europe_e2"
for name in ("in_lres_o1280.zarr", "hres_o2560.zarr"):
    z = zarr.open(f"{D}/{name}", "r")
    a = dict(z.attrs)
    print("==", name, z["data"].shape, z["data"].chunks)
    print("variables", a.get("variables"))
    print({k: a[k] for k in a if k in ("start_date", "end_date", "frequency", "resolution", "field_shape")})
    vm = a.get("variables_metadata", {})
    for v in ("tp", "cp"):
        print(v, json.dumps(vm.get(v, {}))[:500])
    print("keys", list(a.keys()))
    d = np.asarray(z["dates"][:])
    print("dates", d[:3], d[-3:], len(d))
    for k in ("latitudes", "longitudes"):
        x = np.asarray(z[k][:]); print(k, x.size, float(x.min()), float(x.max()))
