import json, numpy as np, zarr, glob
D = "/e/data1/jureap-data/ecmwf/users/jdumont/datasets"
z = zarr.open(f"{D}/o1280_o2560_europe_e2/in_lres_o1280.zarr", "r"); a = dict(z.attrs)
r = a.get("recipe", {}); print("in_lres recipe dates", r.get("dates")); print("in_lres input", json.dumps(r.get("input"))[:800]); print("regional", r.get("regional_materialization"))
for p in sorted(glob.glob(f"{D}/o320_o1280_europe_e2/*.zarr")):
    z = zarr.open(p, "r"); a = dict(z.attrs); r = a.get("recipe", {})
    print("==", p, z["data"].shape, "vars", a.get("variables")[-14:], "dates", r.get("dates", {}).get("start"), r.get("dates", {}).get("end"),
          "src", (r.get("regional_materialization") or {}).get("source"))
