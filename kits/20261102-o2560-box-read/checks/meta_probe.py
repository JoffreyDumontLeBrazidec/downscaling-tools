import json, sys, zipfile
p = sys.argv[1]
with zipfile.ZipFile(p) as z:
    names = [n for n in z.namelist() if n.endswith(".json")]
    print(names[:10])
    md = json.loads(z.read([n for n in names if n.endswith("anemoi.json")][0]))
print(list(md.keys()))
di = md.get("data_indices", {}); print("data_indices keys", list(di.keys())[:10])
s = json.dumps(di)
i = s.find("out_hres"); print(s[i:i+1500])
ds = md.get("dataset", {}); print("dataset keys", list(ds.keys())[:20] if isinstance(ds, dict) else type(ds))
print(json.dumps(ds)[:800])
