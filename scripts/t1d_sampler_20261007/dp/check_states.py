"""Smoke check of one trajectory_states_s<seed>.npz (and its trajectory.json): contents,
call bookkeeping against the requested schedule, finiteness, size, and the projected
size of the full diagnostic.

  python -m scripts.t1d_sampler_20261007.dp.check_states <out-dir> --schedule dense12.json \
      --project-levels 240 --project-draws 16
Exit code 1 on any failed check.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import IDALIA_WINDOWS, box_checks  # type: ignore
else:
    from .common import IDALIA_WINDOWS, box_checks


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--schedule", required=True, help="the noise-scheduler JSON that was passed")
    ap.add_argument("--project-levels", type=int, default=240)
    ap.add_argument("--project-draws", type=int, default=16)
    ap.add_argument("--window", default=None,
                    help="lat0,lat1,lon0,lon1 the centre must lie strictly inside (default: the Idalia "
                         "window of the bundle's date and lead, common.IDALIA_WINDOWS)")
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    sched = json.loads(Path(a.schedule).read_text())["sigmas"]
    L = len(sched)
    files = sorted(out.glob("trajectory_states_s*.npz"))
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + msg)
        ok &= bool(cond)

    check(len(files) > 0, f"{len(files)} trajectory_states files in {out}")
    for f in files:
        z = np.load(f)
        n = z["x_in"].shape[0]
        V, N = z["x_in"].shape[1], z["x_in"].shape[2]
        print(f"{f.name}: keys {sorted(z.files)}")
        print(f"  vars {[str(v) for v in z['vars']]}, cells {N} of {int(z['n_box_full'])} (stride {int(z['stride'])}), "
              f"dtype {z['x_in'].dtype}, {f.stat().st_size / 1e6:.1f} MB")
        check(n == 2 * L - 1, f"calls {n} == 2*{L}-1")
        sig = z["sigma"]
        check(np.allclose(sig[0::2], sched, rtol=1e-6), "first-evaluation sigmas == the custom schedule (fp32 tol)")
        check(np.allclose(sig[1::2], sched[1:], rtol=1e-6), "second-evaluation sigmas == next level (churn off)")
        check(set(np.unique(z["heun_eval"]).tolist()) == {1, 2}, "heun_eval in {1,2}")
        for k in ("x_in", "D", "final", "truth_residual"):
            check(bool(np.all(np.isfinite(z[k]))), f"{k} finite")
        check(abs(float(np.abs(z["x_in"][0]).std()) / float(sched[0]) - 0.6) < 0.2,
              f"call-0 input ~ sigma_max * N(0,1): std(|x|)/sigma_max = {float(np.abs(z['x_in'][0]).std()) / sched[0]:.3f}")
        d_last, fin = z["D"][-1].astype(np.float64), z["final"].astype(np.float64)
        rel = float(np.abs(d_last - fin).max() / max(np.abs(fin).max(), 1e-12))
        check(rel < 1e-3, f"final state == last D (Euler to 0): max rel diff {rel:.2e}")
        for v, name in enumerate(z["vars"]):
            print(f"  {name}: std final {fin[v].std():.3f}  std truth residual {z['truth_residual'][v].std():.3f}  "
                  f"lat {z['lat'].min():.2f}..{z['lat'].max():.2f} lon {z['lon'].min():.2f}..{z['lon'].max():.2f}")
        meta = {k: str(z[k]) for k in z.files if k.startswith("meta_") and k != "meta_noise_scheduler"}
        print(f"  meta {meta}")
        clat, clon, rad = float(z["meta_center_lat"]), float(z["meta_center_lon"]), float(z["meta_radius_km"])
        m = re.search(r"date(\d{8}).*step(\d{3})h", Path(str(z["meta_bundle"])).name)
        win = (tuple(float(x) for x in a.window.split(",")) if a.window
               else IDALIA_WINDOWS.get((m.group(1), m.group(2))) if m else None)
        print(f"  box centre {clat:.2f}N {clon:.2f}E, radius {rad:.0f} km; cells lat {z['lat'].min():.2f}..{z['lat'].max():.2f}"
              f" lon {z['lon'].min():.2f}..{z['lon'].max():.2f}; expected window {win}")
        check(win is not None, "an expected Idalia window is known for this bundle")
        for good, msg in box_checks(clat, clon, rad, win):
            check(good, msg)
        cells_full = int(z["n_box_full"]) * 0.85 < np.pi * rad ** 2 / 77.3 < int(z["n_box_full"]) * 1.15
        check(cells_full, f"disc not truncated: {int(z['n_box_full'])} cells vs about {np.pi * rad ** 2 / 77.3:.0f} expected")
        per_draw = (2 * a.project_levels - 1) * 2 * V * N * 4
        print(f"  projection: {a.project_levels} levels -> {per_draw / 1e6:.0f} MB per draw, "
              f"{a.project_draws * per_draw / 1e9:.2f} GB for {a.project_draws} draws (cap 30 GB)")
        check(a.project_draws * per_draw < 30e9, "projected total under 30 GB")
    tj = out / "trajectory.json"
    if tj.exists():
        d = json.load(open(tj))
        nso = d.get("noise_scheduler_override") or {}
        check(nso.get("schedule_type") == "custom", f"trajectory.json schedule_type {nso.get('schedule_type')}")
        check(d.get("local_scope") is not None, f"cut graph recorded: {d.get('local_scope')}")
        lk = [t for t in d["trajectories"] if t.get("lockin")]
        check(len(lk) == len(files), f"lock-in payload in {len(lk)} trajectories")
        print(f"  box {d['box']}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
