"""CPU tests of the T1d DP package (run: python test_dp.py, or pytest).

- DP against brute force on random matrices, and equal spacing on a convex additive cost;
- the trajectory_states writer (the real code of interp/tools/trajectory.py) on fake arrays,
  read back with common.Trajectory, plus the size estimate of the real diagnostic;
- the numpy piecewise schedules against the fork's scheduler, and the custom-schedule JSON
  accepted by the fork's CustomScheduler;
- capture_denoiser keeps the two-argument call when pass_input is off;
- cost_matrix -> dp_schedule -> lockin_read end to end on two fake draws (two dates).
"""
from __future__ import annotations

import itertools
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _harness import load_fork_samplers, load_trajectory_funcs  # noqa: E402
import common  # noqa: E402
import cost_matrix  # noqa: E402
import dp_schedule  # noqa: E402
import lockin_read  # noqa: E402


def test_dp_bruteforce():
    rng = np.random.default_rng(0)
    for N, K in ((9, 3), (10, 4), (12, 5), (11, 11)):
        C = np.triu(rng.random((N, N)) + 0.01, 1)
        C[np.tril_indices(N)] = np.nan
        path, val = dp_schedule.dp_path(C, K)
        best = min((dp_schedule.path_cost(C, [0, *mid, N - 1]), [0, *mid, N - 1])
                   for mid in itertools.combinations(range(1, N - 1), K - 2))
        assert abs(best[0] - val) < 1e-12 and best[1] == path, (N, K, best, path, val)


def test_dp_convex_equal_spacing():
    N = 241
    i = np.arange(N)
    C = (i[None, :] - i[:, None]).astype(float) ** 2
    C[np.tril_indices(N)] = np.nan
    path, _ = dp_schedule.dp_path(C, 9)                   # 8 intervals of 30
    assert path == list(range(0, 241, 30)), path


def fake_box(center=(25.0, 290.0), radius_km=300.0, res=0.07, seed=0):
    rng = np.random.default_rng(seed)
    lat0, lon0 = center
    dl = radius_km / common.KM_PER_DEG
    dlon = dl / math.cos(math.radians(lat0))
    la, lo = np.meshgrid(np.arange(lat0 - dl, lat0 + dl, res), np.arange(lon0 - dlon, lon0 + dlon, res), indexing="ij")
    la = la.ravel() + rng.normal(0, res / 10, la.size)
    lo = lo.ravel() + rng.normal(0, res / 10, lo.size)
    dist = common.KM_PER_DEG * np.hypot(la - lat0, (lo - lon0) * math.cos(math.radians(lat0)))
    keep = dist <= radius_km
    return la[keep], lo[keep]


def write_fake(path, write, sig, lat, lon, date, seed, rng, names=("10u", "10v", "2t", "msl", "tp")):
    """Fake but structured trajectory: x_i = D + sigma_i * eps with a smooth D path."""
    V, n = len(names), lat.size
    base = rng.normal(size=(V, n)).astype(np.float64)
    eps = rng.normal(size=(V, n))
    calls = []
    for i, s in enumerate(sig):
        t = i / (len(sig) - 1)
        D = base * t + 0.1 * rng.normal(size=(V, n))
        x = D + s * eps
        calls.append((float(s), x.astype(np.float32), D.astype(np.float32)))
        if i < len(sig) - 1:
            s2 = sig[i + 1]
            calls.append((float(s2), (D + s2 * eps).astype(np.float32), D.astype(np.float32)))
    final = calls[-1][2]
    write(path, calls, final, base.astype(np.float32), list(names), list(range(V)), lat, lon,
          np.arange(n), 1, n, meta={"seed": seed, "checkpoint": "fake", "bundle": f"x_date{date}_step024h.nc",
                                    "center_lat": 25.0, "center_lon": 290.0, "radius_km": 300.0,
                                    "local_scope": "", "noise_scheduler": "", "sampler_params": "",
                                    "num_steps": len(sig)})
    return calls


def test_writer_and_reader(tmp=None):
    fx = load_trajectory_funcs()
    tmp = Path(tmp or tempfile.mkdtemp())
    sig = common.dense_levels(12)
    lat, lon = fake_box()
    rng = np.random.default_rng(1)
    calls = write_fake(tmp / "d1" / "trajectory_states_s1000.npz", fx["_write_trajectory_states"], sig, lat, lon,
                       "20230826", 1000, rng)
    z = np.load(tmp / "d1" / "trajectory_states_s1000.npz")
    assert z["x_in"].shape == (23, 5, lat.size) and z["x_in"].dtype == np.float32
    assert z["D"].shape == (23, 5, lat.size)
    assert list(z["heun_eval"][:4]) == [1, 2, 1, 2] and z["heun_eval"][-1] == 1
    assert list(z["step_idx"][:4]) == [0, 0, 1, 1]
    assert str(z["format"]) == "trajectory_states/v1"
    tr = common.Trajectory(tmp / "d1" / "trajectory_states_s1000.npz")
    assert np.allclose(tr.sigma, sig, rtol=1e-12) and tr.x.shape == (12, 5, lat.size)
    assert tr.x2.shape == (11, 5, lat.size) and tr.seed == 1000
    assert np.allclose(tr.x[3], calls[6][1])
    # size of the real diagnostic: 479 calls x (x_in + D) x 5 vars x n_box cells x 4 bytes
    n_box = int(round(math.pi * 500.0 ** 2 / (510.07e6 / 6_599_680)))
    per_draw = 479 * 2 * 5 * n_box * 4
    measured = (tmp / "d1" / "trajectory_states_s1000.npz").stat().st_size
    expect_small = 23 * 2 * 5 * lat.size * 4
    assert 0.95 < measured / expect_small < 1.2, (measured, expect_small)
    return {"n_box_500km_o1280": n_box, "bytes_per_draw": per_draw, "GB_16_draws": 16 * per_draw / 1e9,
            "fake_file_bytes": measured, "fake_payload_bytes": expect_small}


def test_capture_two_arg_call():
    import torch
    fx = load_trajectory_funcs()
    seen = []

    class Inner:
        def fwd_with_preconditioning(self, x, y, sigma, mcg=None, gss=None):
            return {"out_hres": y["out_hres"] * 0.5}

    inner = Inner()
    y = {"out_hres": torch.ones(1, 1, 1, 4, 2)}
    s = {"out_hres": torch.full((1, 1, 1, 1, 1), 3.0)}
    with fx["capture_denoiser"](inner, lambda sig, D: seen.append((sig, float(D.sum())))):
        out = inner.fwd_with_preconditioning({}, y, s)
    assert seen == [(3.0, 4.0)] and float(out["out_hres"].sum()) == 4.0
    seen3 = []
    with fx["capture_denoiser"](inner, lambda sig, D, x: seen3.append(float(x.sum())), pass_input=True):
        inner.fwd_with_preconditioning({}, y, s)
    assert seen3 == [8.0]
    assert "fwd_with_preconditioning" not in vars(inner)   # restored to the class method


def test_schedules_match_fork():
    import torch
    ds, _ = load_fork_samplers()
    for name, kw in common.REFERENCE_SCHEDULES.items():
        sch = ds.ExperimentalSamplerScheduler(sigma_max=kw["sigma_max"], sigma_min=kw["sigma_min"],
                                              num_steps=kw["n_high"] + kw["n_low"], sigma_transition=10.0,
                                              num_steps_high=kw["n_high"], num_steps_low=kw["n_low"], rho=7.0)
        ref = sch.get_schedule(None, torch.float64).numpy()[:-1]
        mine = common.reference_levels(name)
        assert np.allclose(ref, mine, rtol=1e-12), (name, ref, mine)
    blk = common.custom_scheduler_json(common.dense_levels(240))
    kw = {k: v for k, v in blk.items() if k != "schedule_type"}
    got = ds.NOISE_SCHEDULERS[blk["schedule_type"]](**kw).get_schedule(None, torch.float64).numpy()
    assert got.shape == (241,) and got[-1] == 0.0 and got[0] == 1e5 and abs(got[-2] - 0.03) < 1e-15
    # the trajectory tool sets num_steps and sigma_min=0.03 itself; both must agree with the list
    ds.NOISE_SCHEDULERS["custom"](**dict(kw, num_steps=240, sigma_min=0.03))


def test_end_to_end(tmp=None):
    fx = load_trajectory_funcs()
    tmp = Path(tmp or tempfile.mkdtemp())
    sig = common.dense_levels(12)
    lat, lon = fake_box()
    rng = np.random.default_rng(2)
    files = []
    for date, seeds in (("20230826", (1000, 1001)), ("20230828", (1020, 1021))):
        for sd in seeds:
            p = tmp / f"d{date}_l024" / f"trajectory_states_s{sd}.npz"
            write_fake(p, fx["_write_trajectory_states"], sig, lat, lon, date, sd, rng)
            files.append(str(p))
    cost_matrix.main(["--inputs", str(tmp / "d*" / "trajectory_states_s*.npz"), "--out-dir", str(tmp / "cost")])
    for s in ("all", "20230826", "20230828"):
        z = np.load(tmp / "cost" / f"cost_mean_{s}.npz")
        C = z["C_band_heun_ref"]
        assert C.shape == (12, 12) and np.all(np.isfinite(C[np.triu_indices(12, 1)]))
        assert np.all(C[np.triu_indices(12, 1)] >= 0)
    dp_schedule.main(["--cost-dir", str(tmp / "cost"), "--out-dir", str(tmp / "dp"), "--budgets", "4", "6", "8"])
    res = json.load(open(tmp / "dp" / "dp_schedules.json"))
    d = res["schedules"]["dp_C_band_heun_ref_K6_fit20230826"]
    assert d["calls"] == 11 and d["num_steps"] == 6 and d["holdout_sets"] == ["20230828"]
    assert d["noise_scheduler"]["schedule_type"] == "custom" and len(d["noise_scheduler"]["sigmas"]) == 6
    assert d["noise_scheduler"]["sigmas"][0] == 1e5 and d["noise_scheduler"]["sigmas"][-1] == 0.03
    assert (tmp / "dp" / "dp_summary.md").exists() and (tmp / "dp" / "c0_30_running_cost.json").exists()
    lockin_read.main(["--from-states", "--inputs", *files, "--out", str(tmp / "lockin.json")])
    lk = json.load(open(tmp / "lockin.json"))
    assert lk["n_draws"] == 4 and set(lk["vars"]) == {"10u", "10v", "2t", "msl", "tp"}
    return tmp


if __name__ == "__main__":
    test_dp_bruteforce()
    test_dp_convex_equal_spacing()
    print("dp: brute-force and equal-spacing OK")
    size = test_writer_and_reader()
    print("writer/reader OK; size:", json.dumps(size))
    test_capture_two_arg_call()
    print("capture_denoiser OK (2-arg default, 3-arg with pass_input, method restored)")
    test_schedules_match_fork()
    print("piecewise schedules match the fork; custom JSON accepted by the fork's CustomScheduler")
    out = test_end_to_end()
    print("end to end OK:", out)
    print((out / "dp" / "dp_summary.md").read_text()[:1500])
