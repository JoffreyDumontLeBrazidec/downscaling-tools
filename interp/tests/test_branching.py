"""Tests of the U3 branching driver and read on a dummy denoiser (no checkpoint, no GPU).

The driver tests use the REAL ladder and Heun sampler classes of anemoi.models (pristine core), with
a Gaussian posterior-mean denoiser D(y, s) = (y + s^2 m) / (1 + s^2) (prior N(m, 1) per point), for
which the share still undecided at noise level s is known exactly: s^2 / (1 + s^2).

    python -m pytest interp/tests/test_branching.py -q
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("anemoi.models.samplers.diffusion_samplers")

from interp.tools import branching as B  # noqa: E402
from interp.tools import branching_read as R  # noqa: E402

XDTYPE = torch.float32


def _setup(n_points=400, n_vars=3, seed=0):
    nsc, sc = B.merged_sampler_config(None, B.PW30_C0_NOISE_SCHEDULER, B.PW30_C0_SAMPLER)
    sigmas, sampler = B.build_schedule_and_sampler(nsc, sc, torch.device("cpu"))
    g = torch.Generator().manual_seed(seed)
    m = torch.randn((1, 1, 1, n_points, n_vars), generator=g, dtype=XDTYPE)
    x = {"in_lres": m, "in_hres": torch.zeros_like(m)}
    calls = []

    def fn(x, y, sigma, mcg=None, gss=None):
        s = sigma["out_hres"].to(torch.float64)
        yy = y["out_hres"]
        calls.append(float(s.reshape(-1)[0]))
        d = (yy.to(torch.float64) + s ** 2 * x["in_lres"].to(torch.float64)) / (1.0 + s ** 2)
        return {"out_hres": d.to(yy.dtype)}

    shape = (1, 1, 1, n_points, n_vars)
    return sigmas, sampler, x, fn, calls, shape


def test_ladder_is_the_lane_pw30():
    sigmas, sampler, *_ = _setup()
    s = sigmas.numpy()
    assert len(s) == 31 and s[-1] == 0.0
    assert s[0] == pytest.approx(1e5) and s[10] == pytest.approx(10.0) and s[29] == pytest.approx(0.03)
    assert float(sampler.S_churn) == 0.0


def test_branch_nodes_and_call_count():
    sigmas, *_ = _setup()
    nodes = B.branch_nodes(sigmas, 10, B.DEFAULT_BRANCH_SIGMAS)
    assert [n[0] for n in nodes[:10]] == list(range(10))
    assert len(nodes) == 15
    low = nodes[10:]
    for (i, s, lab), t in zip(low, B.DEFAULT_BRANCH_SIGMAS):
        assert i >= 10 and abs(math.log(s / t)) < 0.25, (i, s, t)
    assert B.calls_from_node(0, 30) == 59
    with pytest.raises(ValueError):
        B.branch_nodes(sigmas, 10, [5.0, 5.01])


def test_segments_chain_bit_for_bit():
    sigmas, sampler, x, fn, calls, shape = _setup()
    torch.manual_seed(1)
    y0 = torch.randn(shape, dtype=torch.float64) * sigmas[0]
    full = sampler.sample(x, {"out_hres": y0}, sigmas, fn, dtype=XDTYPE)["out_hres"]
    drv = B.Branching(sampler, sigmas, x, fn, XDTYPE)
    y, cur = y0, 0
    for i in (3, 10, 13, 17, 22, 26, 30):
        y = drv.segment(y, cur, i)
        cur = i
    assert torch.equal(full, y)
    assert full.dtype == XDTYPE


def test_branch_denoise_equals_trunk_call():
    """D(x_k, sigma_k) of a branch is the trunk's own first call at node k."""
    sigmas, sampler, x, fn, calls, shape = _setup()
    torch.manual_seed(2)
    y0 = torch.randn(shape, dtype=torch.float64) * sigmas[0]
    drv = B.Branching(sampler, sigmas, x, fn, XDTYPE)
    y13 = drv.segment(y0, 0, 13)
    d = drv.denoise(y13, 13)
    first = {}

    def rec(x_, y_, s_, *a):
        out = fn(x_, y_, s_, *a)
        first.setdefault("d", out["out_hres"].clone())
        return out

    sampler.sample(x, {"out_hres": y13}, sigmas[13:15], rec, dtype=XDTYPE)
    assert torch.equal(d, first["d"])


def test_churn_is_refused():
    nsc, sc = B.merged_sampler_config(None, B.PW30_C0_NOISE_SCHEDULER, dict(B.PW30_C0_SAMPLER, S_churn=2.5))
    sigmas, sampler = B.build_schedule_and_sampler(nsc, sc, torch.device("cpu"))
    with pytest.raises(ValueError):
        B.Branching(sampler, sigmas, {}, None, XDTYPE)


def _run(n_branch, n_full, shape_kw=None, done=None, store=None):
    sigmas, sampler, x, fn, calls, shape = _setup(**(shape_kw or {}))
    nodes = B.branch_nodes(sigmas, 10, B.DEFAULT_BRANCH_SIGMAS)
    store = {} if store is None else store

    def noise(kind, node, j):
        torch.manual_seed(B.draw_seed(4242, kind, node, j))
        return torch.randn(shape, dtype=torch.float64)

    def sink(kind, node, j, y):
        store[B.draw_name(kind, node, j)] = y.clone()

    drv = B.Branching(sampler, sigmas, x, fn, XDTYPE)
    drv.run(noise, nodes, n_branch, n_full, sink, done)
    return drv, nodes, store, sigmas, sampler, x, fn, shape


def test_run_outputs_calls_and_trunk_equals_model_sample():
    drv, nodes, store, sigmas, sampler, x, fn, shape = _run(2, 2)
    assert len(store) == 15 * 2 + 1 + 2
    assert drv.n_calls == B.expected_calls(nodes, 30, 2, 2, verify=False)
    # the trunk equals what model.sample does with the same global seed
    torch.manual_seed(B.draw_seed(4242, "trunk", -1, 0))
    y_init = torch.randn(shape, dtype=torch.float64) * sigmas[0]
    ref = sampler.sample(x, {"out_hres": y_init}, sigmas, fn, dtype=XDTYPE)["out_hres"]
    assert torch.equal(store["trunk"], ref)
    # a branch at node 0 is a full draw from D(x_0, sigma_max) + sigma_max eps'
    assert not torch.equal(store["branch_n00_j00"], store["trunk"])


def test_resume_skips_and_reproduces():
    _, _, first, *_ = _run(2, 1)
    keep = {k: v for k, v in first.items() if k.startswith("branch_n00") or k == "full_j00"}
    store = dict(keep)
    drv, nodes, second, *_ = _run(2, 1, done=lambda k, n, j: B.draw_name(k, n, j) in keep, store=store)
    for k in first:
        assert torch.equal(first[k], second[k]), k
    assert drv.n_calls < B.expected_calls(nodes, 30, 2, 1, verify=False)


def test_gaussian_undecided_share_matches_theory():
    """Per point, var over eps' at node k / var over full draws -> s_k^2 / (1 + s_k^2)."""
    n_b = n_f = 48
    drv, nodes, store, sigmas, *_ = _run(n_b, n_f, shape_kw={"n_points": 3000, "n_vars": 1, "seed": 3})
    full = torch.stack([store[f"full_j{j:02d}"] for j in range(n_f)]).double().reshape(n_f, -1)
    vf = full.var(dim=0, unbiased=True).mean().item()
    assert vf == pytest.approx(1.0, rel=0.08)          # the sampler reproduces the prior N(m, 1)
    for i, s, _ in nodes:
        br = torch.stack([store[f"branch_n{i:02d}_j{j:02d}"] for j in range(n_b)]).double().reshape(n_b, -1)
        share = br.var(dim=0, unbiased=True).mean().item() / vf
        theory = s * s / (1.0 + s * s)
        assert share == pytest.approx(theory, abs=0.06), (i, s, share, theory)


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------

def test_monotone_fit():
    s = [100.0, 10.0, 1.0, 0.1]
    fit = R.monotone_fit(s, [0.9, 1.1, 0.3, 0.4])
    assert np.allclose(fit, [1.0, 1.0, 0.35, 0.35])


def test_sigma_star_interpolation_and_flags():
    s = [100.0, 10.0, 1.0, 0.1]
    assert R.sigma_star(s, [1.0, 0.9, 0.3, 0.1]) == (pytest.approx(math.exp(math.log(10) + 0.4 / 0.6 * math.log(0.1))), "crossed")
    assert R.sigma_star(s, [0.4, 0.3, 0.2, 0.1])[1] == "above"
    assert R.sigma_star(s, [1.0, 0.9, 0.8, 0.7])[1] == "below"


PST = Path(R.__file__).resolve().parent / "_box_instrument" / "plot_sampler_texture.py"


@pytest.mark.skipif(not PST.exists(), reason="box instrument copy missing")
def test_read_end_to_end_on_synthetic_case(tmp_path):
    """Synthetic storm case: draws = truth + a(sigma) x smooth noise, a^2 = s^2 / (1 + s^2); the read must
    find every feature's sigma* near 1 (where a^2 = 0.5) for features that respond smoothly to the noise."""
    rng = np.random.default_rng(0)
    lat1 = np.arange(18.0, 32.0, 0.07)
    lon1 = np.arange(-76.0, -60.0, 0.07)
    LON, LAT = np.meshgrid(lon1, lat1)
    lat, lon = LAT.ravel(), LON.ravel()
    r2 = ((lat - 25.0) ** 2 + ((lon + 68.0) * math.cos(math.radians(25))) ** 2) / 1.0 ** 2
    msl = 101000.0 - 3000.0 * np.exp(-r2)
    dy, dx = lat - 25.0, (lon + 68.0) * math.cos(math.radians(25))
    sp = 40.0 * np.sqrt(r2) * np.exp(-r2) / (np.sqrt(r2) + 1e-6)
    u, v = -sp * dy / (np.hypot(dx, dy) + 1e-6), sp * dx / (np.hypot(dx, dy) + 1e-6)
    t2 = 300.0 - 0.5 * (lat - 25.0) + np.sin(lon)
    truth = np.stack([u, v, t2, msl], axis=1).astype(np.float32)
    xi = truth.copy()
    xi[:, 2] = 300.0 - 0.5 * (lat - 25.0)
    case = tmp_path / "root" / "synthetic_case"
    (case / "finals").mkdir(parents=True)
    sig = [1e5, 3e4, 1e4, 3e3, 1e3, 300.0, 100.0, 30.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.25, 0.1]
    np.savez(case / "case.npz", lat=lat, lon=lon, truth=truth, x_interp=xi, var_names=np.asarray(["10u", "10v", "2t", "msl"]),
             sigmas=np.asarray(sig + [0.0]), node_index=np.arange(len(sig)), node_sigma=np.asarray(sig),
             node_label=np.asarray([f"n{i}" for i in range(len(sig))]))
    from scipy.ndimage import gaussian_filter

    def noise():
        f = gaussian_filter(rng.standard_normal(LAT.shape), 3.0).ravel()
        f /= f.std()
        return np.stack([2.0 * f, -2.0 * f, 0.5 * f, 150.0 * f], axis=1)

    def draw(a):
        return (truth + a * noise()).astype(np.float32)

    n = 24
    for j in range(n):
        np.save(case / "finals" / f"full_j{j:02d}.npy", draw(1.0))
    np.save(case / "finals" / "trunk.npy", draw(1.0))
    for i, s in enumerate(sig):
        a = s / math.sqrt(1.0 + s * s)
        for j in range(n):
            np.save(case / "finals" / f"branch_n{i:02d}_j{j:02d}.npy", draw(a))
    out = tmp_path / "read"
    assert R.main(["--root", str(tmp_path / "root"), "--out", str(out), "--pst", str(PST), "--n-boot", "50"]) == 0
    stars = {r["feature"]: r for r in __import__("csv").DictReader(open(out / "sigma_star.csv"))}
    assert "storm_pmin_hpa" in stars and "grad2t_p99_K_per_100km" in stars
    assert any(k.startswith("amp_resid_msl_sub") for k in stars) and any(k.startswith("coh_2t_") for k in stars)
    for k in ("storm_pmin_hpa", "amp_resid_msl_sub40-60km", "amp_resid_2t_sub60-100km"):
        s_star = float(stars[k]["sigma_star_monotone"])
        assert stars[k]["flag_monotone"] == "crossed" and 0.2 < s_star < 5.0, (k, stars[k])
    md = (out / "sigma_star.md").read_text()
    assert "Storm features" in md and "Per wavelength bin" in md
