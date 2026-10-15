"""Branching read (U3, TD-221): at which noise level sigma is each feature of a field decided?

For one case (one input/truth pair) this runs the production sampler once (the TRUNK) and,
at chosen nodes k of its sigma ladder, BRANCHES it: the trunk state x_k is replaced by
D(x_k, sigma_k) + sigma_k * eps' for n_branch fresh standard normals eps', and each branch is
finished from sigma_k down to 0 with the same sampler and the same ladder. It also runs n_full
independent full draws from pure noise. The final fields (physical units, the box points of
the cut graph, a few surface variables) are written one .npy per draw; the CPU read
``interp.tools.branching_read`` turns them into features and the share of each feature's
variance that is still undecided at sigma_k:

    share(F, k) = var_eps'(F | branches at k) / var(F | independent full draws)

How the trajectory is driven. The model's own ``sample`` builds the ladder and the sampler
and calls ``sampler.sample(x, y, sigmas, fwd_with_preconditioning, dtype=x.dtype)``. Here the
same scheduler and sampler classes are built from the same merged configuration
(``inference_defaults`` updated with the lane block, as ``predict_step`` and ``sample`` do), and
``sampler.sample`` is called on SEGMENTS of the ladder: with churn off the Heun loop draws no
random numbers and does the same operations per step, so running nodes [a..b] then [b..c]
is the same trajectory as running [a..c] (checked bit for bit in interp/tests/test_branching.py
on a dummy denoiser, and on the GPU against ``predict_step`` itself with ``--verify-production``).
Churn > 0 is refused: the per-step churn factor depends on the segment length and draws noise.

The production check (``--verify-production``, on by default): with the same global torch seed,
the trunk's final physical field must equal ``inference_model.predict_step``'s output (the eval
predict route's own call). The difference is printed and stored in meta.json.

Inputs. ``--bundle`` reads one eval bundle (eefo O320 member input + O1280 target) exactly as
``manual_inference.prediction.predict.predict_from_bundle`` does. ``--dataset-case YYYYMMDD:STEP``
reads the row of the checkpoint's own hindcast stores on Atos (fake-hindcast mapping of the
fork anemoi-datasets), for cases with no bundle (the Alps date 2023-02-26 has no eefo forecast:
the extended ensemble ran on Mondays and Thursdays before cycle 48r1).

Usage (one case per call; see the U3 job files):
    python -m interp branching --checkpoint CKPT --case-label franklin_20230829_024 \
        --bundle .../eefo_o320_0001_date20230829_time0000_mem01_step024h_input_bundle.nc \
        --local-scope-json '{"mode": "bbox", "cut_graph": true, ...}' --out-dir OUT
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_DT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_DT_ROOT) not in sys.path:
    sys.path.insert(0, str(_DT_ROOT))

LOG = logging.getLogger("interp.branching")

DEFAULT_BRANCH_SIGMAS = (5.0, 2.0, 1.0, 0.5, 0.25)
DEFAULT_SAVE_VARS = ("10u", "10v", "2t", "msl", "tp")
# The 30-step churn-off piecewise sampler of the o320_o1280 lane (sigma_max 1e5, 10 exponential
# high steps to sigma 10, 20 Karras low steps to 0.03, rho 7, Heun): the diffusion baseline of
# handoff-standing-rules rule 6 (59 network calls). S_noise is inert at churn 0.
PW30_C0_NOISE_SCHEDULER = {
    "schedule_type": "experimental_piecewise", "num_steps": 30, "sigma_max": 100000.0,
    "sigma_transition": 10.0, "sigma_min": 0.03, "high_schedule_type": "exponential",
    "low_schedule_type": "karras", "num_steps_high": 10, "num_steps_low": 20, "rho": 7.0,
}
PW30_C0_SAMPLER = {"sampler": "heun", "S_churn": 0.0, "S_min": 0.75, "S_max": 100000.0, "S_noise": 1.05}
# The checkpoint's hindcast stores (Jupiter paths in its config) mirrored on Atos.
ATOS_DATASETS = {
    "in_lres": "/home/mlx/ai-ml/datasets/downscaling-od-cf-eefh-0001-mars-o320-2003-2023-12h-v3.zarr",
    "in_hres": "/home/mlx/ai-ml/datasets/downscaling-od-cf-enfh-0001-mars-o1280-2003-2023-12h-v3-forcings.zarr",
    "out_hres": "/home/mlx/ai-ml/datasets/downscaling-od-cf-enfh-0001-mars-o1280-2003-2023-12h-v3.zarr",
}
# Seed offsets inside one case (added to --seed-base). The trunk uses seed_base itself.
SEED_FULL = 1_000
SEED_BRANCH = 100_000


# ---------------------------------------------------------------------------
# pure pieces (no model; unit-tested on a dummy denoiser)
# ---------------------------------------------------------------------------

def merged_sampler_config(inference_defaults, noise_scheduler_params=None, sampler_params=None):
    """The configuration ``predict_step`` + ``sample`` end up with: inference_defaults updated with
    the overrides (dict copies; nothing is mutated)."""
    nsc = dict(inference_defaults.noise_scheduler) if inference_defaults is not None else {}
    sc = dict(inference_defaults.diffusion_sampler) if inference_defaults is not None else {}
    nsc.update(noise_scheduler_params or {})
    sc.update(sampler_params or {})
    return nsc, sc


def build_schedule_and_sampler(nsc: dict, sc: dict, device):
    """Ladder (float64, terminal 0) and sampler instance, built as the model's ``sample`` builds them."""
    import torch
    from anemoi.models.samplers import diffusion_samplers as ds

    nsc, sc = dict(nsc), dict(sc)
    stype = nsc.pop("schedule_type")
    sigmas = ds.NOISE_SCHEDULERS[stype](**nsc).get_schedule(device, torch.float64)
    sname = sc.pop("sampler")
    sampler = ds.DIFFUSION_SAMPLERS[sname](dtype=sigmas.dtype, **sc)
    return sigmas, sampler


def branch_nodes(sigmas, n_high: int, extra_sigmas) -> list[tuple[int, float, str]]:
    """Ladder nodes to branch at: the first ``n_high`` nodes (the high segment, sigma_max down to just
    above the transition) and, for each target in ``extra_sigmas``, the low-segment node nearest to it
    in log sigma. Returns sorted (node index, actual sigma, label)."""
    s = np.asarray(sigmas.detach().double().cpu().numpy() if hasattr(sigmas, "detach") else sigmas,
                   dtype=np.float64)
    n_steps = len(s) - 1
    nodes: dict[int, str] = {i: f"high{i:02d}" for i in range(min(int(n_high), n_steps))}
    low = np.arange(min(int(n_high), n_steps), n_steps)
    for t in extra_sigmas or ():
        if low.size == 0:
            raise ValueError("no low-segment node to branch at")
        i = int(low[np.argmin(np.abs(np.log(s[low]) - math.log(float(t))))])
        if i in nodes:
            raise ValueError(f"target sigma {t} maps to node {i} (sigma {s[i]:.4g}), already a branch node "
                             f"({nodes[i]}); pick distinct targets")
        nodes[i] = f"near{float(t):g}"
    return [(i, float(s[i]), nodes[i]) for i in sorted(nodes)]


def calls_from_node(i: int, n_steps: int) -> int:
    """Denoiser calls of a Heun trajectory from node i to the end (the last step, to sigma 0, is Euler)."""
    return 2 * (n_steps - i) - 1


def expected_calls(nodes, n_steps: int, n_branch: int, n_full: int, verify: bool) -> int:
    """Total denoiser calls of one case (trunk + one D per branch node + branches + full draws (+ check))."""
    trunk = calls_from_node(0, n_steps)
    br = sum(1 + n_branch * calls_from_node(i, n_steps) for i, _, _ in nodes)
    return trunk + br + n_full * trunk + (trunk if verify else 0)


class Branching:
    """Drives one case: trunk, branches, full draws, on the library sampler called per ladder segment."""

    def __init__(self, sampler, sigmas, x: dict, denoise_fn, xdtype, *, model_comm_group=None,
                 grid_shard_sizes=None, dataset="out_hres"):
        churn = float(getattr(sampler, "S_churn", 0.0) or 0.0)
        if churn > 0.0:
            raise ValueError(f"branching needs a churn-off sampler (S_churn={churn}): the churn factor depends "
                             "on the segment length and churn draws noise, so segments would not chain")
        self.sampler = sampler
        self.sigmas = sigmas
        self.x = x
        self.xdtype = xdtype
        self.mcg = model_comm_group
        self.gss = grid_shard_sizes
        self.dataset = dataset
        self.n_steps = len(sigmas) - 1
        self.n_calls = 0
        self.t_calls = 0.0
        self._base = denoise_fn

    def _fn(self, *args, **kwargs):
        import torch
        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.synchronize()
        t0 = time.time()
        out = self._base(*args, **kwargs)
        if cuda:
            torch.cuda.synchronize()
        self.t_calls += time.time() - t0
        self.n_calls += 1
        return out

    def segment(self, y, i0: int, i1: int):
        """Sampler state at node i1 from state ``y`` at node i0 (i1 == n_steps: the final sample)."""
        if i1 <= i0:
            return y
        out = self.sampler.sample(self.x, {self.dataset: y}, self.sigmas[i0:i1 + 1], self._fn,
                                  model_comm_group=self.mcg, grid_shard_sizes=self.gss, dtype=self.xdtype)
        return out[self.dataset]

    def denoise(self, y, i: int):
        """D(y, sigma_i), called exactly as the Heun loop calls it at the start of iteration i (churn off)."""
        s = self.sigmas[i]
        b, t, e = y.shape[0], y.shape[1], y.shape[2]
        sig = s.view(1, 1, 1, 1, 1).expand(b, t, e, 1, 1).to(self.xdtype)
        out = self._fn(self.x, {self.dataset: y.to(self.xdtype)}, {self.dataset: sig}, self.mcg, self.gss)
        return out[self.dataset].to(self.xdtype)

    def run(self, noise, nodes, n_branch: int, n_full: int, sink, done=None):
        """noise(kind, node, j) -> standard normal tensor of the state shape (float64);
        sink(kind, node, j, final_state); done(kind, node, j) -> True to skip a draw already on disk.
        kind is "trunk" (node -1, j 0), "branch" (node = ladder index, j = 0..n_branch-1) or
        "full" (node -1, j = 0..n_full-1)."""
        done = done or (lambda *_: False)
        s0 = self.sigmas[0]
        n = self.n_steps
        y = noise("trunk", -1, 0) * s0                 # as model.sample: randn(float64) * sigmas[0]
        cur = 0
        for i, s_i, label in nodes:
            y = self.segment(y, cur, i)
            cur = i
            todo = [j for j in range(n_branch) if not done("branch", i, j)]
            if not todo:
                continue
            t0, c0 = time.time(), self.n_calls
            d = self.denoise(y, i)
            for j in todo:
                yb = d + self.sigmas[i] * noise("branch", i, j)
                sink("branch", i, j, self.segment(yb, i, n))
            LOG.info("node %d (%s, sigma %.4g): %d branches, %d calls, %.1f s", i, label, s_i, len(todo),
                     self.n_calls - c0, time.time() - t0)
        if not done("trunk", -1, 0):
            sink("trunk", -1, 0, self.segment(y, cur, n))
        for j in range(n_full):
            if not done("full", -1, j):
                sink("full", -1, j, self.segment(noise("full", -1, j) * s0, 0, n))


def draw_name(kind: str, node: int, j: int) -> str:
    if kind == "trunk":
        return "trunk"
    if kind == "full":
        return f"full_j{j:02d}"
    return f"branch_n{node:02d}_j{j:02d}"


def draw_seed(seed_base: int, kind: str, node: int, j: int) -> int:
    if kind == "trunk":
        return int(seed_base)
    if kind == "full":
        return int(seed_base) + SEED_FULL + int(j)
    return int(seed_base) + SEED_BRANCH + 1_000 * int(node) + int(j)


def save_atomic(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# model side
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(_DT_ROOT), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _init_single_rank_group():
    """1-rank gloo default group, as eval.predict.model_loader.setup_distributed does at world size 1
    (edges-sharding code calls collectives with group=None even on one rank)."""
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", store=dist.HashStore(), rank=0, world_size=1)


def load_bundle_case(datamodule, bundle_nc: str, out_names: list[str]):
    """Inputs and target of one eval bundle, as predict_from_bundle reads them (full grids, physical)."""
    from manual_inference.input_data_construction.bundle import (
        extract_target_from_bundle_dataset, load_inputs_from_bundle_numpy, open_bundle_dataset,
        previous_step_bundle_path)
    n2i_lres = datamodule.data_indices["in_lres"].data.input.name_to_index
    n2i_hres = datamodule.data_indices["in_hres"].data.input.name_to_index
    bundle = open_bundle_dataset(bundle_nc)
    try:
        x_lres, x_hres, lon_l, lat_l, lon_h, lat_h = load_inputs_from_bundle_numpy(
            bundle, n2i_lres, n2i_hres, prev_bundle=previous_step_bundle_path(bundle_nc))
        target, found = extract_target_from_bundle_dataset(bundle, out_names)
    finally:
        bundle.close()
    if target is None:
        raise SystemExit(f"no target in bundle {bundle_nc}")
    return dict(x_lres=x_lres, x_hres=x_hres, target=target, lat_h=np.asarray(lat_h, np.float64),
                lon_h=np.asarray(lon_h, np.float64), source=f"bundle {bundle_nc}",
                target_found=int(found))


def load_dataset_case(datamodule, date: str, step: int, paths: dict[str, str], out_names: list[str]):
    """Inputs and target of one hindcast row of the checkpoint's stores (control forecast, 12-hourly)."""
    from anemoi.datasets import open_dataset
    raws = {k: open_dataset(p) for k, p in paths.items()}
    when = _dt.datetime.strptime(date, "%Y%m%d")
    fakes = {k: r.date_step_to_fake_date(when, int(step)) for k, r in raws.items()}
    if len({str(v) for v in fakes.values()}) != 1:
        raise SystemExit(f"the three stores map {date}+{step} to different rows: {fakes}")
    fake = np.datetime64(next(iter(fakes.values())), "s")
    rows = {}
    for k, r in raws.items():
        hit = np.flatnonzero(np.asarray(r.dates).astype("datetime64[s]") == fake)
        if hit.size != 1:
            raise SystemExit(f"{k}: {hit.size} rows at fake date {fake}")
        rows[k] = int(hit[0])
    real = raws["out_hres"].fake_date_to_date_step(fake.astype(object))

    def take(k, n2i):
        r = raws[k]
        arr = np.asarray(r[rows[k]], dtype=np.float32)          # (variables, ensemble, points)
        v2i = r.name_to_index
        out = np.zeros((arr.shape[-1], max(n2i.values()) + 1), dtype=np.float32)
        for name, idx in n2i.items():
            out[:, idx] = arr[v2i[name], 0, :]
        return out

    n2i_lres = datamodule.data_indices["in_lres"].data.input.name_to_index
    n2i_hres = datamodule.data_indices["in_hres"].data.input.name_to_index
    x_lres = take("in_lres", n2i_lres)
    x_hres = take("in_hres", n2i_hres)
    tgt = take("out_hres", {n: i for i, n in enumerate(out_names)})
    lat = np.asarray(raws["out_hres"].latitudes, np.float64)
    lon = np.asarray(raws["out_hres"].longitudes, np.float64)
    return dict(x_lres=x_lres, x_hres=x_hres, target=tgt, lat_h=lat, lon_h=lon,
                source=(f"hindcast stores, row {rows['out_hres']} (fake date {fake}; model date, hindcast date, "
                        f"step = {[str(v) for v in real]})"), target_found=len(out_names), stores=paths)


def run_case(args) -> int:
    import torch

    from eval.predict.graph_cut import activate_local_graph_cut
    from interp.core.model import load_model

    out = Path(args.out_dir) / args.case_label
    (out / "finals").mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.setdefault("runs", [])

    _init_single_rank_group()
    torch.backends.cuda.matmul.allow_tf32 = False      # fp32 as the eval predict route (no TF32)
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda" and not args.allow_cpu:
        raise SystemExit("no CUDA device (pass --allow-cpu only for a smoke test)")
    t_load = time.time()
    bundle = load_model(args.checkpoint, device=device, precision="fp32")
    im, dm = bundle.model, bundle.datamodule
    inner = im.model
    if device == "cpu":
        # CPU smoke only: the pickled graph-transformer blocks carry the triton backend chosen at training time
        # (GPU); switch them to the pyg backend the class itself falls back to when triton cannot run.
        from anemoi.models.layers.block import GraphTransformerConv
        n_sw = 0
        for m in im.modules():
            if getattr(m, "graph_attention_backend", None) == "triton":
                m.graph_attention_backend = "pyg"
                m.conv = GraphTransformerConv(out_channels=m.out_channels_conv)
                n_sw += 1
        LOG.warning("CPU smoke: %d graph-transformer blocks switched from triton to pyg", n_sw)
    scope = json.loads(args.local_scope_json) if args.local_scope_json else None
    cut = activate_local_graph_cut(im, scope) if scope else {"mode": "disabled"}
    LOG.info("model %s.%s loaded in %.0f s; cut graph %s", type(inner).__module__, type(inner).__name__,
             time.time() - t_load, cut)
    n2i_out = dm.data_indices["out_hres"].model.output.name_to_index
    out_names = list(n2i_out.keys())
    save_vars = [v for v in args.save_vars if v in n2i_out]
    missing_vars = [v for v in args.save_vars if v not in n2i_out]
    if missing_vars:
        LOG.warning("not in this checkpoint's output, not saved: %s", missing_vars)

    if args.bundle:
        case = load_bundle_case(dm, args.bundle, out_names)
    else:
        date, step = args.dataset_case.split(":")
        case = load_dataset_case(dm, date, int(step), json.loads(args.dataset_paths_json), out_names)
    LOG.info("case %s: %s; x_lres %s x_hres %s target %s", args.case_label, case["source"],
             case["x_lres"].shape, case["x_hres"].shape, case["target"].shape)

    mask = getattr(im, "_local_graph_cut_data_mask", None)
    mask_np = (mask.detach().cpu().numpy().astype(bool) if mask is not None
               else np.ones(case["target"].shape[0], dtype=bool))
    lat, lon = case["lat_h"][mask_np], case["lon_h"][mask_np]
    tgt_cols = [out_names.index(v) for v in save_vars]
    truth = case["target"][mask_np][:, tgt_cols].astype(np.float32)

    x_in = torch.from_numpy(case["x_lres"]).to(device)[None, None, None, ...]
    x_in_hres = torch.from_numpy(case["x_hres"]).to(device)[None, None, None, ...]
    batch = {"in_lres": x_in, "in_hres": x_in_hres}
    ppt = getattr(im, "post_processors_tendencies", None)
    nsp = json.loads(args.noise_scheduler_json)
    spp = json.loads(args.sampler_params_json)
    nsc, sc = merged_sampler_config(inner.inference_defaults, nsp, spp)

    with torch.inference_mode():
        bsd, gss = inner._before_sampling(batch, im.pre_processors, im.n_step_input, None,
                                          pre_processors_tendencies=getattr(im, "pre_processors_tendencies", None),
                                          post_processors_tendencies=ppt)
        x_up, x_hr = bsd[0], bsd[1]
        xdtype = x_up.dtype
        sigmas, sampler = build_schedule_and_sampler(nsc, sc, x_up.device)
        n_steps = len(sigmas) - 1
        nodes = branch_nodes(sigmas, int(nsc.get("num_steps_high", args.n_high)), args.branch_sigmas)
        shape = (x_up.shape[0], 1, x_up.shape[2], x_up.shape[-2], inner.num_output_channels["out_hres"])
        LOG.info("ladder (%d steps): %s", n_steps, [float(f"{v:.5g}") for v in sigmas.tolist()])
        LOG.info("branch nodes: %s", [(i, round(s, 5), lab) for i, s, lab in nodes])
        LOG.info("expected denoiser calls: %d", expected_calls(nodes, n_steps, args.n_branch, args.n_full,
                                                               args.verify_production))

        # x_interp (physical, cut-graph points) the way the prediction writer exports it
        n2i_lres = dm.data_indices["in_lres"].data.input.name_to_index
        xi = inner.apply_interpolate_to_high_res(x_in[:, 0], grid_shard_sizes=None, model_comm_group=None)
        xi = xi.reshape(xi.shape[-2], xi.shape[-1]).detach().cpu().numpy()
        x_interp = np.stack([xi[:, n2i_lres[v]] if v in n2i_lres else np.zeros(xi.shape[0], np.float32)
                             for v in save_vars], axis=1).astype(np.float32)
        if x_interp.shape[0] != lat.size:
            raise RuntimeError(f"x_interp has {x_interp.shape[0]} points, the cut graph {lat.size}")
        np.savez(out / "case.npz", lat=lat, lon=lon, truth=truth, x_interp=x_interp,
                 var_names=np.asarray(save_vars), sigmas=sigmas.cpu().numpy(),
                 node_index=np.asarray([n[0] for n in nodes]), node_sigma=np.asarray([n[1] for n in nodes]),
                 node_label=np.asarray([n[2] for n in nodes]))

        out_idx = [n2i_out[v] for v in save_vars]

        def phys(y):
            p = inner._after_sampling(y.to(xdtype), im.post_processors, bsd, None, gss, True,
                                      post_processors_tendencies=ppt)
            p = p["out_hres"] if isinstance(p, dict) else p
            return p[0, 0, 0][:, out_idx].detach().cpu().numpy().astype(np.float32)

        def noise(kind, node, j):
            torch.manual_seed(draw_seed(args.seed_base, kind, node, j))
            return torch.randn(shape, device=x_up.device, dtype=torch.float64)

        def path_of(kind, node, j):
            return out / "finals" / f"{draw_name(kind, node, j)}.npy"

        written = []

        def sink(kind, node, j, y):
            arr = phys(y)
            if not np.isfinite(arr).all():
                LOG.error("%s: %d non-finite values", draw_name(kind, node, j), int((~np.isfinite(arr)).sum()))
            save_atomic(path_of(kind, node, j), arr)
            written.append(draw_name(kind, node, j))
            if kind == "trunk":
                state["trunk"] = arr

        def done(kind, node, j):
            return (not args.overwrite) and path_of(kind, node, j).exists()

        state: dict = {}
        drv = Branching(sampler, sigmas, {"in_lres": x_up, "in_hres": x_hr}, inner.fwd_with_preconditioning,
                        xdtype, grid_shard_sizes=gss)
        if args.smoke:
            # plumbing check without the sampler loop: one Heun step from sigma_max (2 calls), one D at node 1
            t0 = time.time()
            y1 = drv.segment(noise("trunk", -1, 0) * sigmas[0], 0, 1)
            d1 = drv.denoise(y1, 1)
            p_d = phys(d1)
            smoke = {"seconds": time.time() - t0, "calls": drv.n_calls, "s_per_call": drv.t_calls / drv.n_calls,
                     "phys_D_node1_finite": bool(np.isfinite(p_d).all()),
                     "phys_D_node1_mean": dict(zip(save_vars, [float(v) for v in p_d.mean(axis=0)])),
                     "truth_mean": dict(zip(save_vars, [float(v) for v in truth.mean(axis=0)])),
                     "x_interp_mean": dict(zip(save_vars, [float(v) for v in x_interp.mean(axis=0)])),
                     "n_points": int(lat.size), "lat_range": [float(lat.min()), float(lat.max())],
                     "lon_range": [float(lon.min()), float(lon.max())]}
            LOG.info("SMOKE %s", json.dumps(smoke))
            (out / "smoke.json").write_text(json.dumps(smoke, indent=1))
            return 0
        t_run = time.time()
        drv.run(noise, nodes, args.n_branch, args.n_full, sink, done)
        run_s = time.time() - t_run

        verify = None
        if args.verify_production:
            trunk = state.get("trunk")
            if trunk is None:
                trunk = np.load(path_of("trunk", -1, 0))
            torch.manual_seed(draw_seed(args.seed_base, "trunk", -1, 0))
            t0 = time.time()
            pred = im.predict_step(batch, model_comm_group=None, noise_scheduler_params=nsp, sampler_params=spp)
            pred = pred["out_hres"] if isinstance(pred, dict) else pred
            prod = pred[0, 0, 0][:, out_idx].detach().cpu().numpy().astype(np.float32)
            diff = np.abs(prod.astype(np.float64) - trunk.astype(np.float64)).max(axis=0)
            rel = diff / np.maximum(trunk.astype(np.float64).std(axis=0), 1e-30)
            verify = {"max_abs_diff_per_var": dict(zip(save_vars, [float(v) for v in diff])),
                      "max_abs_diff_over_std_per_var": dict(zip(save_vars, [float(v) for v in rel])),
                      "bit_identical": bool(np.array_equal(prod, trunk)), "seconds": time.time() - t0}
            LOG.info("PRODUCTION CHECK trunk vs predict_step: %s", verify)

    s_per_call = drv.t_calls / max(drv.n_calls, 1)
    run = {
        "utc": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z", "host": socket.gethostname(),
        "gpu": (torch.cuda.get_device_name(0) if device == "cuda" else "cpu"),
        "code_sha": _git_sha(), "checkpoint": args.checkpoint, "case_label": args.case_label,
        "source": case["source"], "target_channels_found": case["target_found"], "local_scope": scope,
        "cut_graph": {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v))
                      for k, v in cut.items()},
        "n_points": int(lat.size), "save_vars": save_vars, "vars_not_in_checkpoint": missing_vars,
        "noise_scheduler": nsc, "sampler": sc, "sigmas": sigmas.tolist(),
        "branch_nodes": [{"index": i, "sigma": s, "label": lab} for i, s, lab in nodes],
        "n_branch": args.n_branch, "n_full": args.n_full, "seed_base": args.seed_base,
        "seed_rule": {"trunk": "seed_base", "full": f"seed_base + {SEED_FULL} + j",
                      "branch": f"seed_base + {SEED_BRANCH} + 1000 * node + j"},
        "written": written, "denoiser_calls": drv.n_calls, "seconds_in_denoiser": drv.t_calls,
        "seconds_per_call": s_per_call, "run_seconds": run_s, "production_check": verify,
    }
    meta["runs"].append(run)
    meta_path.write_text(json.dumps(meta, indent=1, default=str))
    LOG.info("DONE %s: %d draws written, %d calls, %.3f s per call, %.0f s", args.case_label, len(written),
             drv.n_calls, s_per_call, run_s)
    if verify is not None and max(verify["max_abs_diff_over_std_per_var"].values()) > args.verify_tolerance:
        LOG.error("PRODUCTION CHECK FAILED (tolerance %g)", args.verify_tolerance)
        return 5
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="training checkpoint (the inference- file beside it is loaded)")
    p.add_argument("--case-label", required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--bundle", help="one eval input bundle with target (eefo O320 member + O1280 truth)")
    src.add_argument("--dataset-case", help="YYYYMMDD:STEP row of the checkpoint's hindcast stores")
    p.add_argument("--dataset-paths-json", default=json.dumps(ATOS_DATASETS))
    p.add_argument("--local-scope-json", default=None, help="eval local_scope (bbox + cut_graph) as JSON")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--noise-scheduler-json", default=json.dumps(PW30_C0_NOISE_SCHEDULER))
    p.add_argument("--sampler-params-json", default=json.dumps(PW30_C0_SAMPLER))
    p.add_argument("--n-high", type=int, default=10, help="high-segment nodes (used if the scheduler has no num_steps_high)")
    p.add_argument("--branch-sigmas", nargs="*", type=float, default=list(DEFAULT_BRANCH_SIGMAS))
    p.add_argument("--n-branch", type=int, default=8)
    p.add_argument("--n-full", type=int, default=8)
    p.add_argument("--seed-base", type=int, default=756000)
    p.add_argument("--save-vars", nargs="+", default=list(DEFAULT_SAVE_VARS))
    p.add_argument("--verify-production", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--verify-tolerance", type=float, default=1e-4,
                   help="largest trunk-vs-predict_step difference, in units of the field's box standard deviation, "
                        "accepted (0 on a deterministic GPU; atomics in the graph layers may leave float noise)")
    p.add_argument("--overwrite", action="store_true", help="redo draws already on disk (default: resume)")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--smoke", action="store_true", help="load, cut, read the case, 3 denoiser calls; no draws")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        stream=sys.stdout, force=True)
    return run_case(args)


if __name__ == "__main__":
    sys.exit(main())
