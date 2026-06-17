"""Diffusion trajectory (A1) — the storm's birth, commitment and erasure.

Runs the REAL Heun sampler and, at every denoiser evaluation, captures the
model's current clean-field estimate x̂₀ = D(x_t, σ) (the Tweedie estimate).
Each x̂₀ is reconstructed to PHYSICAL units (mirroring the model's own
`_after_sampling`) and reduced to a storm-core intensity over an auto-detected
box. Plotting intensity vs σ shows WHEN, along the generative trajectory, the
model commits to an extreme mode — and whether the low-σ steps erase it (the
'knows-but-forgets' failure seen in the o320→o1280 TC diagnosis).

Four series, all physical, over the SAME storm box, per surface target:
  realized  — per-seed trajectory of x̂₀'s storm-core metric vs its own σ.
  ceiling   — teacher-forced denoiser probe: feed the TRUE residual + noise at
              each σ and read x̂₀ ("what the model knows"). denoise_at_sigma.
  target    — storm-core metric of the OBSERVED target field.
  x_interp  — storm-core metric of the coarse interpolated input.
The ceiling-minus-realized-final gap is the knows-but-forgets diagnostic.

Capture: the DS-runtime Heun loop calls `self.fwd_with_preconditioning` as its
denoising_fn. We temporarily wrap that one method on the model instance (no
library edits); the real sampler dynamics (churn, Heun correction, schedule)
are untouched. The x̂₀ is reduced to scalars INSIDE the callback, so full
O320/O1280 fields are never stored.

Run with ONE bundle (batch-1): the encoder mis-assembles batches > 1.

Usage
-----
    cd ~/dev/downscaling-tools
    python -m interp trajectory \
        --checkpoint /path/to/checkpoint.ckpt \
        --output-dir ~/perm/interp/<ckpt_id>/trajectory \
        --event franklin_o96_o320 --num-steps 30 --n-seeds 8
"""

from __future__ import annotations

import contextlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

_DT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_DT_ROOT) not in sys.path:
    sys.path.insert(0, str(_DT_ROOT))

from interp.cli import add_event_args, add_model_args, setup_logging
from interp.core.data import collect_event_bundles, resolve_event_args
from interp.core.geometry import DEFAULT_AUTO_WINDOW, box_mask_km, detect_min_center
from interp.core.model import (
    denoise_at_sigma,
    get_surface_target_indices,
    get_variable_names,
    load_model,
    prepare_batch,
    sample_full,
)
from interp.core.runmeta import ckpt_id_from_path, write_run_meta

# Multi-GPU grid-sharding (only needed when launched with world_size > 1). Imported
# lazily inside the sharded branch so the single-GPU path has no hard dependency.

LOGGER = logging.getLogger(__name__)

PA_TO_HPA = 1.0 / 100.0


# ---------------------------------------------------------------------------
# physical reconstruction (an exact replica of the DS model's _after_sampling)
# ---------------------------------------------------------------------------

def reconstruct_phys_box(bundle, x_interp_raw_box, residual, box_t):
    """Normalized residual x̂₀ -> physical output state, RESTRICTED TO THE BOX.

    Mirrors the model's own _after_sampling (denorm the residual + add the RAW
    interpolated input on the matching channels; direct-prediction vars are not
    added back). add_interp_to_state is purely per-cell/per-channel, so slicing
    `residual` and the raw interp to the box cells FIRST is exact and avoids
    materializing the full O1280 field (~2.8 GB) at every denoiser call.
    Returns (1,1,1,Nbox,V_out) in physical units.
    """
    inner = bundle.inner_model
    ppt = getattr(bundle.model, "post_processors_tendencies", None)
    dp = getattr(inner, "direct_prediction_indices", None)
    res_box = residual[:, :, :, box_t, :].to(x_interp_raw_box.dtype)
    return inner.add_interp_to_state(
        x_interp_raw_box, res_box,
        bundle.post_processors, ppt, direct_prediction_indices=dp,
    )


def reconstruct_phys(bundle, x_interp_raw, residual):
    """Normalized residual -> physical output state on the FULL given grid (or a
    per-rank grid SHARD under model-parallel inference). add_interp_to_state is
    per-cell/per-channel, so it is exact on a shard; the caller gathers afterward."""
    inner = bundle.inner_model
    ppt = getattr(bundle.model, "post_processors_tendencies", None)
    dp = getattr(inner, "direct_prediction_indices", None)
    return inner.add_interp_to_state(
        x_interp_raw, residual.to(x_interp_raw.dtype),
        bundle.post_processors, ppt, direct_prediction_indices=dp,
    )


# ---------------------------------------------------------------------------
# storm-core reduction (per surface target; never aggregated)
# ---------------------------------------------------------------------------

# Storm-core reduction. msl uses the deep-core MIN (the eye is a few cells; a
# percentile washes it out, and the realized field's min is clean at every
# sigma). The wind/temp/precip MAXIMA use a robust p99 instead of raw max, so the
# single-cell fine-scale spikes the diffusion injects at low sigma don't dominate.
CORE_Q_HIGH = 0.99


def _q(col, q):
    return float(torch.quantile(col.float(), q))


def _side_reduce(col, name):
    """Storm-core value of one channel over the box. msl = the deep-core MINIMUM
    (the eye is only a few cells; a percentile washes it out — and the realized
    field's min is clean at every sigma). Winds/2t/tp = robust p99 (the diffusion
    injects single-cell fine-scale spikes at low sigma that raw max would chase)."""
    if name == "msl":
        return float(col.min()) * PA_TO_HPA
    if name in ("10u", "10v"):
        return _q(col.abs(), CORE_Q_HIGH)
    return _q(col, CORE_Q_HIGH)


def reduce_field(field5d, indices, has_wind):
    """field5d: (1,1,1,N,V) physical, ALREADY restricted to the box. Reduce over
    all N cells. Returns {name: storm-core value} plus 'wind10m' (p99 speed)."""
    fb = field5d[0, 0, 0]                                 # (Nbox, V)
    out = {name: _side_reduce(fb[:, idx], name) for name, idx in indices.items()}
    if has_wind:
        u, v = fb[:, indices["10u"]], fb[:, indices["10v"]]
        out["wind10m"] = _q(torch.sqrt(u * u + v * v), CORE_Q_HIGH)
    return out


def reduce_box(field5d, indices, box_t, has_wind):
    """field5d: (1,1,1,G,V) physical on the FULL grid; box_t: (G,) bool mask.
    Slice to the box then reduce. Used on the gathered full field in the sharded
    path (where slicing before reconstruction is not possible)."""
    return reduce_field(field5d[:, :, :, box_t, :], indices, has_wind)


# ---------------------------------------------------------------------------
# denoiser capture (instance-level wrap; restored on exit, no library edits)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def capture_denoiser(inner, on_call):
    """Temporarily wrap inner.fwd_with_preconditioning so every x̂₀ estimate
    along the REAL sampler trajectory is handed to on_call(sigma_scalar, D)."""
    orig = inner.fwd_with_preconditioning

    def wrapped(x_interp, x_hres, y_noised, sigma, *a, **k):
        D = orig(x_interp, x_hres, y_noised, sigma, *a, **k)
        try:
            on_call(float(sigma.reshape(-1)[0].item()), D)
        except Exception:
            LOGGER.exception("trajectory capture reducer failed (continuing)")
        return D

    inner.fwd_with_preconditioning = wrapped
    try:
        yield
    finally:
        try:
            del inner.fwd_with_preconditioning            # restore class method
        except AttributeError:
            inner.fwd_with_preconditioning = orig


def _fp32_get_schedule(orig):
    def get_schedule(self, device, dtype=None):
        return orig(self, device, torch.float32).to(torch.float32)
    return get_schedule


@contextlib.contextmanager
def force_fp32_sampler():
    """Run the diffusion sampler in fp32 instead of its default float64.

    `inner.sample` builds the schedule (and thus the sampler dtype, y_init, and
    the whole Heun state y/D1/D2/d/y_next) in float64. At O1280 those are ~5
    full-grid float64 tensors (~3.7 GB each, V_out≈70) and OOM a 40 GB GPU.
    Patching every noise scheduler's get_schedule to emit fp32 sigmas propagates
    fp32 through the sampler (its dtype is taken from sigmas.dtype) and halves
    that state. Numerically negligible for storm-core hPa.
    """
    from anemoi.models.samplers import diffusion_samplers as _ds
    saved = []
    for cls in set(_ds.NOISE_SCHEDULERS.values()):
        orig = cls.get_schedule
        cls.get_schedule = _fp32_get_schedule(orig)
        saved.append((cls, orig))
    try:
        yield
    finally:
        for cls, orig in saved:
            cls.get_schedule = orig


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def _extreme(values, name):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return None, None
    i = int(np.nanargmin(arr)) if name == "msl" else int(np.nanargmax(arr))
    return float(arr[i]), i


def _summarize(metrics, references, ceiling, trajectories):
    """Per-metric headline scalars. Seeds share the deterministic σ sequence, so
    the realized curve is averaged across seeds per call index."""
    summary = {}
    n_calls = min((len(t["steps"]) for t in trajectories), default=0)
    for name in metrics:
        cvals = [c["metrics"][name] for c in ceiling if name in c["metrics"]]
        csig = [c["sigma"] for c in ceiling if name in c["metrics"]]
        c_ext, c_i = _extreme(cvals, name)

        r_ext = r_ext_sigma = None
        if n_calls:
            mean_vals, sigs = [], []
            for k in range(n_calls):
                vk = [t["steps"][k]["metrics"].get(name) for t in trajectories
                      if name in t["steps"][k]["metrics"]]
                if not vk:
                    continue
                mean_vals.append(float(np.mean(vk)))
                sigs.append(float(np.mean([t["steps"][k]["sigma"] for t in trajectories])))
            r_ext, r_i = _extreme(mean_vals, name)
            if r_i is not None:
                r_ext_sigma = sigs[r_i]
        r_final = ([t["final"][name] for t in trajectories if name in t["final"]] or [None])
        r_final = float(np.mean(r_final)) if r_final[0] is not None else None

        entry = {
            "target": references["target"].get(name),
            "x_interp": references["x_interp"].get(name),
            "ceiling_extreme": c_ext,
            "ceiling_extreme_sigma": (csig[c_i] if c_i is not None else None),
            "realized_extreme": r_ext,
            "realized_extreme_sigma": r_ext_sigma,
            "realized_final": r_final,
        }
        if c_ext is not None and r_final is not None:
            entry["ceiling_minus_final"] = c_ext - r_final
        summary[name] = entry
    return summary


def _g(v):
    return "n/a" if v is None else f"{v:.2f}"


def _print_summary(result):
    print("\n" + "=" * 78)
    print("TRAJECTORY — storm-core intensity (target / x_interp / ceiling / realized)")
    print("=" * 78)
    s = result["summary"]
    for name in result.get("metrics_reported", []):
        m = s.get(name, {})
        print(f"  {name:8s} target={_g(m.get('target'))} x_interp={_g(m.get('x_interp'))} "
              f"ceiling={_g(m.get('ceiling_extreme'))}@σ{_g(m.get('ceiling_extreme_sigma'))} "
              f"realized*={_g(m.get('realized_extreme'))}@σ{_g(m.get('realized_extreme_sigma'))} "
              f"realized_final={_g(m.get('realized_final'))}")


# ---------------------------------------------------------------------------
# seeding-sigma sweep (A2): plant the TRUE storm at sigma_seed, sample free below
# ---------------------------------------------------------------------------

def _karras_sigmas(sigma_max, sigma_min, num_steps, device, rho=7.0):
    """Clean EDM/Karras sigma ladder (descending, num_steps levels + a final 0). Fixed
    num_steps per call so every seeded run has the SAME integration resolution."""
    ramp = torch.linspace(0.0, 1.0, num_steps, device=device, dtype=torch.float32)
    mn, mx = float(sigma_min) ** (1.0 / rho), float(sigma_max) ** (1.0 / rho)
    sig = (mx + ramp * (mn - mx)) ** rho
    return torch.cat([sig, sig.new_zeros(1)])


def _build_sampler(inner, device):
    """Rebuild the model's production Heun sampler (fp32) + its sigma_min/sigma_max, so we
    can drive it from a custom y_init and a custom (per-seed) Karras ladder."""
    from anemoi.models.samplers import diffusion_samplers as ds
    nsc = dict(inner.inference_defaults.noise_scheduler)
    sc = dict(inner.inference_defaults.diffusion_sampler)
    sampler = ds.DIFFUSION_SAMPLERS[sc.pop("sampler")](dtype=torch.float32, **sc)
    return sampler, float(nsc.get("sigma_min", 0.03)), float(nsc.get("sigma_max", 1000.0))


def _seeded_sample(inner, sampler, num_steps, sigma_min, x_interp_cond, x_hres_cond,
                   y_residual_cond, start_sigma, seed, mcg, gss, free=False):
    """Karras ladder from start_sigma down to ~0 (num_steps levels). y_init = the TRUE storm
    re-noised to start_sigma (free=False) or pure noise (free=True). Returns the final residual
    (this rank's shard under model-parallel inference)."""
    device = y_residual_cond.device
    sigmas = _karras_sigmas(start_sigma, sigma_min, num_steps, device)
    gen = torch.Generator(device=device.type).manual_seed(int(seed))
    eps = torch.randn(y_residual_cond.shape, device=device, dtype=y_residual_cond.dtype, generator=gen)
    y_init = (float(start_sigma) * eps) if free else (y_residual_cond + float(start_sigma) * eps)
    with torch.no_grad():                    # must not retain the Heun-loop autograd graph (OOM)
        return sampler.sample(x_interp_cond, x_hres_cond, y_init, sigmas,
                              inner.fwd_with_preconditioning,
                              model_comm_group=mcg, grid_shard_shapes=gss)


def _run_seeding(args, bundle, inner, global_rank, world_size, mcg, gss_arg, target_indices,
                 x_interp_cond, x_hres_cond, y_residual_cond, metrics_of, references,
                 clat, clon, box_np, window, eb, out_path):
    """A2 seeding-sigma sweep: how far down the noise schedule must the TRUE storm be planted
    for the free sampler to commit to the deep mode? Final storm-core depth vs sigma_seed
    reveals the critical window where storm depth is decided (and whether the fix is the
    noise schedule [a threshold] or low-sigma guidance [a smooth ramp])."""
    device = y_residual_cond.device
    sampler, sigma_min, sigma_max = _build_sampler(inner, device)
    seed_sigmas = sorted({float(s) for s in args.seed_sigmas})
    seeds = (list(args.seeds) if args.seeds
             else list(range(args.seed_base, args.seed_base + args.n_seeds)))

    def _mean(finals):
        return {k: float(np.mean([f[k] for f in finals])) for k in finals[0]}

    runs = []
    for ss in seed_sigmas:
        finals = []
        for seed in seeds:
            torch.manual_seed(int(seed))
            y_final = _seeded_sample(inner, sampler, args.num_steps, sigma_min, x_interp_cond,
                                     x_hres_cond, y_residual_cond, ss, seed, mcg, gss_arg)
            m = metrics_of(y_final)                      # collective; rank 0 gets the dict
            if m is not None:
                finals.append(m)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if global_rank == 0:
            runs.append({"seed_sigma": ss, "n_seeds": len(finals),
                         "final_mean": _mean(finals), "finals": finals})
            LOGGER.info("seed_sigma=%.3g -> realized msl=%.1f hPa (mean of %d)",
                        ss, runs[-1]["final_mean"].get("msl", float("nan")), len(finals))

    # free baseline = pure noise at sigma_max (the sigma_seed -> infinity asymptote).
    free_finals = []
    for seed in seeds:
        torch.manual_seed(int(seed))
        yf = _seeded_sample(inner, sampler, args.num_steps, sigma_min, x_interp_cond, x_hres_cond,
                            y_residual_cond, sigma_max, seed, mcg, gss_arg, free=True)
        m = metrics_of(yf)
        if m is not None:
            free_finals.append(m)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = None
    if global_rank == 0:
        free_mean = _mean(free_finals) if free_finals else None
        metrics_reported = list(references["target"].keys())
        summary = {}
        for name in metrics_reported:
            pts = sorted((r["seed_sigma"], r["final_mean"].get(name)) for r in runs
                         if name in r["final_mean"])
            if not pts:
                continue
            ys = [y for _, y in pts]
            committed = min(ys) if name == "msl" else max(ys)
            free_v = (free_mean or {}).get(name)
            cross = None
            if free_v is not None:
                half = 0.5 * (committed + free_v)
                for s, y in pts:                          # ascending sigma_seed
                    deep = (y <= half) if name == "msl" else (y >= half)
                    if deep:
                        cross = s                         # largest sigma_seed still committed
            summary[name] = {"committed": committed, "free": free_v, "crossover_seed_sigma": cross,
                             "target": references["target"].get(name),
                             "x_interp": references["x_interp"].get(name)}
        result = {
            "checkpoint": args.checkpoint, "ckpt_id": ckpt_id_from_path(args.checkpoint),
            "mode": "seeding", "units": "physical", "world_size": world_size,
            "metric_rule": {"msl": "box-min (hPa)", "wind10m": "box p99 speed (m/s)"},
            "bundle_paths": [str(p) for p in eb.paths],
            "surface_targets": list(target_indices.keys()), "metrics_reported": metrics_reported,
            "num_steps": args.num_steps, "fp32_sampler": bool(args.fp32_sampler),
            "seeds": [int(s) for s in seeds], "seed_sigmas": seed_sigmas,
            "box": {"name": "storm", "lat": clat, "lon": clon % 360.0,
                    "radius_km": args.eye_radius_km, "n_cells": int(box_np.sum())},
            "probe_field": "msl", "window": list(window),
            "references": references, "free": free_mean, "runs": runs, "summary": summary,
        }
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "seeding.json", "w") as f:
            json.dump(result, f, indent=2)
        LOGGER.info("Results saved to %s", out_path / "seeding.json")
        write_run_meta(out_path, "trajectory_seeding", args)
        print("\nSEEDING SWEEP — final storm-core msl vs seeding sigma "
              "(deep = committed, shallow = free):")
        for r in runs:
            print(f"  seed_sigma={r['seed_sigma']:>8.2f}  msl={r['final_mean'].get('msl', float('nan')):.1f} hPa")
        if free_mean:
            print(f"  free (no seed)            msl={free_mean.get('msl', float('nan')):.1f} hPa")

    if mcg is not None:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
    return result


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run_trajectory(args):
    out_path = Path(args.output_dir)

    # ---- parallel setup: world_size>1 (srun) => grid-shard the model across ranks ----
    from manual_inference.prediction.predict import _get_parallel_info, _init_model_comm_group
    global_rank, local_rank, world_size = _get_parallel_info()
    sharded = world_size > 1
    device = args.device
    mcg = None
    if sharded:
        if str(device).startswith("cuda"):
            torch.cuda.set_device(int(local_rank))
            device = f"cuda:{int(local_rank)}"
        mcg = _init_model_comm_group(device, global_rank, world_size)
        from anemoi.models.distributed.graph import gather_tensor, shard_tensor
        from anemoi.models.distributed.shapes import apply_shard_shapes, get_shard_shapes
        LOGGER.info("rank %d/%d (local %d) on %s — GRID-SHARDED inference",
                    global_rank, world_size, local_rank, device)

    LOGGER.info("Loading model from %s", args.checkpoint)
    bundle = load_model(args.checkpoint, device=device, precision=args.precision,
                        num_gpus_per_model=world_size)
    inner = bundle.inner_model
    target_indices = get_surface_target_indices(bundle)
    if "msl" not in target_indices:
        raise SystemExit("trajectory needs msl in the output schema (storm-core probe)")
    in_names = get_variable_names(bundle)["input_lres"]       # idx -> name
    name2in = {v: k for k, v in in_names.items()}
    has_wind = "10u" in target_indices and "10v" in target_indices
    LOGGER.info("surface targets: %s", target_indices)

    bundle_dir, dates, members, steps, _ = resolve_event_args(args)
    eb = collect_event_bundles(bundle, bundle_dir, dates, members, steps)
    _, _, lat_hres, lon_hres = eb.coords
    y0 = eb.y[0:1].to(device)                                 # observed, physical, FULL grid

    # Storm box from the OBSERVED msl (deterministic -> identical on every rank).
    window = (tuple(float(x) for x in args.auto_window.split(","))
              if args.auto_window else DEFAULT_AUTO_WINDOW)
    probe = y0[0, 0, 0, :, target_indices["msl"]].cpu().numpy()
    clat, clon = detect_min_center(probe, lat_hres, lon_hres, window)
    box_np = box_mask_km(lat_hres, lon_hres, clat, clon, args.eye_radius_km)
    box_t = torch.from_numpy(box_np).to(device)
    LOGGER.info("storm box: center=(%.2f,%.2f) R=%.0fkm -> %d hres cells",
                clat, clon, args.eye_radius_km, int(box_np.sum()))

    # ---- conditioning tensors + the per-step reconstruct->reduce step (mode-specific) ----
    # metrics_of(residual) maps a (possibly sharded) residual x̂₀ to the storm-core metrics
    # dict on rank 0 (None elsewhere). In sharded mode it GATHERS the full grid (a
    # collective — EVERY rank must call it) then box-reduces on rank 0.
    if sharded:
        (x_interp_cond, x_hres_cond, x_interp_raw_sh), gss = inner._before_sampling(
            eb.x_lres[0:1].to(device), eb.x_hres[0:1].to(device),
            bundle.pre_processors, 1, model_comm_group=mcg)

        def _gather_full(field_sh):
            return gather_tensor(field_sh, -2, apply_shard_shapes(field_sh, -2, gss), mcg)

        y_sh = shard_tensor(y0, -2, get_shard_shapes(y0, -2, mcg), mcg)
        y_residual_cond = inner.compute_residuals(
            y_sh[:, 0, ...], x_interp_raw_sh[:, 0, ...])[:, None, ...]
        offs = [0]
        for s in gss:
            offs.append(offs[-1] + int(s))
        LOGGER.info("rank %d holds %d / %d box cells", global_rank,
                    int(box_t[offs[global_rank]:offs[global_rank + 1]].sum()), int(box_np.sum()))

        # Gather only the SURFACE-target channels (not the full ~70-channel output) — keeps
        # the per-call all-gather tiny (~5 channels) instead of ~2.8 GB at O1280.
        surf_idx = torch.tensor(list(target_indices.values()), device=device)
        surf_remap = {name: i for i, name in enumerate(target_indices)}

        def metrics_of(residual):
            phys_sh = reconstruct_phys(bundle, x_interp_raw_sh, residual)[..., surf_idx]
            phys_full = _gather_full(phys_sh)
            return reduce_box(phys_full, surf_remap, box_t, has_wind) if global_rank == 0 else None

        xir_full = _gather_full(x_interp_raw_sh)
    else:
        prepared = prepare_batch(bundle, eb.x_lres[0:1], eb.x_hres[0:1], eb.y[0:1])
        x_interp_cond, x_hres_cond = prepared["x_interp"], prepared["x_hres"]
        y_residual_cond = prepared["y_residual"]
        x_interp_raw_box = prepared["x_interp_raw"][:, :, :, box_t, :]

        def metrics_of(residual):
            return reduce_field(
                reconstruct_phys_box(bundle, x_interp_raw_box, residual, box_t),
                target_indices, has_wind)

    gss_arg = gss if sharded else None

    # References (rank 0): target from the full observed y; x_interp from the raw interp.
    references = None
    if global_rank == 0:
        references = {"target": reduce_box(y0, target_indices, box_t, has_wind)}
        fb_xi = (xir_full[:, :, :, box_t, :] if sharded else x_interp_raw_box)[0, 0, 0]
        xi_metrics = {name: _side_reduce(fb_xi[:, name2in[name]], name)
                      for name in target_indices if name in name2in}
        if has_wind and "10u" in name2in and "10v" in name2in:
            u, v = fb_xi[:, name2in["10u"]], fb_xi[:, name2in["10v"]]
            xi_metrics["wind10m"] = _q(torch.sqrt(u * u + v * v), CORE_Q_HIGH)
        references["x_interp"] = xi_metrics

    # A2 seeding-sigma sweep reuses the whole setup above (load / bundle / box / sharded
    # conditioning / metrics_of / references), then sweeps where the storm is planted.
    if args.mode == "seeding":
        return _run_seeding(args, bundle, inner, global_rank, world_size, mcg, gss_arg,
                            target_indices, x_interp_cond, x_hres_cond, y_residual_cond,
                            metrics_of, references, clat, clon, box_np, window, eb, out_path)

    # Teacher-forced ceiling: feed the TRUE residual + noise at each sigma.
    ceiling = []
    for sigma in args.ceiling_sigmas:
        noise = torch.randn_like(y_residual_cond)
        D = denoise_at_sigma(bundle, x_interp_cond, x_hres_cond, y_residual_cond,
                             sigma, noise, model_comm_group=mcg, grid_shard_shapes=gss_arg)
        m = metrics_of(D)                                    # collective in sharded mode
        if global_rank == 0:
            ceiling.append({"sigma": float(sigma), "metrics": m})
            LOGGER.info("ceiling σ=%.3g -> msl=%.1f hPa", sigma, m.get("msl", float("nan")))

    # Realized trajectories: capture x̂₀ along the real sampler.
    seeds = (list(args.seeds) if args.seeds
             else list(range(args.seed_base, args.seed_base + args.n_seeds)))
    trajectories = []
    for seed in seeds:
        records = []

        def on_call(sigma_scalar, D, _rec=records):
            m = metrics_of(D)                                # collective; runs on all ranks
            if m is not None:                                # only rank 0 records
                _rec.append({"call_idx": len(_rec), "sigma": sigma_scalar, "metrics": m})

        torch.manual_seed(int(seed))
        sampler_ctx = force_fp32_sampler() if args.fp32_sampler else contextlib.nullcontext()
        with capture_denoiser(inner, on_call), sampler_ctx:
            final_resid = sample_full(bundle, x_interp_cond, x_hres_cond,
                                      num_steps=args.num_steps, seed=int(seed),
                                      model_comm_group=mcg, grid_shard_shapes=gss_arg)
        final_metrics = metrics_of(final_resid) or {}        # collective; {} on non-zero ranks
        if global_rank == 0:
            trajectories.append({"seed": int(seed), "steps": records, "final": final_metrics})
            if records:
                d = records[-1]["metrics"].get("msl", float("nan")) - final_metrics.get("msl", float("nan"))
                LOGGER.info("seed %d: %d denoiser calls, final msl=%.1f hPa "
                            "(last x̂₀ − final = %+.2f hPa; should be small)",
                            seed, len(records), final_metrics.get("msl", float("nan")), d)

    result = None
    if global_rank == 0:
        metrics_reported = list(references["target"].keys())
        result = {
            "checkpoint": args.checkpoint,
            "ckpt_id": ckpt_id_from_path(args.checkpoint),
            "units": "physical",
            "world_size": world_size,
            "metric_rule": {"msl": "box-min (hPa)", "10u": "box p99 |·| (m/s)",
                            "10v": "box p99 |·| (m/s)", "2t": "box p99 (K)",
                            "tp": "box p99", "wind10m": "box p99 speed (m/s)"},
            "bundle_paths": [str(p) for p in eb.paths],
            "surface_targets": list(target_indices.keys()),
            "metrics_reported": metrics_reported,
            "num_steps": args.num_steps,
            "fp32_sampler": bool(args.fp32_sampler),
            "seeds": [int(s) for s in seeds],
            "box": {"name": "storm", "lat": clat, "lon": clon % 360.0,
                    "radius_km": args.eye_radius_km, "n_cells": int(box_np.sum())},
            "probe_field": "msl",
            "window": list(window),
            "references": references,
            "ceiling": ceiling,
            "trajectories": trajectories,
            "summary": _summarize(metrics_reported, references, ceiling, trajectories),
        }
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "trajectory.json", "w") as f:
            json.dump(result, f, indent=2)
        LOGGER.info("Results saved to %s", out_path / "trajectory.json")
        write_run_meta(out_path, "trajectory", args)
        _print_summary(result)

    if sharded:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
    return result


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        description="Diffusion trajectory (A1) — storm-core x̂₀ vs σ (birth/commit/erase)")
    add_model_args(p)
    add_event_args(p)
    p.add_argument("--mode", default="trajectory", choices=["trajectory", "seeding"],
                   help="trajectory = ceiling + realized x̂₀ vs σ (default); "
                        "seeding = A2 sweep: plant the TRUE storm at σ_seed, sample free below, "
                        "and report final storm depth vs σ_seed (the critical-window test)")
    p.add_argument("--seed-sigmas", nargs="+", type=float,
                   default=[2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 300.0],
                   help="[seeding] σ_seed grid — plant the true storm at each, then sample free below")
    p.add_argument("--num-steps", type=int, default=30,
                   help="Heun steps for the realized sampler trajectory")
    p.add_argument("--n-seeds", type=int, default=8,
                   help="number of realized trajectories (distinct noise seeds)")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="explicit seed list (overrides --n-seeds / --seed-base)")
    p.add_argument("--seed-base", type=int, default=1000)
    p.add_argument("--ceiling-sigmas", nargs="+", type=float,
                   default=[5.0, 10.0, 20.0, 40.0, 80.0, 150.0, 300.0],
                   help="sigmas for the teacher-forced denoiser ceiling probe; the "
                        "mid-to-high 'storm-laying' regime. Below ~sigma 5 the probe is "
                        "degenerate (teacher-forcing the true residual at low noise is "
                        "near-trivial and the reconstruction min picks up sharp artifacts)")
    p.add_argument("--eye-radius-km", type=float, default=500.0,
                   help="radius of the storm-core box for the intensity reduction")
    p.add_argument("--auto-window", default=None,
                   help="lat0,lat1,lon0,lon1 (deg, lon 0..360) for storm auto-detect")
    p.add_argument("--fp32-sampler", action=argparse.BooleanOptionalAction, default=True,
                   help="run the diffusion sampler in fp32 (halves its float64 state; "
                        "required to fit the realized path at O1280). --no-fp32-sampler "
                        "keeps the production float64 sampler.")
    args = p.parse_args(argv)
    setup_logging()
    return run_trajectory(args)


if __name__ == "__main__":
    main()
