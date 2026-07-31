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
    is_dict_api,
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
    res_box = residual[:, :, :, box_t, :].to(x_interp_raw_box.dtype)
    if is_dict_api(inner):
        return inner.add_interp_to_state(x_interp_raw_box, res_box, bundle.post_processors, ppt,
                                         target_dataset="out_hres", source_dataset="in_lres")
    dp = getattr(inner, "direct_prediction_indices", None)
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
    if is_dict_api(inner):
        return inner.add_interp_to_state(x_interp_raw, residual.to(x_interp_raw.dtype),
                                         bundle.post_processors, ppt,
                                         target_dataset="out_hres", source_dataset="in_lres")
    dp = getattr(inner, "direct_prediction_indices", None)
    return inner.add_interp_to_state(
        x_interp_raw, residual.to(x_interp_raw.dtype),
        bundle.post_processors, ppt, direct_prediction_indices=dp,
    )


# ---------------------------------------------------------------------------
# storm-core reduction (per surface target; never aggregated)
# ---------------------------------------------------------------------------

# Storm-core reduction. msl uses the deep-core MIN over PHYSICALLY-REALIZABLE cells
# (see MSL_PHYS_FLOOR_HPA): the eye is only a few cells so a percentile washes it out,
# but a raw min() chases the sharp single-cell reconstruction artifacts the diffusion
# injects at low sigma (the same artifacts the wind p99 already guards against — only
# msl, being a min, needs a floor instead of a high percentile). The wind/temp/precip
# MAXIMA use a robust p99 instead of raw max for the same reason.
CORE_Q_HIGH = 0.99

# Physical floor (hPa) for the msl deep-core min. The deepest mean-sea-level pressure
# ever observed in a storm is ~870 hPa (Typhoon Tip) and this model's own teacher-forced
# ceiling bottoms out ~914 hPa, so any box cell below this is a low-sigma reconstruction
# artifact, never a real eye. Flooring the min here recovers the genuine planted eye at
# small sigma_seed (where a couple of garbage cells otherwise drove box-min to -244/+827).
MSL_PHYS_FLOOR_HPA = 870.0

# Diagnostic-only keys emitted alongside the real metrics; excluded from metrics_reported.
_DIAG_KEYS = ("msl_raw_min_hpa", "msl_n_below_floor")

# Schedule floor for the realized/seeded samplers — keeps the Heun loop finite for
# checkpoints whose inference_defaults.sigma_min is 0.0 (the unified o96->o320).
SAMPLER_SIGMA_MIN = 0.03


def _q(col, q):
    return float(torch.quantile(col.float(), q))


def _side_reduce(col, name):
    """Storm-core value of one channel over the box. msl = the deep-core MINIMUM over
    physically-realizable cells (>= MSL_PHYS_FLOOR_HPA), which keeps the genuine few-cell
    eye but rejects the sharp single-cell artifacts the diffusion injects at low sigma.
    Winds/2t/tp = robust p99 (raw max would chase those same single-cell spikes)."""
    if name == "msl":
        hpa = col.float() * PA_TO_HPA
        phys = hpa[hpa >= MSL_PHYS_FLOOR_HPA]
        return float(phys.min()) if phys.numel() else float(hpa.min())  # fallback: never empty
    if name in ("10u", "10v"):
        return _q(col.abs(), CORE_Q_HIGH)
    return _q(col, CORE_Q_HIGH)


def reduce_field(field5d, indices, has_wind):
    """field5d: (1,1,1,N,V) physical, ALREADY restricted to the box. Reduce over
    all N cells. Returns {name: storm-core value} plus 'wind10m' (p99 speed) and, for
    msl, two diagnostics: the un-floored raw box-min and the count of sub-floor cells."""
    fb = field5d[0, 0, 0]                                 # (Nbox, V)
    out = {name: _side_reduce(fb[:, idx], name) for name, idx in indices.items()}
    if "msl" in indices:
        mhpa = fb[:, indices["msl"]].float() * PA_TO_HPA
        out["msl_raw_min_hpa"] = float(mhpa.min())                       # un-floored (artifact-prone)
        out["msl_n_below_floor"] = int((mhpa < MSL_PHYS_FLOOR_HPA).sum())  # # artifact cells clipped
    if has_wind:
        u, v = fb[:, indices["10u"]], fb[:, indices["10v"]]
        speed = torch.sqrt(u * u + v * v)
        out["wind10m"] = _q(speed, 0.999)          # p99.9 (p99 useless for the tail)
        out["wind10m_max"] = float(speed.max())    # raw max (spiky; user-requested)
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

    def wrapped(*args, **kwargs):
        D = orig(*args, **kwargs)
        try:
            if args and isinstance(args[0], dict):       # unified: (x_dict, y_dict, sigma_dict, ...)
                sig = next(iter(args[2].values()))
                D_t = D["out_hres"] if isinstance(D, dict) else D
            else:                                        # ds: (x_interp, x_hres, y_noised, sigma, ...)
                sig, D_t = args[3], D
            on_call(float(sig.reshape(-1)[0].item()), D_t)
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
# residual_diag: normalized residual-eye SNR (NO forward / NO sampler)
# ---------------------------------------------------------------------------

def _run_residual_diag(args, bundle, target_indices, name2in, box_t, y_residual_cond,
                       references, eb, out_path, global_rank, world_size):
    """Measure the TRUE residual eye in the model's NORMALIZED units (the space the
    EDM noise sigma lives in), per surface channel. No denoiser/sampler call: only
    load + interpolate + subtract (already done by run_trajectory's setup) + the
    output normalizer. SNR(sigma)=|norm_residual_eye|/sigma, so sigma_star=|norm_eye|
    is the sigma where the eye signal equals the per-cell noise floor. Cross-lane
    sigma_star contrast tests M-owner (eye buried earlier at o1280 => smaller sigma_star)."""
    if world_size != 1:
        raise SystemExit("residual_diag is single-GPU only (no forward needed)")
    device = y_residual_cond.device
    dtype = y_residual_cond.dtype
    V = y_residual_cond.shape[-1]

    def out_denorm(t):
        return bundle.post_processors(t, dataset="output", in_place=False)

    zeros = torch.zeros((1, 1, 1, 1, V), device=device, dtype=dtype)
    ones = torch.ones((1, 1, 1, 1, V), device=device, dtype=dtype)
    add = out_denorm(zeros)[0, 0, 0, 0, :]                 # (V,) per-channel mean shift
    mul = out_denorm(ones)[0, 0, 0, 0, :] - add            # (V,) per-channel scale
    r_norm = (y_residual_cond - add) / mul                 # model-space normalized residual
    rb = r_norm[0, 0, 0][box_t]                            # (Nbox, V) normalized
    yb = y_residual_cond[0, 0, 0][box_t]                   # (Nbox, V) physical delta

    out = {
        "mode": "residual_diag",
        "ckpt_id": ckpt_id_from_path(args.checkpoint),
        "checkpoint": args.checkpoint,
        "event": args.event,
        "bundle_paths": [str(p) for p in eb.paths],
        "box_n_cells": int(box_t.sum().item()),
        "references": references,
        "channels": {},
    }
    for name, idx in target_indices.items():
        phys = yb[:, idx]
        norm = rb[:, idx]
        if name == "msl":
            phys_eye = float(phys.min()) * PA_TO_HPA       # deepest residual, hPa (negative=deeper)
            norm_eye = float(norm.min())
            # eye-core spatial extent: cells deeper than half the peak (coherence proxy)
            thr = 0.5 * float(norm.min())
            core_cells = int((norm <= thr).sum().item())
        else:
            i = int(norm.abs().argmax())
            phys_eye = float(phys[i])
            norm_eye = float(norm[i])
            thr = 0.5 * float(norm.abs().max())
            core_cells = int((norm.abs() >= thr).sum().item())
        out["channels"][name] = {
            "mul": float(mul[idx]), "add": float(add[idx]),
            "phys_residual_eye": phys_eye,
            "norm_residual_eye": norm_eye,
            "abs_norm_eye": abs(norm_eye),
            "sigma_star_SNR1": abs(norm_eye),
            "eye_core_cells_half_peak": core_cells,
            "eye_core_frac_of_box": core_cells / max(1, int(box_t.sum().item())),
            "norm_residual_box_rms": float(norm.pow(2).mean().sqrt()),
            "norm_residual_box_p99abs": float(torch.quantile(norm.abs(), 0.99)),
            "norm_residual_grid_rms": float(r_norm[0, 0, 0][:, idx].pow(2).mean().sqrt()),
        }
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "residual_diag.json", "w") as fh:
        json.dump(out, fh, indent=2)
    write_run_meta(out_path, "trajectory_residual_diag", args)
    print("\n" + "=" * 78)
    print("RESIDUAL DIAG — normalized residual eye / SNR (sigma_star = |norm eye|)")
    print("=" * 78)
    for name, c in out["channels"].items():
        print(f"  {name:6s} mul={c['mul']:.4g} add={c['add']:.4g} | phys_eye={c['phys_residual_eye']:.3g} "
              f"norm_eye={c['norm_residual_eye']:.3g} sigma*={c['sigma_star_SNR1']:.3g} "
              f"core_cells={c['eye_core_cells_half_peak']} box_rms={c['norm_residual_box_rms']:.3g} "
              f"grid_rms={c['norm_residual_grid_rms']:.3g}")
    LOGGER.info("residual_diag saved to %s", out_path / "residual_diag.json")
    return out


# ---------------------------------------------------------------------------
# seeding-sigma sweep (A2): plant the TRUE storm at sigma_seed, sample free below
# ---------------------------------------------------------------------------

def _karras_sigmas(sigma_max, sigma_min, num_steps, device, rho=7.0):
    """Clean EDM/Karras sigma ladder (descending, num_steps levels + a final 0)
    spanning the FULL [sigma_min, sigma_max] -- the production-schedule resolution."""
    ramp = torch.linspace(0.0, 1.0, num_steps, device=device, dtype=torch.float32)
    mn, mx = float(sigma_min) ** (1.0 / rho), float(sigma_max) ** (1.0 / rho)
    sig = (mx + ramp * (mn - mx)) ** rho
    return torch.cat([sig, sig.new_zeros(1)])


def _seeded_ladder(sigma_max, sigma_min, num_steps, start_sigma, device, rho=7.0):
    """Seeded sigma ladder that PRESERVES the production schedule step density below
    start_sigma, instead of recompressing a fixed num_steps into [sigma_min, start_sigma].

    The free / realized sampler integrates the full Karras ladder sigma_max -> sigma_min
    (num_steps levels), and its low-sigma tail is stable (clean box-min at every sigma).
    Planting the true storm at start_sigma and sampling free below it must follow that
    SAME ladder from start_sigma down. The old _karras_sigmas(start_sigma, ...) instead
    crammed the full num_steps budget into [sigma_min, start_sigma]; for small start_sigma
    that is a wildly over-dense integration of the stiff low-sigma score d=(y-D)/sigma,
    which compounds into non-physical single-cell residual blow-ups (the -244 / +821 hPa
    box-min seen at sigma_seed in {1, 5}). Here we build the full production ladder and keep
    only its nodes strictly below start_sigma (including the terminal 0), prepending the
    planting level so y_init is re-noised at exactly start_sigma. At start_sigma == sigma_max
    this reproduces the full free ladder exactly."""
    full = _karras_sigmas(sigma_max, sigma_min, num_steps, device, rho)   # descending + terminal 0
    tail = full[full < float(start_sigma)]                                # production-density tail (incl. 0)
    head = full.new_tensor([float(start_sigma)])
    return torch.cat([head, tail])


def _build_sampler(inner, device):
    """Rebuild the model's production Heun sampler (fp32) + its sigma_min/sigma_max, so we
    can drive it from a custom y_init and a custom (per-seed) Karras ladder."""
    from anemoi.models.samplers import diffusion_samplers as ds
    nsc = dict(inner.inference_defaults.noise_scheduler)
    sc = dict(inner.inference_defaults.diffusion_sampler)
    sampler = ds.DIFFUSION_SAMPLERS[sc.pop("sampler")](dtype=torch.float32, **sc)
    # Clamp the schedule floor: some checkpoints (unified o96->o320) ship sigma_min=0.0,
    # which makes the Karras ladder end at 0 and the Heun step divide by ~0 -> NaN.
    sigma_min = max(float(nsc.get("sigma_min", 0.03)), SAMPLER_SIGMA_MIN)
    return sampler, sigma_min, float(nsc.get("sigma_max", 1000.0))


def _guided_denoiser(base_fn, lam, sig_lo, sig_hi):
    """σ-banded score amplification (pure inference; no model/sampler/training change).

    The Heun sampler steps along the score d=(y−D)/σ where D=base_fn(...) is the denoised
    x̂₀ and y is the current noised state x_σ. Returning D'=(1−λ)y+λD IN-BAND makes the
    in-band score exactly λ·d — amplifying the denoising pull and de-anchoring the shallow
    input-conditioned conditional mean (the lever for the o320→o1280 eye under-commitment).
    λ=1 is the identity (returns base_fn unchanged); outside [sig_lo, sig_hi], D is untouched.
    Handles both the ds-API call (x_interp, x_hres, y_noised, sigma, ...) and the unified
    dict-API call (x_dict, y_dict, sigma_dict, ...)."""
    if float(lam) == 1.0:
        return base_fn
    lam = float(lam)
    sig_lo, sig_hi = float(sig_lo), float(sig_hi)

    def guided(*args, **kwargs):
        D = base_fn(*args, **kwargs)
        try:
            if args and isinstance(args[0], dict):     # unified: (x_dict, y_dict, sigma_dict, ...)
                sig = float(next(iter(args[2].values())).reshape(-1)[0].item())
                if not (sig_lo <= sig <= sig_hi):
                    return D
                y = args[1]
                if isinstance(D, dict):
                    return {k: (1.0 - lam) * y[k].to(D[k].dtype) + lam * D[k] for k in D}
                yt = next(iter(y.values())) if isinstance(y, dict) else y
                return (1.0 - lam) * yt.to(D.dtype) + lam * D
            sig = float(args[3].reshape(-1)[0].item())  # ds: (x_interp, x_hres, y_noised, sigma, ...)
            if not (sig_lo <= sig <= sig_hi):
                return D
            return (1.0 - lam) * args[2].to(D.dtype) + lam * D
        except Exception:
            LOGGER.exception("guidance wrap failed (returning unguided D)")
            return D

    return guided


def _autoguided_denoiser(strong_fn, weak_fn, w, sig_lo, sig_hi):
    """Karras-style autoguidance: D' = w*D_strong - (w-1)*D_weak = D + (w-1)*(D - D_weak)
    in [sig_lo, sig_hi]. Amplifies what the strong model learned beyond an undertrained
    D_bad while preserving diversity (score-bias correction, not mode truncation).
    Handles both the ds-API call (x_interp, x_hres, y_noised, sigma, ...) and the
    unified dict-API call (x_dict, y_dict, sigma_dict, ...)."""
    if float(w) == 1.0:
        return strong_fn
    w = float(w)
    sig_lo, sig_hi = float(sig_lo), float(sig_hi)

    def fn(*args, **kwargs):
        D = strong_fn(*args, **kwargs)
        try:
            if args and isinstance(args[0], dict):
                sig = float(next(iter(args[2].values())).reshape(-1)[0].item())
                if not (sig_lo <= sig <= sig_hi):
                    return D
                Dw = weak_fn(*args, **kwargs)
                if isinstance(D, dict):
                    return {k: D[k] + (w - 1.0) * (D[k] - Dw[k].to(D[k].dtype)) for k in D}
                Dwt = next(iter(Dw.values())) if isinstance(Dw, dict) else Dw
                return D + (w - 1.0) * (D - Dwt.to(D.dtype))
            sig = float(args[3].reshape(-1)[0].item())
            if not (sig_lo <= sig <= sig_hi):
                return D
            Dw = weak_fn(*args, **kwargs)
            return D + (w - 1.0) * (D - Dw.to(D.dtype))
        except Exception:
            LOGGER.exception("autoguidance wrap failed (returning unguided D)")
            return D

    return fn


def _seeded_sample(inner, sampler, num_steps, sigma_min, sigma_max, x_interp_cond, x_hres_cond,
                   y_residual_cond, start_sigma, seed, mcg, gss, free=False,
                   guidance=None, sampler_kwargs=None, denoise_fn=None):
    """Production-density Karras ladder from start_sigma down to ~0 (see _seeded_ladder).
    y_init = the TRUE storm re-noised to start_sigma (free=False) or pure noise (free=True).
    `guidance=(lam, sig_lo, sig_hi)` wraps the denoiser in σ-banded score amplification;
    `sampler_kwargs` (e.g. {"S_churn":.., "S_min":.., "S_max":..}) are forwarded to the
    sampler (None/{} → checkpoint defaults).
    Returns the final residual (this rank's shard under model-parallel inference)."""
    device = y_residual_cond.device
    sigmas = _seeded_ladder(sigma_max, sigma_min, num_steps, start_sigma, device)
    gen = torch.Generator(device=device.type).manual_seed(int(seed))
    eps = torch.randn(y_residual_cond.shape, device=device, dtype=y_residual_cond.dtype, generator=gen)
    y_init = (float(start_sigma) * eps) if free else (y_residual_cond + float(start_sigma) * eps)
    denoise_fn = denoise_fn if denoise_fn is not None else inner.fwd_with_preconditioning
    if guidance is not None:
        denoise_fn = _guided_denoiser(denoise_fn, *guidance)
    skw = dict(sampler_kwargs or {})
    with torch.no_grad():                    # must not retain the Heun-loop autograd graph (OOM)
        if is_dict_api(inner):               # unified sampler takes per-dataset dicts
            out = sampler.sample({"in_lres": x_interp_cond, "in_hres": x_hres_cond},
                                 {"out_hres": y_init}, sigmas, denoise_fn,
                                 model_comm_group=mcg, **skw)
            return out["out_hres"] if isinstance(out, dict) else out
        return sampler.sample(x_interp_cond, x_hres_cond, y_init, sigmas,
                              denoise_fn,
                              model_comm_group=mcg, grid_shard_shapes=gss, **skw)


# Physical sanity band for a seeded FINAL msl box-min (hPa). The deepest real Atlantic TC
# is ~870 hPa and the model's teacher-forced ceiling tops out ~914, so anything below this
# (or above ~1080) is a numerical artifact, never a storm. Used ONLY to keep the auto-summary
# (committed / crossover) from being driven by a corrupt row -- the per-row values are still
# reported verbatim in `runs`. Non-blocking: excluded rows are logged and recorded, not dropped.
MSL_PHYS_MIN_HPA = 870.0
MSL_PHYS_MAX_HPA = 1080.0

# A seeded msl row is DEGENERATE when more than this fraction of box cells fall below the
# physical floor. At very small sigma_seed the low-sigma denoiser, fed a sharp planted eye,
# rings out over a whole REGION (not a few isolated spikes): job 30926929 saw 675/9938 cells
# (~7%) sub-floor at sigma_seed=1 and 49 at sigma_seed=5, vs 0 at sigma_seed>=20. For such a
# row the floored box-min is meaningless (pegged at the floor), so it must be flagged and kept
# out of the summary committed/crossover -- not reported as if it were a real ~870 hPa eye.
MSL_DEGENERATE_FRAC = 0.001  # > ~0.1% of box cells sub-floor => regional breakdown, not an eye


def _seed_row_physical(name, y):
    """Is a seeded final storm-core value physically plausible? Guards the summary against
    low-sigma injection artifacts (e.g. msl box-min of -244 / +821 hPa)."""
    if y is None or not np.isfinite(y):
        return False
    if name == "msl":
        return MSL_PHYS_MIN_HPA <= y <= MSL_PHYS_MAX_HPA
    return True


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

    # --- plant selection: what gets re-noised at sigma_seed -------------------
    seed_source = getattr(args, "seed_source", "truth")
    plant = y_residual_cond
    plant_info = {"seed_source": seed_source}
    if seed_source == "self_deepest":
        if world_size != 1:
            raise SystemExit("--seed-source self_deepest requires single-rank inference "
                             "(plant selection uses rank-0 metrics)")
        n_free = int(getattr(args, "n_free_draws", 20))
        best, best_m = None, None
        for i in range(n_free):
            fseed = int(args.seed_base) + 50000 + i
            torch.manual_seed(fseed)
            yf = _seeded_sample(inner, sampler, args.num_steps, sigma_min, sigma_max,
                                x_interp_cond, x_hres_cond, y_residual_cond, sigma_max,
                                fseed, mcg, gss_arg, free=True)
            m = metrics_of(yf)
            if (m is not None and np.isfinite(m.get("msl", np.nan))
                    and _seed_row_physical("msl", m["msl"])
                    and (best_m is None or m["msl"] < best_m["msl"])):
                best_m, best = m, yf.detach().clone()
            del yf
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if best is None:
            raise SystemExit("self_deepest: no physically-valid free draw found")
        plant = best
        plant_info.update({"n_free_draws": n_free, "plant_metrics": best_m})
        LOGGER.info("self_deepest plant: deepest of %d free draws, box msl %.1f hPa",
                    n_free, best_m["msl"])
    elif seed_source == "det_prior":
        if world_size != 1:
            raise SystemExit("--seed-source det_prior requires single-rank inference")
        if not getattr(args, "det_checkpoint", None):
            raise SystemExit("--seed-source det_prior requires --det-checkpoint")
        from interp.core.model import load_model as _load_det
        LOGGER.info("det_prior: loading det checkpoint %s", args.det_checkpoint)
        det_bundle = _load_det(args.det_checkpoint, device=str(device),
                               precision=args.precision, num_gpus_per_model=world_size)
        det_inner = det_bundle.inner_model
        det_sampler, _, _ = _build_sampler(det_inner, device)
        DET_SIGMA = 5.0e5   # D(x_T, 5e5) protocol: ladder [5e5, 5e5, 0], y_init = randn*5e5
        det_sigmas = torch.tensor([DET_SIGMA, DET_SIGMA, 0.0], device=device, dtype=torch.float32)
        gen = torch.Generator(device=device.type).manual_seed(int(args.seed_base))
        eps = torch.randn(y_residual_cond.shape, device=device,
                          dtype=y_residual_cond.dtype, generator=gen)
        y_init = DET_SIGMA * eps
        with torch.no_grad():
            if is_dict_api(det_inner):
                out = det_sampler.sample({"in_lres": x_interp_cond, "in_hres": x_hres_cond},
                                         {"out_hres": y_init}, det_sigmas,
                                         det_inner.fwd_with_preconditioning,
                                         model_comm_group=mcg, S_churn=0.0)
                plant = out["out_hres"] if isinstance(out, dict) else out
            else:
                plant = det_sampler.sample(x_interp_cond, x_hres_cond, y_init, det_sigmas,
                                           det_inner.fwd_with_preconditioning,
                                           model_comm_group=mcg, grid_shard_shapes=gss_arg,
                                           S_churn=0.0)
        plant = plant.detach()
        m_plant = metrics_of(plant)
        plant_info.update({"det_checkpoint": str(args.det_checkpoint),
                           "det_sigma": DET_SIGMA, "plant_metrics": m_plant})
        LOGGER.info("det_prior plant: box msl %.1f hPa",
                    (m_plant or {}).get("msl", float("nan")))
        del det_bundle, det_inner, det_sampler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elif seed_source == "file":
        if not getattr(args, "plant_file", None):
            raise SystemExit("--seed-source file requires --plant-file")
        blob = torch.load(args.plant_file, map_location=device, weights_only=False)
        plant = blob["plant"].to(device=device, dtype=y_residual_cond.dtype)
        if tuple(plant.shape) != tuple(y_residual_cond.shape):
            raise SystemExit("plant shape %s != expected %s" % (tuple(plant.shape),
                                                                tuple(y_residual_cond.shape)))
        m_here = metrics_of(plant)
        m_src = blob.get("metrics") or {}
        plant_info.update({"plant_file": str(args.plant_file),
                           "source_metrics": m_src, "plant_metrics": m_here,
                           "source_meta": blob.get("meta")})
        if m_here is not None and m_src.get("msl") is not None:
            drift = abs(float(m_here["msl"]) - float(m_src["msl"]))
            lvl = LOGGER.warning if drift > 1.0 else LOGGER.info
            lvl("plant-file residual-space check: msl here %.2f vs source %.2f (drift %.2f hPa)",
                m_here["msl"], m_src["msl"], drift)

    if getattr(args, "plant_dump", None) and global_rank == 0:
        m_plant = plant_info.get("plant_metrics") or metrics_of(plant)
        torch.save({"plant": plant.detach().to("cpu"), "metrics": m_plant,
                    "meta": {"seed_source": seed_source,
                             "checkpoint": str(args.checkpoint),
                             "det_checkpoint": str(getattr(args, "det_checkpoint", None)),
                             "bundle_paths": [str(b) for b in getattr(eb, "paths", [])] if "eb" in dir() else None}},
                   args.plant_dump)
        LOGGER.info("plant dumped to %s", args.plant_dump)
    if getattr(args, "plant_only", False):
        LOGGER.info("--plant-only: exiting after plant construction")
        return {"plant": plant_info} if global_rank == 0 else None

    def _mean(finals):
        return {k: float(np.mean([f[k] for f in finals])) for k in finals[0]}

    runs = []
    for ss in seed_sigmas:
        finals = []
        for seed in seeds:
            torch.manual_seed(int(seed))
            y_final = _seeded_sample(inner, sampler, args.num_steps, sigma_min, sigma_max,
                                     x_interp_cond, x_hres_cond, plant, ss, seed, mcg, gss_arg)
            m = metrics_of(y_final)                      # collective; rank 0 gets the dict
            if m is not None:
                finals.append(m)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if global_rank == 0:
            fm = _mean(finals)
            # Flag regional low-sigma breakdown: too many box cells below the physical floor
            # means the floored msl box-min is pegged at the floor, not a real eye.
            deg_tol = MSL_DEGENERATE_FRAC * int(box_np.sum())
            degenerate = bool(fm.get("msl_n_below_floor", 0.0) > deg_tol)
            runs.append({"seed_sigma": ss, "n_seeds": len(finals),
                         "final_mean": fm, "finals": finals, "msl_degenerate": degenerate})
            LOGGER.info("seed_sigma=%.3g -> realized msl=%.1f hPa (mean of %d; raw_min=%.1f, "
                        "n<floor=%.0f%s)", ss, fm.get("msl", float("nan")), len(finals),
                        fm.get("msl_raw_min_hpa", float("nan")), fm.get("msl_n_below_floor", 0.0),
                        " DEGENERATE" if degenerate else "")

    # free baseline = pure noise at sigma_max (the sigma_seed -> infinity asymptote).
    free_finals = []
    for seed in seeds:
        torch.manual_seed(int(seed))
        yf = _seeded_sample(inner, sampler, args.num_steps, sigma_min, sigma_max, x_interp_cond,
                            x_hres_cond, y_residual_cond, sigma_max, seed, mcg, gss_arg, free=True)
        m = metrics_of(yf)
        if m is not None:
            free_finals.append(m)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = None
    if global_rank == 0:
        free_mean = _mean(free_finals) if free_finals else None
        metrics_reported = [k for k in references["target"].keys() if k not in _DIAG_KEYS]
        # Seed sigmas whose msl row is a regional low-sigma breakdown (floored value meaningless).
        degenerate_ss = sorted({r["seed_sigma"] for r in runs if r.get("msl_degenerate")})
        summary = {}
        for name in metrics_reported:
            all_pts = sorted((r["seed_sigma"], r["final_mean"].get(name)) for r in runs
                             if name in r["final_mean"])
            if not all_pts:
                continue
            # Only trustworthy rows drive `committed` / `crossover`: drop non-physical values
            # AND (for msl) rows flagged degenerate -- a floor-pegged regional breakdown is not
            # an eye. Excluded rows are logged + recorded in the summary, never silently folded in.
            def _trust(s, y, _name=name):
                if not _seed_row_physical(_name, y):
                    return False
                return not (_name == "msl" and s in degenerate_ss)
            pts = [(s, y) for s, y in all_pts if _trust(s, y)]
            excluded = [s for s, y in all_pts if not _trust(s, y)]
            if excluded:
                LOGGER.warning("seeding summary[%s]: excluded seed_sigma rows %s "
                               "from committed/crossover (non-physical or degenerate)", name, excluded)
            free_v = (free_mean or {}).get(name)
            base = {"target": references["target"].get(name),
                    "x_interp": references["x_interp"].get(name),
                    "excluded_seed_sigmas": excluded,
                    "degenerate_seed_sigmas": (degenerate_ss if name == "msl" else [])}
            if not pts:
                summary[name] = {"committed": None, "free": free_v,
                                 "crossover_seed_sigma": None, **base}
                continue
            ys = [y for _, y in pts]
            committed = min(ys) if name == "msl" else max(ys)
            cross = None
            if free_v is not None:
                half = 0.5 * (committed + free_v)
                for s, y in pts:                          # ascending sigma_seed
                    deep = (y <= half) if name == "msl" else (y >= half)
                    if deep:
                        cross = s                         # largest sigma_seed still committed
            summary[name] = {"committed": committed, "free": free_v,
                             "crossover_seed_sigma": cross, **base}
        result = {
            "checkpoint": args.checkpoint, "ckpt_id": ckpt_id_from_path(args.checkpoint),
            "mode": "seeding", "units": "physical", "world_size": world_size,
            "seed_source": seed_source, "plant": plant_info,
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


def _run_guidance(args, bundle, inner, global_rank, world_size, mcg, gss_arg,
                  target_indices, x_interp_cond, x_hres_cond, y_residual_cond,
                  metrics_of, references, clat, clon, box_np, window, eb, out_path, fields_of=None):
    """Inference-time FIX SCREEN for the o320→o1280 deep-eye under-commitment. Runs the FREE
    sampler (pure noise at sigma_max — the deployed path) over n seeds with two optional knobs
    and reports the resulting storm-core distribution vs the unguided baseline:
      - knob-2 GUIDANCE: σ-banded score amplification, lambda in [guidance_sigma_lo, _hi];
      - knob-1 S_churn: stochastic injection (S_churn, gated to [s_churn_min, s_churn_max]).
    One job = one (lambda, band, churn) config; launch lambda/churn variants as SIBLINGS and
    compare the eye-depth distribution. NB: this storm-box screen is NOT the mandatory
    full-field gap-closed guard — a promising config must still pass the canonical
    methods_tc_extreme_compare 250-sample per-field check before any 'fix' claim."""
    device = y_residual_cond.device
    sampler, sigma_min, sigma_max = _build_sampler(inner, device)
    seeds = (list(args.seeds) if args.seeds
             else list(range(args.seed_base, args.seed_base + args.n_seeds)))

    lam = float(args.guidance_lambda)
    band = (lam, float(args.guidance_sigma_lo), float(args.guidance_sigma_hi))
    guidance = band if lam != 1.0 else None
    skw = {}
    if args.s_churn is not None:
        skw["S_churn"] = float(args.s_churn)
        if args.s_churn_min is not None:
            skw["S_min"] = float(args.s_churn_min)
        if args.s_churn_max is not None:
            skw["S_max"] = float(args.s_churn_max)

    ag_fn, ag_info = None, None
    if getattr(args, "autoguide_weight", None):
        if not getattr(args, "autoguide_checkpoint", None):
            raise SystemExit("--autoguide-weight requires --autoguide-checkpoint")
        from interp.core.model import load_model as _load_weak
        LOGGER.info("autoguidance: loading D_bad %s", args.autoguide_checkpoint)
        weak_bundle = _load_weak(args.autoguide_checkpoint, device=str(device),
                                 precision=args.precision, num_gpus_per_model=world_size)
        ag_fn = _autoguided_denoiser(inner.fwd_with_preconditioning,
                                     weak_bundle.inner_model.fwd_with_preconditioning,
                                     float(args.autoguide_weight),
                                     float(args.autoguide_sigma_lo),
                                     float(args.autoguide_sigma_hi))
        ag_info = {"checkpoint": str(args.autoguide_checkpoint),
                   "weight": float(args.autoguide_weight),
                   "sigma_lo": float(args.autoguide_sigma_lo),
                   "sigma_hi": float(args.autoguide_sigma_hi)}

    finals = []
    dumped = {}
    for seed in seeds:
        torch.manual_seed(int(seed))
        yf = _seeded_sample(inner, sampler, args.num_steps, sigma_min, sigma_max,
                            x_interp_cond, x_hres_cond, y_residual_cond, sigma_max, seed,
                            mcg, gss_arg, free=True, guidance=guidance,
                            sampler_kwargs=(skw or None), denoise_fn=ag_fn)
        if (getattr(args, "dump_fields", 0) and fields_of is not None
                and len(dumped) < int(args.dump_fields)):
            dumped[int(seed)] = fields_of(yf)
        m = metrics_of(yf)                                   # collective; rank 0 gets the dict
        if m is not None:
            finals.append(m)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if global_rank == 0 and m is not None:
            LOGGER.info("seed %d (lam=%.2g churn=%s): msl=%.1f hPa  wind_max=%.1f m/s",
                        seed, lam, skw.get("S_churn", "ckpt"),
                        m.get("msl", float("nan")), m.get("wind10m_max", float("nan")))

    if dumped and global_rank == 0:
        _, _, lat_hres, lon_hres = eb.coords
        arrs = {"lat": np.asarray(lat_hres)[box_np], "lon": np.asarray(lon_hres)[box_np]}
        for name, vals in (fields_of(y_residual_cond) or {}).items():
            arrs["truth_%s" % name] = vals
        for seed, fields in dumped.items():
            for name, vals in fields.items():
                arrs["s%d_%s" % (seed, name)] = vals
        out_path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path / "guidance_fields.npz", **arrs)
        LOGGER.info("dumped %d seed field-sets", len(dumped))

    result = None
    if global_rank == 0:
        metrics_reported = [k for k in references["target"].keys() if k not in _DIAG_KEYS]

        def _dist(name, deep_is_min):
            vals = [f[name] for f in finals if name in f and np.isfinite(f[name])]
            if not vals:
                return None
            arr = np.array(vals, dtype=float)
            best = float(arr.min() if deep_is_min else arr.max())
            worst = float(arr.max() if deep_is_min else arr.min())
            return {"n": len(vals), "best": best, "worst": worst,
                    "mean": float(arr.mean()), "std": float(arr.std()),
                    "deep_is_min": bool(deep_is_min),
                    "target": references["target"].get(name),
                    "x_interp": references["x_interp"].get(name)}

        def _gc(name, val):                                  # gap-closed (model−input)/(truth−input)
            tgt = references["target"].get(name)
            xi = references["x_interp"].get(name)
            if val is None or tgt is None or xi is None or (tgt - xi) == 0:
                return None
            return float((val - xi) / (tgt - xi))

        dist = {name: _dist(name, name == "msl") for name in metrics_reported}
        gap_closed = {}
        for name in metrics_reported:
            d = dist.get(name)
            if not d:
                continue
            gap_closed[name] = {"best": _gc(name, d["best"]), "mean": _gc(name, d["mean"])}

        result = {
            "checkpoint": args.checkpoint, "ckpt_id": ckpt_id_from_path(args.checkpoint),
            "mode": "guidance", "units": "physical", "world_size": world_size,
            "config": {"guidance_lambda": lam, "guidance_sigma_lo": band[1],
                       "guidance_sigma_hi": band[2], "sampler_kwargs": skw, "free": True,
                       "autoguide": ag_info},
            "metric_rule": {"msl": "box-min (hPa)", "wind10m": "box p99 speed (m/s)",
                            "wind10m_max": "box max speed (m/s)"},
            "bundle_paths": [str(p) for p in eb.paths],
            "surface_targets": list(target_indices.keys()), "metrics_reported": metrics_reported,
            "num_steps": args.num_steps, "fp32_sampler": bool(args.fp32_sampler),
            "seeds": [int(s) for s in seeds],
            "box": {"name": "storm", "lat": clat, "lon": clon % 360.0,
                    "radius_km": args.eye_radius_km, "n_cells": int(box_np.sum())},
            "window": list(window), "references": references,
            "distribution": dist, "gap_closed": gap_closed, "finals": finals,
        }
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "guidance.json", "w") as f:
            json.dump(result, f, indent=2)
        LOGGER.info("Results saved to %s", out_path / "guidance.json")
        write_run_meta(out_path, "trajectory_guidance", args)
        md = dist.get("msl") or {}
        wd = dist.get("wind10m_max") or dist.get("wind10m") or {}
        tgt_w = references["target"].get("wind10m_max", references["target"].get("wind10m"))
        print(f"\nGUIDANCE SCREEN — lam={lam} band=[{band[1]:.0f},{band[2]:.0f}] "
              f"churn={skw.get('S_churn', 'ckpt-default')}  (free sampler, n={len(finals)})")
        print(f"  msl   deepest={md.get('best', float('nan')):.1f}  mean={md.get('mean', float('nan')):.1f}"
              f"  (truth {references['target'].get('msl', float('nan')):.1f}, "
              f"input {references['x_interp'].get('msl', float('nan')):.1f})  "
              f"gc_best={(gap_closed.get('msl') or {}).get('best')}")
        print(f"  wind  strongest={wd.get('best', float('nan')):.1f}  mean={wd.get('mean', float('nan')):.1f}"
              f"  (truth {tgt_w if tgt_w is not None else float('nan'):.1f})")

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
    if sharded and not is_dict_api(inner):
        from anemoi.models.distributed.shapes import apply_shard_shapes, get_shard_shapes
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
    elif sharded:  # unified dict-API, grid-sharded across model_comm_group
        from anemoi.models.distributed.graph import gather_tensor, shard_tensor
        from anemoi.models.distributed.shapes import get_shard_sizes
        batch = {"in_lres": eb.x_lres[0:1].to(device), "in_hres": eb.x_hres[0:1].to(device)}
        (x_interp_cond, x_hres_cond), dss = inner._before_sampling(
            batch, bundle.pre_processors, 1, model_comm_group=mcg)
        gss = dss  # DatasetShardSizes -> threaded into denoise/sample via gss_arg
        out_sizes = dss["out_hres"]
        # apply_interpolate_to_high_res with grid_shard_sizes set drives the
        # InterpolationConnection all_to_all path, which expects a GRID-SHARDED input.
        # Passing the FULL lres grid + hres shard sizes makes the all_to_all split counts
        # disagree across ranks -> NCCL all_to_all deadlock (the tier1 b785 hang). Match
        # _before_sampling: upsample the full grid collective-free (grid_shard_sizes=None),
        # then take this rank's grid shard locally.
        xir_full = inner.apply_interpolate_to_high_res(
            eb.x_lres[0:1].to(device)[:, 0, ...],
            grid_shard_sizes=None, model_comm_group=mcg)[:, None, ...]
        x_interp_raw_sh = shard_tensor(xir_full, -2, out_sizes, mcg)
        y_sh = shard_tensor(y0, -2, get_shard_sizes(y0, -2, mcg), mcg)
        prt = getattr(bundle.model, "pre_processors_tendencies", None)
        y_residual_cond = inner.compute_residuals(
            y_sh, x_interp_raw_sh, bundle.pre_processors["out_hres"], prt["out_hres"],
            target_dataset="out_hres")

        def _gather_full(field_sh):
            return gather_tensor(field_sh, -2, out_sizes, mcg)

        surf_idx = torch.tensor(list(target_indices.values()), device=device)
        surf_remap = {name: i for i, name in enumerate(target_indices)}

        def metrics_of(residual):
            # The unified (dict-API) add_interp_to_state adds back the NORMALIZED interp
            # (x_interp_cond), NOT the raw physical interp -- this matches the single-GPU
            # dict-API path. Passing x_interp_raw_sh (physical, msl ~98300 Pa) inflated every
            # reconstructed field to ~1e6 (the b785 faithfulness bug). compute_residuals above
            # still correctly uses the RAW interp; only reconstruct takes the normalized state.
            phys_sh = reconstruct_phys(bundle, x_interp_cond, residual)[..., surf_idx]
            phys_full = _gather_full(phys_sh)
            return reduce_box(phys_full, surf_remap, box_t, has_wind) if global_rank == 0 else None

        xir_full = _gather_full(x_interp_raw_sh)
    elif is_dict_api(inner):
        # Unified (dict-API) single-GPU prep: _before_sampling gives the NORMALIZED interp
        # (which the unified add_interp_to_state adds back), compute_residuals (with the
        # SINGLE out_hres state/tendency normalizers) gives the true residual for the ceiling.
        batch = {"in_lres": eb.x_lres[0:1].to(device), "in_hres": eb.x_hres[0:1].to(device)}
        (x_interp_cond, x_hres_cond), _ = inner._before_sampling(batch, bundle.pre_processors, 1)
        x_interp_raw = inner.apply_interpolate_to_high_res(
            eb.x_lres[0:1].to(device)[:, 0, ...])[:, None, ...]
        prt = getattr(bundle.model, "pre_processors_tendencies", None)
        y_residual_cond = inner.compute_residuals(
            y0, x_interp_raw, bundle.pre_processors["out_hres"], prt["out_hres"],
            target_dataset="out_hres")
        recon_state_box = x_interp_cond[:, :, :, box_t, :]    # NORMALIZED interp = what's added back
        xi_phys_box = x_interp_raw[:, :, :, box_t, :]         # RAW physical interp for x_interp ref

        def metrics_of(residual):
            return reduce_field(
                reconstruct_phys_box(bundle, recon_state_box, residual, box_t),
                target_indices, has_wind)
    else:
        prepared = prepare_batch(bundle, eb.x_lres[0:1], eb.x_hres[0:1], eb.y[0:1])
        x_interp_cond, x_hres_cond = prepared["x_interp"], prepared["x_hres"]
        y_residual_cond = prepared["y_residual"]
        recon_state_box = prepared["x_interp_raw"][:, :, :, box_t, :]
        xi_phys_box = recon_state_box

        def metrics_of(residual):
            return reduce_field(
                reconstruct_phys_box(bundle, recon_state_box, residual, box_t),
                target_indices, has_wind)

    gss_arg = gss if sharded else None

    # --- P1 lock-in capture (single-GPU only): per-call box FIELDS per surface target ---
    fields_of = None
    if not sharded:
        def fields_of(residual):
            phys = reconstruct_phys_box(bundle, recon_state_box, residual, box_t)
            return {name: phys[0, 0, 0, :, i].detach().float().cpu().numpy()
                    for name, i in target_indices.items()}


    # References (rank 0): target from the full observed y; x_interp from the raw interp.
    references = None
    if global_rank == 0:
        references = {"target": reduce_box(y0, target_indices, box_t, has_wind)}
        fb_xi = (xir_full[:, :, :, box_t, :] if sharded else xi_phys_box)[0, 0, 0]
        xi_metrics = {name: _side_reduce(fb_xi[:, name2in[name]], name)
                      for name in target_indices if name in name2in}
        if has_wind and "10u" in name2in and "10v" in name2in:
            u, v = fb_xi[:, name2in["10u"]], fb_xi[:, name2in["10v"]]
            xi_metrics["wind10m"] = _q(torch.sqrt(u * u + v * v), CORE_Q_HIGH)
        references["x_interp"] = xi_metrics

    # PARITY SELF-CHECK (env-gated, b785 faithfulness debug): reconstruct the TRUE residual
    # -> must equal the observed target storm-core. metrics_of gathers (collective), so ALL
    # ranks must call it. Garbage here => reconstruct/gather/layout bug; fine here => the
    # model FORWARD (ceiling/realized) is the bug.
    import os as _pos
    if _pos.environ.get("INTERP_PARITY_CHECK"):
        _chk = metrics_of(y_residual_cond)
        if global_rank == 0:
            _t = references["target"]
            LOGGER.info("PARITY reconstruct(true_residual): msl=%.4f  target_msl=%.4f  "
                        "x_interp_msl=%.4f  (msl match => reconstruct OK, model-forward is the bug)",
                        _chk.get("msl"), _t.get("msl"), references["x_interp"].get("msl"))

    # A2 seeding-sigma sweep reuses the whole setup above (load / bundle / box / sharded
    # conditioning / metrics_of / references), then sweeps where the storm is planted.
    if args.mode == "residual_diag":
        return _run_residual_diag(args, bundle, target_indices, name2in, box_t,
                                  y_residual_cond, references, eb, out_path,
                                  global_rank, world_size)

    if args.mode == "seeding":
        return _run_seeding(args, bundle, inner, global_rank, world_size, mcg, gss_arg,
                            target_indices, x_interp_cond, x_hres_cond, y_residual_cond,
                            metrics_of, references, clat, clon, box_np, window, eb, out_path)

    if args.mode == "guidance":
        return _run_guidance(args, bundle, inner, global_rank, world_size, mcg, gss_arg,
                             target_indices, x_interp_cond, x_hres_cond, y_residual_cond,
                             metrics_of, references, clat, clon, box_np, window, eb, out_path, fields_of=fields_of)

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
        lock_fields = []                                     # [(sigma, {var: box np array})]

        def on_call(sigma_scalar, D, _rec=records, _lf=lock_fields):
            m = metrics_of(D)                                # collective; runs on all ranks
            if m is not None:                                # only rank 0 records
                _rec.append({"call_idx": len(_rec), "sigma": sigma_scalar, "metrics": m})
            if getattr(args, "lockin", False) and fields_of is not None:
                _lf.append((float(sigma_scalar), fields_of(D)))

        torch.manual_seed(int(seed))
        sampler_ctx = force_fp32_sampler() if args.fp32_sampler else contextlib.nullcontext()
        with capture_denoiser(inner, on_call), sampler_ctx:
            final_resid = sample_full(bundle, x_interp_cond, x_hres_cond,
                                      num_steps=args.num_steps, seed=int(seed),
                                      model_comm_group=mcg, grid_shard_shapes=gss_arg,
                                      sigma_min=SAMPLER_SIGMA_MIN)
        final_metrics = metrics_of(final_resid) or {}        # collective; {} on non-zero ranks
        lockin = None
        if getattr(args, "lockin", False) and fields_of is not None and global_rank == 0:
            fin = fields_of(final_resid)
            tgt = {name: y0[0, 0, 0, box_t, i].float().cpu().numpy()
                   for name, i in target_indices.items()}

            def _corr(a, b):
                a = a - a.mean(); b = b - b.mean()
                d = float(np.sqrt((a * a).sum() * (b * b).sum()))
                return float((a * b).sum() / d) if d > 0 else float("nan")

            # ref = FIRST-call x-hat-0 (sigma~sigma_max) ~= conditional mean given input:
            # anomaly curves isolate when the SAMPLE-SPECIFIC (generative) part commits.
            ref = lock_fields[0][1]
            lockin = {"sigmas": [s for s, _ in lock_fields], "vars": {}}
            for name in target_indices:
                fin_a = fin[name] - ref[name]
                tgt_a = tgt[name] - ref[name]
                lockin["vars"][name] = {
                    "corr_final": [_corr(f[name], fin[name]) for _, f in lock_fields],
                    "corr_target": [_corr(f[name], tgt[name]) for _, f in lock_fields],
                    "amp_ratio": [float(np.std(f[name]) / max(np.std(fin[name]), 1e-12))
                                    for _, f in lock_fields],
                    "corr_final_anom": [_corr(f[name] - ref[name], fin_a) for _, f in lock_fields],
                    "corr_target_anom": [_corr(f[name] - ref[name], tgt_a) for _, f in lock_fields],
                    "amp_ratio_anom": [float(np.std(f[name] - ref[name]) / max(np.std(fin_a), 1e-12))
                                         for _, f in lock_fields],
                }
        if global_rank == 0:
            trajectories.append({"seed": int(seed), "steps": records, "final": final_metrics,
                                 "lockin": lockin})
            if records:
                d = records[-1]["metrics"].get("msl", float("nan")) - final_metrics.get("msl", float("nan"))
                LOGGER.info("seed %d: %d denoiser calls, final msl=%.1f hPa "
                            "(last x̂₀ − final = %+.2f hPa; should be small)",
                            seed, len(records), final_metrics.get("msl", float("nan")), d)

    result = None
    if global_rank == 0:
        metrics_reported = [k for k in references["target"].keys() if k not in _DIAG_KEYS]
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
    p.add_argument("--lockin", action="store_true",
                   help="P1: capture per-call box fields and emit per-variable lock-in "
                        "(pattern-correlation vs own final / vs target) curves")
    p.add_argument("--mode", default="trajectory",
                   choices=["trajectory", "seeding", "residual_diag", "guidance"],
                   help="trajectory = ceiling + realized x̂₀ vs σ (default); "
                        "seeding = A2 sweep: plant the TRUE storm at σ_seed, sample free below, "
                        "and report final storm depth vs σ_seed (the critical-window test); "
                        "guidance = FIX SCREEN: free sampler + σ-banded score amplification "
                        "(--guidance-lambda in [--guidance-sigma-lo,-hi]) and/or --s-churn, "
                        "report the storm-core eye-depth distribution vs the unguided baseline")
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
    p.add_argument("--seed-source", choices=["truth", "self_deepest", "det_prior", "file"],
                   default="truth",
                   help="[seeding] what to plant at sigma_seed: truth = the observed target "
                        "residual (A2 diagnostic, truth-leaks); self_deepest = deepest box-msl "
                        "of --n-free-draws free draws (truth-free self-restart); det_prior = "
                        "single-forward prediction of a det-supervised checkpoint "
                        "(--det-checkpoint, D(x_T, 5e5) protocol of the 20260723 det study)")
    p.add_argument("--n-free-draws", type=int, default=20,
                   help="[seeding/self_deepest] free draws to search for the deepest plant")
    p.add_argument("--det-checkpoint", default=None,
                   help="[seeding/det_prior] training-format det-supervised checkpoint path")
    p.add_argument("--plant-file", default=None,
                   help="[seeding/file] torch.save'd plant dict from a --plant-dump run "
                        "(cross-runtime handoff when det and diffusion ckpts need different envs)")
    p.add_argument("--plant-dump", default=None,
                   help="[seeding] save the constructed plant (tensor+metrics) to this path")
    p.add_argument("--plant-only", action="store_true", default=False,
                   help="[seeding] exit after constructing (and dumping) the plant - no sweep")
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
    p.add_argument("--guidance-lambda", type=float, default=1.0,
                   help="[guidance] σ-banded score amplification factor λ (1.0 = identity / "
                        "unguided control; >1 amplifies the in-band denoising pull toward the "
                        "deep mode). D'=(1−λ)y+λD in-band ⇒ in-band score = λ·d")
    p.add_argument("--guidance-sigma-lo", type=float, default=40.0,
                   help="[guidance] lower σ of the guidance band (default 40, the storm-laying "
                        "regime from residual_diag σ*)")
    p.add_argument("--guidance-sigma-hi", type=float, default=120.0,
                   help="[guidance] upper σ of the guidance band (default 120)")
    p.add_argument("--s-churn", type=float, default=None,
                   help="[guidance] sampler S_churn (stochastic injection; None = checkpoint "
                        "default). Raise to populate the deep basin (knob-1)")
    p.add_argument("--s-churn-min", type=float, default=None,
                   help="[guidance] S_min — lower σ where churn applies (None = ckpt default)")
    p.add_argument("--s-churn-max", type=float, default=None,
                   help="[guidance] S_max — upper σ where churn applies (None = ckpt default)")
    p.add_argument("--dump-fields", type=int, default=0,
                   help="[guidance] save box physical fields (surface targets) for the first "
                        "N seeds + truth to guidance_fields.npz (for offline box-FFT spectra)")
    p.add_argument("--autoguide-checkpoint", default=None,
                   help="[guidance] D_bad checkpoint for autoguidance (same lane + API family; "
                        "loaded in-process). Karras 2024: D' = w*D_strong - (w-1)*D_weak")
    p.add_argument("--autoguide-weight", type=float, default=None,
                   help="[guidance] autoguidance weight w (>1 amplifies what the strong model "
                        "learned beyond D_bad; None = off)")
    p.add_argument("--autoguide-sigma-lo", type=float, default=0.0,
                   help="[guidance] lower sigma of the autoguidance band (default 0 = full range)")
    p.add_argument("--autoguide-sigma-hi", type=float, default=1.0e9,
                   help="[guidance] upper sigma of the autoguidance band (default unbounded)")
    p.add_argument("--fp32-sampler", action=argparse.BooleanOptionalAction, default=True,
                   help="run the diffusion sampler in fp32 (halves its float64 state; "
                        "required to fit the realized path at O1280). --no-fp32-sampler "
                        "keeps the production float64 sampler.")
    args = p.parse_args(argv)
    setup_logging()
    return run_trajectory(args)


if __name__ == "__main__":
    main()
