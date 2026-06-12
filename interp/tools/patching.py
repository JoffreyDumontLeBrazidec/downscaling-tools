"""Causal activation patching for AIFSDD (Tier-3 interpretability).

Determines which network stage commits each surface field, and at which noise
level sigma, by causally splicing CLEAN activations into a CORRUPTED forward
pass and measuring per-surface-target recovery.

Background: why naive per-block patching is degenerate here
-----------------------------------------------------------
The model is encoder -> processor -> decoder. The conditioning corruption
(`zero_lres`: x_interp := 0, mirroring ablation.ablate_lres) enters ONLY
through the encoder. The processor is a `GraphTransformerProcessor` whose 16
blocks are chained as pure functions `x, edge_attr = block(x, edge_attr,
geometry, cond=c_hidden)` where geometry and `c_hidden` (sigma-only) are
corruption-independent. Therefore splicing the clean activation into the FULL
hidden state of ANY single block forces every downstream block to recompute the
exact reference trajectory -> single-block == chunk == patch-all (verified
empirically: all identical). Full-hidden-state single-block patching cannot
resolve individual blocks for this architecture.

Three non-degenerate modes (all implemented)
--------------------------------------------
1. residual     -- the PRIMARY per-block localizer. The corruption is injected
   INTO THE PROCESSOR RESIDUAL STREAM: a fixed, per-block pseudo-random vector is
   added to EVERY block's output[0] (scaled to a fixed fraction of that output's
   norm). Because the corruption is re-introduced at every block, splicing the
   clean reference into block L's FULL output restores L exactly while blocks
   0..L-1 stay corrupted and blocks L+1..15 re-corrupt -> recovery rises
   monotonically toward the decoder. The conditioning is left CLEAN, so the
   decoder/skip path is uncorrupted and:
       patch_none -> 0  (every block corrupted)
       patch_all  -> 1  (every block clean == exact reference)
   This is the mode whose `sanity`/`recovery_per_block` are the headline result.

2. grid_region  -- patch only a STORM-CENTERED SPATIAL SUBSET of the hidden-grid
   nodes at each block under the CONDITIONING corruption (`zero_lres`). Downstream
   graph-attention then mixes patched (clean) and un-patched (corrupted) nodes, so
   the effect becomes block-dependent. Recovery is measured over the storm region
   on the output (data) grid. NOTE: because the conditioning corruption also feeds
   the decoder directly, patch_all here CANNOT reach 1.0 (skip-path ceiling); use
   `residual` for the clean per-block localization signal.

3. stage        -- patch reference activations at network CUT POINTS:
     enc_data  : encoder output #0 (data-grid latent feeding the decoder)
     enc_hidden: encoder output #1 (hidden-grid latent feeding the processor)
     enc_both  : both encoder outputs (must recover ~1.0 -- corruption is an
                 encoder input, so restoring both encoder outputs restores all)
     proc_out  : processor output (== full patch-all; decoder still sees the
                 corrupted data-grid latent + additive skip x_latent)
   Recovery measured globally. Answers "which STAGE commits each field at sigma".

The diffusion noise tensor is generated once and reused for every reference /
corrupted / patched call, so the only things that differ are (a) the conditioning
corruption and (b) the spliced activation.

Usage
-----
    cd ~/dev/downscaling-tools
    python -m interp patching \
        --checkpoint /path/to/checkpoint.ckpt \
        --output-dir ~/perm/interp/<ckpt_id>/activation_patching \
        --modes residual grid_region stage --corruption zero_lres
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import torch

_DT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_DT_ROOT) not in sys.path:
    sys.path.insert(0, str(_DT_ROOT))

from interp.cli import add_model_args, add_sigma_args, setup_logging
from interp.core.data import EVENTS, load_single_bundle
from interp.core.geometry import (
    ATLANTIC_WINDOW,
    detect_min_center,
    great_circle_mask_deg,
    node_latlon_deg,
    norm_lon,
)
from interp.core.hooks import iter_processor_blocks
from interp.core.model import (
    denoise_at_sigma,
    get_surface_target_indices,
    load_model,
    prepare_batch,
)
from interp.core.runmeta import write_run_meta

LOGGER = logging.getLogger(__name__)

STAGE_NAMES = ["enc_data", "enc_hidden", "enc_both", "proc_out"]

# Real TC event bundle (Franklin window; valid 2023-08-29 = peak near Bermuda).
# Storm-centered interp MUST feed real bundles, never val_dataloader.
_DEFAULT_BUNDLE = EVENTS["franklin_o96_o320"]["peak_bundle"]


# ---------------------------------------------------------------------------
# Corruption
# ---------------------------------------------------------------------------

def corrupt_conditioning(prepared: dict, mode: str):
    x_interp = prepared["x_interp"]
    x_hres = prepared["x_hres"]
    if mode == "zero_lres":
        return torch.zeros_like(x_interp), x_hres
    if mode == "zero_hres":
        return x_interp, torch.zeros_like(x_hres)
    if mode == "zero_both":
        return torch.zeros_like(x_interp), torch.zeros_like(x_hres)
    raise ValueError(f"unknown corruption mode: {mode}")


# ---------------------------------------------------------------------------
# Per-target MSE, optionally restricted to a node subset (grid axis = -2)
# ---------------------------------------------------------------------------

def per_target_mse_region(pred, ref, target_indices: dict, node_mask_t=None) -> dict:
    out = {}
    p = pred.float()
    r = ref.float()
    for name, idx in target_indices.items():
        pf = p[..., idx]  # (..., grid)
        rf = r[..., idx]
        if node_mask_t is not None:
            pf = pf[..., node_mask_t]
            rf = rf[..., node_mask_t]
        out[name] = float(((pf - rf) ** 2).mean().item())
    return out


def _recovery(mse_patched: dict, mse_corrupted: dict, ti) -> dict:
    rec = {}
    for t in ti:
        d = mse_corrupted[t]
        rec[t] = float(1.0 - mse_patched[t] / d) if d > 0 else float("nan")
    return rec


# ---------------------------------------------------------------------------
# Hooks: reference capture (block activations + stage activations)
# ---------------------------------------------------------------------------

class ReferenceCapture:
    """Capture each processor block's output[0] AND encoder/processor outputs."""

    def __init__(self, inner, blocks):
        self.inner = inner
        self.blocks = blocks
        self.block_acts: dict[str, torch.Tensor] = {}
        self.enc = None       # tuple (x_data_latent, x_latent)
        self.proc = None      # tensor
        self.proc_is_tuple = False
        self._hooks = []

    def __enter__(self):
        for idx, module in self.blocks:
            self._hooks.append(module.register_forward_hook(self._make_block(f"L{idx}")))
        self._hooks.append(self.inner.encoder.register_forward_hook(self._enc))
        self._hooks.append(self.inner.processor.register_forward_hook(self._proc))
        return self

    def _make_block(self, name):
        def hook(module, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            self.block_acts[name] = t.detach().clone()
        return hook

    def _enc(self, module, inp, out):
        if isinstance(out, tuple):
            self.enc = tuple(t.detach().clone() if torch.is_tensor(t) else t for t in out)
        else:
            self.enc = out.detach().clone()

    def _proc(self, module, inp, out):
        self.proc_is_tuple = isinstance(out, tuple)
        t = out[0] if isinstance(out, tuple) else out
        self.proc = t.detach().clone()

    def __exit__(self, *exc):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        return False


# ---------------------------------------------------------------------------
# Hooks: patching
# ---------------------------------------------------------------------------

class BlockRegionPatch:
    """Replace only storm-region rows of selected blocks' output[0] with reference."""

    def __init__(self, blocks, ref_acts: dict, patch_names, node_mask_t):
        self.blocks = blocks
        self.ref_acts = ref_acts
        self.patch_names = set(patch_names)
        self.node_mask_t = node_mask_t  # bool tensor over node dim (axis 0)
        self._hooks = []

    def __enter__(self):
        for idx, module in self.blocks:
            name = f"L{idx}"
            if name in self.patch_names:
                self._hooks.append(module.register_forward_hook(self._make(name)))
        return self

    def _make(self, name):
        ref = self.ref_acts[name]
        m = self.node_mask_t

        def hook(module, inp, out):
            cur = out[0] if isinstance(out, tuple) else out
            patched = cur.clone()
            patched[m] = ref[m]
            if isinstance(out, tuple):
                return (patched,) + tuple(out[1:])
            return patched
        return hook

    def __exit__(self, *exc):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        return False


class StagePatch:
    """Patch reference activations at an encoder/processor cut point."""

    def __init__(self, inner, enc_ref, proc_ref, which: str):
        self.inner = inner
        self.enc_ref = enc_ref
        self.proc_ref = proc_ref
        self.which = which
        self._hooks = []

    def __enter__(self):
        if self.which.startswith("enc"):
            self._hooks.append(self.inner.encoder.register_forward_hook(self._enc))
        if self.which == "proc_out":
            self._hooks.append(self.inner.processor.register_forward_hook(self._proc))
        return self

    def _enc(self, module, inp, out):
        a, b = out[0], out[1]
        if self.which in ("enc_data", "enc_both"):
            a = self.enc_ref[0]
        if self.which in ("enc_hidden", "enc_both"):
            b = self.enc_ref[1]
        return (a, b) + tuple(out[2:])

    def _proc(self, module, inp, out):
        if isinstance(out, tuple):
            return (self.proc_ref,) + tuple(out[1:])
        return self.proc_ref

    def __exit__(self, *exc):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        return False


class ResidualStreamPatch:
    """Inject corruption into the processor residual stream, splice clean refs.

    For every block a fixed pseudo-random perturbation vector (seeded per block,
    so it is identical across the reference / corrupted / patched calls) is added
    to output[0], scaled to `noise_scale` * ||output||. Blocks whose name is in
    `patch_names` instead have their output[0] REPLACED by the captured clean
    reference activation (no perturbation).
    """

    def __init__(self, blocks, ref_acts: dict, patch_names, noise_scale: float, seed: int):
        self.blocks = blocks
        self.ref_acts = ref_acts
        self.patch_names = set(patch_names)
        self.noise_scale = noise_scale
        self.seed = seed
        self._hooks = []
        self._pert: dict[str, torch.Tensor] = {}

    def __enter__(self):
        for idx, module in self.blocks:
            name = f"L{idx}"
            self._hooks.append(module.register_forward_hook(self._make(name, idx)))
        return self

    def _make(self, name, idx):
        is_patched = name in self.patch_names

        def hook(module, inp, out):
            cur = out[0] if isinstance(out, tuple) else out
            if is_patched:
                new = self.ref_acts[name]
            else:
                pert = self._pert.get(name)
                if pert is None:
                    gen = torch.Generator(device=cur.device)
                    gen.manual_seed(self.seed + idx)
                    pert = torch.randn(cur.shape, generator=gen, device=cur.device,
                                       dtype=cur.dtype)
                    self._pert[name] = pert
                scale = self.noise_scale * cur.norm() / (pert.norm() + 1e-9)
                new = cur + scale * pert
            if isinstance(out, tuple):
                return (new,) + tuple(out[1:])
            return new
        return hook

    def __exit__(self, *exc):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        return False


# ---------------------------------------------------------------------------
# Per-sigma compute
# ---------------------------------------------------------------------------

def _run(bundle, x_interp, x_hres, y_residual, sigma, noise):
    return denoise_at_sigma(bundle, x_interp, x_hres, y_residual, sigma, noise=noise)


def compute_at_sigma(bundle, prepared, blocks, ti, sigma, noise, corruption, modes,
                     hidden_mask_t, data_mask_t, residual_noise_scale=0.15,
                     residual_seed=1000):
    x_interp = prepared["x_interp"]
    x_hres = prepared["x_hres"]
    y_residual = prepared["y_residual"]
    x_interp_c, x_hres_c = corrupt_conditioning(prepared, corruption)
    inner = bundle.inner_model

    block_names = [f"L{idx}" for idx, _ in blocks]
    cb = len(block_names) // 2
    chunk_a, chunk_b = block_names[:cb], block_names[cb:]

    # Reference run (captures block + stage activations)
    with ReferenceCapture(inner, blocks) as cap:
        ref_out = _run(bundle, x_interp, x_hres, y_residual, sigma, noise)
        ref_blocks = dict(cap.block_acts)
        enc_ref = cap.enc
        proc_ref = cap.proc

    # Corrupted run (no hooks)
    corr_out = _run(bundle, x_interp_c, x_hres_c, y_residual, sigma, noise)

    res = {"sigma": float(sigma)}

    # ---- residual mode (primary per-block localizer) ----------------------
    if "residual" in modes:
        def residual_patched(patch_names):
            with ResidualStreamPatch(blocks, ref_blocks, patch_names,
                                     residual_noise_scale, residual_seed):
                out = _run(bundle, x_interp, x_hres, y_residual, sigma, noise)
            return per_target_mse_region(out, ref_out, ti, None)

        mse_corr_res = residual_patched([])            # all blocks corrupted
        rec_block, mse_block = {}, {}
        for name in block_names:
            m = residual_patched([name])
            mse_block[name] = m
            rec_block[name] = _recovery(m, mse_corr_res, ti)
        rec_chunk, mse_chunk = {}, {}
        for label, names in [("A", chunk_a), ("B", chunk_b)]:
            m = residual_patched(names)
            mse_chunk[label] = m
            rec_chunk[label] = _recovery(m, mse_corr_res, ti)
        rec_all = _recovery(residual_patched(block_names), mse_corr_res, ti)
        rec_none = _recovery(mse_corr_res, mse_corr_res, ti)  # == 0 by construction
        res["residual"] = {
            "noise_scale": float(residual_noise_scale),
            "mse_corrupted_per_target": mse_corr_res,
            "recovery_per_block": rec_block,
            "mse_patched_per_block": mse_block,
            "recovery_per_chunk": rec_chunk,
            "mse_patched_per_chunk": mse_chunk,
            "sanity": {"patch_all": rec_all, "patch_none": rec_none},
        }

    # ---- grid_region mode -------------------------------------------------
    if "grid_region" in modes:
        mse_corr_region = per_target_mse_region(corr_out, ref_out, ti, data_mask_t)
        ref_energy = {t: float((ref_out[..., idx].float() ** 2)[..., data_mask_t].mean().item())
                      for t, idx in ti.items()}

        def region_patched(patch_names):
            with BlockRegionPatch(blocks, ref_blocks, patch_names, hidden_mask_t):
                out = _run(bundle, x_interp_c, x_hres_c, y_residual, sigma, noise)
            return per_target_mse_region(out, ref_out, ti, data_mask_t)

        rec_block, mse_block = {}, {}
        for name in block_names:
            m = region_patched([name])
            mse_block[name] = m
            rec_block[name] = _recovery(m, mse_corr_region, ti)
        rec_chunk, mse_chunk = {}, {}
        for label, names in [("A", chunk_a), ("B", chunk_b)]:
            m = region_patched(names)
            mse_chunk[label] = m
            rec_chunk[label] = _recovery(m, mse_corr_region, ti)
        rec_all = _recovery(region_patched(block_names), mse_corr_region, ti)
        rec_none = _recovery(per_target_mse_region(corr_out, ref_out, ti, data_mask_t),
                             mse_corr_region, ti)
        res["grid_region"] = {
            "mse_corrupted_region_per_target": mse_corr_region,
            "ref_meansq_region_per_target": ref_energy,
            "recovery_per_block": rec_block,
            "mse_patched_per_block": mse_block,
            "recovery_per_chunk": rec_chunk,
            "mse_patched_per_chunk": mse_chunk,
            "sanity": {"patch_all": rec_all, "patch_none": rec_none},
        }

    # ---- stage mode -------------------------------------------------------
    if "stage" in modes:
        mse_corr_global = per_target_mse_region(corr_out, ref_out, ti, None)
        rec_stage, mse_stage = {}, {}
        for which in STAGE_NAMES:
            with StagePatch(inner, enc_ref, proc_ref, which):
                out = _run(bundle, x_interp_c, x_hres_c, y_residual, sigma, noise)
            m = per_target_mse_region(out, ref_out, ti, None)
            mse_stage[which] = m
            rec_stage[which] = _recovery(m, mse_corr_global, ti)
        res["stage"] = {
            "stage_names": STAGE_NAMES,
            "mse_corrupted_global_per_target": mse_corr_global,
            "recovery_per_stage": rec_stage,
            "mse_patched_per_stage": mse_stage,
        }

    return res


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def detect_atlantic_low(y, lat_d, lon_d, msl_ch: int) -> dict:
    """Deepest N. Atlantic low in the real-bundle target field -> storm center."""
    y5 = y if y.dim() == 5 else y[:, None]
    msl = y5[0, 0, 0, :, msl_ch].float().cpu().numpy()
    lat0, lon0 = detect_min_center(msl, lat_d, lon_d, window=ATLANTIC_WINDOW)
    return {"lat": lat0, "lon": float(norm_lon(lon0)), "msl": float(msl.min())}


def run_activation_patching(args):
    sigmas = args.sigmas
    modes = list(args.modes)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading model from %s", args.checkpoint)
    bundle = load_model(args.checkpoint, device=args.device, precision=args.precision)

    inner = bundle.inner_model
    blocks = iter_processor_blocks(inner)
    block_names = [f"L{idx}" for idx, _ in blocks]
    ti = get_surface_target_indices(bundle)
    LOGGER.info("Processor blocks: %s ; surface targets: %s", block_names, ti)

    # REAL event bundle (never val_dataloader) + storm-centered region masks
    LOGGER.info("Loading input from real bundle: %s", args.bundle)
    eb = load_single_bundle(bundle, args.bundle)
    x_lres, x_hres = eb.x_lres.to(args.device), eb.x_hres.to(args.device)
    y = eb.y.to(args.device)

    lat_h, lon_h = node_latlon_deg(bundle.graph_data, inner._graph_name_hidden)
    lat_d, lon_d = node_latlon_deg(bundle.graph_data, inner._graph_name_data)
    if args.storm_center is not None:
        storm = {"lat": float(args.storm_center[0]), "lon": float(args.storm_center[1]),
                 "msl": None, "override": True}
    else:
        storm = detect_atlantic_low(y, lat_d, lon_d, ti["msl"])
    LOGGER.info("Storm scene: %s", storm)

    hidden_mask = great_circle_mask_deg(lat_h, lon_h, storm["lat"], storm["lon"],
                                        args.region_radius_deg)
    data_mask = great_circle_mask_deg(lat_d, lon_d, storm["lat"], storm["lon"],
                                      args.region_radius_deg)
    LOGGER.info("region radius=%.1f deg -> hidden nodes=%d/%d  data nodes=%d/%d",
                args.region_radius_deg, int(hidden_mask.sum()), len(lat_h),
                int(data_mask.sum()), len(lat_d))
    hidden_mask_t = torch.as_tensor(hidden_mask, device=args.device)
    data_mask_t = torch.as_tensor(data_mask, device=args.device)

    prepared = prepare_batch(bundle, x_lres, x_hres, y)
    torch.manual_seed(args.seed)
    noise = torch.randn_like(prepared["y_residual"])

    results = {
        "checkpoint": args.checkpoint,
        "corruption": args.corruption,
        "modes": modes,
        "surface_targets": list(ti.keys()),
        "block_names": block_names,
        "chunk_boundary": len(block_names) // 2,
        "stage_names": STAGE_NAMES,
        "residual_noise_scale": args.residual_noise_scale,
        "seed": args.seed,
        "storm": {
            "label": args.storm_label,
            "bundle": args.bundle,
            "center_lat": storm["lat"],
            "center_lon": storm["lon"],
            "msl_pa": storm.get("msl"),
            "radius_deg": args.region_radius_deg,
            "n_hidden_region": int(hidden_mask.sum()),
            "n_data_region": int(data_mask.sum()),
        },
        "sigma_results": [],
    }

    for sigma in sigmas:
        LOGGER.info("Patching at sigma=%.4g", sigma)
        results["sigma_results"].append(
            compute_at_sigma(bundle, prepared, blocks, ti, sigma, noise,
                             args.corruption, modes, hidden_mask_t, data_mask_t,
                             residual_noise_scale=args.residual_noise_scale))

    out_file = output_path / "activation_patching.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    LOGGER.info("Results saved to %s", out_file)
    write_run_meta(output_path, "patching", args)
    _print_summary(results)
    return results


def _print_summary(results: dict):
    ti = results["surface_targets"]
    st = results["storm"]
    print("\n" + "=" * 84)
    print("ACTIVATION PATCHING  corruption=%s  storm=%s @ (%.2f,%.2f) %sPa  region r=%.1f deg"
          % (results["corruption"], st["label"], st["center_lat"], st["center_lon"],
             ("%.0f" % st["msl_pa"]) if st.get("msl_pa") else "?", st["radius_deg"]))
    print("=" * 84)
    for res in results["sigma_results"]:
        print(f"\nSigma = {res['sigma']:g}")
        for mode, denom_key in (("residual", "mse_corrupted_per_target"),
                                ("grid_region", "mse_corrupted_region_per_target")):
            if mode not in res:
                continue
            rr = res[mode]
            print(f"  [{mode}] per-block recovery (denom MSE_corr: "
                  + " ".join(f"{t}={rr[denom_key][t]:.3e}" for t in ti) + ")")
            print(f"  {'block':>8s} " + " ".join(f"{t:>8s}" for t in ti))
            for name in results["block_names"]:
                r = rr["recovery_per_block"][name]
                print(f"  {name:>8s} " + " ".join(f"{r[t]:8.3f}" for t in ti))
            for label in ("A", "B"):
                r = rr["recovery_per_chunk"][label]
                print(f"  {'chunk-' + label:>8s} " + " ".join(f"{r[t]:8.3f}" for t in ti))
            ra, rn = rr["sanity"]["patch_all"], rr["sanity"]["patch_none"]
            print(f"  {'ALL':>8s} " + " ".join(f"{ra[t]:8.3f}" for t in ti))
            print(f"  {'NONE':>8s} " + " ".join(f"{rn[t]:8.3f}" for t in ti))
        if "stage" in res:
            sg = res["stage"]
            print("  [stage] global recovery")
            print(f"  {'stage':>10s} " + " ".join(f"{t:>8s}" for t in ti))
            for which in results["stage_names"]:
                r = sg["recovery_per_stage"][which]
                print(f"  {which:>10s} " + " ".join(f"{r[t]:8.3f}" for t in ti))
    print("\nNotes: residual NONE~0 / ALL~1.0; per-block recovery rises toward the decoder;")
    print("  grid_region recovery ~1 => that block carries the storm field;")
    print("  stage enc_both must be ~1.0 (corruption only enters the encoder);")
    print("  stage proc_out == full patch-all (decoder still sees corrupted data latent + skip).")


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Causal activation patching for AIFSDD")
    add_model_args(p)
    add_sigma_args(p)
    p.add_argument("--modes", nargs="+", default=["residual", "grid_region", "stage"],
                   choices=["residual", "grid_region", "stage"])
    p.add_argument("--corruption", default="zero_lres",
                   choices=["zero_lres", "zero_hres", "zero_both"])
    p.add_argument("--region-radius-deg", type=float, default=8.0)
    p.add_argument("--residual-noise-scale", type=float, default=0.15,
                   help="residual-stream corruption magnitude as a fraction of each "
                        "block output's norm (residual mode)")
    p.add_argument("--storm-label", default="franklin")
    p.add_argument("--storm-center", nargs=2, type=float, default=None,
                   metavar=("LAT", "LON"),
                   help="override storm center; else auto-detect deepest Atlantic low")
    p.add_argument("--bundle", default=_DEFAULT_BUNDLE,
                   help="real TC event input bundle (.nc); never val_dataloader")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    setup_logging()
    return run_activation_patching(args)


if __name__ == "__main__":
    main()
