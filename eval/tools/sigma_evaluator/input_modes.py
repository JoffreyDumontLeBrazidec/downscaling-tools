"""Noisy-input modes of the per-sigma probe (experiment U2, registry TD-220, 2026-10-15).

Question: at high noise levels, does the denoiser D read its noisy input, or does it return the
conditional mean of the residual given the coarse input whatever the noisy input holds?

For every bundle (one date) and every sigma the probe evaluates D(x, sigma) ONCE per mode, on
inputs that share one noise draw eps (same eps for the three modes of a (date, sigma), and for
every checkpoint and rank count, because eps is drawn on the full grid from a CPU generator
seeded by (base seed, date, sigma index)):

  truth       x = y0 + sigma * eps               y0 = normalised truth residual of this date
  noise_only  x = sigma * eps                    what a sampler started at sigma_max = sigma feeds
  other_date  x = y0(other date) + sigma * eps   another date's normalised truth residual

y0 is read off the model's own output map: post(out) = a + b * out per point and channel, with
a = post(0) and b = post(1) - post(0) from ``_after_sampling``; y0 = (truth - a) / b. D is the
model's ``fwd_with_preconditioning`` (the function the sampler calls), and the saved field is
post(D) in physical units, i.e. what a one-step sampler from sigma would return.

Opt-in only: ``run_sigma_evaluator --noisy-input-mode ...`` dispatches here before anything of
the default path runs, so the default sigma evaluator is unchanged.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

INPUT_MODES = ("truth", "noise_only", "other_date")
DEFAULT_EPS_SEED = 756
_DATE_RE = re.compile(r"date(\d{8})")


def _log(msg: str) -> None:
    print(f"[u2-probe] {msg}", flush=True)


# --------------------------------------------------------------------------------------------
# pure helpers (unit-tested)
# --------------------------------------------------------------------------------------------
def parse_input_modes(raw: str) -> list[str]:
    """Comma-separated subset of INPUT_MODES, order kept, no duplicates."""
    modes = [m.strip() for m in str(raw or "").split(",") if m.strip()]
    if not modes:
        raise ValueError("--noisy-input-mode needs at least one of " + ", ".join(INPUT_MODES))
    bad = [m for m in modes if m not in INPUT_MODES]
    if bad:
        raise ValueError(f"unknown noisy-input mode(s) {bad}; choose from {list(INPUT_MODES)}")
    if len(set(modes)) != len(modes):
        raise ValueError(f"duplicate noisy-input modes in {modes}")
    return modes


def bundle_date(path) -> str:
    m = _DATE_RE.search(Path(str(path)).name)
    if not m:
        raise ValueError(f"cannot read a date (dateYYYYMMDD) from bundle name {Path(str(path)).name}")
    return m.group(1)


def resolve_other_dates(dates: list[str], raw_map: str = "") -> dict[str, str]:
    """Other date per date.

    ``raw_map`` "d1:d2,d3:d4" gives the pairs explicitly (every date needs one, the other date must
    be in ``dates`` and differ from it). Without it the rule is deterministic: the date
    len(dates)//2 places further along the sorted list, cyclically (5 dates: +2).
    """
    dates = list(dates)
    if len(set(dates)) != len(dates):
        raise ValueError(f"other_date needs one bundle per date, got {dates}")
    if raw_map and raw_map.strip():
        out = {}
        for tok in raw_map.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if ":" not in tok:
                raise ValueError(f"--other-date-map entry {tok!r} is not 'date:other'")
            d, o = (t.strip() for t in tok.split(":", 1))
            out[d] = o
        missing = [d for d in dates if d not in out]
        if missing:
            raise ValueError(f"--other-date-map has no entry for {missing}")
        for d in dates:
            if out[d] not in dates:
                raise ValueError(f"--other-date-map: other date {out[d]} of {d} is not among the bundles {dates}")
            if out[d] == d:
                raise ValueError(f"--other-date-map: {d} is paired with itself")
        return {d: out[d] for d in dates}
    if len(dates) < 2:
        raise ValueError("other_date needs at least two bundles (dates)")
    order = sorted(dates)
    shift = max(len(order) // 2, 1)
    return {d: order[(i + shift) % len(order)] for i, d in enumerate(order)}


def eps_seed(base_seed: int, date: str, sigma_index: int) -> int:
    """Seed of the noise draw of one (date, sigma): independent of mode, checkpoint and rank count."""
    return (int(base_seed) * 1_000_003 + int(date) * 101 + int(sigma_index)) % (2**63 - 1)


def draw_eps(n_points: int, n_channels: int, seed: int) -> torch.Tensor:
    """Full-grid standard normal noise (float64, CPU), reproducible from ``seed`` alone."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return torch.randn((n_points, n_channels), generator=gen, dtype=torch.float64)


def noisy_state(mode: str, sigma: float, eps: torch.Tensor, y0_self: torch.Tensor | None,
                y0_other: torch.Tensor | None) -> torch.Tensor:
    """The denoiser input of one mode (all tensors on the same grid slice)."""
    if mode == "truth":
        return y0_self + float(sigma) * eps
    if mode == "noise_only":
        return float(sigma) * eps
    if mode == "other_date":
        return y0_other + float(sigma) * eps
    raise ValueError(f"unknown noisy-input mode {mode!r}")


def shard_bounds(shard_sizes, rank: int, n_points: int) -> tuple[int, int]:
    """[start, end) of this rank's contiguous grid shard (shard_tensor splits dim -2 in rank order)."""
    if shard_sizes is None:
        return 0, int(n_points)
    sizes = [int(s) for s in shard_sizes]
    if sum(sizes) != int(n_points):
        raise ValueError(f"shard sizes {sizes} do not add up to {n_points} grid points")
    start = sum(sizes[:rank])
    return start, start + sizes[rank]


def _rms(x: np.ndarray) -> np.ndarray:
    """Per-channel root mean square over points, ignoring NaN."""
    return np.sqrt(np.nanmean(np.asarray(x, dtype=np.float64) ** 2, axis=0))


# --------------------------------------------------------------------------------------------
# model-facing pieces
# --------------------------------------------------------------------------------------------
def _out_names(datamodule) -> list[str]:
    n2i = datamodule.data_indices["out_hres"].model.output.name_to_index
    return [n for n, _ in sorted(n2i.items(), key=lambda kv: kv[1])]


def _tendency_kwargs(inference_model) -> dict:
    kw = {}
    if hasattr(inference_model, "pre_processors_tendencies"):
        kw["pre_processors_tendencies"] = inference_model.pre_processors_tendencies
    if hasattr(inference_model, "post_processors_tendencies"):
        kw["post_processors_tendencies"] = inference_model.post_processors_tendencies
    return kw


def _force_edges_sharding(inference_model) -> int:
    """heads -> edges on every module, as eval.predict.model_loader does for model-parallel inference."""
    n = 0
    for mod in inference_model.modules():
        if getattr(mod, "shard_strategy", None) == "heads":
            mod.shard_strategy = "edges"
            n += 1
        if hasattr(mod, "_cached_halo_info"):
            mod._cached_halo_info = None
        if hasattr(mod, "_cached_partition"):
            mod._cached_partition = None
    return n


def _load_bundle_inputs(datamodule, inference_model, bundle_nc: Path, names: list[str]) -> dict:
    """Inputs and truth of one bundle, read exactly as manual_inference predict_from_bundle reads them."""
    from manual_inference.input_data_construction import bundle as mib

    n2i_lres = datamodule.data_indices["in_lres"].data.input.name_to_index
    n2i_hres = datamodule.data_indices["in_hres"].data.input.name_to_index
    ds = mib.open_bundle_dataset(bundle_nc)
    try:
        x_lres, x_hres, _lon_l, _lat_l, lon_h, lat_h = mib.load_inputs_from_bundle_numpy(
            ds, n2i_lres, n2i_hres, prev_bundle=mib.previous_step_bundle_path(bundle_nc))
        mib.check_input_distribution(inference_model, x_lres, n2i_lres, label=Path(str(bundle_nc)).name)
        truth, found = mib.extract_target_from_bundle_dataset(ds, names)
    finally:
        try:
            ds.close()
        except Exception:  # noqa: BLE001
            pass
    if truth is None:
        raise ValueError(f"bundle has no target_hres_* truth: {bundle_nc}")
    truth, lon_h, lat_h = np.asarray(truth, dtype=np.float32), np.asarray(lon_h, dtype=np.float64), np.asarray(lat_h, dtype=np.float64)
    # a model with an eval.predict local graph cut predicts only its data-mask points (as in predict_from_bundle)
    gmask = getattr(inference_model, "_local_graph_cut_data_mask", None)
    if gmask is not None:
        m = gmask.detach().cpu().numpy().astype(bool)
        truth, lon_h, lat_h = truth[m], lon_h[m], lat_h[m]
    return {"x_lres": np.asarray(x_lres), "x_hres": np.asarray(x_hres), "truth": truth,
            "found": int(found), "lon": lon_h, "lat": lat_h}


class _Denoiser:
    """D(x, sigma) and the output map of one inference model, on the model's own call path."""

    def __init__(self, inference_model, device, model_comm_group):
        self.iface = inference_model
        self.inner = getattr(inference_model, "model", inference_model)
        self.device = device
        self.mcg = model_comm_group
        self.tend = _tendency_kwargs(inference_model)

    def before(self, x_lres: np.ndarray, x_hres: np.ndarray):
        batch = {"in_lres": torch.from_numpy(x_lres).to(self.device)[None, None, None, ...],
                 "in_hres": torch.from_numpy(x_hres).to(self.device)[None, None, None, ...]}
        before, gss = self.inner._before_sampling(batch, self.iface.pre_processors, self.iface.n_step_input,
                                                  self.mcg, **self.tend)
        return before, gss

    def after(self, out: torch.Tensor, before, gss) -> torch.Tensor:
        return self.inner._after_sampling(out, self.iface.post_processors, before, self.mcg, gss, True, **self.tend)

    def affine(self, n_local: int, n_ch: int, before, gss):
        """a = post(0), b = post(1) - post(0) on the full (gathered) grid, float64 where possible."""
        shape = (1, 1, 1, n_local, n_ch)
        try:
            p0 = self.after(torch.zeros(shape, dtype=torch.float64, device=self.device), before, gss)
            p1 = self.after(torch.ones(shape, dtype=torch.float64, device=self.device), before, gss)
        except Exception as exc:  # noqa: BLE001
            _log(f"WARNING float64 output map failed ({type(exc).__name__}: {exc}); using float32")
            dt = before[0].dtype
            p0 = self.after(torch.zeros(shape, dtype=dt, device=self.device), before, gss)
            p1 = self.after(torch.ones(shape, dtype=dt, device=self.device), before, gss)
        a = p0[0, 0, 0].double()
        return a, p1[0, 0, 0].double() - a

    def denoise(self, state_local: torch.Tensor, sigma: float, before, gss) -> torch.Tensor:
        """post(D(state, sigma)) on the full grid; state is this rank's (n_local, C) slice."""
        dtype = before[0].dtype
        y = state_local.to(self.device).to(dtype)[None, None, None, ...]
        sig = torch.tensor(float(sigma), dtype=torch.float64, device=self.device)
        sig = sig.view(1, 1, 1, 1, 1).expand(y.shape[0], y.shape[1], y.shape[2], 1, 1).to(dtype)
        x = {"in_lres": before[0], "in_hres": before[1]}
        D = self.inner.fwd_with_preconditioning(x, {"out_hres": y}, {"out_hres": sig}, self.mcg, gss)["out_hres"]
        return self.after(D.to(dtype), before, gss)[0, 0, 0]


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


# --------------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------------
def run_input_mode_probe(args) -> Path:
    from manual_inference.prediction import predict as mip

    modes = parse_input_modes(args.noisy_input_mode)
    bundle_root = (getattr(args, "bundle_root", "") or "").strip()
    if not bundle_root:
        raise ValueError("--noisy-input-mode needs --bundle-root (one *_input_bundle.nc per date)")
    if (getattr(args, "checkpoint_compat_profile", "") or "").strip():
        raise ValueError("--checkpoint-compat-profile is not supported with --noisy-input-mode")
    if not args.sigmas.strip():
        raise ValueError("--noisy-input-mode needs an explicit --sigmas list")
    sigma_values = [float(x.strip()) for x in args.sigmas.split(",") if x.strip()]

    bundle_paths = sorted(Path(bundle_root).expanduser().glob("*_input_bundle.nc"))
    if args.n_samples > 0:
        bundle_paths = bundle_paths[: args.n_samples]
    if not bundle_paths:
        raise FileNotFoundError(f"no *_input_bundle.nc under {bundle_root}")
    dates = [bundle_date(p) for p in bundle_paths]
    if len(set(dates)) != len(dates):
        raise ValueError(f"--noisy-input-mode needs one bundle per date, got dates {dates}")
    other = resolve_other_dates(dates, getattr(args, "other_date_map", "")) if "other_date" in modes else {}

    out_csv = Path(args.out_csv) if args.out_csv else Path(args.ckpt_root) / args.name_exp / args.out_file
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    save_dir = out_csv.parent / "denoised"
    base_seed = int(os.environ.get("ANEMOI_BASE_SEED", DEFAULT_EPS_SEED))
    save_scope = (getattr(args, "save_scope_json", "") or "").strip() or None

    global_rank, local_rank, world_size = mip._get_parallel_info()
    requested = int(args.num_gpus_per_model) if int(args.num_gpus_per_model) > 0 else world_size
    if requested != world_size:
        raise RuntimeError(f"--num-gpus-per-model {requested} but world_size {world_size}; launch one task per GPU")
    if args.device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        requested_device = args.device
    device = mip._resolve_device(requested_device, local_rank)
    if str(device).startswith("cuda"):
        torch.cuda.set_device(int(str(device).split(":")[1]))
    rank0 = global_rank == 0
    if rank0:
        save_dir.mkdir(parents=True, exist_ok=True)
        _log(f"modes {modes}; sigmas {sigma_values}; dates {dates}; other {other}; eps base seed {base_seed}; "
             f"world_size {world_size}; device {device}; save {save_dir}; save scope {save_scope}")

    model_comm_group = mip._init_model_comm_group(device, global_rank, world_size)
    try:
        ckpt_path = str(Path(args.ckpt_root) / args.name_exp / args.name_ckpt)
        t0 = time.time()
        inference_model, datamodule, _d, _n = mip._load_objects(
            ckpt_path=ckpt_path, device=device, validation_frequency=args.validation_frequency,
            precision="fp32", num_gpus_per_model_override=requested)
        if world_size > 1:
            n_edges = _force_edges_sharding(inference_model)
            if rank0:
                _log(f"forced shard_strategy heads->edges on {n_edges} modules (model-parallel inference)")
        inference_model.eval()
        if mip._model_takes_lead_hours(inference_model):
            raise NotImplementedError("lead-conditioned checkpoints are not supported by the input-mode probe")
        names = _out_names(datamodule)
        n_ch = len(names)
        if rank0:
            _log(f"loaded {ckpt_path} in {time.time() - t0:.0f} s; {n_ch} output channels")
        save_ws = [w.strip() for w in (getattr(args, "save_weather_states", "") or "").split(",") if w.strip()] or names
        miss = [w for w in save_ws if w not in names]
        if miss:
            raise ValueError(f"--save-weather-states not in the model outputs: {miss}")
        save_idx = [names.index(w) for w in save_ws]
        den = _Denoiser(inference_model, device, model_comm_group)

        # pass 1: inputs, truth and y0 of every date (this rank's grid slice kept on CPU)
        prep: dict[str, dict] = {}
        y0_local: dict[str, torch.Tensor] = {}
        checks: dict[str, dict] = {}
        n_points = None
        mask = None
        for path, date in zip(bundle_paths, dates):
            with torch.inference_mode():
                inp = _load_bundle_inputs(datamodule, inference_model, path, names)
                before, gss = den.before(inp["x_lres"], inp["x_hres"])
                n_full = int(inp["truth"].shape[0])
                if n_points is None:
                    n_points = n_full
                    from eval.predict.local_scope import hres_mask_for_scope
                    mask = hres_mask_for_scope(inp["lon"], inp["lat"], save_scope)
                elif n_full != n_points:
                    raise ValueError(f"bundle {path} has {n_full} points, the first had {n_points}")
                lo, hi = shard_bounds(None if gss is None else gss["out_hres"], global_rank, n_full)
                if int(before[0].shape[-2]) != hi - lo:
                    raise RuntimeError(f"rank {global_rank}: conditioning has {before[0].shape[-2]} grid points, "
                                       f"shard bounds give {hi - lo}")
                a, b = den.affine(hi - lo, n_ch, before, gss)
                truth = torch.from_numpy(inp["truth"].astype(np.float64)).to(a.device)
                y0 = (truth - a) / b
                nan_ch = [names[c] for c in range(n_ch) if bool(torch.isnan(y0[:, c]).any())]
                if nan_ch:
                    y0[:, [names.index(n) for n in nan_ch]] = 0.0
                recon = (a + b * y0 - truth).abs().amax(dim=0) / truth.std(dim=0).clamp_min(1e-30)
                recon[torch.isnan(recon)] = 0.0
                checks[date] = {"truth_channels_found": inp["found"], "truth_nan_channels_set_to_0": nan_ch,
                                "recon_truth_maxrel_to_std": float(recon.max().item()),
                                "b_spread_rel_max": float(((b.amax(0) - b.amin(0)) / b.abs().median(0).values
                                                           .clamp_min(1e-30)).max().item()),
                                "shard": [lo, hi]}
                y0_local[date] = y0[lo:hi].float().cpu()
                if rank0:
                    m = mask
                    np.savez_compressed(
                        save_dir / f"ref_{date}.npz",
                        y=inp["truth"][m][:, save_idx], a=a[m][:, save_idx].float().cpu().numpy(),
                        b=b[m][:, save_idx].float().cpu().numpy(), lat=inp["lat"][m],
                        lon=((inp["lon"][m] + 180.0) % 360.0) - 180.0, weather_states=np.asarray(save_ws),
                        date=np.asarray(date), bundle=np.asarray(str(path)),
                        other_date=np.asarray(other.get(date, "")))
                    _log(f"pass 1 {date}: {json.dumps(checks[date])}")
                prep[date] = {"x_lres": inp["x_lres"], "x_hres": inp["x_hres"]}
                del a, b, truth, y0, before
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # pass 2: D on the three inputs per (date, sigma), same eps for every mode
        rows = []
        for date in dates:
            with torch.inference_mode():
                before, gss = den.before(prep[date]["x_lres"], prep[date]["x_hres"])
                lo, hi = shard_bounds(None if gss is None else gss["out_hres"], global_rank, n_points)
                y_box = np.load(save_dir / f"ref_{date}.npz")["y"] if rank0 else None
                for s_idx, sigma in enumerate(sigma_values):
                    seed = eps_seed(base_seed, date, s_idx)
                    eps = draw_eps(n_points, n_ch, seed)[lo:hi]
                    fields = {}
                    for mode in modes:
                        t1 = time.time()
                        state = noisy_state(mode, sigma, eps, y0_local[date].double(),
                                            y0_local[other[date]].double() if mode == "other_date" else None)
                        post = den.denoise(state, sigma, before, gss)
                        if rank0:
                            fld = post[torch.from_numpy(mask).to(post.device)][:, save_idx].float().cpu().numpy()
                            finite = bool(np.isfinite(fld).all())
                            np.savez_compressed(save_dir / f"D_{date}_s{sigma:g}_{mode}.npz", y_pred=fld,
                                                sigma=np.asarray(sigma), mode=np.asarray(mode),
                                                eps_seed=np.asarray(seed), date=np.asarray(date),
                                                other_date=np.asarray(other.get(date, "") if mode == "other_date" else ""))
                            fields[mode] = fld
                            _log(f"{date} sigma {sigma:g} {mode}: {time.time() - t1:.1f} s, finite {finite}")
                        del post, state
                    if rank0:
                        ref = fields.get("truth")
                        err_ref = _rms(ref - y_box) if ref is not None else None
                        for mode, fld in fields.items():
                            d_ref = _rms(fld - ref) if ref is not None else np.full(len(save_ws), np.nan)
                            err = _rms(fld - y_box)
                            for k, w in enumerate(save_ws):
                                rows.append({
                                    "checkpoint": ckpt_path, "date": date, "sigma": sigma, "mode": mode,
                                    "other_date": other.get(date, "") if mode == "other_date" else "",
                                    "weather_state": w, "eps_seed": seed,
                                    "rms_D_minus_D_truth": float(d_ref[k]),
                                    "rms_D_minus_y": float(err[k]),
                                    "rms_D_truth_minus_y": float(err_ref[k]) if err_ref is not None else float("nan"),
                                })
                del before
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if rank0:
            with open(out_csv, "w", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["checkpoint"])
                wr.writeheader()
                wr.writerows(rows)
            manifest = {
                "checkpoint": ckpt_path, "modes": modes, "sigmas": sigma_values, "dates": dates,
                "bundles": [str(p) for p in bundle_paths], "other_date": other, "eps_base_seed": base_seed,
                "eps_seed_rule": "(base*1000003 + int(date)*101 + sigma_index) mod (2**63-1), CPU torch.Generator, "
                                 "float64 full grid, same for every mode",
                "save_scope": save_scope, "save_points": int(mask.sum()), "grid_points": int(n_points),
                "weather_states": save_ws, "world_size": world_size, "checks": checks,
                "downscaling_tools_commit": _git_commit(),
                "fields": "y_pred = post(D(x, sigma)) in physical units (a one-step sampler from sigma), "
                          "ref_<date>.npz: truth y and the output map a, b (post(out) = a + b*out)",
            }
            (save_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
            _log(f"wrote {out_csv} ({len(rows)} rows) and {save_dir}")
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        return out_csv
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
