"""Build the model and the CASES (date x sigma) the GPU checks run on.

Backend ``bundles`` (the default): the model through ``interp.core.model.load_model`` (manual_inference's loader, the
one every interp tool uses; ds and unified APIs), the data from prepml ``*input_bundle.nc`` files (x_lres, x_hres and
the target y), prepared exactly as ``interp.core.model.prepare_batch`` does (interpolated low-res, normalised
conditioning, residual target). Each case is one EDM denoiser call at a fixed sigma on one bundle:
loss = mean((D(y + sigma n; sigma) - y)^2) in the normalised residual space, the training objective up to the
per-sigma weight (which cannot turn a zero gradient into a nonzero one).

A checkpoint trained on a regional dataset (the E2 box: 55,871 o320 and 814,810 o1280 points) reads global bundles
as they are: the bundle points are subset to the checkpoint's own point sets (its ``<dataset>/latitudes`` supporting
arrays), matched by position. ``--local-scope-json`` (eval.predict's graph cut) is for a GLOBAL checkpoint run on part
of the globe; it cuts only the output side and cannot shrink a regional model's low-res input.
"""
from __future__ import annotations

import glob
import logging
import os

import numpy as np
import torch

from .forward_checks import Case, CaseResult

LOGGER = logging.getLogger(__name__)
DEFAULT_SIGMAS = (0.2, 2.0, 20.0, 80.0)
FLOW_SIGMAS = (0.2, 5.0, 80.0)


def _expand(paths: list[str]) -> list[str]:
    out = []
    for p in paths:
        hits = sorted(glob.glob(os.path.expanduser(p)))
        out.extend(hits if hits else [p])
    return out


def _supporting_arrays(checkpoint: str) -> dict:
    from anemoi.utils.checkpoints import load_metadata

    for path in (checkpoint, os.path.join(os.path.dirname(checkpoint), "inference-" + os.path.basename(checkpoint))):
        if not os.path.exists(path):
            continue
        try:
            _, arrays = load_metadata(path, supporting_arrays=True)
        except Exception:  # noqa: BLE001 - a file without metadata: try the next one
            continue
        if arrays:
            return flatten_arrays(arrays)
    return {}


def flatten_arrays(arrays, prefix: str = "") -> dict:
    """anemoi-utils returns supporting arrays nested per dataset ({'in_lres': {'latitudes': ...}}): flatten to 'in_lres/latitudes'."""
    out = {}
    for k, v in dict(arrays).items():
        if isinstance(v, dict):
            out |= flatten_arrays(v, f"{prefix}{k}/")
        else:
            out[f"{prefix}{k}"] = v
    return out


def match_points(lat_src, lon_src, lat_dst, lon_dst, tol_deg: float = 1e-3) -> np.ndarray:
    """Index into the source points of each destination point (same grid, given in degrees); stops on any miss.

    Points are keyed by position rounded to tol_deg (longitudes wrapped to [-180, 180)); a point that rounds across a
    key boundary is found among the 8 neighbouring keys. numpy only (6.6M points in a few seconds).
    """
    def keys(lat, lon, dla=0, dlo=0):
        la = np.round(np.asarray(lat, np.float64) / tol_deg).astype(np.int64) + dla
        lo = np.round(((np.asarray(lon, np.float64) + 180.0) % 360.0 - 180.0) / tol_deg).astype(np.int64) + dlo
        return la * 10**9 + lo

    ks = keys(lat_src, lon_src)
    order = np.argsort(ks, kind="stable")
    ks_sorted = ks[order]
    if len(ks_sorted) > 1 and np.any(ks_sorted[1:] == ks_sorted[:-1]):
        raise SystemExit(f"two bundle points share a position at {tol_deg} deg: grid too fine for the key")
    idx = np.full(len(np.asarray(lat_dst)), -1, dtype=np.int64)
    for dla, dlo in [(0, 0)] + [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]:
        todo = np.where(idx < 0)[0]
        if not todo.size:
            break
        kd = keys(np.asarray(lat_dst)[todo], np.asarray(lon_dst)[todo], dla, dlo)
        pos = np.clip(np.searchsorted(ks_sorted, kd), 0, len(ks_sorted) - 1)
        hit = ks_sorted[pos] == kd
        idx[todo[hit]] = order[pos[hit]]
    if np.any(idx < 0):
        raise SystemExit(f"bundle points do not contain {int((idx < 0).sum())} of the model's points")
    if len(np.unique(idx)) != len(idx):
        raise SystemExit("two model points matched one bundle point: not the same grid")
    return idx


class BundlesBackend:
    def __init__(self, checkpoint: str, bundles: list[str], device: str = "cuda", precision: str = "bf16",
                 local_scope_json: str | None = None):
        from interp.core.data import load_single_bundle
        from interp.core.model import get_variable_names, is_dict_api, load_model, prepare_batch

        self.device = device
        # load in fp32 and run the forward under autocast, as training does (bf16-mixed)
        self.bundle = load_model(checkpoint, device=device, precision="fp32")
        if local_scope_json:
            from eval.predict.graph_cut import activate_local_graph_cut

            # eval.predict takes the scope as JSON text; accept a file path too
            if os.path.isfile(local_scope_json):
                with open(local_scope_json) as f:
                    local_scope_json = f.read()
            activate_local_graph_cut(self.bundle.model, local_scope_json)
        self.autocast = torch.bfloat16 if precision == "bf16" else (torch.float16 if precision == "fp16" else None)
        self.dict_api = is_dict_api(self.bundle.inner_model)
        vn = get_variable_names(self.bundle)
        self.out_vars = [vn["output"][i] for i in sorted(vn["output"])]
        self.in_vars = {
            "x_interp": [vn["input_lres"][i] for i in sorted(vn["input_lres"])],
            "x_hres": [vn["input_hres"][i] for i in sorted(vn["input_hres"])],
        }
        self.batches = []
        self.subset = None
        for p in _expand(bundles):
            bb = load_single_bundle(self.bundle, p)
            bb = self._to_model_points(checkpoint, bb)
            with torch.no_grad():
                if self.dict_api:
                    prep = self._prepare_dict(bb.x_lres, bb.x_hres, bb.y)
                else:
                    prep = prepare_batch(self.bundle, bb.x_lres[:, 0], bb.x_hres[:, 0], bb.y[:, 0])
            self.batches.append((os.path.basename(p), prep))
        if not self.batches:
            raise SystemExit("no bundles found")

    def _prepare_dict(self, x_lres, x_hres, y) -> dict:
        """The unified (dict-API) downscaler: the training task's data flow (tasks/diffusiondownscaler.py), step by step.

        interp's prepare_batch is written for the legacy `ds` model (one pre_processors callable taking dataset=...);
        here every dataset has its own normaliser and the residual target needs y in the dataset's DATA layout.
        Inputs are (batch, time, ensemble, grid, vars), as load_single_bundle returns them.
        """
        root, inner, dev = self.bundle.model, self.bundle.inner_model, self.bundle.device
        x_lres, x_hres, y = x_lres.to(dev), x_hres.to(dev), y.to(dev)
        xi_raw = inner.residual["in_lres"](x_lres, grid_shard_sizes=None, model_comm_group=None)[:, :, None]
        ti = inner.data_indices["out_hres"]
        full = ti.data.output.full.to(dev)
        if y.shape[-1] != len(full):
            raise SystemExit(f"bundle target has {y.shape[-1]} channels, the model outputs {len(full)}")
        y_data = torch.zeros(*y.shape[:-1], len(ti.name_to_index), dtype=y.dtype, device=dev)
        y_data[..., full] = y  # bundle targets come in model-output order = data.output.full order
        tend = getattr(root, "pre_processors_tendencies", None)
        target = inner.compute_residuals(
            y=y_data,
            x_interp=xi_raw[..., inner.get_matching_channel_indices("out_hres").to(dev)],
            pre_processors_state=root.pre_processors["out_hres"],
            pre_processors_tendencies=tend["out_hres"] if tend is not None and "out_hres" in tend else None,
            target_dataset="out_hres",
        )
        return {
            "x_interp": root.pre_processors["in_lres"](xi_raw, in_place=False),
            "x_hres": root.pre_processors["in_hres"](x_hres, in_place=False),
            "x_interp_raw": xi_raw,
            "y_residual": target,
        }

    def _to_model_points(self, checkpoint: str, bb):
        """A regional checkpoint (e.g. the E2 box) on global bundles: keep the bundle points the model was trained on.

        The model's own point sets come from the checkpoint's supporting arrays (``<dataset>/latitudes``); each bundle
        point is matched to them by position, so the subset is exact or the run stops.
        """
        n_l, n_h, n_y = bb.x_lres.shape[-2], bb.x_hres.shape[-2], bb.y.shape[-2]
        arrays = _supporting_arrays(checkpoint)
        want = {ds: arrays.get(f"{ds}/latitudes") for ds in ("in_lres", "in_hres", "out_hres")}
        if all(v is None for v in want.values()):
            LOGGER.warning("no supporting arrays in %s: bundle points used as they are (%d, %d, %d)", checkpoint, n_l, n_h, n_y)
            return bb
        if all(v is not None and len(v) == n for v, n in zip(want.values(), (n_l, n_h, n_y))):
            return bb
        lat_l, lon_l, lat_h, lon_h = bb.coords
        idx = {}
        for ds, lat, lon in (("in_lres", lat_l, lon_l), ("in_hres", lat_h, lon_h), ("out_hres", lat_h, lon_h)):
            if want[ds] is None:
                raise SystemExit(f"bundle grid differs from the model's but the checkpoint has no {ds}/latitudes")
            idx[ds] = match_points(lat, lon, want[ds], arrays[f"{ds}/longitudes"])
        self.subset = {ds: len(i) for ds, i in idx.items()}
        LOGGER.info("bundle points subset to the checkpoint's: %s", self.subset)
        sel = lambda t, i: t[..., torch.as_tensor(i), :]
        lat_y = np.asarray(lat_h)[idx["out_hres"]]
        return type(bb)(x_lres=sel(bb.x_lres, idx["in_lres"]), x_hres=sel(bb.x_hres, idx["in_hres"]),
                        y=sel(bb.y, idx["out_hres"]), paths=bb.paths,
                        coords=(np.asarray(lat_l)[idx["in_lres"]], np.asarray(lon_l)[idx["in_lres"]],
                                lat_y, np.asarray(lon_h)[idx["out_hres"]]))

    @property
    def root(self) -> torch.nn.Module:
        return self.bundle.model

    @property
    def graph(self):
        m = self.bundle.model
        for owner in (m, getattr(m, "model", None)):
            for name in ("graph_data", "_graph_data"):
                g = getattr(owner, name, None) if owner is not None else None
                if g is not None and len(getattr(g, "edge_types", None) or (g if hasattr(g, "items") else [])):
                    return g
        return None

    def _forward(self, prep: dict, sigma: float, input_grad: bool, perturb: float, seed: int) -> CaseResult:
        inner = self.bundle.inner_model
        x_interp = prep["x_interp"].detach().clone()
        x_hres = prep["x_hres"].detach().clone()
        y = prep["y_residual"].detach()
        if perturb:
            g = torch.Generator(device="cpu").manual_seed(seed + 1)
            x_interp = x_interp + perturb * torch.randn(x_interp.shape, generator=g).to(x_interp)
            x_hres = x_hres + perturb * torch.randn(x_hres.shape, generator=g).to(x_hres)
        if input_grad:
            x_interp.requires_grad_(True)
            x_hres.requires_grad_(True)
        g = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(y.shape, generator=g).to(y)
        sig = torch.tensor(sigma, device=y.device, dtype=y.dtype)
        y_noised = y + sig * noise
        ens = x_interp.shape[2]
        device_type = "cuda" if "cuda" in str(self.device) else "cpu"
        with torch.autocast(device_type=device_type, dtype=self.autocast or torch.float32, enabled=self.autocast is not None):
            if self.dict_api:
                D = inner.fwd_with_preconditioning(
                    {"in_lres": x_interp, "in_hres": x_hres}, {"out_hres": y_noised},
                    {"out_hres": sig.view(1, 1, 1, 1, 1).expand(1, 1, ens, 1, 1)}, None, None,
                )["out_hres"]
            else:
                D = inner.fwd_with_preconditioning(x_interp, x_hres, y_noised, sig.view(1, 1, 1, 1).expand(1, ens, 1, 1), None, None)
        D = D.float()
        loss = (D - y.float()).pow(2).mean()
        return CaseResult(loss=loss, output=D, inputs={"x_interp": x_interp, "x_hres": x_hres},
                          input_vars=self.in_vars, output_vars=self.out_vars)

    def cases(self, sigmas=DEFAULT_SIGMAS) -> list[Case]:
        out = []
        for name, prep in self.batches:
            for s in sigmas:
                out.append(Case(label=f"{name}:s{s:g}", sigma=float(s),
                                run=lambda input_grad, perturb, seed, prep=prep, s=s: self._forward(prep, s, input_grad, perturb, seed)))
        return out
