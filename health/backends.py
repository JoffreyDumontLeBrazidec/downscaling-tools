"""Build the model and the CASES (date x sigma) the GPU checks run on.

Backend ``bundles`` (the default): the model through ``interp.core.model.load_model`` (manual_inference's loader, the
one every interp tool uses; ds and unified APIs), the data from prepml ``*input_bundle.nc`` files (x_lres, x_hres and
the target y), prepared exactly as ``interp.core.model.prepare_batch`` does (interpolated low-res, normalised
conditioning, residual target). Each case is one EDM denoiser call at a fixed sigma on one bundle:
loss = mean((D(y + sigma n; sigma) - y)^2) in the normalised residual space, the training objective up to the
per-sigma weight (which cannot turn a zero gradient into a nonzero one). Regional (box) checkpoints evaluated on a
global bundle take ``--local-scope-json`` (eval.predict's graph cut).
"""
from __future__ import annotations

import glob
import logging
import os

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
        for p in _expand(bundles):
            bb = load_single_bundle(self.bundle, p)
            with torch.no_grad():
                prep = prepare_batch(self.bundle, bb.x_lres[:, 0], bb.x_hres[:, 0], bb.y[:, 0])
            self.batches.append((os.path.basename(p), prep))
        if not self.batches:
            raise SystemExit("no bundles found")

    @property
    def root(self) -> torch.nn.Module:
        return self.bundle.model

    @property
    def graph(self):
        return getattr(self.bundle.model, "graph_data", None)

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
