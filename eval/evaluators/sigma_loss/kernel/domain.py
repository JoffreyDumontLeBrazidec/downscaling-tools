"""sigma_loss kernel — fixed-sigma install, per-variable F-space loss, batch capture."""
from __future__ import annotations

import logging
import math
import types
from typing import Any, Callable

import torch

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# API detection
# ---------------------------------------------------------------------------
def detect_api(downscaler: Any) -> str:
    """Return "dict" (unified multi-ds) or "tensor" (legacy single-ds).

    The unified task stores its loss as a ``torch.nn.ModuleDict`` keyed by the
    target dataset name; the legacy task stores a single loss module. We key off
    that plus the ``_decoder_datasets`` attribute on the underlying model.
    """
    loss = getattr(downscaler, "loss", None)
    if isinstance(loss, torch.nn.ModuleDict):
        return "dict"
    inner = getattr(getattr(downscaler, "model", None), "model", None)
    if inner is not None and hasattr(inner, "_decoder_datasets"):
        return "dict"
    return "tensor"


def target_dataset_name(downscaler: Any) -> str:
    """Resolve the unified target dataset key (e.g. "out_hres")."""
    inner = downscaler.model.model
    decoders = getattr(inner, "_decoder_datasets", None)
    if decoders:
        return decoders[0]
    return "out_hres"


def model_sigma_data(downscaler: Any) -> float:
    return float(downscaler.model.model.sigma_data)


# ---------------------------------------------------------------------------
# Fixed-sigma monkeypatch
# ---------------------------------------------------------------------------
def fixed_sigma_weight(sigma: float, sigma_data: float) -> float:
    """The EDM loss weight == 1 / c_out^2 == (s^2 + sd^2) / (s * sd)^2."""
    return (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2


def install_fixed_sigma(downscaler: Any, sigma: float, api: str) -> Callable[[], None]:
    """Monkeypatch ``_get_noise_level`` to return a deterministic sigma.

    Returns a restore callable that puts the original method back.

    The patched method honours the original signature for each API and returns
    the documented weight ``(sigma^2 + sigma_data^2)/(sigma*sigma_data)^2``.
    """
    original = downscaler._get_noise_level

    if api == "dict":
        def _fixed_dict(self, shape, sigma_max, sigma_min, sigma_data, rho, device):
            sig_dict: dict[str, torch.Tensor] = {}
            w_dict: dict[str, torch.Tensor] = {}
            w_val = fixed_sigma_weight(sigma, sigma_data)
            for name, shp in shape.items():
                bs, ens = shp[0], shp[2]
                # broadcast shape (batch, time=1, ensemble, grid=1, vars=1)
                sig_dict[name] = torch.full((bs, 1, ens, 1, 1), float(sigma), device=device)
                w_dict[name] = torch.full((bs, 1, ens, 1, 1), float(w_val), device=device)
            return sig_dict, w_dict

        downscaler._get_noise_level = types.MethodType(_fixed_dict, downscaler)
    else:
        def _fixed_tensor(self, shape, sigma_max, sigma_min, sigma_data, rho, device):
            sig = torch.full(shape, float(sigma), device=device)
            w = torch.full(shape, fixed_sigma_weight(sigma, sigma_data), device=device)
            return sig, w

        downscaler._get_noise_level = types.MethodType(_fixed_tensor, downscaler)

    def _restore() -> None:
        downscaler._get_noise_level = original

    return _restore


# ---------------------------------------------------------------------------
# Loss-arg capture (to recover per-variable F-space loss faithfully)
# ---------------------------------------------------------------------------
class _LossCapture:
    """Wrap a loss module's forward to record (pred, target, weights).

    We then re-invoke the *same* module with ``squash=False`` to obtain the
    per-variable vector whose mean equals the squashed (task) total. This keeps
    every scaler/node-weighting term identical to the task loss.
    """

    def __init__(self, loss_module: torch.nn.Module):
        self.loss_module = loss_module
        self._orig_forward = loss_module.forward
        self.captured: dict[str, Any] | None = None

    def __enter__(self):
        capture = self

        def _wrapped(pred, target, *args, **kwargs):
            capture.captured = {"pred": pred, "target": target, "args": args, "kwargs": dict(kwargs)}
            return capture._orig_forward(pred, target, *args, **kwargs)

        self.loss_module.forward = _wrapped
        return self

    def __exit__(self, *exc):
        self.loss_module.forward = self._orig_forward
        return False

    def per_variable_loss(self) -> torch.Tensor:
        """Re-run captured loss with squash=False -> per-variable vector."""
        if self.captured is None:
            raise RuntimeError("Loss was not invoked during _step; cannot read per-variable loss.")
        kwargs = dict(self.captured["kwargs"])
        kwargs["squash"] = False
        with torch.inference_mode():
            out = self._orig_forward(
                self.captured["pred"], self.captured["target"], *self.captured["args"], **kwargs
            )
        return out.detach().reshape(-1).float().cpu()


def _loss_module_for(downscaler: Any, api: str) -> torch.nn.Module:
    if api == "dict":
        ds_name = target_dataset_name(downscaler)
        return downscaler.loss[ds_name]
    return downscaler.loss


# ---------------------------------------------------------------------------
# Single (sigma, batch) F-space loss
# ---------------------------------------------------------------------------
def _call_step(downscaler: Any, batch: Any, api: str):
    if api == "dict":
        return downscaler._step(batch, validation_mode=True)
    return downscaler._step(batch, 0, training_mode=True, validation_mode=True)


def fspace_loss_for_batch(
    downscaler: Any,
    batch: Any,
    api: str,
    *,
    seed: int,
) -> tuple[float, torch.Tensor]:
    """Run a single forward at the currently-installed fixed sigma.

    Returns (total_fspace_loss, per_variable_fspace_loss_vector).

    A fixed manual seed is set immediately before ``_step`` so the noise draw
    (``torch.randn_like`` inside ``_noise_target``) is identical across sigma and
    checkpoint for a given batch — the comparability contract.
    """
    loss_module = _loss_module_for(downscaler, api)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        with _LossCapture(loss_module) as cap:
            step_out = _call_step(downscaler, batch, api)
            total = step_out[0]
            per_var = cap.per_variable_loss()
    total_f = float(total.detach().float().cpu().item())
    return total_f, per_var


# ---------------------------------------------------------------------------
# Output variable names (per-variable channel mapping)
# ---------------------------------------------------------------------------
def output_variable_names(downscaler: Any, api: str) -> list[str]:
    """Ordered output-channel variable names via data_indices.model.output.

    Mirrors the probe's per-channel slicing of ``data_indices ... output``: we
    use the *model* output ordering, which is the order the loss vector is
    produced in.
    """
    di = downscaler.data_indices
    if api == "dict":
        ds_name = target_dataset_name(downscaler)
        nti = di[ds_name].model.output.name_to_index
    else:
        nti = di.model.output.name_to_index
    ordered = sorted(nti.items(), key=lambda kv: kv[1])
    return [name for name, _idx in ordered]


# ---------------------------------------------------------------------------
# Sigma grid
# ---------------------------------------------------------------------------
def log_spaced_grid(sigma_min: float, sigma_max: float, n: int) -> list[float]:
    if n < 2:
        return [float(sigma_min)]
    lo, hi = math.log10(sigma_min), math.log10(sigma_max)
    return [float(10 ** (lo + (hi - lo) * i / (n - 1))) for i in range(n)]


# ---------------------------------------------------------------------------
# Fixed batch capture
# ---------------------------------------------------------------------------
def capture_fixed_batches(datamodule: Any, k: int, device: str, api: str) -> list[Any]:
    """Capture K validation batches once, moved to device.

    These exact batches are reused for every sigma and checkpoint so the loss
    profile is comparable. The val dataloader order is deterministic for a fixed
    validation frequency + base seed.
    """
    dl = datamodule.val_dataloader()
    batches: list[Any] = []
    for i, batch in enumerate(dl):
        if i >= k:
            break
        batches.append(_to_device(batch, device))
    if not batches:
        raise RuntimeError("val_dataloader yielded no batches; cannot capture fixed batches.")
    return batches


def _to_device(batch: Any, device: str) -> Any:
    if isinstance(batch, dict):
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return [v.to(device) if torch.is_tensor(v) else v for v in batch]
    return batch.to(device) if torch.is_tensor(batch) else batch
