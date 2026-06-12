"""Forward-hook utilities: processor block discovery + activation capture.

Single source of truth for "give me the 16 processor blocks" — previously
re-implemented (with diverging fallbacks) in activation_profiling,
cka_analysis and activation_patching.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

import torch

LOGGER = logging.getLogger(__name__)


def iter_processor_blocks(inner) -> list[tuple[int, object]]:
    """Return [(global_idx, block_module), ...] for every processor block.

    Anemoi's processor splits num_layers into num_chunks ProcessorChunks
    (stored in ``proc.proc``); each chunk holds the actual transformer layers
    in ``chunk.blocks``. For o96->o320: 2 chunks * 8 blocks = L0..L15.
    """
    proc = inner.processor
    items: list[tuple[int, object]] = []
    if hasattr(proc, "proc") and len(proc.proc) > 0:
        gi = 0
        for chunk in proc.proc:
            blocks = getattr(chunk, "blocks", None)
            if blocks is None:
                LOGGER.warning("Chunk %s has no .blocks", type(chunk).__name__)
                continue
            for block in blocks:
                items.append((gi, block))
                gi += 1
    elif hasattr(proc, "blocks"):
        items = list(enumerate(proc.blocks))
    elif hasattr(proc, "layers"):
        items = list(enumerate(proc.layers))
    else:
        raise RuntimeError(f"Cannot find processor sub-layers (type={type(proc).__name__})")
    return items


def _first_tensor(out):
    if isinstance(out, tuple):
        for t in out:
            if isinstance(t, torch.Tensor):
                return t
        return None
    return out if isinstance(out, torch.Tensor) else None


@contextmanager
def captured_block_activations(inner, layer_fmt: str = "layer_{:02d}"):
    """Capture every processor block's output[0] (detached, fp32, on-device).

    Yields a dict that fills as the forward pass runs:
        {"layer_00": tensor, ..., "layer_15": tensor}
    """
    store: dict[str, torch.Tensor] = {}
    handles = []

    def _make(name):
        def hook(module, inp, out):
            t = _first_tensor(out)
            if t is not None:
                store[name] = t.detach().float()
        return hook

    for idx, block in iter_processor_blocks(inner):
        handles.append(block.register_forward_hook(_make(layer_fmt.format(idx))))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


@contextmanager
def captured_module_stats(inner, modules=("encoder", "processor", "decoder")):
    """Capture summary stats (l2/mean/std/max_abs/shape) of whole-module outputs.

    Yields a dict {module_name: stats_dict} filled during the forward pass.
    """
    store: dict[str, dict] = {}
    handles = []

    def _make(name):
        def hook(module, inp, out):
            t = _first_tensor(out)
            if t is None:
                return
            ft = t.float()
            store[name] = {
                "l2_norm": ft.norm().item(),
                "mean": ft.mean().item(),
                "std": ft.std().item(),
                "max_abs": ft.abs().max().item(),
                "shape": list(t.shape),
            }
        return hook

    for name in modules:
        mod = getattr(inner, name, None)
        if mod is not None:
            handles.append(mod.register_forward_hook(_make(name)))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


def tensor_stats(t: torch.Tensor) -> dict:
    """Summary stats for one activation tensor (matches captured_module_stats)."""
    ft = t.float()
    return {
        "l2_norm": ft.norm().item(),
        "mean": ft.mean().item(),
        "std": ft.std().item(),
        "max_abs": ft.abs().max().item(),
        "shape": list(t.shape),
    }
