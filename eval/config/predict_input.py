"""Where ``eval.predict`` takes its inputs from, and which weights it loads.

Two lane keys of the ``predict`` block, checked here so that ``eval.config.loader``,
``eval.cli predict`` and ``eval.predict.main`` read them the same way. Pure Python on
purpose (no torch): the loader and the CLI import it.

``predict.input_source`` (default: bundles, the behaviour before 2026-11-03)::

    input_source:
      type: zarr_split      # take every case from a split of the checkpoint's own datamodule
      split: test           # the datamodule split to read: training, validation or test

The zarr_split route is for checkpoints that were never given eval-format bundles, such as
the E2 Europe box checkpoints: (init date, lead) is mapped to a row of the split through the
dataset's fake-hindcast mapping, and the inputs and the truth of that row are read through
the datamodule the checkpoint was trained with. No bundle is required, resolved or verified.

``predict.ema`` (default: none, the raw weights ``inference-<name>``): ``ema2000`` or
``ema10000`` loads the EMA weights ``inference-<ema>-<name>`` saved next to the base
checkpoint, and refuses when that file is missing.
"""
from __future__ import annotations

import json
import re

BUNDLES = "bundles"
ZARR_SPLIT = "zarr_split"
ZARR_SPLITS = ("training", "validation", "test")
_EMA_RE = re.compile(r"^ema[0-9]+$")


class InputSourceError(ValueError):
    """A predict.input_source or predict.ema value that cannot be run as written."""


def normalize_input_source(value) -> dict | None:
    """Return ``{"type": "zarr_split", "split": ...}``, or None for the bundle route.

    Accepts None, "bundles", {"type": "bundles"}, a zarr_split mapping, or the JSON string
    of any of these (the form ``eval.predict.main --input-source-json`` receives).
    """
    if isinstance(value, str):
        text = value.strip()
        if not text or text == BUNDLES:
            return None
        try:
            value = json.loads(text)
        except ValueError:
            raise InputSourceError(
                f"input_source must be {BUNDLES!r} or a mapping with type {ZARR_SPLIT!r}, got {value!r}"
            ) from None
    if value is None:
        return None
    if not isinstance(value, dict):
        raise InputSourceError(f"input_source must be a mapping, got {type(value).__name__}")
    kind = value.get("type", BUNDLES)
    if kind == BUNDLES:
        extra = sorted(set(value) - {"type"})
        if extra:
            raise InputSourceError(f"input_source type {BUNDLES!r} takes no other key, got {extra}")
        return None
    if kind != ZARR_SPLIT:
        raise InputSourceError(f"input_source type must be {BUNDLES!r} or {ZARR_SPLIT!r}, got {kind!r}")
    unknown = sorted(set(value) - {"type", "split"})
    if unknown:
        raise InputSourceError(
            f"input_source type {ZARR_SPLIT!r} takes only 'split' (the checkpoint's own config names "
            f"the stores), got also {unknown}"
        )
    split = value.get("split", "test")
    if split not in ZARR_SPLITS:
        raise InputSourceError(f"input_source split must be one of {list(ZARR_SPLITS)}, got {split!r}")
    return {"type": ZARR_SPLIT, "split": split}


def normalize_ema(value) -> str:
    """Return the EMA tag (``ema2000``, ``ema10000``, ...) or "" for the raw weights."""
    if value in (None, "", "raw", False):
        return ""
    text = str(value).strip()
    if not _EMA_RE.match(text):
        raise InputSourceError(f"predict.ema must be an EMA tag such as ema2000 or ema10000, got {value!r}")
    return text
