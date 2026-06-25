"""Schema-aware indirection to the manual-inference predict/dataset modules.

The eval harness can run two checkpoint families that use *different* data_indices
schemas in their predict path:

  * unified multi-ds (default): ``manual_inference.prediction.*`` — dict-batch
    ``data_indices["in_lres"]`` / ``["in_hres"]`` / ``["out_hres"]``.
  * legacy single-dataset "ds" (cfec83a3-era): ``manual_inference_legacy_ds.prediction.*``
    — list-indexed ``data_indices.data.input[0]`` and positional ``predict_step``.

Selection is by the ``KEYSTONE_LEGACY_DS`` env var (set to "1" by the cfec83a3
sbatches, which already source the *ds* runtime). Everything else keeps the
unified path unchanged. This is import-time indirection only; no behavior change
when the flag is unset.
"""
from __future__ import annotations

import importlib
import os

_LEGACY = os.environ.get("KEYSTONE_LEGACY_DS", "").strip() in ("1", "true", "True", "yes")
_PKG = "manual_inference_legacy_ds" if _LEGACY else "manual_inference"

predict = importlib.import_module(f"{_PKG}.prediction.predict")
dataset = importlib.import_module(f"{_PKG}.prediction.dataset")

USING_LEGACY_DS = _LEGACY
MANUAL_INFERENCE_PKG = _PKG
