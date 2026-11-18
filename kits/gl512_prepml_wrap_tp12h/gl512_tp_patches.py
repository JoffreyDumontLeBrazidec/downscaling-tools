"""GL512 prepml tp/cp fix (2026-11-18), loaded by sitecustomize of gl512-20261118-prepml-wrap-fast-tp12h.

Cause (docs epics/misc/20261117_quaver_scorecards_for_paper.md s5.8-s5.9): the GL512 checkpoint's out_hres tp and cp are
12-h accumulations (variables_metadata period ["0h", "12h"]: each model call predicts the rain of the 12 h ending at its
valid time). anemoi-inference's Accumulate post-processor (accumulate_from_start_of_forecast) adds max(0, x) of EVERY
processed state into a running sum, and the downscaling runner calls the model at step 0 too (forecast_stepper includes
step 0). With time_step 24h only every other 12-h window entered the sum (half the rain), and step 0 added the window
-12 h..0 h, which lies before the forecast start (non-zero tp at step 0).

Fix, in two parts:
  1. the lane steps every 12 h (prepml.time_step 12h, steps 12..240/12), so the 12-h windows tile 0..240 h;
  2. this patch: at step 0 the accumulated fields are set to exactly 0 before Accumulate adds them, so the sum starts at
     the forecast start, as in od eefo/enfo (tp = cp = 0 at step 0). Negative windows are already clipped at 0 by
     Accumulate (np.maximum(0, x)).
Guarded: a failure to import anemoi.inference (prepml CLI tasks without torch) leaves everything unchanged.
"""
import logging
from datetime import timedelta

LOG = logging.getLogger(__name__)

try:
    import numpy as np
    from anemoi.inference.post_processors.accumulate import Accumulate

    if not getattr(Accumulate, "_gl512_step0_zero", False):
        _orig_process = Accumulate.process

        def _process_step0_zero(self, state):
            step = state.get("step")
            if step is not None and step == timedelta(0):
                state = state.copy()
                state["fields"] = dict(state["fields"])
                for name in self.accumulations:
                    if name in state["fields"]:
                        state["fields"][name] = np.zeros_like(state["fields"][name])
                LOG.info("gl512_tp_patches: step 0, accumulations %s set to 0 before accumulating", self.accumulations)
            return _orig_process(self, state)

        Accumulate.process = _process_step0_zero
        Accumulate._gl512_step0_zero = True
        LOG.info("gl512_tp_patches: Accumulate.process patched (tp/cp = 0 at step 0)")
except Exception as _e:  # pragma: no cover
    LOG.warning("gl512_tp_patches not applied: %s", _e)
