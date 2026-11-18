"""sitecustomize for gl512-20261118-prepml-wrap-fast-tp12h (2026-11-11) = the gl512-20261109 wrap sitecustomize plus speed_patches.

(1) the shared prepml_sitecustomize_unified/sitecustomize.py unchanged, (2) gl512_runner_patches (sampler dicts, rule-31 seeds,
global noise; same file as the 20261109 wrap), (3) speed_patches with SPEED_TF32=1 SPEED_LNROW=1 SPEED_EDGEPAD=32 exported by
bin/activate. Every step is guarded: a failure here must not break the prepml CLI tasks that run in this venv without torch.
"""
import importlib.util
import logging

_SHARED = "/home/ecm5702/dev/anemoi-inference-configs/prepml_sitecustomize_unified/sitecustomize.py"

try:
    _spec = importlib.util.spec_from_file_location("_prepml_sitecustomize_unified", _SHARED)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
except Exception as _e:  # pragma: no cover
    logging.getLogger(__name__).warning("shared prepml sitecustomize failed: %s", _e)

try:
    import gl512_runner_patches  # noqa: F401  patches downscaling_unified_runner
except Exception as _e:  # pragma: no cover
    logging.getLogger(__name__).warning("gl512 runner patches not applied: %s", _e)

try:
    import speed_patches  # noqa: F401  TF32, one-row ConditionalLayerNorm, lin_edge padding (env knobs from bin/activate)
except Exception as _e:  # pragma: no cover
    logging.getLogger(__name__).warning("speed patches not applied: %s", _e)

# gl512-20261118-prepml-wrap-fast-tp12h (2026-11-18): the fast wrap plus gl512_tp_patches (tp/cp = 0 at step 0 in Accumulate);
# used with lane steps every 12 h so the checkpoint 12-h tp/cp windows tile the forecast (docs quaver scorecards note s5.9).
try:
    import gl512_tp_patches  # noqa: F401
except Exception as _e:  # pragma: no cover
    logging.getLogger(__name__).warning("gl512 tp patches not applied: %s", _e)
