"""Deprecated path, kept as a forwarder: this module moved to eval.evaluators.tc.core.experiment_config.

Delete this file once nothing outside the repository refers to eval._backends.tc.experiment_config any more
(see eval/_backends/README.md for who still does).
"""
import importlib
import logging
import runpy
import sys
import warnings

_OLD = "eval._backends.tc.experiment_config"
_NEW = "eval.evaluators.tc.core.experiment_config"
_MESSAGE = f"{_OLD} moved to {_NEW}; update the caller."

logging.getLogger(__name__).warning("DEPRECATED: %s", _MESSAGE)

if __name__ == "__main__":
    runpy.run_module(_NEW, run_name="__main__", alter_sys=True)
else:
    warnings.warn(_MESSAGE, DeprecationWarning, stacklevel=2)
    sys.modules[__name__] = importlib.import_module(_NEW)
