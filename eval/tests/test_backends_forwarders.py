"""The forwarding modules left in eval/_backends keep working for callers outside the repository."""
from __future__ import annotations

import importlib
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

FORWARDERS = {
    "eval._backends.precip.tp_histogram_comparison": "eval.evaluators.precip_dist.core.tp_histogram_comparison",
    "eval._backends.tc.data_types": "eval.evaluators.tc.core.data_types",
    "eval._backends.tc.events": "eval.evaluators.tc.core.events",
    "eval._backends.tc.experiment_config": "eval.evaluators.tc.core.experiment_config",
    "eval._backends.tc.grid": "eval.evaluators.tc.core.grid",
    "eval._backends.tc.loading_grib": "eval.evaluators.tc.core.loading_grib",
    "eval._backends.tc.loading_predictions": "eval.evaluators.tc.core.loading_predictions",
    "eval._backends.tc.plot_config": "eval.evaluators.tc.core.plot_config",
    "eval._backends.tc.workflows": "eval.evaluators.tc.core.workflows",
    "eval._backends.region_plotting.plot_regions": "eval.evaluators.region_plot.core.plot_regions",
    "eval._backends.region_plotting.plot_one_date_local": "eval.evaluators.region_plot.core.plot_one_date_local",
    "eval._backends.weight_diagnostics.plot_checkpoint_weights": "eval.tools.weight_diagnostics.plot_checkpoint_weights",
    "eval._backends.spread_proxy.scoring": "eval.evaluators.spread_proxy.core.scoring",
    "eval._backends.storm_maps.render": "eval.evaluators.storm_maps.core.render",
    "eval._backends.sigma_evaluator.run_sigma_evaluator": "eval.tools.sigma_evaluator.run_sigma_evaluator",
}


@pytest.mark.parametrize("old,new", sorted(FORWARDERS.items()))
def test_old_path_gives_the_new_module_and_warns(old, new):
    try:
        target = importlib.import_module(new)
    except ImportError as exc:  # a heavy optional dependency is missing in this environment
        pytest.skip(f"{new} cannot be imported here: {exc}")
    sys.modules.pop(old, None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module(old)
    assert module is target
    assert any(issubclass(w.category, DeprecationWarning) and new in str(w.message) for w in caught)


def test_python_dash_m_on_an_old_path_runs_the_new_module():
    """The runbook style call ``python -m eval._backends...`` still reaches the command line of the module."""
    result = subprocess.run(
        [sys.executable, "-m", "eval._backends.precip.tp_histogram_comparison", "--help"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "--predictions-dir" in result.stdout
    assert "DEPRECATED" in result.stderr
