"""TC-score proxy evaluator — the cheap TC ladder column.

Emits the two numbers the experiment panel needs per event, on the model's NATIVE support:
  * ``eye_deepest``      -- deepest eye MSLP over the whole budget (the headline)
  * ``eye_casemin_mean`` -- mean over cases of the per-case ensemble minimum (the distributional
                            companion; the headline alone is an instance lottery)
plus the peak-wind analogues, for the model AND for the target (ENFO) and input (EEFO) curves
carried in the same prediction files -- so every line on a panel is guaranteed same-support.

Metric prefix is ``tcproxy_`` and NOT ``tc_`` on purpose: the `tc` evaluator scores on the lane's
`support_mode` (regridded on most lanes) while this reads the model's own grid. TC extremes are
support-dependent, so the two families must never be mixed in one column.
"""

from .runner import run
from .scorer import score
from .plotter import plot

EVALUATOR_SPEC = {
    "name": "tc_proxy",
    "default_enabled": False,
    "scoreboard": False,
    "requires": ["predictions"],
}
