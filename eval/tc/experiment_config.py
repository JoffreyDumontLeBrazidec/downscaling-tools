"""Default experiment configuration per TC event. Overridable at runtime via CLI flags."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TCExperimentConfig:
    """Default experiment IDs for a TC event.

    CLI --analysis-expid overrides analysis_expid at runtime.
    CLI --extra-reference-expids adds references at runtime.
    """

    analysis_expid: str
    reference_expids: tuple[str, ...] = ()
    base_tc_dir: str = "/home/ecm5702/hpcperm/data/tc"


# One default config per event.
# Events without an entry (like franklin_idalia) are handled gracefully
# by workflows: a warning is logged and the event is skipped.
EXPERIMENT_CONFIGS: dict[str, TCExperimentConfig] = {
    "franklin": TCExperimentConfig(
        analysis_expid="OPER_O320_0001",
        reference_expids=("ENFO_O320_0001", "EEFO_O96_0001"),
    ),
    "idalia": TCExperimentConfig(
        analysis_expid="OPER_O320_0001",
        reference_expids=("ENFO_O320_0001", "EEFO_O96_0001"),
    ),
    "hilary": TCExperimentConfig(
        analysis_expid="OPER_O320_0001",
        reference_expids=("ENFO_O320_0001", "EEFO_O96_0001"),
    ),
    "dora": TCExperimentConfig(
        analysis_expid="OPER_O320_0001",
        reference_expids=("ENFO_O320_0001", "EEFO_O96_0001"),
    ),
    "fernanda": TCExperimentConfig(
        analysis_expid="OPER_O320_0001",
        reference_expids=("ENFO_O320_0001", "EEFO_O96_0001"),
    ),
    "humberto": TCExperimentConfig(
        analysis_expid="OPER_O96_0001",
        reference_expids=("ENFO_O48_0001",),
    ),
}
