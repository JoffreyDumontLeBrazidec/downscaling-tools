"""TC (tropical cyclone extremes) evaluator.

For each tropical cyclone event of the lane (`tc.events`), extracts raw extremes
from the prediction files: minimum sea-level pressure (and its 0.01 percentile)
and maximum 10 m wind (and its 99.99 percentile), for the model and
for the reference sources present on the same grid (OPER, ENFO, EEFO). By the
run-trust contract of 2026-06-21 the verdict is the raw extremes, read by eye:
there is no composite score, no ratio and no anchor. The grid used is chosen by
`tc.support_mode` (native, regridded or both; regridded needs metview).
Scoreboard rows are named tc_<event>_<extreme> for the model and
tc_<event>_<source>_<extreme> for each reference.

The per-member maps that tc used to draw when a lane's `tc.member_maps` block
asked for them were retired on 2026-09-30. The block is now ignored with a
warning; `zoom_maps` draws single-member maps of a storm.
"""
from .runner import run
from .scorer import score
from .plotter import plot

EVALUATOR_SPEC = {
    "name": "tc",
    "requires": ["predictions"],
    "outputs": [
        "stats.json: raw extremes per event for the model and each reference.",
        "plots/all_tc_distributions.pdf: the distribution figure (written by plot), promoted to the run root as tc_pdf_distributions.pdf.",
    ],
    # Promoted to the run root by eval.lean_layout.
    "deliverables": {
        "top_level": [
            {
                "src": "plots/all_tc_distributions.pdf",
                "as": "tc_pdf_distributions.pdf"
            }
        ],
        # member_maps/ is no longer written (retired 2026-09-30) but stays
        # listed so that re-projecting an old run keeps its plots/tc_member_maps view.
        "plots": [
            "plots",
            "member_maps"
        ]
    },
}


__all__ = ["run", "score", "plot", "EVALUATOR_SPEC"]
