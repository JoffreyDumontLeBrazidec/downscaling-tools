"""The one list of evaluators.

Every evaluator the evaluation CLI knows about is listed here exactly once.
``eval.cli`` derives the names it accepts from this file, and the scoreboard
aggregator derives the evaluators whose rows reach ``scores.csv`` from it, so
there is no second hand-written list to keep in step.

Each entry records:

``group``
    ``scored``      produces rows that the lane scoreboard ranks runs by;
    ``standard``    runs by default on a lane but produces no scoreboard row;
    ``diagnostic``  asked for explicitly (a lane ``diagnostics`` group or
                    ``--only``), to explain a result rather than to rank it;
    ``retired``     can no longer be run. Its package has been moved to
                    ``eval/_quarantine/<retired_on>/`` and is not importable
                    as ``eval.evaluators.<name>``.
``feeds_scoreboard``
    True when the aggregator reads the evaluator's ``score()`` records into the
    run's ``scoreboard/scores.csv``.
``question``
    One plain sentence saying what question the evaluator answers.
``replacement`` / ``retired_on``
    For retired evaluators only: what to use instead (None when nothing
    replaces it) and the date it was retired.
``host_prefix``
    When set, the evaluator can only run on a machine whose host name starts
    with this prefix (``"ac"`` for the ECMWF spectral transform, which exists
    on Atos AC only). Elsewhere the CLI skips it with a warning.

What an evaluator needs in order to run (``requires``) and which files it
promotes (``deliverables``) stay in the ``EVALUATOR_SPEC`` of its package,
because they describe the code rather than the evaluator's role.

``eval/evaluators/tctracks`` is deliberately absent: it is the comparison
library behind ``eval.cli tccompare``, not an evaluator that ``evaluate`` can
run, and it has no ``run()``.

Retirement convention: selecting a retired evaluator explicitly (``--only``)
prints its replacement and exits with status 1; a retired name still listed in
a lane YAML group is skipped with a warning, so old or untracked lanes keep
running.
"""
from __future__ import annotations

from dataclasses import dataclass

SCORED = "scored"
STANDARD = "standard"
DIAGNOSTIC = "diagnostic"
RETIRED = "retired"
GROUPS = (SCORED, STANDARD, DIAGNOSTIC, RETIRED)

QUARANTINE_ROOT = "eval/_quarantine"


@dataclass(frozen=True)
class Evaluator:
    name: str
    group: str
    feeds_scoreboard: bool
    question: str
    replacement: str | None = None
    retired_on: str | None = None
    host_prefix: str | None = None
    # Retired evaluators only: the ``deliverables`` block their spec declared,
    # kept so that re-projecting an old run directory (eval.lean_layout) still
    # gives the same top-level file names after the package was quarantined.
    legacy_deliverables: dict | None = None


_ENTRIES: tuple[Evaluator, ...] = (
    # ------------------------------------------------------------------ scored
    Evaluator(
        "tc", SCORED, True,
        "How deep and how strong are the model's tropical cyclones, read as raw "
        "extremes of minimum sea-level pressure and maximum 10 m wind next to the "
        "truth and the operational references on the same grid?",
    ),
    Evaluator(
        "surface", SCORED, True,
        "How large is the model's error on the surface variables, as a normalised "
        "mean squared error per variable and an area- and variable-weighted total?",
    ),
    Evaluator(
        "spectra_ecmwf_v2", SCORED, True,
        "Does the model's power spectrum, computed with the ECMWF spectral transform "
        "on the complete grid, match the truth's spectrum at fine scales "
        "(wavenumbers above 100)?",
        host_prefix="ac",
    ),
    Evaluator(
        "precip_scores", SCORED, True,
        "How accurate is the model's six-hour precipitation, per member and for the "
        "ensemble mean, against the truth and against the interpolated input?",
    ),
    Evaluator(
        "sigma_loss", SCORED, True,
        "How large is the denoiser's loss at each noise level, per variable, when the "
        "checkpoint is run once on a noised truth field?",
    ),
    # ---------------------------------------------------------------- standard
    Evaluator(
        "region_plot", STANDARD, False,
        "What do the model, truth and input fields look like side by side over the "
        "lane's fixed regions?",
    ),
    Evaluator(
        "probabilistic", STANDARD, False,
        "How good is the model ensemble as a probabilistic forecast, measured by "
        "CRPS, spread and ensemble-mean error per variable and region?",
    ),
    # -------------------------------------------------------------- diagnostic
    Evaluator(
        "texture", DIAGNOSTIC, False,
        "Does the fine-scale texture of the model's fields, measured on the native "
        "O1280 grid, have the same statistics as the truth's?",
    ),
    Evaluator(
        "wind_extremes", DIAGNOSTIC, False,
        "Is the strongest 10 m wind in the model a coherent weather feature or "
        "isolated grid-scale noise?",
    ),
    Evaluator(
        "displacement", DIAGNOSTIC, False,
        "Does the model move weather features away from where its driving input "
        "puts them?",
    ),
    Evaluator(
        "spectra_coherence", DIAGNOSTIC, False,
        "At each spatial scale, does the model have the truth's amplitude and is it "
        "in phase with the truth?",
    ),
    Evaluator(
        "membermaps", DIAGNOSTIC, False,
        "What do the driving input, the truth and a model member look like on a map, "
        "as full fields and as high-pass fine-scale views?",
    ),
    Evaluator(
        "spread_proxy", DIAGNOSTIC, False,
        "Is the spread of the model ensemble similar to the spread of the ENFO "
        "truth ensemble?",
    ),
    Evaluator(
        "precip_dist", DIAGNOSTIC, False,
        "Does the distribution of the model's precipitation values match the truth's "
        "at each lead time?",
    ),
    Evaluator(
        "precip_events", DIAGNOSTIC, False,
        "What do the model and the truth look like at the heaviest precipitation "
        "events of the window?",
    ),
    Evaluator(
        "local_global", DIAGNOSTIC, False,
        "Does the model run on a local cut-out of the globe give the same answer as "
        "the model run on the whole globe?",
    ),
    Evaluator(
        "lane_diagnostics", DIAGNOSTIC, False,
        "Which figures explain a result that has already been scored on this lane, "
        "with the support, sample size and arm stated for every number?",
    ),
    Evaluator(
        "mlflow", DIAGNOSTIC, False,
        "How did the training and validation losses evolve while this checkpoint "
        "was trained?",
    ),
    Evaluator(
        "quaver", DIAGNOSTIC, False,
        "How does the ensemble published to FDB score in ECMWF's quaver scorecard, "
        "which is the canonical probabilistic verdict and reaches the scoreboard "
        "through its own ingest?",
    ),
    Evaluator(
        "storm_maps", DIAGNOSTIC, False,
        "What does the deepest storm look like in the truth, the model and the input, "
        "and how does its regional power spectrum compare in the 40 to 150 km band?",
    ),
    Evaluator(
        "shape", DIAGNOSTIC, False,
        "Are the model's fine-scale 10 m wind structures shaped like the truth's, in "
        "elongation, aspect ratio and orientation relative to the flow?",
    ),
    Evaluator(
        "tc_structure", DIAGNOSTIC, False,
        "Does the model's tropical cyclone have the truth's structure: centre, central "
        "pressure, wind profile, radius of maximum wind, wind radii, vorticity and "
        "asymmetry?",
    ),
    # ----------------------------------------------------------------- retired
    Evaluator(
        "spectra", RETIRED, False,
        "Did the model's power spectrum, estimated with a fast HEALPix proxy "
        "transform, match the truth's above wavenumber 100?",
        replacement="spectra_ecmwf_v2", retired_on="20260928",
        legacy_deliverables={
            "top_level": [{"src": "all_spectra_proxy.pdf", "as": "spectra_proxy.pdf"}],
        },
    ),
    Evaluator(
        "spectra_ecmwf", RETIRED, False,
        "Did the model's power spectrum, computed with the ECMWF spectral transform on "
        "templates with the polar rows removed, match the truth's?",
        replacement="spectra_ecmwf_v2", retired_on="20260928",
    ),
    Evaluator(
        "mechanistic", RETIRED, False,
        "What do the checkpoint's weights look like? (It was a stub that only created "
        "empty output directories.)",
        replacement=None, retired_on="20260928",
    ),
    Evaluator(
        "interp", RETIRED, False,
        "What do the interpretability analyses computed outside the CLI show for this "
        "checkpoint?",
        replacement=None, retired_on="20260928",
    ),
    Evaluator(
        "leadtime", RETIRED, False,
        "How do the surface scores and spectra change with lead time? (It was never "
        "registered and never ran.)",
        replacement=None, retired_on="20260928",
    ),
    Evaluator(
        "sigma", RETIRED, False,
        "How does the denoiser's loss change with the noise level, computed with the "
        "older sigma-sweep script?",
        replacement="sigma_loss", retired_on="20260928",
    ),
    Evaluator(
        "obs_crps", RETIRED, False,
        "What is the fair CRPS of the published ensemble against surface station "
        "observations, as a cheap stand-in for quaver?",
        replacement="quaver", retired_on="20260928",
    ),
    Evaluator(
        "intermediate", RETIRED, False,
        "What do the intermediate steps of the diffusion sampler look like?",
        replacement=None, retired_on="20260928",
    ),
)

REGISTRY: dict[str, Evaluator] = {e.name: e for e in _ENTRIES}

if len(REGISTRY) != len(_ENTRIES):  # pragma: no cover - guards a hand edit
    raise RuntimeError("eval.evaluators.registry lists an evaluator twice")
for _e in _ENTRIES:  # pragma: no cover - guards a hand edit
    if _e.group not in GROUPS:
        raise RuntimeError(f"evaluator {_e.name!r} has unknown group {_e.group!r}")
    if (_e.group == RETIRED) != (_e.retired_on is not None):
        raise RuntimeError(f"evaluator {_e.name!r}: retired_on must be set exactly when retired")
    if _e.feeds_scoreboard and _e.group != SCORED:
        raise RuntimeError(f"evaluator {_e.name!r}: only scored evaluators feed the scoreboard")


def get(name: str) -> Evaluator | None:
    return REGISTRY.get(name)


def names(*groups: str) -> list[str]:
    """Evaluator names in registry order, restricted to ``groups`` when given."""
    return [e.name for e in _ENTRIES if not groups or e.group in groups]


def runnable_names() -> list[str]:
    """Every evaluator that ``eval.cli`` can run (all groups except retired)."""
    return names(SCORED, STANDARD, DIAGNOSTIC)


def scoreboard_names() -> list[str]:
    """Evaluators whose ``score()`` records the aggregator writes to scores.csv."""
    return [e.name for e in _ENTRIES if e.feeds_scoreboard]


def is_retired(name: str) -> bool:
    entry = REGISTRY.get(name)
    return entry is not None and entry.group == RETIRED


def retired_message(name: str) -> str:
    """The tombstone text for a retired evaluator."""
    entry = REGISTRY[name]
    where = f"{QUARANTINE_ROOT}/{entry.retired_on}/{name}/"
    if entry.replacement:
        instead = f"Use '{entry.replacement}' instead."
    else:
        instead = "Nothing replaces it."
    return (
        f"Evaluator '{name}' was retired on {entry.retired_on}. {instead} "
        f"Its code is kept, unimportable, under {where}"
    )
