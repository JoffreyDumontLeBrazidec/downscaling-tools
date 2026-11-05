"""Provenance of a score: what every table has to say before a number is quoted (audit of 5 Nov 2026,
docs ``epics/misc/20261105_score_and_eval_computation_audit.md``; standing rule 34(e)).

Six facts travel with every score written by ``eval.cli``: the verifying field, how the ensemble was built,
the area-weight form, the spread convention, the domain bounds and the date spacing. The evaluators build the
block with :func:`provenance` and write it into their summary JSON / ``run_meta.json`` and their ``summary.md``;
``eval.cli evaluate`` collects the blocks of a run into ``evaluators/provenance.md``.

The defects the block guards against (audit numbering): D1 fair CRPS against the forecast target quoted as if
against observations; D2 "ensembles" made of M generative draws from the one control input; D3 cos(lat) per
point on reduced Gaussian grids; D4 single fields (input, truth, HRES) scored as if they were ensembles; D7
three different "Europe" boxes; D8 consecutive dates with the SE over dates read as independent; D10 two spread
conventions quoted as one.
"""
from __future__ import annotations

from datetime import date as _date
from typing import Any, Iterable, Mapping

import numpy as np

RULE_34 = ("rule 34: the headline fair CRPS, spread/skill, twCRPS, Brier and rank histograms are against "
           "observations (obs_points, obs_imerg, tc_obs); gridded scores against the forecast target are diagnostics "
           "labelled 'vs forecast target (not observations)'")
ENSEMBLE_RULE = ("4 Nov 2026 rule: a presented model ensemble is ONE draw per input ensemble member (input variability "
                 "plus generative variability); M draws from the one control input are a single-input diffusion "
                 "diagnostic, never an ensemble mean, spread or CRPS to present")

PER_INPUT_MEMBER = "one draw per input member (input + generative variability)"
SINGLE_INPUT_DRAWS = "single-input draws: {m} generative draws from one input field (diagnostic only, not an ensemble)"
SINGLE_DRAW = "single draw (deterministic row)"
SINGLE_FIELD = "single field (fair CRPS = MAE of one field, spread 0; not like-for-like with an ensemble)"


def ensemble_construction(attrs: Mapping[str, Any], n_members: int, *, input_members_differ: bool | None = None,
                          source_bundles: Iterable[str] | None = None) -> str:
    """How the members of a prediction file were made, from what the file says about itself.

    ``attrs`` are the file's global attributes (``member_mode``, ``competitor_row`` of the member box kit; ``seed_rule``
    of eval.cli predict); ``input_members_differ`` says whether the per-member input fields in the file differ from
    each other (``x_interp_members`` or a member axis on ``x`` / ``x_interp``); ``source_bundles`` are the input bundle
    paths per member when the file carries them. The first fact that settles it wins.
    """
    m = int(n_members)
    row = attrs.get("competitor_row")
    if row:
        return f"competitor row: {row}"
    mode = str(attrs.get("member_mode", "") or "").lower()
    if mode == "pf":
        return PER_INPUT_MEMBER
    if source_bundles is not None:
        paths = {str(p) for p in source_bundles}
        if m > 1 and len(paths) == m:
            return PER_INPUT_MEMBER
        if m > 1 and len(paths) == 1:
            return SINGLE_INPUT_DRAWS.format(m=m)
    if input_members_differ is not None and m > 1:
        return PER_INPUT_MEMBER if input_members_differ else SINGLE_INPUT_DRAWS.format(m=m)
    if mode == "control":
        return SINGLE_INPUT_DRAWS.format(m=m)
    if m <= 1:
        return SINGLE_DRAW
    return SINGLE_INPUT_DRAWS.format(m=m) + " [construction not recorded in the file; assumed single input]"


def is_single_input(construction: str) -> bool:
    return construction.startswith("single-input draws")


def members_differ(a: np.ndarray, axis: int = 0, tol: float = 0.0) -> bool:
    """True when at least two slices along ``axis`` differ (the per-member inputs are different members)."""
    a = np.asarray(a)
    if a.shape[axis] < 2:
        return False
    first = np.take(a, 0, axis=axis)
    rest = np.moveaxis(a, axis, 0)[1:]
    with np.errstate(invalid="ignore"):
        return bool(np.nanmax(np.abs(rest - first[None])) > tol)


def date_spacing(dates: Iterable[str]) -> dict[str, Any]:
    """Number of dates and their spacing. Consecutive dates share weather, so the SE over dates reads too small."""
    ds = sorted({str(d)[:8] for d in dates if str(d)[:8].isdigit()})
    out: dict[str, Any] = {"n_dates": len(ds), "first": ds[0] if ds else None, "last": ds[-1] if ds else None,
                           "min_gap_days": None, "median_gap_days": None, "consecutive": False}
    if len(ds) > 1:
        days = [_date(int(d[:4]), int(d[4:6]), int(d[6:8])).toordinal() for d in ds]
        gaps = np.diff(days)
        out.update(min_gap_days=int(gaps.min()), median_gap_days=float(np.median(gaps)),
                   consecutive=bool(np.median(gaps) <= 1.0))
    out["note"] = ("consecutive dates: the dates are not independent samples, so the SE over dates is optimistic; "
                   "quote paired differences and wins, not the SE alone" if out["consecutive"] else
                   (f"{out['n_dates']} dates, median gap {out['median_gap_days']:g} days" if len(ds) > 1 else
                    f"{out['n_dates']} date(s)"))
    return out


def provenance(*, verifying_field: str, ensemble: str, weight_form: str, spread_form: str,
               domain_bounds: Mapping[str, Any], dates: Iterable[str], competitor_rows: Mapping[str, str] | None = None,
               against_observations: bool, **extra: Any) -> dict[str, Any]:
    """The block every scorer writes. ``domain_bounds`` maps region name to [lat_min, lat_max, lon_min, lon_max]
    (or None for everything); ``competitor_rows`` maps row name to its construction (ensemble or single field)."""
    block: dict[str, Any] = {
        "verifying_field": verifying_field,
        "against_observations": bool(against_observations),
        "ensemble_construction": ensemble,
        "competitor_rows": dict(competitor_rows or {}),
        "weight_form": weight_form,
        "spread_form": spread_form,
        "domain_bounds": {k: (list(v) if v is not None else None) for k, v in dict(domain_bounds).items()},
        "date_spacing": date_spacing(dates),
        "status": "headline" if against_observations else "diagnostic (vs forecast target, not observations)",
        "rules": [RULE_34, ENSEMBLE_RULE],
    }
    block.update(extra)
    block["warnings"] = provenance_warnings(block)
    return block


def provenance_warnings(block: Mapping[str, Any]) -> list[str]:
    """What a reader has to say next to the numbers of this table."""
    out = []
    if not block.get("against_observations"):
        out.append("scores are against the forecast target, not observations: diagnostic rows only (rule 34)")
    if is_single_input(str(block.get("ensemble_construction", ""))):
        out.append("model members are generative draws from ONE input: not an ensemble under the 4 Nov rule; "
                   "do not present its mean, spread or CRPS as the model ensemble")
    single = [r for r, k in dict(block.get("competitor_rows", {})).items() if str(k).startswith("single field")]
    if single:
        out.append(f"rows {', '.join(single)} are single fields: compare them with the ensemble mean (MAE) only, "
                   "never their 'fair CRPS' with an ensemble's")
    if dict(block.get("date_spacing", {})).get("consecutive"):
        out.append("consecutive dates: SE over dates is optimistic")
    return out


def provenance_lines(block: Mapping[str, Any]) -> list[str]:
    """Markdown lines for the head of a summary table."""
    ds = dict(block.get("date_spacing", {}))
    bounds = ", ".join(f"{k}: {'all' if v is None else v}" for k, v in dict(block.get("domain_bounds", {})).items())
    rows = "; ".join(f"{r}: {k}" for r, k in dict(block.get("competitor_rows", {})).items())
    lines = ["> Provenance (audit 5 Nov 2026, rule 34(e)).",
             f"> Verifying field: {block.get('verifying_field')} [{block.get('status')}].",
             f"> Model ensemble: {block.get('ensemble_construction')}." + (f" Other rows: {rows}." if rows else ""),
             f"> Area weights: {block.get('weight_form')}. Spread: {block.get('spread_form')}.",
             f"> Domains: {bounds or 'n/a'}.",
             f"> Dates: {ds.get('n_dates')} ({ds.get('first')}..{ds.get('last')}); {ds.get('note')}."]
    for w in block.get("warnings", []):
        lines.append(f"> WARNING: {w}.")
    return lines
