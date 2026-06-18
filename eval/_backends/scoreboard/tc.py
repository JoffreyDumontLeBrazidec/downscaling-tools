"""TC (tropical cyclone) extreme scoring — analysis-anchored multi-depth scoring."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from eval._backends.scoreboard._utils import finite_float as _finite_float
from eval._backends.scoreboard.row_matching import (
    classify_row,
    find_model_row,
    find_row_by_predicate,
    is_analysis_row,
    is_eefo_row,
    is_reference_row,
)
from eval.scoreboard.types import RowClassification

LOG = logging.getLogger(__name__)

MSLP_REFERENCE_HPA = 1013.25
_OVERSHOOT_BETA = 0.5

# --- Documented o96->o320 TC scoring contract (decided 2026-06-17; incident
# epics/ds-multi-ds-parity/in-progress/20260617_o96_o320_unified_oldlike_tc_regression.md;
# knowledge: docs/knowledge/results/tc-extreme-score-support-contract.md) -----------
# A single, documented support so extreme_score is comparable across all runs:
#   * MODEL + EEFO floor are measured on the REGRIDDED support (eval.cli default;
#     lane support_mode "both" -> the bare "idalia"/"franklin" events).
#   * ANALYSIS target = the CURATED canonical AN (canonical_analysis.yaml "o320"),
#     which is deliberately DEEPER than the embedded regridded OPER. The regridded
#     OPER is over-smoothed (idalia mslp_min 985.4 / wind_max 24.5 vs curated
#     971.3 / 32.0), so anchoring on it lets models trivially overshoot and the
#     EEFO floor collapses the rescaled score (~0.05). Anchoring on the deep NATIVE
#     OPER (946.8) is the opposite failure (franklin floors to 0.0). The curated
#     mid-target is the only well-conditioned, discriminating anchor; it is NOT a
#     same-support OPER percentile and must NOT be "re-derived" to either embedded
#     support. Keep it pinned and documented.
# Because the curated AN is intentionally != the embedded OPER, the guardrail below
# does NOT compare canonical-vs-embedded (that gap is by design). Instead it detects
# the real failure mode: a run whose EMBEDDED OPER was computed on a DIFFERENT
# TC-stats support (native / legacy manual "strict") than the regridded contract,
# which is the documented cause of the false "Idalia 21/21" regression. WARN-only.
_OFFCONTRACT_MSLP_MIN_TOL_HPA = 3.0   # embedded vs contract regridded OPER mslp_min
_OFFCONTRACT_WIND_P99_TOL_MS = 2.5    # embedded vs contract regridded OPER wind_p99


def _warn_on_offcontract_support(
    embedded: dict[str, Any] | None,
    contract_oper: dict[str, Any] | None,
    *,
    event_name: str | None,
) -> None:
    """WARN when a run's embedded OPER row does not match the documented o96->o320
    REGRIDDED contract OPER — i.e. it was measured on a different TC-stats support
    (native / manual "strict") and its extreme_score is not comparable to the public
    scoreboard. mslp_min flags native; wind_p99 (a broad percentile) additionally
    flags the legacy manual "strict" support. Purely observational; no score change."""
    if not isinstance(embedded, dict) or not isinstance(contract_oper, dict):
        return
    flags = []
    for key, tol, unit in (
        ("mslp_min", _OFFCONTRACT_MSLP_MIN_TOL_HPA, "hPa"),
        ("wind_p99", _OFFCONTRACT_WIND_P99_TOL_MS, "m/s"),
    ):
        e = _finite_float(embedded.get(key))
        c = _finite_float(contract_oper.get(key))
        if e is None or c is None:
            continue
        if abs(e - c) > tol:
            flags.append("%s embedded=%.3f vs contract=%.3f (%.3f %s)" % (key, e, c, abs(e - c), unit))
    if flags:
        LOG.warning(
            "TC off-contract support for event=%s: embedded OPER does not match the "
            "documented o96->o320 REGRIDDED contract [%s] — extreme_score is NOT "
            "comparable to the public scoreboard (run scored on a different TC-stats "
            "support, e.g. native / manual strict). See "
            "docs/knowledge/results/tc-extreme-score-support-contract.md.",
            event_name, "; ".join(flags),
        )


def _asymmetric_ratio_score(ratio: float) -> float:
    """Score a model/analysis ratio with asymmetric penalty: overshoot penalized at half rate."""
    if ratio <= 1.0:
        return ratio
    return max(0.0, 1.0 - _OVERSHOOT_BETA * (ratio - 1.0))


def _symmetric_match_score(ratio: float) -> float:
    """Score a model/reference ratio with symmetric penalty: 1.0 at ratio=1, linear falloff, clipped to [0, 1]."""
    return max(0.0, 1.0 - abs(ratio - 1.0))



def mslp_depth(value: float) -> float:
    """Convert MSLP (hPa) to depth below standard reference: deeper = more extreme."""
    return max(MSLP_REFERENCE_HPA - value, 0.0)


def multi_depth_tc_score(
    model: dict[str, Any],
    analysis: dict[str, Any],
) -> float | None:
    """Compute analysis-anchored TC score using multi-depth tail percentiles.

    Returns 0–1 score: 1.0 = model matches analysis extremes, 0.0 = no extreme signal.
    """
    mslp_keys = ("mslp_p1", "mslp_p01", "mslp_min")
    wind_keys = ("wind_p99", "wind_p999", "wind_max")

    mslp_ratios: list[float] = []
    for key in mslp_keys:
        m_val = _finite_float(model.get(key))
        a_val = _finite_float(analysis.get(key))
        if m_val is None or a_val is None:
            continue
        a_depth = mslp_depth(a_val)
        if a_depth <= 0.0:
            continue
        m_depth = mslp_depth(m_val)
        mslp_ratios.append(_asymmetric_ratio_score(m_depth / a_depth))

    wind_ratios: list[float] = []
    for key in wind_keys:
        m_val = _finite_float(model.get(key))
        a_val = _finite_float(analysis.get(key))
        if m_val is None or a_val is None:
            continue
        if a_val <= 0.0:
            continue
        wind_ratios.append(_asymmetric_ratio_score(m_val / a_val))

    if not mslp_ratios and not wind_ratios:
        return None

    scores: list[float] = []
    if mslp_ratios:
        scores.append(sum(mslp_ratios) / len(mslp_ratios))
    if wind_ratios:
        scores.append(sum(wind_ratios) / len(wind_ratios))
    return sum(scores) / len(scores)


def multi_depth_enfo_deviation(
    model: dict[str, Any],
    enfo: dict[str, Any],
    analysis: dict[str, Any],
) -> float | None:
    """Compute ENFO deviation: how far the model diverges from ENFO, normalized by analysis."""
    mslp_keys = ("mslp_p1", "mslp_p01", "mslp_min")
    wind_keys = ("wind_p99", "wind_p999", "wind_max")

    mslp_devs: list[float] = []
    for key in mslp_keys:
        m_val = _finite_float(model.get(key))
        e_val = _finite_float(enfo.get(key))
        a_val = _finite_float(analysis.get(key))
        if m_val is None or e_val is None or a_val is None:
            continue
        a_depth = mslp_depth(a_val)
        if a_depth <= 0.0:
            continue
        mslp_devs.append(abs(mslp_depth(m_val) - mslp_depth(e_val)) / a_depth)

    wind_devs: list[float] = []
    for key in wind_keys:
        m_val = _finite_float(model.get(key))
        e_val = _finite_float(enfo.get(key))
        a_val = _finite_float(analysis.get(key))
        if m_val is None or e_val is None or a_val is None:
            continue
        if a_val <= 0.0:
            continue
        wind_devs.append(abs(m_val - e_val) / a_val)

    if not mslp_devs and not wind_devs:
        return None

    devs: list[float] = []
    if mslp_devs:
        devs.append(sum(mslp_devs) / len(mslp_devs))
    if wind_devs:
        devs.append(sum(wind_devs) / len(wind_devs))
    return sum(devs) / len(devs)


def multi_depth_enfo_match_score(
    model: dict[str, Any],
    enfo: dict[str, Any],
) -> float | None:
    """Compute ENFO-anchored symmetric tail-extreme match score.

    Uses only the deepest tail keys per variable (mslp_p01, mslp_min, wind_p999,
    wind_max). The shallower-tail keys (mslp_p1, wind_p99) are intentionally
    excluded because at 10-member parity the ML/ENFO ratios there sit at ~0.95–
    0.99 across the board and mask the real gap that shows up at min/max.

    Returns 0–1 score: 1.0 = model tail extremes match ENFO exactly,
    0.0 = ratio outside [0, 2] on every key. Penalizes undershoot and overshoot
    equally. None if no usable tail keys.
    """
    mslp_keys = ("mslp_p01", "mslp_min")
    wind_keys = ("wind_p999", "wind_max")

    mslp_scores: list[float] = []
    for key in mslp_keys:
        m_val = _finite_float(model.get(key))
        e_val = _finite_float(enfo.get(key))
        if m_val is None or e_val is None:
            continue
        e_depth = mslp_depth(e_val)
        if e_depth <= 0.0:
            continue
        m_depth = mslp_depth(m_val)
        mslp_scores.append(_symmetric_match_score(m_depth / e_depth))

    wind_scores: list[float] = []
    for key in wind_keys:
        m_val = _finite_float(model.get(key))
        e_val = _finite_float(enfo.get(key))
        if m_val is None or e_val is None:
            continue
        if e_val <= 0.0:
            continue
        wind_scores.append(_symmetric_match_score(m_val / e_val))

    if not mslp_scores and not wind_scores:
        return None

    group_means: list[float] = []
    if mslp_scores:
        group_means.append(sum(mslp_scores) / len(mslp_scores))
    if wind_scores:
        group_means.append(sum(wind_scores) / len(wind_scores))
    return sum(group_means) / len(group_means)


_MIN_AN_EXTREME = 0.1  # hPa or m/s; floor below which AN's extreme value isn't meaningful for ratio normalization.


def _safe_ratio(model_extreme: float, an_extreme: float) -> float | None:
    if an_extreme is None or model_extreme is None:
        return None
    if an_extreme < _MIN_AN_EXTREME:
        return None
    return model_extreme / an_extreme


def tail_extreme_ratios(
    model: dict[str, Any],
    analysis: dict[str, Any],
) -> dict[str, float | None]:
    """Compute per-key tail-extreme ratios vs OPER analysis.

    Returns dict with keys mslp_p001_ratio, mslp_min_ratio, wind_p9999_ratio,
    wind_max_ratio. Each is ``model_extreme / AN_extreme`` in 'extremeness' coords
    (mslp_depth for MSLP keys; raw value for wind). AN row = 1.0 by construction.

    Semantics:
        1.0  -> model matches AN at this percentile
        > 1  -> model more extreme than AN
        < 1  -> model less extreme than AN
        None -> AN value missing or too small to ratio against

    Requires the input rows to have ``mslp_p001`` and ``wind_p9999`` keys
    (added to extreme_tail_table for the 0.01 / 99.99 percentiles). Older stats
    rows without those keys will return None for the corresponding ratios.
    """
    out: dict[str, float | None] = {}

    # MSLP — depth coords
    for key in ("mslp_p001", "mslp_min"):
        m_val = _finite_float(model.get(key))
        a_val = _finite_float(analysis.get(key))
        if m_val is None or a_val is None:
            out[f"{key}_ratio"] = None
            continue
        out[f"{key}_ratio"] = _safe_ratio(mslp_depth(m_val), mslp_depth(a_val))

    # Wind — raw
    for key in ("wind_p9999", "wind_max"):
        m_val = _finite_float(model.get(key))
        a_val = _finite_float(analysis.get(key))
        if m_val is None or a_val is None:
            out[f"{key}_ratio"] = None
            continue
        out[f"{key}_ratio"] = _safe_ratio(max(m_val, 0.0), max(a_val, 0.0))

    return out


_REACH_MIN_DENOMINATOR = 1.0  # hPa or m/s; below this the AN→ENFO range is too narrow to be meaningful.


def _position(model_extreme: float, an_extreme: float, enfo_extreme: float) -> float | None:
    """Locate model_extreme on the AN→ENFO axis. 0=AN, 1=ENFO; <0 below AN; >1 beyond ENFO.

    Returns None when ENFO is not strictly more extreme than AN at this key, i.e.
    when ``enfo_extreme - an_extreme < _REACH_MIN_DENOMINATOR``. That guard catches
    two failure modes:
      - degenerate denominator (AN ≈ ENFO),
      - inverted axis where AN's small-sample tail exceeds ENFO's large-sample tail
        (e.g. Idalia wind_p999 where AN n≈50k yields p999 above ENFO n≈3.5M p999).
    Inverted-axis interpolation is semantically broken — position values flip sign.
    """
    denom = enfo_extreme - an_extreme
    if denom < _REACH_MIN_DENOMINATOR:
        return None
    return (model_extreme - an_extreme) / denom


def multi_depth_an_enfo_position(
    model: dict[str, Any],
    analysis: dict[str, Any],
    enfo: dict[str, Any],
) -> dict[str, float | None]:
    """Compute AN→ENFO position scores for the deepest tail keys.

    Returns {"mslp_position": float|None, "wind_position": float|None}.
    Each is the mean (over the 2 keys per variable) of:

        (extreme(model) - extreme(AN)) / (extreme(ENFO) - extreme(AN))

    where extreme = mslp_depth for MSLP keys, raw value for wind keys.

    Semantics:
        < 0  -> model less extreme than AN
        0    -> model matches AN
        1    -> model matches ENFO
        > 1  -> model beyond ENFO

    Keys are skipped when |ENFO_extreme − AN_extreme| is below 1 hPa / 1 m/s
    (degenerate denominator: AN and ENFO are too close to interpolate).
    """
    mslp_keys = ("mslp_p01", "mslp_min")
    wind_keys = ("wind_p999", "wind_max")

    mslp_positions: list[float] = []
    for key in mslp_keys:
        m_val = _finite_float(model.get(key))
        a_val = _finite_float(analysis.get(key))
        e_val = _finite_float(enfo.get(key))
        if m_val is None or a_val is None or e_val is None:
            continue
        pos = _position(mslp_depth(m_val), mslp_depth(a_val), mslp_depth(e_val))
        if pos is not None:
            mslp_positions.append(pos)

    wind_positions: list[float] = []
    for key in wind_keys:
        m_val = _finite_float(model.get(key))
        a_val = _finite_float(analysis.get(key))
        e_val = _finite_float(enfo.get(key))
        if m_val is None or a_val is None or e_val is None:
            continue
        pos = _position(max(m_val, 0.0), max(a_val, 0.0), max(e_val, 0.0))
        if pos is not None:
            wind_positions.append(pos)

    return {
        "mslp_position": (sum(mslp_positions) / len(mslp_positions)) if mslp_positions else None,
        "wind_position": (sum(wind_positions) / len(wind_positions)) if wind_positions else None,
    }


def rescale_with_eefo_floor(raw_score: float, eefo_raw: float | None) -> float:
    """Rescale a raw TC score so EEFO maps to 0 and analysis maps to 1."""
    if eefo_raw is not None and eefo_raw < 1.0:
        return max(0.0, (raw_score - eefo_raw) / (1.0 - eefo_raw))
    return raw_score


def normalize_tc_rows(
    rows: list[dict[str, Any]],
    *,
    canonical_analysis: dict[str, Any] | None = None,
    canonical_eefo: dict[str, Any] | None = None,
    event_name: str | None = None,
    extreme_reference_expid: str | None = None,
    contract_oper: dict[str, Any] | None = None,
) -> None:
    """Analysis-anchored TC scoring with multi-depth tail percentiles.

    Parameters
    ----------
    rows : list of row dicts from TC stats JSON
    canonical_analysis : dict, optional
        Explicit canonical analysis values for this event. When provided and the
        embedded analysis is O320-class, overrides the embedded analysis row.
    event_name : str, optional
        Used only for backward compat with the old interface that looked up
        canonical values internally. Prefer passing canonical_analysis directly.
    extreme_reference_expid : str, optional
        Exact expid of the reference whose tail extremes we want to match (e.g.
        "ENFO_O320_0001"). When provided and the row is found, populates
        ``_enfo_match_value`` on each row via ``multi_depth_enfo_match_score``.
        When absent or the row is not found, ``_enfo_match_value`` is None.

    Scores are rescaled so EEFO maps to 0 and analysis maps to 1.
    Falls back to legacy batch-relative normalization when analysis row or
    tail percentiles are not available.
    """
    embedded_analysis = find_row_by_predicate(rows, is_analysis_row)

    # Use canonical only when the embedded analysis is O320-class (not O1280)
    analysis_row: dict[str, Any] | None = None
    if canonical_analysis is not None:
        embedded_exp = str(embedded_analysis.get("exp", "")).upper() if embedded_analysis else ""
        if "O1280" not in embedded_exp:
            analysis_row = canonical_analysis
            # Off-support guardrail (WARN-only). The documented contract scores the model
            # on the REGRIDDED support against the curated canonical AN, so canonical !=
            # embedded OPER is BY DESIGN. Flag instead any run whose embedded OPER was
            # computed on a different support (native / manual "strict") and is therefore
            # not comparable to the public scoreboard. See the contract block up top.
            _warn_on_offcontract_support(
                embedded_analysis, contract_oper, event_name=event_name
            )

    if analysis_row is None:
        analysis_row = embedded_analysis if isinstance(embedded_analysis, dict) else None

    enfo_row = find_row_by_predicate(rows, is_reference_row)

    extreme_ref_row: dict[str, Any] | None = None
    if extreme_reference_expid:
        extreme_ref_row = find_row_by_predicate(
            rows,
            lambda exp, _expid=extreme_reference_expid: exp == _expid,
        )

    if analysis_row is not None and _finite_float(analysis_row.get("mslp_p1")) is not None:
        eefo_raw = None
        if canonical_eefo is not None:
            eefo_raw = multi_depth_tc_score(canonical_eefo, analysis_row)
        else:
            eefo_row = find_row_by_predicate(rows, is_eefo_row)
            if eefo_row is not None:
                eefo_raw = multi_depth_tc_score(eefo_row, analysis_row)

        for row in rows:
            exp = str(row.get("exp", "")).strip()
            if is_analysis_row(exp):
                row["_extreme_score_value"] = 1.0
                row["_enfo_deviation_value"] = None
                row["_enfo_match_value"] = None
                row["_mslp_reach_value"] = None
                row["_wind_reach_value"] = None
                row["_mslp_p001_ratio_value"] = 1.0
                row["_mslp_min_ratio_value"] = 1.0
                row["_wind_p9999_ratio_value"] = 1.0
                row["_wind_max_ratio_value"] = 1.0
                continue
            raw_score = multi_depth_tc_score(row, analysis_row)
            if raw_score is not None:
                row["_extreme_score_value"] = rescale_with_eefo_floor(raw_score, eefo_raw)
            else:
                row["_extreme_score_value"] = None
            if enfo_row is not None:
                row["_enfo_deviation_value"] = multi_depth_enfo_deviation(row, enfo_row, analysis_row)
            else:
                row["_enfo_deviation_value"] = None
            if extreme_ref_row is not None and row is not extreme_ref_row:
                row["_enfo_match_value"] = multi_depth_enfo_match_score(row, extreme_ref_row)
                positions = multi_depth_an_enfo_position(row, analysis_row, extreme_ref_row)
                row["_mslp_reach_value"] = positions["mslp_position"]
                row["_wind_reach_value"] = positions["wind_position"]
            else:
                row["_enfo_match_value"] = None
                row["_mslp_reach_value"] = None
                row["_wind_reach_value"] = None
            # AN-anchored tail ratios use the **embedded** OPER row (same support_mode,
            # bbox, member-clip as every other row in this stats JSON) rather than the
            # canonical analysis (which may not carry mslp_p001 / wind_p9999 fields and
            # was computed with different parameters).
            ratios_ref = embedded_analysis if isinstance(embedded_analysis, dict) else analysis_row
            ratios = tail_extreme_ratios(row, ratios_ref or {})
            row["_mslp_p001_ratio_value"] = ratios["mslp_p001_ratio"]
            row["_mslp_min_ratio_value"] = ratios["mslp_min_ratio"]
            row["_wind_p9999_ratio_value"] = ratios["wind_p9999_ratio"]
            row["_wind_max_ratio_value"] = ratios["wind_max_ratio"]
        return

    # Legacy fallback: batch-relative normalization
    m_values = [v for v in (_finite_float(row.get("mslp_980_990_fraction")) for row in rows) if v is not None]
    w_values = [v for v in (_finite_float(row.get("wind_gt_25_fraction")) for row in rows) if v is not None]
    m_min = min(m_values) if m_values else None
    m_max = max(m_values) if m_values else None
    w_min = min(w_values) if w_values else None
    w_max = max(w_values) if w_values else None

    for row in rows:
        score = _finite_float(row.get("extreme_score"))
        if score is not None:
            row["_extreme_score_value"] = score
            row["_enfo_deviation_value"] = None
            row["_enfo_match_value"] = None
            row["_mslp_reach_value"] = None
            row["_wind_reach_value"] = None
            row["_mslp_p001_ratio_value"] = None
            row["_mslp_min_ratio_value"] = None
            row["_wind_p9999_ratio_value"] = None
            row["_wind_max_ratio_value"] = None
            continue

        m_val = _finite_float(row.get("mslp_980_990_fraction"))
        w_val = _finite_float(row.get("wind_gt_25_fraction"))
        if m_val is None or w_val is None or m_min is None or m_max is None or w_min is None or w_max is None:
            row["_extreme_score_value"] = None
            row["_enfo_deviation_value"] = None
            row["_enfo_match_value"] = None
            row["_mslp_reach_value"] = None
            row["_wind_reach_value"] = None
            row["_mslp_p001_ratio_value"] = None
            row["_mslp_min_ratio_value"] = None
            row["_wind_p9999_ratio_value"] = None
            row["_wind_max_ratio_value"] = None
            continue

        m_norm = 0.0 if m_max <= m_min else (m_val - m_min) / (m_max - m_min)
        w_norm = 0.0 if w_max <= w_min else (w_val - w_min) / (w_max - w_min)
        row["_extreme_score_value"] = 0.5 * (m_norm + w_norm)
        row["_enfo_deviation_value"] = None
        row["_enfo_match_value"] = None
        row["_mslp_reach_value"] = None
        row["_wind_reach_value"] = None
        row["_mslp_p001_ratio_value"] = None
        row["_mslp_min_ratio_value"] = None
        row["_wind_p9999_ratio_value"] = None
        row["_wind_max_ratio_value"] = None


def load_tc_extreme_scores_from_json(
    stats_path: Path,
    *,
    run_id: str,
    event_names: tuple[str, ...] | list[str] | None = None,
    canonical_analysis_by_event: dict[str, dict[str, Any]] | None = None,
    canonical_eefo_by_event: dict[str, dict[str, Any]] | None = None,
    extreme_reference_expid: str | None = None,
) -> dict[str, float]:
    """Load TC extreme scores and ENFO deviation from a stats JSON.

    Parameters
    ----------
    stats_path : Path to the TC stats JSON file.
    run_id : The experiment run ID to match.
    event_names : Which events to score. Defaults to ("idalia", "franklin").
    canonical_analysis_by_event : dict mapping event name -> canonical analysis dict.
        When provided, used instead of internal lookup. Pass None to use the
        old behavior of looking up CANONICAL_OPER_O320_ANALYSIS by event name.
    canonical_eefo_by_event : dict mapping event name -> canonical EEFO dict.
        When provided, used as the EEFO floor instead of finding an EEFO row.
    extreme_reference_expid : str, optional
        Exact expid of the reference ensemble whose tail extremes we want to match
        (e.g. "ENFO_O320_0001" for the o96_o320 lane). When provided and the row is
        present in each event's stats, populates "<event>_enfo_match" keys in the
        returned scores dict.

    Returns
    -------
    dict with keys like "idalia", "franklin" (scores),
    "idalia_enfo_dev", "franklin_enfo_dev" (deviations, when available),
    and "idalia_enfo_match", "franklin_enfo_match" (ENFO match scores, when
    extreme_reference_expid is provided).
    """
    with stats_path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        data = {}
    events = data.get("events")
    if not isinstance(events, dict):
        return {}

    # Backward compat: if no canonical provided, load from YAML
    if canonical_analysis_by_event is None:
        from eval._backends.scoreboard.canonical_data import load_canonical_analysis

        canonical_analysis_by_event = load_canonical_analysis("o320")

    # Off-support guardrail reference: the documented regridded-contract embedded OPER
    # per event (WARN-only; never used in scoring). See _warn_on_offcontract_support.
    from eval._backends.scoreboard.canonical_data import load_contract_oper

    contract_oper_by_event = load_contract_oper("o320")

    requested_events = tuple(event_names or ("idalia", "franklin"))
    scores: dict[str, float] = {}
    for event_name in requested_events:
        event_data = events.get(event_name)
        if not isinstance(event_data, dict):
            continue
        rows = event_data.get("extreme_tail", {}).get("rows", [])
        if not isinstance(rows, list):
            rows = event_data.get("rows", [])
        if not isinstance(rows, list):
            continue
        norm_rows = [row for row in rows if isinstance(row, dict)]

        # Support variants share a base event ("idalia__native" -> "idalia"); strip the
        # support suffix so native/regridded look up the SAME canonical AN/EEFO instead of
        # silently falling back to the embedded (different-support) analysis row. (P3 fix,
        # 2026-06-17 — previously "idalia__native" missed the YAML key and self-anchored.)
        base_event = event_name.split("__", 1)[0]
        canonical = canonical_analysis_by_event.get(base_event) if canonical_analysis_by_event else None
        eefo = canonical_eefo_by_event.get(base_event) if canonical_eefo_by_event else None
        contract_oper = contract_oper_by_event.get(base_event) if contract_oper_by_event else None
        normalize_tc_rows(
            norm_rows,
            canonical_analysis=canonical,
            canonical_eefo=eefo,
            event_name=event_name,
            extreme_reference_expid=extreme_reference_expid,
            contract_oper=contract_oper,
        )

        chosen = find_model_row(norm_rows, run_id)
        if chosen is None:
            continue
        score = _finite_float(chosen.get("_extreme_score_value"))
        if score is not None:
            scores[event_name] = score
        enfo_dev = _finite_float(chosen.get("_enfo_deviation_value"))
        if enfo_dev is not None:
            scores[f"{event_name}_enfo_dev"] = enfo_dev
        enfo_match = _finite_float(chosen.get("_enfo_match_value"))
        if enfo_match is not None:
            scores[f"{event_name}_enfo_match"] = enfo_match
        mslp_reach = _finite_float(chosen.get("_mslp_reach_value"))
        if mslp_reach is not None:
            scores[f"{event_name}_mslp_reach"] = mslp_reach
        wind_reach = _finite_float(chosen.get("_wind_reach_value"))
        if wind_reach is not None:
            scores[f"{event_name}_wind_reach"] = wind_reach
        for ratio_key in ("mslp_p001_ratio", "mslp_min_ratio", "wind_p9999_ratio", "wind_max_ratio"):
            val = _finite_float(chosen.get(f"_{ratio_key}_value"))
            if val is not None:
                scores[f"{event_name}_{ratio_key}"] = val

        # Raw physical extremes (hPa / m/s) — support-robust sanity anchors for the
        # fragile extreme_score. Emit the matched row's own deepest MSLP and strongest
        # wind, plus the OPER analysis / ENFO reference / EEFO input-baseline anchors so
        # each event is self-checkable at a glance ("model 976.3 vs OPER 985.4"). The
        # anchors are the same for every row in an event (one TC-stats support), so the
        # scoreboard surfaces the matched-row pair per run and the anchors on the
        # reference rows. (2026-06-18, incident 20260617.)
        for raw_key in ("mslp_min", "wind_max"):
            v = _finite_float(chosen.get(raw_key))
            if v is not None:
                scores[f"{event_name}_{raw_key}"] = v
        for anchor_label, anchor_predicate in (
            ("oper", is_analysis_row),
            ("enfo", is_reference_row),
            ("eefo", is_eefo_row),
        ):
            anchor_row = find_row_by_predicate(norm_rows, anchor_predicate)
            if not isinstance(anchor_row, dict):
                continue
            for raw_key in ("mslp_min", "wind_max"):
                v = _finite_float(anchor_row.get(raw_key))
                if v is not None:
                    scores[f"{event_name}_{anchor_label}_{raw_key}"] = v
    return scores
