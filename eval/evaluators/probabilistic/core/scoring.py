"""Local spread/CRPS scoring for eval.cli prediction NetCDFs.

What is computed, per prediction file (one date and lead), variable and region:

* the four historical metrics, unchanged in name and meaning (``scores_by_lead.csv``,
  ``summary_by_lead.csv``): ``crps``, ``fcrps`` (Ferro's fair CRPS), ``spread`` (area mean of
  the pointwise member standard deviation, ddof 1) and ``rmse_ens_mean`` (root of the area
  mean squared error of the ensemble mean);
* the calibration read of the owner's rule (``calibration_by_lead.csv``, see
  ``calibration.py``): spread as the root of the pooled mean ensemble variance, RMSE as the
  root of the pooled mean squared error, their ratio and the ratio times ``sqrt((m+1)/m)``,
  pooled over points (area-weighted) and dates, with ``m``, ``n_dates`` and ``n_points``;
* rank histograms with ``m+1`` bins, ties broken at random with a fixed seed
  (``rank_histograms.csv``);
* when ``scale_split`` is on, the same scores for the large part ``L = P_L(field)`` and the
  fine part ``F = field - L`` of members and truth (``band_scores_by_lead.csv``, band column
  ``full`` / ``large`` / ``fine``; ``scale_split.py`` describes ``P_L``), and the inter-member
  spread of the low-resolution input in each band when the files carry the input members
  (``x_interp`` with two or more members), so the read can say how much of the large-scale
  spread comes from the input.

House conventions. Precipitation (``tp``, ``cp``) is clipped at zero in members and truth
before any score: a sampler can write small negative rain, and the scores are defined on
the physical field (``PRECIP_FIELDS``). The low-pass of rain is a band split of the SCORES
only (the L part is the synoptic-scale rain mass, the F part its placement and intensity);
no filtered field is written back, so the negative values a Gaussian of rain makes in the F
part are harmless and nothing is clipped after the split. ``10ff`` is derived from ``10u``
and ``10v`` when the file has no wind speed. The truth is member 0 of ``y``.
"""
from __future__ import annotations

import csv
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import xarray as xr

from eval.evaluators.surface.core.compute import (
    _area_weights,
    _to_member_point_weather,
    _weather_state_index,
)
from eval.discovery.predictions import find_predictions

from .calibration import (
    RANK_TIE_SEED,
    PooledMoments,
    histogram_shape,
    rank_counts,
    summarize_histogram,
    tie_rng,
    truth_ranks,
)
from .scale_split import DEFAULT_CUT_KM, DEFAULT_GRID_KM, LowPassOperator, box_edge_mask, inner_mask

# Strip dropped at a regional box edge in every band, as in the wide box kits (degrees).
DEFAULT_BOX_EDGE_DEG = 1.0

LOG = logging.getLogger(__name__)

# tp is in every run (owner, 30 Oct 2026); a field absent from a file is skipped with a warning.
DEFAULT_WEATHER_STATES = ["2t", "10ff", "tp", "2d", "msl", "t_850", "z_500"]
DEFAULT_DOMAINS = ["n.hem", "tropics", "s.hem", "europe"]
METRICS = ("crps", "fcrps", "spread", "rmse_ens_mean")
# clipped at zero before scoring (house convention, see module docstring)
PRECIP_FIELDS = ("tp", "cp")
BANDS = ("full", "large", "fine")
BAND_METRICS = ("fcrps", "crps", "spread_rms", "rmse_ens_mean", "ms_spread", "ms_err")
INPUT_VARIABLE = "x_interp"


def _as_list(value: Any, *, cast=str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        if not value:
            return []
        return [cast(part.strip()) for part in value.split(",") if part.strip()]
    if isinstance(value, Iterable):
        return [cast(v) for v in value]
    return [cast(value)]


def _grid_point_values(da: xr.DataArray, n_points: int) -> np.ndarray | None:
    if da.size == n_points:
        return np.asarray(da.values, dtype=np.float64).reshape(-1)
    if "grid_point_hres" in da.dims and int(da.sizes["grid_point_hres"]) == n_points:
        indexers = {dim: 0 for dim in da.dims if dim != "grid_point_hres"}
        return np.asarray(da.isel(**indexers).values, dtype=np.float64).reshape(-1)
    return None


def _lat_lon(ds: xr.Dataset, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    if "lat_hres" not in ds or "lon_hres" not in ds:
        raise ValueError("Dataset is missing lat_hres/lon_hres needed for domain masks")
    lat = _grid_point_values(ds["lat_hres"], n_points)
    lon = _grid_point_values(ds["lon_hres"], n_points)
    if lat is None or lon is None:
        raise ValueError("Could not align lat_hres/lon_hres with grid_point_hres")
    lon = ((lon + 180.0) % 360.0) - 180.0
    return lat, lon


def _domain_mask(name: str, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    key = name.lower()
    if key in {"global", "all"}:
        return np.ones(lat.shape, dtype=bool)
    if key in {"n.hem", "nhem", "north_hemisphere"}:
        return lat >= 20.0
    if key == "tropics":
        return (lat > -20.0) & (lat < 20.0)
    if key in {"s.hem", "shem", "south_hemisphere"}:
        return lat <= -20.0
    if key == "europe":
        return (lat >= 35.0) & (lat <= 75.0) & (lon >= -25.0) & (lon <= 45.0)
    raise ValueError(f"Unknown probabilistic domain: {name!r}")


def _select_field(da: xr.DataArray, ws_index: dict[str, int], field: str) -> np.ndarray | None:
    if field in ws_index:
        return np.asarray(da.isel(weather_state=ws_index[field]).values, dtype=np.float64)
    if field == "10ff":
        if "10u" not in ws_index or "10v" not in ws_index:
            return None
        u = np.asarray(da.isel(weather_state=ws_index["10u"]).values, dtype=np.float64)
        v = np.asarray(da.isel(weather_state=ws_index["10v"]).values, dtype=np.float64)
        return np.hypot(u, v)
    return None


def _clip_precip(field: str, values: np.ndarray | None) -> np.ndarray | None:
    """Negative precipitation is set to zero before scoring (house convention)."""
    if values is None or field not in PRECIP_FIELDS:
        return values
    return np.where(np.isfinite(values), np.maximum(values, 0.0), values)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    finite = np.isfinite(values) & np.isfinite(weights)
    if not np.any(finite):
        return math.nan
    w = weights[finite]
    total = float(w.sum())
    if total <= 0.0:
        return math.nan
    return float(np.sum(values[finite] * w) / total)


def crps_ensemble_components(
    forecasts: np.ndarray,
    truth: np.ndarray,
    *,
    spread_ddof: int = 1,
) -> dict[str, np.ndarray]:
    """Return pointwise CRPS, fair CRPS, spread, and ensemble-mean RMSE terms.

    Parameters use shape ``(member, point)`` for forecasts and ``(point,)`` for
    truth. Points with non-finite truth or any non-finite ensemble member are
    marked NaN in all returned arrays.
    """
    forecasts = np.asarray(forecasts, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    if forecasts.ndim != 2:
        raise ValueError(f"forecasts must have shape (member, point), got {forecasts.shape}")
    if forecasts.shape[1] != truth.size:
        raise ValueError(
            f"forecast/truth point mismatch: {forecasts.shape[1]} vs {truth.size}"
        )

    member_count = int(forecasts.shape[0])
    valid = np.isfinite(truth) & np.all(np.isfinite(forecasts), axis=0)
    out = {name: np.full(truth.shape, np.nan, dtype=np.float64) for name in METRICS}
    if member_count <= 0 or not np.any(valid):
        return out

    f = forecasts[:, valid]
    y = truth[valid]
    term1 = np.mean(np.abs(f - y[np.newaxis, :]), axis=0)

    sorted_f = np.sort(f, axis=0)
    ranks = np.arange(1, member_count + 1, dtype=np.float64)
    coeff = (2.0 * ranks - member_count - 1.0)[:, np.newaxis]
    pair_sum = 2.0 * np.sum(coeff * sorted_f, axis=0)

    out["crps"][valid] = term1 - pair_sum / (2.0 * member_count * member_count)
    if member_count > 1:
        out["fcrps"][valid] = term1 - pair_sum / (2.0 * member_count * (member_count - 1))
        ddof = min(max(int(spread_ddof), 0), member_count - 1)
    else:
        out["fcrps"][valid] = np.nan
        ddof = 0
    out["spread"][valid] = np.std(f, axis=0, ddof=ddof)
    out["rmse_ens_mean"][valid] = np.square(np.mean(f, axis=0) - y)
    return out


def _pointwise_variance(members: np.ndarray) -> np.ndarray:
    """Ensemble variance per point, ddof 1 (the quantity the calibration pools), NaN where a member is."""
    members = np.asarray(members, dtype=np.float64)
    if members.shape[0] < 2:
        return np.full(members.shape[1], np.nan)
    out = np.var(members, axis=0, ddof=1)
    out[~np.all(np.isfinite(members), axis=0)] = np.nan
    return out


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    n_points: dict[tuple[Any, ...], int] = defaultdict(int)
    for row in rows:
        key = (row["step"], row["weather_state"], row["domain"], row["metric"])
        value = float(row["value"])
        if math.isfinite(value):
            grouped[key].append(value)
            n_points[key] += int(row.get("n_points", 0))

    summaries: list[dict[str, Any]] = []
    for (step, weather_state, domain, metric), values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=np.float64)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        stderr = float(std / math.sqrt(arr.size)) if arr.size > 1 else 0.0
        summaries.append({
            "step": int(step),
            "weather_state": weather_state,
            "domain": domain,
            "metric": metric,
            "mean": mean,
            "std": std,
            "stderr": stderr,
            "n_dates": int(arr.size),
            "n_points_total": int(n_points[(step, weather_state, domain, metric)]),
        })
    return summaries


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _headline_metrics(summary_rows: list[dict[str, Any]],
                      calibration_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in summary_rows:
        key = (str(row["weather_state"]), str(row["domain"]), str(row["metric"]))
        value = float(row["mean"])
        if math.isfinite(value):
            grouped[key].append(value)
    for (weather_state, domain, metric), values in sorted(grouped.items()):
        metrics[f"probabilistic_{weather_state}_{domain}_{metric}_mean"] = float(np.mean(values))
    # the corrected spread/RMSE of the full field, mean over leads (a calibrated ensemble reads 1)
    ratios: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in calibration_rows or []:
        value = float(row.get("spread_skill_ratio_corrected", math.nan))
        if row.get("band") == "full" and math.isfinite(value):
            ratios[(str(row["weather_state"]), str(row["domain"]))].append(value)
    for (weather_state, domain), values in sorted(ratios.items()):
        metrics[f"probabilistic_{weather_state}_{domain}_spread_skill_ratio_corrected_mean"] = float(np.mean(values))
    return metrics


# --------------------------------------------------------------------------- bands

def _band_fields(
    pred_values: np.ndarray,
    truth_point: np.ndarray,
    input_values: np.ndarray | None,
    operator: LowPassOperator | None,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]]:
    """``{band: (members, truth, input members or None)}``; only ``full`` without an operator."""
    bands = {"full": (pred_values, truth_point, input_values)}
    if operator is None:
        return bands
    large_pred = np.stack([operator.apply(member) for member in pred_values])
    large_truth = operator.apply(truth_point)
    large_input = None
    if input_values is not None:
        large_input = np.stack([operator.apply(member) for member in input_values])
    bands["large"] = (large_pred, large_truth, large_input)
    bands["fine"] = (pred_values - large_pred, truth_point - large_truth,
                     None if input_values is None else input_values - large_input)
    return bands


def _score_band(
    members: np.ndarray,
    truth: np.ndarray,
    input_members: np.ndarray | None,
    *,
    weights: np.ndarray,
    masks: dict[str, np.ndarray],
    spread_ddof: int,
) -> dict[str, dict[str, Any]]:
    """Per domain: the band metrics and the pooled moments of one date (``_weighted_mean`` of each)."""
    components = crps_ensemble_components(members, truth, spread_ddof=spread_ddof)
    variance = np.square(components["spread"]) if int(spread_ddof) == 1 else _pointwise_variance(members)
    input_variance = _pointwise_variance(input_members) if input_members is not None else None
    valid = np.isfinite(truth) & np.all(np.isfinite(members), axis=0)
    out: dict[str, dict[str, Any]] = {}
    for domain_name, domain_mask in masks.items():
        mask = valid & domain_mask
        if not np.any(mask):
            continue
        w = np.where(mask, weights, 0.0)
        ms_spread = _weighted_mean(variance, w)
        ms_err = _weighted_mean(components["rmse_ens_mean"], w)
        values = {
            "fcrps": _weighted_mean(components["fcrps"], w),
            "crps": _weighted_mean(components["crps"], w),
            "ms_spread": ms_spread,
            "ms_err": ms_err,
            "spread_rms": math.sqrt(ms_spread) if math.isfinite(ms_spread) else math.nan,
            "rmse_ens_mean": math.sqrt(ms_err) if math.isfinite(ms_err) else math.nan,
            "n_points": int(mask.sum()),
            "components": components,
            "mask": mask,
        }
        if input_variance is not None:
            ms_input = _weighted_mean(input_variance, w)
            values["ms_spread_input"] = ms_input
            values["spread_input_rms"] = math.sqrt(ms_input) if math.isfinite(ms_input) else math.nan
        out[domain_name] = values
    return out


def _input_members(ds: xr.Dataset, include_input: bool) -> tuple[xr.DataArray | None, dict[str, Any]]:
    """The low-resolution input members on the output grid, and a note for the summary."""
    if not include_input:
        return None, {"variable": INPUT_VARIABLE, "n_members": 0, "note": "input spread not requested"}
    if INPUT_VARIABLE not in ds:
        return None, {"variable": INPUT_VARIABLE, "n_members": 0,
                      "note": f"files carry no {INPUT_VARIABLE}: input spread not reported"}
    da = _to_member_point_weather(ds[INPUT_VARIABLE], ds, label=INPUT_VARIABLE)
    n = int(da.sizes["member"])
    if n < 2:
        return None, {"variable": INPUT_VARIABLE, "n_members": n,
                      "note": "files carry one input member only: input spread not reported"}
    return da, {"variable": INPUT_VARIABLE, "n_members": n,
                "note": "inter-member spread of the input members per band (spread_input_rms)"}


# ---------------------------------------------------------------------- main entry

def compute_probabilistic_scores(
    predictions_dir: str | Path,
    output_dir: str | Path,
    *,
    weather_states: Iterable[str] | str | None = None,
    domains: Iterable[str] | str | None = None,
    steps: Iterable[int] | str | None = None,
    dates: Iterable[str] | str | None = None,
    spread_ddof: int = 1,
    scale_split: bool = False,
    scale_cut_km: float = DEFAULT_CUT_KM,
    scale_grid_km: float = DEFAULT_GRID_KM,
    scale_edge_margin_km: float | None = None,
    box_edge_deg: float = DEFAULT_BOX_EDGE_DEG,
    rank_histogram: bool = True,
    rank_tie_seed: int = RANK_TIE_SEED,
    include_input: bool = True,
) -> dict[str, Any]:
    """Compute local probabilistic scores and write CSV/JSON artifacts.

    ``scale_split`` adds the large/fine band scores (``P_L`` with half-power wavelength
    ``scale_cut_km``, built on a ``scale_grid_km`` grid; points within
    ``scale_edge_margin_km`` of a regional box edge, default the cut, are left out of the
    band scores). On a regional box every band, the full field included, also leaves out a
    ``box_edge_deg`` strip at the box edge (default 1 degree, as the wide box kits do; the
    edge is where the models differ most, so keeping it inflates gaps). It costs about three times the full-field scoring (eleven low-pass
    applications and two more CRPS passes per variable), so it is off unless asked for.
    """
    predictions_dir = Path(predictions_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fields = _as_list(weather_states, cast=str) or list(DEFAULT_WEATHER_STATES)
    domain_names = _as_list(domains, cast=str) or list(DEFAULT_DOMAINS)
    step_filter = set(_as_list(steps, cast=int)) if steps else None
    date_filter = set(_as_list(dates, cast=str)) if dates else None
    edge_margin_km = float(scale_cut_km if scale_edge_margin_km is None else scale_edge_margin_km)

    pred_files = find_predictions(predictions_dir)
    if step_filter is not None:
        pred_files = [p for p in pred_files if int(p.step) in step_filter]
    if date_filter is not None:
        pred_files = [p for p in pred_files if str(p.date) in date_filter]
    if not pred_files:
        raise ValueError(f"No prediction files matched filters in {predictions_dir}")

    rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    pooled: dict[tuple[int, str, str, str], PooledMoments] = defaultdict(PooledMoments)
    hist_counts: dict[tuple[int, str, str], np.ndarray] = {}
    hist_dates: dict[tuple[int, str, str], int] = defaultdict(int)
    missing_fields: set[str] = set()
    clipped_fields: set[str] = set()
    input_note: dict[str, Any] = {}
    operator: LowPassOperator | None = None
    operator_key: tuple | None = None
    n_members_seen: set[int] = set()

    for pred in pred_files:
        with xr.open_dataset(pred.path, cache=False, decode_timedelta=False) as ds:
            for required in ("y_pred", "y", "weather_state"):
                if required not in ds:
                    raise ValueError(f"{pred.path}: missing required variable {required!r}")
            y_pred = _to_member_point_weather(ds["y_pred"], ds, label="y_pred")
            y_true = _to_member_point_weather(ds["y"], ds, label="y")
            if y_true.sizes["member"] > 1:
                y_true = y_true.isel(member=0, drop=True).expand_dims(member=[0])
            x_in, note = _input_members(ds, include_input)
            input_note = input_note or note
            ws_index = _weather_state_index(ds)
            n_points = int(y_pred.sizes["grid_point_hres"])
            weights = _area_weights(ds, n_points)
            lat, lon = _lat_lon(ds, n_points)
            edge = box_edge_mask(lat, lon, float(box_edge_deg))
            domain_masks = {name: _domain_mask(name, lat, lon) & edge for name in domain_names}

            inner = np.ones(n_points, dtype=bool)
            if scale_split:
                # one operator per grid: the files of a run share their point set
                key = (n_points, float(lat[0]), float(lon[0]), float(lat[-1]), float(lon[-1]))
                if operator is None or operator_key != key:
                    operator = LowPassOperator(lat, lon, weights, cut_km=float(scale_cut_km),
                                               grid_km=float(scale_grid_km))
                    operator_key = key
                    LOG.info("probabilistic: P_L built for %d points (%s)", n_points, operator.describe()["method"])
                inner = inner_mask(lat, lon, edge_margin_km)
            band_masks = {name: mask & inner for name, mask in domain_masks.items()}

            for field in fields:
                pred_values = _clip_precip(field, _select_field(y_pred, ws_index, field))
                truth_values = _clip_precip(field, _select_field(y_true, ws_index, field))
                if pred_values is None or truth_values is None:
                    missing_fields.add(field)
                    skipped.append({
                        "path": str(pred.path),
                        "date": pred.date,
                        "step": pred.step,
                        "weather_state": field,
                        "reason": "missing field or derived-field inputs",
                    })
                    continue
                if field in PRECIP_FIELDS:
                    clipped_fields.add(field)
                if truth_values.ndim == 2:
                    truth_point = truth_values[0]
                else:
                    truth_point = np.asarray(truth_values).reshape(-1)
                input_values = _clip_precip(field, _select_field(x_in, ws_index, field)) if x_in is not None else None
                n_members = int(pred_values.shape[0])
                n_members_seen.add(n_members)

                bands = _band_fields(pred_values, truth_point, input_values, operator)
                for band, (members, truth, input_members) in bands.items():
                    masks = domain_masks if band == "full" else band_masks
                    scored = _score_band(members, truth, input_members, weights=weights, masks=masks,
                                         spread_ddof=spread_ddof)
                    for domain_name in masks:
                        if domain_name not in scored:
                            skipped.append({
                                "path": str(pred.path),
                                "date": pred.date,
                                "step": pred.step,
                                "weather_state": field,
                                "domain": domain_name,
                                "band": band,
                                "reason": "no valid points" if band == "full" else "no valid points inside the edge margin",
                            })
                            continue
                        s = scored[domain_name]
                        common = {"date": pred.date, "step": int(pred.step), "weather_state": field,
                                  "domain": domain_name, "n_points": s["n_points"], "n_members": n_members,
                                  "source_path": str(pred.path)}
                        if band == "full":
                            # the historical rows, unchanged
                            domain_weights = np.where(s["mask"], weights, 0.0)
                            for metric in METRICS:
                                value = _weighted_mean(s["components"][metric], domain_weights)
                                if metric == "rmse_ens_mean" and math.isfinite(value):
                                    value = math.sqrt(value)
                                rows.append({**common, "metric": metric, "value": value})
                        for metric in BAND_METRICS + (("spread_input_rms",) if "spread_input_rms" in s else ()):
                            band_rows.append({**common, "band": band, "metric": metric, "value": s[metric]})
                        pooled[(int(pred.step), field, domain_name, band)].add_date(
                            mean_variance=s["ms_spread"], mean_sq_error=s["ms_err"], n_points=s["n_points"],
                            n_members=n_members, fcrps=s["fcrps"], crps=s["crps"],
                            input_variance=s.get("ms_spread_input"),
                            input_members=0 if input_members is None else int(input_members.shape[0]))

                if rank_histogram and n_members > 1:
                    ranks = truth_ranks(pred_values, truth_point, tie_rng(pred.date, pred.step, field, seed=rank_tie_seed))
                    for domain_name, domain_mask in domain_masks.items():
                        counts = rank_counts(ranks, n_members, domain_mask)
                        if counts.sum() == 0:
                            continue
                        key = (int(pred.step), field, domain_name)
                        hist_counts[key] = hist_counts.get(key, 0) + counts
                        hist_dates[key] += 1
                del bands

    if not rows:
        raise ValueError(
            f"Probabilistic scoring produced no rows for {predictions_dir}; skipped={skipped[:5]}"
        )
    wanted_missing = sorted(missing_fields & set(fields))
    if "tp" in wanted_missing:
        LOG.warning("probabilistic: the files carry no tp (precipitation is in every run by the owner's rule); "
                    "tp rows are missing from %s", output_dir)

    summary_rows = _summarize(rows)
    calibration_rows = [
        {"step": step, "weather_state": ws, "domain": domain, "band": band, **moments.summary()}
        for (step, ws, domain, band), moments in sorted(pooled.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], BANDS.index(kv[0][3])))
        if moments.n_dates
    ]
    n_bins_max = max((c.size for c in hist_counts.values()), default=0)
    hist_rows = []
    for (step, ws, domain), counts in sorted(hist_counts.items()):
        summary = summarize_histogram(counts)
        hist_rows.append({
            "step": step, "weather_state": ws, "domain": domain, "n_members": int(counts.size - 1),
            "n_bins": int(counts.size), "n_dates": hist_dates[(step, ws, domain)],
            "n_points_total": int(counts.sum()), **summary, "shape": histogram_shape(summary),
            **{f"bin_{i}": int(counts[i]) if i < counts.size else "" for i in range(n_bins_max)},
        })

    score_csv = output_dir / "scores_by_lead.csv"
    summary_csv = output_dir / "summary_by_lead.csv"
    band_csv = output_dir / "band_scores_by_lead.csv"
    calibration_csv = output_dir / "calibration_by_lead.csv"
    hist_csv = output_dir / "rank_histograms.csv"
    skipped_json = output_dir / "skipped.json"
    summary_json = output_dir / "probabilistic_summary.json"

    _write_csv(
        score_csv,
        rows,
        ["date", "step", "weather_state", "domain", "metric", "value", "n_points", "n_members", "source_path"],
    )
    _write_csv(
        summary_csv,
        summary_rows,
        ["step", "weather_state", "domain", "metric", "mean", "std", "stderr", "n_dates", "n_points_total"],
    )
    _write_csv(
        band_csv,
        band_rows,
        ["date", "step", "weather_state", "domain", "band", "metric", "value", "n_points", "n_members", "source_path"],
    )
    _write_csv(
        calibration_csv,
        calibration_rows,
        ["step", "weather_state", "domain", "band", "n_members", "n_dates", "n_points_total", "fcrps_mean",
         "fcrps_stderr", "crps_mean", "spread_rms", "rmse_ens_mean", "spread_skill_ratio",
         "spread_skill_ratio_corrected", "correction_factor", "spread_input_rms", "input_n_members",
         "spread_input_share"],
    )
    _write_csv(
        hist_csv,
        hist_rows,
        ["step", "weather_state", "domain", "n_members", "n_bins", "n_dates", "n_points_total",
         "outer_two_fraction", "outer_two_expected", "flatness_rms", "rank_mean_norm", "chi2_per_bin", "shape",
         *[f"bin_{i}" for i in range(n_bins_max)]],
    )
    skipped_json.write_text(json.dumps(skipped, indent=2) + "\n")
    n_members = max(n_members_seen) if n_members_seen else 0
    payload = {
        "schema_version": "1.1",
        "predictions_dir": str(predictions_dir),
        "score_csv": str(score_csv),
        "summary_csv": str(summary_csv),
        "band_scores_csv": str(band_csv),
        "calibration_csv": str(calibration_csv),
        "rank_histograms_csv": str(hist_csv),
        "n_files": len(pred_files),
        "n_rows": len(rows),
        "n_summary_rows": len(summary_rows),
        "n_band_rows": len(band_rows),
        "weather_states": fields,
        "domains": domain_names,
        "steps": sorted({int(p.step) for p in pred_files}),
        "dates": sorted({str(p.date) for p in pred_files}),
        "skipped_count": len(skipped),
        "missing_fields": wanted_missing,
        "precip_clipped_at_zero": sorted(clipped_fields),
        "n_members": n_members,
        "spread_ddof": int(spread_ddof),
        "correction_factor": math.sqrt((n_members + 1.0) / n_members) if n_members else None,
        "box_edge_deg": float(box_edge_deg),
        "scale_split": {
            "enabled": bool(scale_split),
            "edge_margin_km": edge_margin_km if scale_split else None,
            **(operator.describe() if operator is not None else {"cut_km": float(scale_cut_km)}),
        },
        "rank_histogram": {"enabled": bool(rank_histogram), "tie_seed": int(rank_tie_seed),
                           "n_bins": n_members + 1 if n_members else None,
                           "ties": "all ties broken uniformly at random (covers zero precipitation)"},
        "input_members": input_note,
        "headline_metrics": _headline_metrics(summary_rows, calibration_rows),
    }
    summary_json.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return payload
