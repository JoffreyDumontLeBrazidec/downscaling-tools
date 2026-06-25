"""Local spread/CRPS scoring for eval.cli prediction NetCDFs."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import xarray as xr

from eval._backends.scoreboard._surface_compute import (
    _area_weights,
    _to_member_point_weather,
    _weather_state_index,
)
from eval.discovery.predictions import find_predictions

DEFAULT_WEATHER_STATES = ["2t", "10ff", "2d", "msl", "t_850", "z_500"]
DEFAULT_DOMAINS = ["n.hem", "tropics", "s.hem", "europe"]
METRICS = ("crps", "fcrps", "spread", "rmse_ens_mean")


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


def _headline_metrics(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in summary_rows:
        key = (str(row["weather_state"]), str(row["domain"]), str(row["metric"]))
        value = float(row["mean"])
        if math.isfinite(value):
            grouped[key].append(value)
    for (weather_state, domain, metric), values in sorted(grouped.items()):
        metrics[f"probabilistic_{weather_state}_{domain}_{metric}_mean"] = float(np.mean(values))
    return metrics


def compute_probabilistic_scores(
    predictions_dir: str | Path,
    output_dir: str | Path,
    *,
    weather_states: Iterable[str] | str | None = None,
    domains: Iterable[str] | str | None = None,
    steps: Iterable[int] | str | None = None,
    dates: Iterable[str] | str | None = None,
    spread_ddof: int = 1,
) -> dict[str, Any]:
    """Compute local probabilistic scores and write CSV/JSON artifacts."""
    predictions_dir = Path(predictions_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fields = _as_list(weather_states, cast=str) or list(DEFAULT_WEATHER_STATES)
    domain_names = _as_list(domains, cast=str) or list(DEFAULT_DOMAINS)
    step_filter = set(_as_list(steps, cast=int)) if steps else None
    date_filter = set(_as_list(dates, cast=str)) if dates else None

    pred_files = find_predictions(predictions_dir)
    if step_filter is not None:
        pred_files = [p for p in pred_files if int(p.step) in step_filter]
    if date_filter is not None:
        pred_files = [p for p in pred_files if str(p.date) in date_filter]
    if not pred_files:
        raise ValueError(f"No prediction files matched filters in {predictions_dir}")

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for pred in pred_files:
        with xr.open_dataset(pred.path, cache=False, decode_timedelta=False) as ds:
            for required in ("y_pred", "y", "weather_state"):
                if required not in ds:
                    raise ValueError(f"{pred.path}: missing required variable {required!r}")
            y_pred = _to_member_point_weather(ds["y_pred"], ds, label="y_pred")
            y_true = _to_member_point_weather(ds["y"], ds, label="y")
            if y_true.sizes["member"] > 1:
                y_true = y_true.isel(member=0, drop=True).expand_dims(member=[0])
            ws_index = _weather_state_index(ds)
            n_points = int(y_pred.sizes["grid_point_hres"])
            weights = _area_weights(ds, n_points)
            lat, lon = _lat_lon(ds, n_points)
            domain_masks = {name: _domain_mask(name, lat, lon) for name in domain_names}

            for field in fields:
                pred_values = _select_field(y_pred, ws_index, field)
                truth_values = _select_field(y_true, ws_index, field)
                if pred_values is None or truth_values is None:
                    skipped.append({
                        "path": str(pred.path),
                        "date": pred.date,
                        "step": pred.step,
                        "weather_state": field,
                        "reason": "missing field or derived-field inputs",
                    })
                    continue
                if truth_values.ndim == 2:
                    truth_point = truth_values[0]
                else:
                    truth_point = np.asarray(truth_values).reshape(-1)
                components = crps_ensemble_components(
                    pred_values,
                    truth_point,
                    spread_ddof=spread_ddof,
                )
                valid_base = np.isfinite(truth_point) & np.all(np.isfinite(pred_values), axis=0)
                for domain_name, domain_mask in domain_masks.items():
                    mask = valid_base & domain_mask
                    if not np.any(mask):
                        skipped.append({
                            "path": str(pred.path),
                            "date": pred.date,
                            "step": pred.step,
                            "weather_state": field,
                            "domain": domain_name,
                            "reason": "no valid points",
                        })
                        continue
                    domain_weights = np.where(mask, weights, 0.0)
                    for metric in METRICS:
                        point_values = components[metric]
                        value = _weighted_mean(point_values, domain_weights)
                        if metric == "rmse_ens_mean" and math.isfinite(value):
                            value = math.sqrt(value)
                        rows.append({
                            "date": pred.date,
                            "step": int(pred.step),
                            "weather_state": field,
                            "domain": domain_name,
                            "metric": metric,
                            "value": value,
                            "n_points": int(mask.sum()),
                            "n_members": int(pred_values.shape[0]),
                            "source_path": str(pred.path),
                        })

    if not rows:
        raise ValueError(
            f"Probabilistic scoring produced no rows for {predictions_dir}; skipped={skipped[:5]}"
        )

    summary_rows = _summarize(rows)
    score_csv = output_dir / "scores_by_lead.csv"
    summary_csv = output_dir / "summary_by_lead.csv"
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
    skipped_json.write_text(json.dumps(skipped, indent=2) + "\n")
    payload = {
        "schema_version": "1.0",
        "predictions_dir": str(predictions_dir),
        "score_csv": str(score_csv),
        "summary_csv": str(summary_csv),
        "n_files": len(pred_files),
        "n_rows": len(rows),
        "n_summary_rows": len(summary_rows),
        "weather_states": fields,
        "domains": domain_names,
        "steps": sorted({int(p.step) for p in pred_files}),
        "dates": sorted({str(p.date) for p in pred_files}),
        "skipped_count": len(skipped),
        "headline_metrics": _headline_metrics(summary_rows),
    }
    summary_json.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return payload
