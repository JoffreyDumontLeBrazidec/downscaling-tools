"""Distributional track statistics per (scope, basin, role).

Doctrine hooks:
- ONE support: every loaded source's provenance is checked for a consistent
  (grid_support, steps, vorticity) contract; violations are WARNED and stamped
  into the metrics, never gated.
- Distributional only: no member pairing anywhere (EEFO/ENFO are not paired).
- Count statistics are normalised per forecast (present (date, member) tars),
  so sources with different completeness or member budgets stay comparable.
- Bootstrap CIs resample tracks (distribution stats) or init-dates (counts).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

TS_WIND_MS = 17.5  # tropical-storm threshold used for TC-day counts
DENSITY_BIN_DEG = 2.0
SUPPORT_KEYS = ("grid_support", "steps", "vorticity", "time")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_source(parsed_dir: Path) -> dict[str, Any]:
    """Load one source's parsed tables (parquet preferred, CSV fallback)."""
    parsed_dir = Path(parsed_dir)

    def _read(stem: str) -> pd.DataFrame:
        pq = parsed_dir / f"{stem}.parquet"
        if pq.exists():
            return pd.read_parquet(pq)
        csv = parsed_dir / f"{stem}.csv"
        if csv.exists() and csv.stat().st_size > 1:
            return pd.read_csv(csv, dtype={"init_date": str, "time": str})
        return pd.DataFrame()

    provenance_path = parsed_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    records = _read("tracks")
    summary = _read("track_summary")
    forecasts = _read("forecasts")
    for df in (records, summary, forecasts):
        if not df.empty and "init_date" in df:
            df["init_date"] = df["init_date"].astype(str)
    return {
        "records": records,
        "summary": summary,
        "forecasts": forecasts,
        "provenance": provenance,
    }


def check_support_contract(sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """All sources must share one tracking support. Warn-only."""
    reference: dict[str, Any] = {}
    violations: list[str] = []
    for role, src in sources.items():
        prov = src.get("provenance") or {}
        for key in SUPPORT_KEYS:
            value = prov.get(key)
            if key not in reference:
                reference[key] = value
            elif value != reference[key]:
                violations.append(f"{role}.{key}={value!r} != {reference[key]!r}")
    if violations:
        LOG.warning("SUPPORT CONTRACT VIOLATIONS (figures remain rendered, "
                    "read with care): %s", violations)
    return {"contract": reference, "consistent": not violations, "violations": violations}


# ---------------------------------------------------------------------------
# Filtering / scoping
# ---------------------------------------------------------------------------

def scope_mask(df: pd.DataFrame, months: list[str] | None, basins: list[str] | None) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)
    mask = pd.Series(True, index=df.index)
    if months:
        mask &= df["init_date"].str[:6].isin([str(m) for m in months])
    if basins and "basin" in df:
        mask &= df["basin"].isin(basins)
    return mask


def _scoped(src: dict[str, Any], months: list[str] | None, basins: list[str] | None) -> dict[str, pd.DataFrame]:
    records = src["records"]
    summary = src["summary"]
    forecasts = src["forecasts"]
    return {
        "records": records[scope_mask(records, months, basins)] if not records.empty else records,
        "summary": summary[scope_mask(summary, months, basins)] if not summary.empty else summary,
        "forecasts": forecasts[scope_mask(forecasts, months, None)] if not forecasts.empty else forecasts,
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _bootstrap_ci(values: np.ndarray, stat, n_boot: int = 1000, seed: int = 0) -> tuple[float, float] | None:
    if len(values) < 3:
        return None
    rng = np.random.default_rng(seed)
    stats = [stat(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def _count_ci_per_forecast(summary: pd.DataFrame, forecasts: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> tuple[float, float] | None:
    """Block bootstrap over init-dates for tracks-per-forecast."""
    if forecasts.empty:
        return None
    present = forecasts[forecasts["present"]] if "present" in forecasts else forecasts
    dates = sorted(present["init_date"].unique())
    if len(dates) < 3:
        return None
    per_date_fc = present.groupby("init_date").size()
    per_date_tracks = summary.groupby("init_date").size() if not summary.empty else pd.Series(dtype=int)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        sample = rng.choice(dates, size=len(dates), replace=True)
        n_fc = sum(per_date_fc.get(d, 0) for d in sample)
        n_tr = sum(per_date_tracks.get(d, 0) for d in sample)
        stats.append(n_tr / n_fc if n_fc else 0.0)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def score_role_scope(scoped: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """All scalar statistics for one (role, scope, basin-set) slice."""
    records = scoped["records"]
    summary = scoped["summary"]
    forecasts = scoped["forecasts"]
    present = forecasts[forecasts["present"]] if ("present" in forecasts and not forecasts.empty) else forecasts
    n_forecasts = int(len(present))
    n_tracks = int(len(summary))
    out: dict[str, Any] = {
        "n_forecasts": n_forecasts,
        "n_tracks": n_tracks,
        "tracks_per_forecast": round(n_tracks / n_forecasts, 4) if n_forecasts else None,
        "tracks_per_forecast_ci": _count_ci_per_forecast(summary, forecasts),
    }
    if not summary.empty:
        mslp = summary["mslp_min_hpa"].dropna().to_numpy(dtype=float)
        wind = summary["wind_max_ms"].dropna().to_numpy(dtype=float)
        out.update({
            "mslp_min": float(np.min(mslp)) if len(mslp) else None,
            "mslp_p5": float(np.percentile(mslp, 5)) if len(mslp) else None,
            "mslp_median": float(np.median(mslp)) if len(mslp) else None,
            "mslp_median_ci": _bootstrap_ci(mslp, np.median),
            "mslp_p5_ci": _bootstrap_ci(mslp, lambda v: np.percentile(v, 5)),
            "wind_max": float(np.max(wind)) if len(wind) else None,
            "wind_p95": float(np.percentile(wind, 95)) if len(wind) else None,
            "wind_median": float(np.median(wind)) if len(wind) else None,
            "wind_median_ci": _bootstrap_ci(wind, np.median),
        })
        cls = summary["classification"].fillna("unknown").value_counts().to_dict()
        out["classification_counts"] = {str(k): int(v) for k, v in cls.items()}
    if not records.empty:
        tc_days = int((records["wind_ms"] >= TS_WIND_MS).sum())
        out["tc_days"] = tc_days
        out["tc_days_per_forecast"] = round(tc_days / n_forecasts, 4) if n_forecasts else None
        step_stats = (
            records.groupby("step_h")["mslp_hpa"]
            .agg(median="median",
                 q10=lambda s: float(np.percentile(s, 10)),
                 q90=lambda s: float(np.percentile(s, 90)),
                 n="size")
            .reset_index()
        )
        out["step_intensity"] = step_stats.to_dict(orient="records")
    return out


def density_grid(records: pd.DataFrame, kind: str = "records") -> dict[str, Any]:
    """2-degree track (or genesis) density, normalised per forecast later."""
    lat_edges = np.arange(-60.0, 60.0 + DENSITY_BIN_DEG, DENSITY_BIN_DEG)
    lon_edges = np.arange(0.0, 360.0 + DENSITY_BIN_DEG, DENSITY_BIN_DEG)
    if records.empty:
        hist = np.zeros((len(lat_edges) - 1, len(lon_edges) - 1))
    else:
        hist, _, _ = np.histogram2d(
            records["lat"].to_numpy(dtype=float),
            records["lon_e"].to_numpy(dtype=float) % 360.0,
            bins=[lat_edges, lon_edges],
        )
    return {"hist": hist, "lat_edges": lat_edges, "lon_edges": lon_edges, "kind": kind}


# ---------------------------------------------------------------------------
# Top-level scoring
# ---------------------------------------------------------------------------

def score_sources(
    sources: dict[str, dict[str, Any]],
    months: list[str],
    basins: list[str],
) -> dict[str, Any]:
    """Metrics for every (scope, basin, role). Scopes = each month + 'all'."""
    scopes: list[tuple[str, list[str] | None]] = [(m, [m]) for m in months]
    if len(months) != 1:
        scopes.append(("all", months or None))
    rows: list[dict[str, Any]] = []
    for scope_name, scope_months in scopes:
        for basin in basins:
            for role, src in sources.items():
                scoped = _scoped(src, scope_months, [basin])
                stats = score_role_scope(scoped)
                rows.append({"scope": scope_name, "basin": basin, "role": role, **stats})
    return {
        "support": check_support_contract(sources),
        "scopes": [s for s, _ in scopes],
        "basins": basins,
        "months": months,
        "roles": list(sources),
        "metrics": rows,
    }


# ---------------------------------------------------------------------------
# Case selection (named-storm-like panels without per-storm config)
# ---------------------------------------------------------------------------

def select_cases(
    sources: dict[str, dict[str, Any]],
    months: list[str],
    basin: str,
    *,
    reference_role: str = "target",
    top_k: int = 3,
    match_deg: float = 5.0,
    match_hours: float = 48.0,
) -> list[dict[str, Any]]:
    """Pick the ``top_k`` deepest reference tracks in scope and associate every
    role's tracks by proximity of their deepest point (space AND time). Purely
    diagnostic association — never used for paired statistics."""
    ref = sources.get(reference_role)
    if ref is None or ref["summary"].empty:
        return []
    ref_sum = ref["summary"][scope_mask(ref["summary"], months, [basin])].dropna(subset=["mslp_min_hpa"])
    if ref_sum.empty:
        return []
    # Dedup near-identical reference storms across members/dates: greedy pick
    # deepest, suppress later picks within the match radius/window.
    ref_sorted = ref_sum.sort_values("mslp_min_hpa")
    picked: list[pd.Series] = []
    for _, row in ref_sorted.iterrows():
        if len(picked) >= top_k:
            break
        if any(_close(row, p, match_deg, match_hours) for p in picked):
            continue
        picked.append(row)
    cases = []
    for i, ref_row in enumerate(picked):
        case = {
            "case_id": f"{basin}_case{i + 1}_{ref_row['init_date']}",
            "reference_role": reference_role,
            "mslp_min_hpa": float(ref_row["mslp_min_hpa"]),
            "at": {
                "lat": float(ref_row["mslp_min_lat"]),
                "lon_e": float(ref_row["mslp_min_lon_e"]),
                "valid_time": str(ref_row["mslp_min_valid_time"]),
            },
            "members": {},
        }
        for role, src in sources.items():
            summ = src["summary"]
            if summ.empty:
                case["members"][role] = []
                continue
            in_scope = summ[scope_mask(summ, months, [basin])].dropna(subset=["mslp_min_hpa"])
            matched = in_scope[[_close(r, ref_row, match_deg, match_hours) for _, r in in_scope.iterrows()]]
            case["members"][role] = matched[["init_date", "member", "track_id"]].to_dict(orient="records")
        cases.append(case)
    return cases


def _close(a: pd.Series, b: pd.Series, match_deg: float, match_hours: float) -> bool:
    try:
        ta = pd.to_datetime(str(a["mslp_min_valid_time"]), format="%Y/%m/%d/%H")
        tb = pd.to_datetime(str(b["mslp_min_valid_time"]), format="%Y/%m/%d/%H")
    except (ValueError, TypeError):
        return False
    dlon = abs(float(a["mslp_min_lon_e"]) - float(b["mslp_min_lon_e"])) % 360.0
    dlon = min(dlon, 360.0 - dlon)
    dist_ok = abs(float(a["mslp_min_lat"]) - float(b["mslp_min_lat"])) <= match_deg and dlon <= match_deg
    time_ok = abs((ta - tb).total_seconds()) / 3600.0 <= match_hours
    return dist_ok and time_ok
