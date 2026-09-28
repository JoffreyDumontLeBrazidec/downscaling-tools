"""Tidy track tables for one tracked source (model expver or reference).

Consumes a completed tracker run root (``tars/`` + ``manifests/``) and writes:

- ``parsed/tracks.parquet``        one row per track point, all basins
- ``parsed/track_summary.parquet`` one row per track (lifetime extremes, genesis)
- ``parsed/forecasts.csv``         one row per (init_date, member) tar present —
  the denominator for count statistics (a forecast with zero tracks would
  otherwise be invisible in the records table)
- ``parsed/provenance.json``       support contract: tracker settings, member
  set, date coverage, completeness

CSV mirrors are written next to each parquet for greppability.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .parsing import parse_tar, records_from_tracks

#: Filename contract for tracker tars, shared by pipeline targets and lazy
#: run-root parsing: <expver>_<date>_<time>_m<member>_o<grid>_tracks.tar
TAR_NAME_RE = re.compile(
    r"^(?P<expver>.+)_(?P<date>\d{8})_(?P<time>\d{2})_m(?P<member>\d{3})_o(?P<grid>\d+)_tracks\.tar$"
)

TRACK_SUMMARY_COLUMNS = [
    "init_date", "time", "member", "basin", "track_id", "classification",
    "n_records", "first_step_h", "last_step_h",
    "mslp_min_hpa", "wind_max_ms",
    "mslp_min_valid_time", "mslp_min_lat", "mslp_min_lon_e",
    "genesis_lat", "genesis_lon_e",
]


def parse_source_tars(config: Any) -> tuple[list[dict], list[dict], list[dict]]:
    """Parse every present tar of a TCTrackerConfig into (records, summaries,
    forecasts). Missing tars are skipped and recorded absent in forecasts."""
    records: list[dict] = []
    summaries: list[dict] = []
    forecasts: list[dict] = []
    for target in config.targets:
        present = target.tar_path.exists() and target.tar_path.stat().st_size > 0
        forecasts.append({
            "init_date": target.date,
            "time": config.time,
            "member": target.member,
            "present": present,
        })
        if not present:
            continue
        tracks = parse_tar(target.tar_path, source=target.tag)
        records.extend(records_from_tracks(
            tracks, init_date=target.date, time=config.time, member=target.member,
        ))
        for track in tracks:
            recs = track["records"]
            steps = [r_step for r_step in (
                _safe_step(config, target, r) for r in recs) if r_step is not None]
            summaries.append({
                "init_date": target.date,
                "time": config.time,
                "member": target.member,
                "basin": track["basin"],
                "track_id": f"{track['basin']}-{track['seq']:03d}",
                "classification": track.get("classification"),
                "n_records": len(recs),
                "first_step_h": min(steps) if steps else None,
                "last_step_h": max(steps) if steps else None,
                "mslp_min_hpa": track.get("mslp_min_hpa"),
                "wind_max_ms": track.get("wind_max_ms"),
                "mslp_min_valid_time": track.get("mslp_min_valid_time"),
                "mslp_min_lat": track.get("mslp_min_lat"),
                "mslp_min_lon_e": track.get("mslp_min_lon_e"),
                "genesis_lat": track.get("genesis_lat"),
                "genesis_lon_e": track.get("genesis_lon_e"),
            })
    return records, summaries, forecasts


def _safe_step(config: Any, target: Any, rec: dict) -> int | None:
    from .parsing import step_hours
    try:
        return step_hours(target.date, config.time, rec["valid_time"])
    except ValueError:
        return None


def build_provenance(config: Any, forecasts: list[dict], *, role: str = "", source_id: str = "") -> dict:
    present = [f for f in forecasts if f["present"]]
    dates_present = sorted({f["init_date"] for f in present})
    return {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "role": role,
        "source_id": source_id or config.expver,
        "lane": config.lane,
        "expver": config.expver,
        "fdb_class": config.fdb_class,
        "fdb_type": config.fdb_type,
        "stream": config.stream,
        "grid_support": f"o{config.grid}",
        "time": config.time,
        "steps": f"{config.start_step}-{config.end_step}/{config.step_interval}h",
        "vorticity": config.vorticity,
        "members": sorted({f["member"] for f in forecasts}),
        "dates_requested": list(config.dates),
        "dates_present": dates_present,
        "n_forecasts_requested": len(forecasts),
        "n_forecasts_present": len(present),
        "completeness": round(len(present) / len(forecasts), 4) if forecasts else 0.0,
    }


def write_tables(
    run_root: Path,
    records: list[dict],
    summaries: list[dict],
    forecasts: list[dict],
    provenance: dict,
) -> Path:
    """Write parsed tables under ``<run_root>/parsed/``; returns that dir."""
    import pandas as pd

    parsed_dir = Path(run_root) / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)

    records_df = pd.DataFrame(records)
    summary_df = pd.DataFrame(summaries, columns=TRACK_SUMMARY_COLUMNS if summaries else None)
    forecasts_df = pd.DataFrame(forecasts)

    records_df.to_csv(parsed_dir / "tracks.csv", index=False)
    summary_df.to_csv(parsed_dir / "track_summary.csv", index=False)
    forecasts_df.to_csv(parsed_dir / "forecasts.csv", index=False)
    try:
        records_df.to_parquet(parsed_dir / "tracks.parquet", index=False)
        summary_df.to_parquet(parsed_dir / "track_summary.parquet", index=False)
    except Exception:  # pyarrow missing/old — CSVs above remain authoritative
        pass
    (parsed_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, default=str) + "\n", encoding="utf-8",
    )
    return parsed_dir


def parse_and_write(config: Any, *, role: str = "", source_id: str = "") -> Path:
    """One-call parse of a completed tracker run into tidy tables."""
    records, summaries, forecasts = parse_source_tars(config)
    provenance = build_provenance(config, forecasts, role=role, source_id=source_id)
    return write_tables(config.output_dir, records, summaries, forecasts, provenance)


def parse_run_root(run_root: Path, *, role: str = "", source_id: str = "") -> Path:
    """Parse a tracker run root WITHOUT a TCTrackerConfig, reconstructing
    (date, member, time, grid) from the tar filename contract. Used by
    ``tccompare`` to lazily build tables for any completed run. Completeness
    vs the originally requested target set is unknown here (only present tars
    are visible), so ``completeness`` is null in the provenance."""
    run_root = Path(run_root)
    tars = sorted((run_root / "tars").glob("*.tar"))
    records: list[dict] = []
    summaries: list[dict] = []
    forecasts: list[dict] = []
    grids: set[str] = set()
    expvers: set[str] = set()
    from .parsing import step_hours

    for tar_path in tars:
        match = TAR_NAME_RE.match(tar_path.name)
        if not match:
            continue
        date, time, member = match["date"], match["time"], int(match["member"])
        grids.add(match["grid"])
        expvers.add(match["expver"])
        forecasts.append({"init_date": date, "time": time, "member": member, "present": True})
        tracks = parse_tar(tar_path, source=tar_path.stem)
        records.extend(records_from_tracks(tracks, init_date=date, time=time, member=member))
        for track in tracks:
            recs = track["records"]
            steps = []
            for rec in recs:
                try:
                    steps.append(step_hours(date, time, rec["valid_time"]))
                except ValueError:
                    pass
            summaries.append({
                "init_date": date, "time": time, "member": member,
                "basin": track["basin"],
                "track_id": f"{track['basin']}-{track['seq']:03d}",
                "classification": track.get("classification"),
                "n_records": len(recs),
                "first_step_h": min(steps) if steps else None,
                "last_step_h": max(steps) if steps else None,
                "mslp_min_hpa": track.get("mslp_min_hpa"),
                "wind_max_ms": track.get("wind_max_ms"),
                "mslp_min_valid_time": track.get("mslp_min_valid_time"),
                "mslp_min_lat": track.get("mslp_min_lat"),
                "mslp_min_lon_e": track.get("mslp_min_lon_e"),
                "genesis_lat": track.get("genesis_lat"),
                "genesis_lon_e": track.get("genesis_lon_e"),
            })
    provenance = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "role": role,
        "source_id": source_id or "/".join(sorted(expvers)),
        "expver": "/".join(sorted(expvers)),
        "grid_support": f"o{grids.pop()}" if len(grids) == 1 else sorted(grids),
        "members": sorted({f["member"] for f in forecasts}),
        "dates_present": sorted({f["init_date"] for f in forecasts}),
        "n_forecasts_present": len(forecasts),
        "completeness": None,
        "parsed_from": "run_root_lazy",
    }
    effective = run_root / "effective_config.json"
    if effective.exists():
        try:
            eff = json.loads(effective.read_text())
            tc = (eff.get("tctracker") or {}).get("config") or {}
            if tc:
                provenance["steps"] = (
                    f"{tc.get('start_step')}-{tc.get('end_step')}/{tc.get('step_interval')}h"
                )
                provenance["vorticity"] = tc.get("vorticity")
                provenance["time"] = tc.get("time")
                provenance["fdb_class"] = tc.get("fdb_class")
                provenance["stream"] = tc.get("stream")
        except (json.JSONDecodeError, OSError):
            pass
    return write_tables(run_root, records, summaries, forecasts, provenance)
