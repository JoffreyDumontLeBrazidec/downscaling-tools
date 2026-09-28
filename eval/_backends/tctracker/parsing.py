"""All-basin parsing of ECMWF tctracker archives into track/record structures.

The tracker emits one HURDAT-style text file per basin inside each per-(date,
member) tar. The grammar is identical across basins (verified on j761 tars:
atl/wnp/enp share the same header / record / classification line shapes; an
empty basin is a single ``M= 0`` header). This module generalises the original
Atlantic-only parser in ``pipeline.py`` to every basin and derives tidy
per-record rows suitable for table building (``tables.py``).

Latitude sign convention: tctracker basin files store latitude as an unsigned
packed value; hemisphere is implied by the basin. Southern-hemisphere basins
(sin/aus/spc) are stored positive and negated here. Longitudes are degrees
east in [0, 360). Validate against a southern-hemisphere storm before using
SH basins for verdicts.
"""
from __future__ import annotations

import re
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

KT_TO_MS = 1.0 / 1.94384

#: Basin file suffixes emitted by tctracker, and the implied latitude sign.
BASIN_LAT_SIGN = {
    "atl": 1.0,
    "enp": 1.0,
    "cnp": 1.0,
    "wnp": 1.0,
    "nin": 1.0,
    "sin": -1.0,
    "aus": -1.0,
    "spc": -1.0,
}
BASINS = tuple(BASIN_LAT_SIGN)

_HEADER_RE = re.compile(r"\d{5}\s+(\d+/\d+/\d+)\s+M=\s*(\d+)\s+(\d+)\s+SNBR=\s*(\d+)")
_CLASS_RE = re.compile(r"\d+\s+(HR\d|TS|TD|ET|SSD)\s*$")
_RECORD_RE = re.compile(r"\d{5}\s+(\d{4}/\d{2}/\d{2}/\d{2})\*(\d{7})\s+(\d+)\s+(\d+)\*")


def parse_basin_text(text: str, basin: str, source: str) -> list[dict[str, Any]]:
    """Parse one basin file's text into a list of track dicts.

    Mirrors the proven Atlantic grammar; ``basin`` only affects the latitude
    sign and the ``basin`` field attached to each track.
    """
    lat_sign = BASIN_LAT_SIGN.get(basin, 1.0)
    tracks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in text.splitlines():
        header = _HEADER_RE.match(line)
        if header:
            if current and current["records"]:
                tracks.append(current)
            current = {
                "source": source,
                "basin": basin,
                "start_date": header.group(1),
                "n_steps": int(header.group(2)),
                "seq": int(header.group(3)),
                "snbr": int(header.group(4)),
                "records": [],
                "classification": None,
            }
            continue
        cls = _CLASS_RE.match(line)
        if cls and current is not None:
            current["classification"] = cls.group(1)
            continue
        rec = _RECORD_RE.match(line)
        if rec and current is not None:
            raw = rec.group(2)
            current["records"].append({
                "valid_time": rec.group(1),
                "lat": lat_sign * int(raw[:3]) / 10.0,
                "lon_e": int(raw[3:]) / 10.0,
                "wind_kt": int(rec.group(3)),
                "wind_ms": int(rec.group(3)) * KT_TO_MS,
                "mslp_hpa": int(rec.group(4)),
            })
    if current and current["records"]:
        tracks.append(current)
    for track in tracks:
        _finalise_track(track)
    return tracks


def _finalise_track(track: dict[str, Any]) -> None:
    records = track["records"]
    if not records:
        return
    track["wind_max_kt"] = max(r["wind_kt"] for r in records)
    track["wind_max_ms"] = max(r["wind_ms"] for r in records)
    track["mslp_min_hpa"] = min(r["mslp_hpa"] for r in records)
    track["first_valid_time"] = records[0]["valid_time"]
    track["last_valid_time"] = records[-1]["valid_time"]
    deepest = min(records, key=lambda r: r["mslp_hpa"])
    track["mslp_min_lat"] = deepest["lat"]
    track["mslp_min_lon_e"] = deepest["lon_e"]
    track["mslp_min_valid_time"] = deepest["valid_time"]
    track["genesis_lat"] = records[0]["lat"]
    track["genesis_lon_e"] = records[0]["lon_e"]


def step_hours(init_date: str, time: str, valid_time: str) -> int:
    """Forecast step in hours from init (``YYYYMMDD`` + ``HH``) to a record's
    ``YYYY/MM/DD/HH`` valid time."""
    init = datetime.strptime(f"{init_date}{int(time):02d}", "%Y%m%d%H")
    valid = datetime.strptime(valid_time, "%Y/%m/%d/%H")
    return int(round((valid - init).total_seconds() / 3600.0))


def parse_tar(
    tar_path: Path,
    source: str,
    basins: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Parse every requested basin file inside one tracker tar."""
    wanted = tuple(basins) if basins else BASINS
    tracks: list[dict[str, Any]] = []
    with tarfile.open(tar_path) as tf:
        for name in tf.getnames():
            basin = name.rsplit("_", 1)[-1]
            if basin not in wanted:
                continue
            fh = tf.extractfile(name)
            if fh is None:
                continue
            text = fh.read().decode("utf-8", errors="replace")
            tracks.extend(parse_basin_text(text, basin, source))
    return tracks


def records_from_tracks(
    tracks: list[dict[str, Any]],
    *,
    init_date: str,
    time: str,
    member: int,
) -> list[dict[str, Any]]:
    """Flatten track dicts into tidy per-record rows.

    ``track_id`` is unique within (init_date, member, basin): ``<basin>-<seq>``.
    """
    rows: list[dict[str, Any]] = []
    for track in tracks:
        track_id = f"{track['basin']}-{track['seq']:03d}"
        for rec in track["records"]:
            rows.append({
                "init_date": init_date,
                "time": time,
                "member": member,
                "basin": track["basin"],
                "track_id": track_id,
                "classification": track.get("classification"),
                "valid_time": rec["valid_time"],
                "step_h": step_hours(init_date, time, rec["valid_time"]),
                "lat": rec["lat"],
                "lon_e": rec["lon_e"],
                "wind_ms": rec["wind_ms"],
                "mslp_hpa": rec["mslp_hpa"],
            })
    return rows
