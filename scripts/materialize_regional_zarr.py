#!/usr/bin/env python3
"""Materialize a spatial/time/variable subset of an Anemoi zarr dataset.

This is the fast path for regional downscaling testbeds: use
``anemoi.datasets.open_dataset(..., area=...)`` to read a cropped view from an
existing full-grid zarr, then write a normal zarr store that can be reopened by
``anemoi.datasets.open_dataset``.

The script deliberately defaults to preserving source statistics. For warm-start
or apples-to-apples experiments this keeps normalization compatible with the
full-lane checkpoints. Recomputing regional statistics is a separate scientific
choice and should be done explicitly in a follow-up tool/arm.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from numcodecs import Blosc

try:
    from anemoi.datasets import open_dataset
except Exception as exc:  # pragma: no cover - CLI environment guard
    raise SystemExit(
        "Failed to import anemoi.datasets. Run with a certified/downscaling "
        f"environment such as /home/ecm5702/dev/.ds-260612/bin/python. Error: {exc}"
    ) from exc


BOX_PRESETS: dict[str, list[float]] = {
    # Canonical regional o320->o1280 TC training box, selected from HURDAT2
    # 2003-2023 coverage versus O1280 cell cost.
    "tc_atlantic_mdr_west": [5.0, 35.0, -100.0, -40.0],
}


def _split_csv(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    out: list[str] = []
    for value in values:
        out.extend(part.strip() for part in value.split(",") if part.strip())
    return out or None


def _split_int_csv(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _load_indices_file(path: str | None) -> list[int] | None:
    if path is None:
        return None
    raw = Path(path).expanduser().read_text().strip()
    if not raw:
        return []
    if raw[0] == "[":
        values = json.loads(raw)
    else:
        values = [part for line in raw.splitlines() for part in line.replace(",", " ").split()]
    return [int(value) for value in values]


def _frequency_delta(value: str) -> np.timedelta64:
    value = value.strip().lower()
    if len(value) < 2:
        raise ValueError(f"Invalid frequency: {value!r}")
    amount = int(value[:-1])
    unit = value[-1]
    if unit == "h":
        return np.timedelta64(amount, "h")
    if unit == "d":
        return np.timedelta64(amount, "D")
    if unit == "m":
        return np.timedelta64(amount, "m")
    raise ValueError(f"Unsupported reindex frequency {value!r}; use units h, d, or m")


def _reindex_dates_like(dates: np.ndarray, frequency: str | None) -> np.ndarray:
    dates = np.asarray(dates, dtype="datetime64[s]")
    if not frequency or len(dates) == 0:
        return dates
    step = _frequency_delta(frequency).astype("timedelta64[s]")
    return dates[0] + np.arange(len(dates), dtype="int64") * step


def _infer_uniform_step(dates: np.ndarray) -> np.timedelta64 | None:
    """Return the constant step of a uniformly spaced axis, else None.

    A 0- or 1-element axis is treated as trivially uniform (None step) and must
    be handled by the caller; a multi-element axis with mixed gaps returns None.
    """
    dates = np.asarray(dates, dtype="datetime64[s]")
    if len(dates) < 2:
        return None
    diffs = np.diff(dates)
    step = diffs[0]
    if step <= np.timedelta64(0, "s"):
        return None
    if np.all(diffs == step):
        return step
    return None


def _timedelta_to_freq(step: np.timedelta64) -> str:
    """Express a timedelta as an h/d/m frequency string (the units anemoi uses)."""
    seconds = int(step / np.timedelta64(1, "s"))
    if seconds % 86400 == 0:
        return f"{seconds // 86400}d"
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}m"
    raise ValueError(f"Cannot express step {step} with h/d/m units")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _date_key(value: Any) -> str:
    arr = np.asarray(value, dtype="datetime64[s]")
    return np.datetime_as_string(arr, unit="s")


def _patch_metadata(
    attrs: dict[str, Any],
    *,
    source: Path,
    shape: tuple[int, ...],
    grid_chunk: int,
    variables: list[str],
    dates: np.ndarray,
    source_dates: np.ndarray,
    area: list[float] | None,
    bbox: list[float] | None,
    source_fake_hindcasts: dict[str, Any] | None,
    statistics_policy: str,
    description: str | None,
    frequency: str | None,
    reindex_frequency: str | None,
) -> dict[str, Any]:
    patched = dict(attrs)
    patched["variables"] = list(variables)
    patched["shape"] = [int(x) for x in shape]
    patched["field_shape"] = [int(shape[-1])]
    patched["chunks"] = [1, int(shape[1]), int(shape[2]), int(grid_chunk)]
    patched["total_size"] = int(np.prod(shape) * np.dtype("float32").itemsize)
    patched["start_date"] = _date_key(dates[0]) if len(dates) else None
    patched["end_date"] = _date_key(dates[-1]) if len(dates) else None
    patched["description"] = description or f"Regional materialization of {source}"

    if isinstance(patched.get("variables_metadata"), dict):
        patched["variables_metadata"] = {
            name: patched["variables_metadata"][name]
            for name in variables
            if name in patched["variables_metadata"]
        }
    if isinstance(patched.get("constant_fields"), list):
        patched["constant_fields"] = [name for name in patched["constant_fields"] if name in variables]
    if isinstance(patched.get("variables_with_nans"), list):
        patched["variables_with_nans"] = [name for name in patched["variables_with_nans"] if name in variables]

    if source_fake_hindcasts:
        fake_subset = {}
        for written_date, source_date in zip(dates, source_dates, strict=True):
            source_key = _date_key(source_date)
            if source_key in source_fake_hindcasts:
                fake_subset[_date_key(written_date)] = source_fake_hindcasts[source_key]
        patched["fake_hindcasts"] = fake_subset
    # Always stamp the true axis cadence so a copied-from-source ``frequency``
    # can never disagree with the dates we actually wrote.
    if frequency:
        patched["frequency"] = frequency

    request = dict(patched.get("data_request") or {})
    if area is not None:
        request["area"] = [float(x) for x in area]
    patched["data_request"] = request

    recipe = dict(patched.get("recipe") or {})
    recipe["regional_materialization"] = {
        "source": str(source),
        "area_north_west_south_east": [float(x) for x in area] if area else None,
        "bbox_latmin_latmax_lonmin_lonmax": [float(x) for x in bbox] if bbox else None,
        "statistics_policy": statistics_policy,
        "reindex_dates_frequency": reindex_frequency,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    patched["recipe"] = recipe

    timestamp = datetime.now(timezone.utc).isoformat()
    history = patched.get("history") or []
    event = {
        "action": "regional_materialization",
        "timestamp": timestamp,
        "source": str(source),
        "statistics_policy": statistics_policy,
    }
    if isinstance(history, list):
        patched["history"] = [*history, event]
    else:
        patched["history"] = [
            {"action": "source_history_text", "timestamp": timestamp, "value": str(history)},
            event,
        ]
    return _jsonable(patched)


def _stat_array(
    dataset: Any,
    name: str,
    variables: list[str],
    count_per_var: int,
    vars_with_nans: set[str],
) -> np.ndarray | None:
    nvars = len(variables)
    stats = getattr(dataset, "statistics", {}) or {}
    if name in stats:
        arr = np.asarray(stats[name])
        if arr.shape == (nvars,):
            return arr
    if name == "count":
        # Per-variable valid-value count for the materialized subset
        # (rows x ensemble x grid); only used when source lacks ``count``.
        return np.full((nvars,), int(count_per_var), dtype=np.int64)
    if name == "has_nans":
        return np.array([v in vars_with_nans for v in variables], dtype=bool)
    return None


def materialize(args: argparse.Namespace) -> None:
    source = Path(args.source).expanduser().resolve()
    target = Path(args.target).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source zarr not found: {source}")
    if target.exists():
        if not args.overwrite:
            raise FileExistsError(f"Target exists (use --overwrite): {target}")
        if not args.dry_run:
            shutil.rmtree(target)

    if sum(value is not None for value in (args.box, args.bbox, args.area)) != 1:
        raise ValueError("Use exactly one spatial subset option: --box, --bbox, or --area.")
    bbox = list(BOX_PRESETS[args.box]) if args.box else [float(x) for x in args.bbox] if args.bbox else None
    area = [float(x) for x in args.area] if args.area else None
    if bbox is not None:
        lat_min, lat_max, lon_min, lon_max = bbox
        area = [lat_max, lon_min, lat_min, lon_max]
    if area is None:
        raise ValueError("A spatial subset is required: pass --bbox or --area.")

    open_kwargs: dict[str, Any] = {"area": area}
    variables = _split_csv(args.select)
    if variables:
        open_kwargs["select"] = variables
    steps = _split_int_csv(args.fake_hindcasts_steps)
    if args.fake_hindcasts_start or args.fake_hindcasts_end or steps:
        open_kwargs["fake_hindcasts"] = {
            "start": args.fake_hindcasts_start,
            "end": args.fake_hindcasts_end,
            "steps": steps,
        }
    if args.start:
        open_kwargs["start"] = args.start
    if args.end:
        open_kwargs["end"] = args.end

    dataset = open_dataset(str(source), **open_kwargs)
    row_indices = _load_indices_file(args.indices_file)
    if row_indices is not None:
        max_index = int(dataset.shape[0]) - 1
        bad = [idx for idx in row_indices if idx < 0 or idx > max_index]
        if bad:
            raise IndexError(f"Row indices out of range for {dataset.shape[0]} rows: {bad[:10]}")
    elif args.max_rows is not None and int(dataset.shape[0]) > args.max_rows:
        row_indices = list(range(int(args.max_rows)))
    else:
        row_indices = list(range(int(dataset.shape[0])))

    if args.max_rows is not None and len(row_indices) > args.max_rows:
        row_indices = row_indices[: int(args.max_rows)]

    full_dates = np.asarray(dataset.dates, dtype="datetime64[s]")
    source_dates = full_dates[row_indices]

    # The source axis is regular; capture its native cadence as the default
    # fake-axis step for row subsets that would otherwise be irregular.
    native_step = _infer_uniform_step(full_dates)
    native_freq = _timedelta_to_freq(native_step) if native_step is not None else None

    # Resolve an explicit --reindex-dates-frequency. ``auto`` adopts the
    # source-native cadence; otherwise the literal value is used verbatim.
    requested = args.reindex_dates_frequency
    reindex_frequency: str | None = None
    if requested:
        if requested.strip().lower() == "auto":
            if native_freq is None:
                raise SystemExit(
                    "--reindex-dates-frequency auto could not infer a native step: "
                    "the source date axis is not uniformly spaced. Pass an explicit "
                    "frequency such as 6h."
                )
            reindex_frequency = native_freq
        else:
            reindex_frequency = requested

    dates = _reindex_dates_like(source_dates, reindex_frequency)

    # Correct-default safety net (warn, never gate): a reopenable anemoi store
    # needs a constant frequency, but row subsetting (e.g. discrete TC event
    # windows) easily leaves irregular gaps. Rather than refuse, auto-rewrite the
    # output onto the source-native cadence and warn; source_dates provenance is
    # preserved regardless, and an explicit --reindex-dates-frequency still wins.
    effective_step = _infer_uniform_step(dates)
    if effective_step is None and len(dates) > 1 and reindex_frequency is None:
        if native_freq is not None:
            reindex_frequency = native_freq
            dates = _reindex_dates_like(source_dates, reindex_frequency)
            effective_step = _infer_uniform_step(dates)
            print(
                "WARNING: row subset produced a non-uniform date axis; "
                f"auto-reindexed to source-native frequency {native_freq} so the "
                "store stays reopenable (source_dates preserved). Pass "
                "--reindex-dates-frequency to choose a different axis.",
                file=sys.stderr,
            )
        else:
            gaps = np.unique(np.diff(dates))
            print(
                "WARNING: output date axis is non-uniform "
                f"({len(gaps)} distinct gaps {[str(g) for g in gaps[:5]]}) and the "
                "source cadence could not be inferred, so anemoi.open_dataset may "
                "reject this store. Pass --reindex-dates-frequency <freq> to force "
                "a regular axis.",
                file=sys.stderr,
            )

    resolved_frequency = reindex_frequency or (
        _timedelta_to_freq(effective_step) if effective_step is not None else None
    )

    shape = (len(row_indices), *tuple(int(x) for x in dataset.shape[1:]))
    variables = list(dataset.variables)
    logical_gib = float(np.prod(shape) * np.dtype("float32").itemsize / (1024**3))

    summary = {
        "source": str(source),
        "target": str(target),
        "shape": shape,
        "variables": variables,
        "area": area,
        "bbox": bbox,
        "logical_gib": logical_gib,
        "date_start": _date_key(dates[0]) if len(dates) else None,
        "date_end": _date_key(dates[-1]) if len(dates) else None,
        "frequency": resolved_frequency,
        "reindexed": bool(reindex_frequency),
        "rows": int(shape[0]),
        "grid_points": int(shape[-1]),
    }
    print(json.dumps(_jsonable(summary), indent=2))
    if args.dry_run:
        return
    if logical_gib > args.max_gib and not args.confirm_large:
        raise SystemExit(
            f"Refusing to write {logical_gib:.1f} GiB logical payload. "
            f"Increase --max-gib or pass --confirm-large if intentional."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    source_zarr = zarr.open(str(source), mode="r")
    source_attrs = dict(source_zarr.attrs)
    source_fake = source_attrs.get("fake_hindcasts") if isinstance(source_attrs.get("fake_hindcasts"), dict) else None

    compressor = Blosc(cname=args.compressor, clevel=args.clevel, shuffle=Blosc.BITSHUFFLE)
    grid_chunk = int(args.grid_chunk or shape[-1])
    grid_chunk = max(1, min(grid_chunk, shape[-1]))
    group = zarr.open(str(target), mode="w")
    attrs = _patch_metadata(
        source_attrs,
        source=source,
        shape=shape,
        grid_chunk=grid_chunk,
        variables=variables,
        dates=dates,
        source_dates=source_dates,
        area=area,
        bbox=bbox,
        source_fake_hindcasts=source_fake,
        statistics_policy=args.statistics_policy,
        description=args.description,
        frequency=resolved_frequency,
        reindex_frequency=reindex_frequency,
    )
    for key, value in attrs.items():
        group.attrs[key] = value

    group.create_dataset(
        "data",
        shape=shape,
        chunks=(1, shape[1], shape[2], grid_chunk),
        dtype="float32",
        compressor=compressor,
    )
    for row, source_row in enumerate(row_indices):
        group["data"][row] = np.asarray(dataset[source_row], dtype=np.float32)
        if args.progress_every and (row + 1) % args.progress_every == 0:
            print(f"wrote {row + 1}/{shape[0]} rows", file=sys.stderr)

    lat = np.asarray(dataset.latitudes)
    lon = np.asarray(dataset.longitudes)
    group.create_dataset("latitudes", data=lat, chunks=(min(len(lat), 103120),), dtype=lat.dtype)
    group.create_dataset("longitudes", data=lon, chunks=(min(len(lon), 103120),), dtype=lon.dtype)
    group.create_dataset("dates", data=dates, chunks=(max(1, min(len(dates), 22320)),), dtype=dates.dtype)

    vars_with_nans = set(source_attrs.get("variables_with_nans") or [])
    count_per_var = int(shape[0]) * int(shape[2]) * int(shape[3])
    for name in ("mean", "stdev", "minimum", "maximum", "sums", "squares", "count", "has_nans"):
        arr = _stat_array(dataset, name, variables, count_per_var, vars_with_nans)
        if arr is not None:
            group.create_dataset(name, data=arr, chunks=arr.shape, dtype=arr.dtype)

    marker = group.require_group("_regional_materialization")
    marker.attrs["summary"] = _jsonable(summary)
    marker.create_dataset("source_dates", data=source_dates, chunks=(max(1, min(len(source_dates), 22320)),), dtype=source_dates.dtype)
    if row_indices:
        marker.create_dataset("source_row_indices", data=np.asarray(row_indices, dtype=np.int64), chunks=(max(1, min(len(row_indices), 22320)),), dtype="int64")
    print(f"Wrote {target}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Source Anemoi zarr dataset")
    parser.add_argument("target", help="Target regional zarr dataset")
    parser.add_argument("--box", choices=sorted(BOX_PRESETS), help="Named regional box preset")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"))
    parser.add_argument("--area", nargs=4, type=float, metavar=("NORTH", "WEST", "SOUTH", "EAST"))
    parser.add_argument("--select", nargs="*", help="Variables to keep; supports comma-separated values")
    parser.add_argument("--fake-hindcasts-start", help="Real hindcast date lower bound, e.g. 2023-08-26")
    parser.add_argument("--fake-hindcasts-end", help="Real hindcast date upper bound, e.g. 2023-08-30")
    parser.add_argument("--fake-hindcasts-steps", help="Comma-separated steps, e.g. 0,24,48,72,96,120")
    parser.add_argument("--start", help="Fake date lower bound for simple date slicing")
    parser.add_argument("--end", help="Fake date upper bound for simple date slicing")
    parser.add_argument("--indices-file", help="Optional row indices to materialize after spatial/variable/fake-hindcast filtering; accepts JSON list or whitespace/comma-separated integers")
    parser.add_argument("--max-rows", type=int, help="Prototype limiter after subsetting")
    parser.add_argument("--reindex-dates-frequency", help="Rewrite output dates to a regular fake axis, e.g. 1h or 12h, while preserving source_dates provenance. Pass 'auto' to adopt the source axis' native step. If omitted and a row subset leaves irregular gaps, the output is auto-reindexed to the source-native cadence with a warning (source_dates provenance is preserved)")
    parser.add_argument("--statistics-policy", choices=["source"], default="source")
    parser.add_argument("--description")
    parser.add_argument("--compressor", default="zstd")
    parser.add_argument("--clevel", type=int, default=3)
    parser.add_argument("--grid-chunk", type=int, help="Grid chunk size; default is whole cropped grid")
    parser.add_argument("--max-gib", type=float, default=10.0, help="Safety limit unless --confirm-large is passed")
    parser.add_argument("--confirm-large", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    materialize(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
