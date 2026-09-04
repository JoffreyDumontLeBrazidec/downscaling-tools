#!/usr/bin/env python3
"""Fair CRPS of an FDB ensemble against surface station observations.

This reproduces, without quaver, the surface CRPS that quaver stores for an
experiment.  It is meant to be run under the ECMWF module environment
(``module load python3 vtb ecmwf-toolbox``), NOT inside the eval venv, because
it needs ``vtb`` for the observation database and ``mars`` for the fields.

Calibration (2026-09-04, ja6y, 2t, n.hem, 3 dates x 3 lead times): the station
count matched quaver exactly at every point and the fair CRPS agreed to within
0.17 per cent, mean 0.09 per cent.

The recipe is quaver's, read out of quaver 3.7.7 rather than guessed:

* the reference is the SYNOP station observation from STVL, the same database
  quaver reads through ``surfaceobservations()``;
* the model value at a station is the NEAREST grid point, because quaver's
  ``interpolation_method`` defaults to ``nearestpoint``;
* 2 m temperature is corrected for the height difference between the model
  orography and the station with a 0.0065 K/m lapse rate, and nothing else is;
* observations outside quaver's hard limits are dropped, as are those departing
  from the operational analysis by more than the parameter's gross-error limit;
* the spatial mean is weighted by station density, with the same Gaussian
  kernel quaver uses.

Output is one row per (parameter, domain, date, lead time), plus a lead-time
summary averaged over dates, which is the shape of a quaver scorecard.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

LOG = logging.getLogger("obs_crps")

G_CONST = 9.80665
LAPSE_RATE = 0.0065

# quaver's toss.yaml: hard limits, and the maximum departure from the analysis.
HARD_LIMITS = {
    "2t": (173.0, 333.0),
    "2d": (150.0, 333.0),
    "10ff": (0.0, 100.0),
    "msl": (80000.0, 110000.0),
}
ANALYSIS_DEPARTURE_MAX = {"2t": 30.0, "10ff": 40.0, "msl": 1000.0}

# Only 2 m temperature carries an orography correction in quaver
# (Parameters.TwoMetreTemperature.has_orography_correction).
OROGRAPHY_CORRECTED = {"2t"}

# vtb/etc/domains.yaml, as (south, north); all longitudes except europe.
DOMAINS = {
    "n.hem": (20.0, 90.0, -180.0, 180.0),
    "tropics": (-20.0, 20.0, -180.0, 180.0),
    "s.hem": (-90.0, -20.0, -180.0, 180.0),
    "europe": (35.0, 75.0, -12.5, 42.5),
}

# A verified parameter may need more than one MARS parameter: wind speed is
# stored as its components and quaver derives the speed the same way.
MARS_PARAMETERS = {"10ff": ("10u", "10v")}


def mars_parameters(parameter: str) -> tuple[str, ...]:
    return MARS_PARAMETERS.get(parameter, (parameter,))


# --------------------------------------------------------------------------
# geometry and scoring
# --------------------------------------------------------------------------


def unit_xyz(lat, lon) -> np.ndarray:
    """Cartesian coordinates on the unit sphere, so longitude wrapping is free."""
    la = np.deg2rad(np.asarray(lat, dtype=float))
    lo = np.deg2rad(np.asarray(lon, dtype=float))
    return np.stack(
        [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1
    )


def station_density_weights(lat, lon, scangle: float = 0.75) -> np.ndarray:
    """Station-density weights, equivalent to ``vtb.geo.station_density_weights``.

    Every station gets a weight of one over the sum of a Gaussian kernel
    exp(-(d/scangle)^2) over its neighbours, where d is angular distance and the
    kernel is truncated beyond four times ``scangle``.  Stations packed together
    therefore count for less, which stops the dense European and North American
    networks from dominating a hemispheric mean.

    This is a vectorised rewrite of the vtb function, which loops in Python and
    is the slowest step of the whole calculation.  It was checked against vtb on
    12,765 stations: maximum relative difference 4e-9, about seven times faster.
    """
    sc = float(scangle) * np.pi / 180.0
    xyz = unit_xyz(lat, lon)
    tree = cKDTree(xyz)
    chord = 2.0 * np.sin(min(4.0 * sc, np.pi) / 2.0)
    neighbours = tree.query_ball_tree(tree, r=chord)

    kernel_sum = np.empty(len(xyz), dtype=float)
    for i, nb in enumerate(neighbours):
        d = np.linalg.norm(xyz[np.asarray(nb)] - xyz[i], axis=1)
        ang = 2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))
        k = np.exp(-((ang / sc) ** 2))
        k[ang > 4.0 * sc] = 0.0
        kernel_sum[i] = k.sum()

    w = 1.0 / kernel_sum
    return w / w.sum()


def fair_crps(members: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Fair (ensemble-size debiased) CRPS, one value per station.

    ``members`` is (M, N), ``truth`` is (N,).  The fair form divides the
    ensemble-spread term by M(M-1) rather than M^2, which removes the bias that
    makes a small ensemble look artificially sharp.
    """
    m = members.shape[0]
    if m < 2:
        raise ValueError("fair CRPS needs at least two members")
    skill = np.abs(members - truth[None, :]).mean(axis=0)
    spread = np.abs(members[:, None, :] - members[None, :, :]).sum(axis=(0, 1))
    return skill - spread / (2.0 * m * (m - 1))


# --------------------------------------------------------------------------
# data access
# --------------------------------------------------------------------------


def run_mars(request: str, cache_dir: Path, tag: str) -> None:
    req_path = cache_dir / f"_request_{tag}.mars"
    req_path.write_text(request)
    LOG.info("mars retrieve: %s", tag)
    proc = subprocess.run(
        ["mars", str(req_path)], cwd=cache_dir, capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"mars failed for {tag} (rc={proc.returncode}):\n{proc.stdout[-4000:]}"
        )


def ensure_forecast_fields(
    cache_dir: Path,
    parameter: str,
    dates: list[str],
    steps: list[int],
    nmem: int,
    expver: str,
    class_: str,
    stream: str,
    type_: str,
    database: str,
) -> None:
    """Retrieve the whole window for one parameter in a single MARS call.

    One bulk request is far cheaper than one request per date and lead time, and
    files already present are left alone so a rerun is close to free.
    """
    wanted = [
        cache_dir / f"{expver}_{p}_{d}_{s}.grib"
        for p in mars_parameters(parameter)
        for d in dates
        for s in steps
    ]
    if all(f.exists() for f in wanted):
        LOG.info("forecast fields for %s already cached", parameter)
        return

    numbers = "/".join(str(i) for i in range(1, nmem + 1))
    blocks = []
    for i, p in enumerate(mars_parameters(parameter)):
        target = f'"{expver}_{p}_[date]_[step].grib"'
        if i == 0:
            blocks.append(
                "retrieve,\n"
                f"  class={class_}, expver={expver}, stream={stream}, type={type_},\n"
                f"  number={numbers},\n"
                f"  date={'/'.join(dates)}, time=0000,\n"
                f"  step={'/'.join(str(s) for s in steps)},\n"
                f"  levtype=sfc, param={p}, database={database},\n"
                f"  target={target}"
            )
        else:
            blocks.append(f"retrieve, param={p}, target={target}")
    run_mars("\n".join(blocks) + "\n", cache_dir, f"fc_{parameter}")


def ensure_analysis_fields(
    cache_dir: Path, parameter: str, valid_dates: list[str], grid: str
) -> None:
    """Retrieve the operational analysis used for the gross-error check."""
    wanted = [
        cache_dir / f"an_{p}_{d}.grib"
        for p in mars_parameters(parameter)
        for d in valid_dates
    ]
    if all(f.exists() for f in wanted):
        return
    blocks = []
    for i, p in enumerate(mars_parameters(parameter)):
        target = f'"an_{p}_[date].grib"'
        if i == 0:
            blocks.append(
                "retrieve,\n"
                "  class=od, expver=0001, stream=oper, type=an,\n"
                f"  date={'/'.join(valid_dates)}, time=0000,\n"
                f"  levtype=sfc, param={p}, grid={grid}, database=off,\n"
                f"  target={target}"
            )
        else:
            blocks.append(f"retrieve, param={p}, target={target}")
    run_mars("\n".join(blocks) + "\n", cache_dir, f"an_{parameter}")


def ensure_orography(cache_dir: Path, grid: str) -> Path:
    """Model orography on the output grid, as quaver's correction retrieves it."""
    path = cache_dir / f"orography_{grid}.grib"
    if not path.exists():
        run_mars(
            "retrieve,\n"
            "  class=od, expver=0001, stream=enfo, type=pf, number=1,\n"
            "  date=2024-01-01, time=0000, step=0,\n"
            f"  levtype=sfc, param=z, grid={grid}, database=off,\n"
            f'  target="{path.name}"\n',
            cache_dir,
            f"orography_{grid}",
        )
    return path


def read_field_values(path: Path, number: int | None = None) -> np.ndarray:
    import earthkit.data as ekd

    fields = ekd.from_source("file", str(path)).to_fieldlist()
    if number is None:
        return fields[0].to_numpy().ravel()
    for field in fields:
        if int(field.metadata("number")) == number:
            return field.to_numpy().ravel()
    raise KeyError(f"member {number} not found in {path}")


def read_grid(path: Path):
    import earthkit.data as ekd

    geo = ekd.from_source("file", str(path)).to_fieldlist()[0].geography
    return geo.latitudes(), geo.longitudes()


def retrieve_observations(parameter: str, valid: dt.datetime) -> pd.DataFrame:
    """Station observations valid at one time, straight out of STVL.

    The observation table has no lead time: the forecast reference date and the
    lead time have to be folded into a single valid time, with a forecast length
    of zero.  Asking for a non-zero forecast length silently returns nothing.
    """
    import vtb.media as vmedia

    fieldset = vmedia.stvl_retrieve(
        table="observation",
        parameter=parameter,
        reference_datetimes=[valid.strftime("%Y-%m-%dT%H:%M:%S")],
        forecast_lengths=[dt.timedelta(hours=0)],
    )
    if len(fieldset) == 0:
        return pd.DataFrame()
    df = fieldset[0].to_dataframe().rename(columns={"value_0": "obs"})
    return df.dropna(subset=["obs"])


# --------------------------------------------------------------------------
# the calculation
# --------------------------------------------------------------------------


def domain_mask(df: pd.DataFrame, domain: str) -> np.ndarray:
    south, north, west, east = DOMAINS[domain]
    lat = df["latitude"].to_numpy()
    lon = ((df["longitude"].to_numpy() + 180.0) % 360.0) - 180.0
    return (lat >= south) & (lat <= north) & (lon >= west) & (lon <= east)


class WeightCache:
    """Station-density weights depend only on which stations are present."""

    def __init__(self):
        self._cache: dict[str, np.ndarray] = {}

    def get(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        key = hashlib.sha1(
            np.ascontiguousarray(np.stack([lat, lon])).tobytes()
        ).hexdigest()
        if key not in self._cache:
            self._cache[key] = station_density_weights(lat, lon)
        return self._cache[key]


def model_values_at_stations(
    cache_dir: Path, expver: str, parameter: str, date: str, step: int,
    nmem: int, idx: np.ndarray,
) -> np.ndarray:
    """Ensemble values at the station points, one row per member.

    Wind speed is not archived directly, so it is built from its components the
    same way quaver does: the speed of each member, not the speed of the mean.
    """
    parts = []
    for p in mars_parameters(parameter):
        path = cache_dir / f"{expver}_{p}_{date}_{step}.grib"
        parts.append(
            np.stack([read_field_values(path, n)[idx] for n in range(1, nmem + 1)])
        )
    if parameter == "10ff":
        return np.hypot(parts[0], parts[1])
    return parts[0]


def analysis_values_at_stations(
    cache_dir: Path, parameter: str, valid: dt.datetime, idx: np.ndarray
) -> np.ndarray | None:
    parts = []
    for p in mars_parameters(parameter):
        path = cache_dir / f"an_{p}_{valid.strftime('%Y%m%d')}.grib"
        if not path.exists():
            return None
        parts.append(read_field_values(path)[idx])
    if parameter == "10ff":
        return np.hypot(parts[0], parts[1])
    return parts[0]


def score_window(
    cache_dir: Path,
    expver: str,
    parameters: list[str],
    domains: list[str],
    dates: list[str],
    steps: list[int],
    nmem: int,
    grid: str,
    orography_path: Path,
    gross_error_check: bool = True,
) -> pd.DataFrame:
    grid_lat, grid_lon = read_grid(orography_path)
    tree = cKDTree(unit_xyz(grid_lat, grid_lon))
    orography_height = read_field_values(orography_path) / G_CONST
    weights = WeightCache()

    rows: list[dict] = []
    for parameter in parameters:
        observation_cache: dict[dt.datetime, pd.DataFrame] = {}
        for date in dates:
            for step in steps:
                valid = dt.datetime.strptime(date, "%Y%m%d") + dt.timedelta(hours=step)
                if valid not in observation_cache:
                    observation_cache[valid] = retrieve_observations(parameter, valid)
                obs = observation_cache[valid]
                if obs.empty:
                    LOG.warning("no observations for %s at %s", parameter, valid)
                    continue

                lo, hi = HARD_LIMITS.get(parameter, (-np.inf, np.inf))
                obs = obs[(obs["obs"] >= lo) & (obs["obs"] <= hi)].reset_index(drop=True)

                _, idx = tree.query(
                    unit_xyz(obs["latitude"].to_numpy(), obs["longitude"].to_numpy())
                )

                limit = ANALYSIS_DEPARTURE_MAX.get(parameter)
                if gross_error_check and limit is not None:
                    an = analysis_values_at_stations(cache_dir, parameter, valid, idx)
                    if an is None:
                        LOG.warning(
                            "analysis missing for %s at %s, gross-error check skipped",
                            parameter, valid,
                        )
                    else:
                        keep = np.abs(obs["obs"].to_numpy() - an) <= limit
                        obs = obs[keep].reset_index(drop=True)
                        idx = idx[keep]

                members = model_values_at_stations(
                    cache_dir, expver, parameter, date, step, nmem, idx
                )
                if parameter in OROGRAPHY_CORRECTED:
                    delta = orography_height[idx] - obs["elevation"].to_numpy()
                    members = members + delta[None, :] * LAPSE_RATE

                good = np.isfinite(members).all(axis=0) & np.isfinite(obs["obs"].to_numpy())
                obs = obs[good].reset_index(drop=True)
                members = members[:, good]

                crps = fair_crps(members, obs["obs"].to_numpy())
                error = members.mean(axis=0) - obs["obs"].to_numpy()
                spread = members.std(axis=0, ddof=1)

                for domain in domains:
                    mask = domain_mask(obs, domain)
                    if not mask.any():
                        continue
                    lat = obs["latitude"].to_numpy()[mask]
                    lon = obs["longitude"].to_numpy()[mask]
                    w = weights.get(lat, lon)
                    rows.append({
                        "parameter": parameter,
                        "domain": domain,
                        "date": int(date),
                        "step": int(step),
                        "nstations": int(mask.sum()),
                        "fcrps": float(np.sum(w * crps[mask])),
                        "fcrps_unweighted": float(np.mean(crps[mask])),
                        "bias": float(np.sum(w * error[mask])),
                        "spread": float(np.sum(w * spread[mask])),
                    })
                LOG.info("scored %s %s +%sh (%d stations)", parameter, date, step, len(obs))
    return pd.DataFrame(rows)


def summarise_by_lead(rows: pd.DataFrame) -> pd.DataFrame:
    """Average each score over the dates in the window, per lead time.

    This is the curve a quaver scorecard plots: the score as a function of lead
    time, with the forecast start dates averaged out.  ``ndates`` is carried so a
    lead time that is short of dates cannot be mistaken for a complete one.
    """
    grouped = rows.groupby(["parameter", "domain", "step"], as_index=False).agg(
        fcrps=("fcrps", "mean"),
        fcrps_std=("fcrps", "std"),
        bias=("bias", "mean"),
        spread=("spread", "mean"),
        nstations=("nstations", "mean"),
        ndates=("date", "nunique"),
    )
    return grouped.sort_values(["parameter", "domain", "step"]).reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expver", required=True)
    ap.add_argument("--class", dest="class_", default="rd")
    ap.add_argument("--stream", default="enfo")
    ap.add_argument("--type", dest="type_", default="pf")
    ap.add_argument("--database", default="fdb")
    ap.add_argument("--grid", default="O1280", help="output grid, for the orography")
    ap.add_argument("--dates", required=True, help="comma separated YYYYMMDD")
    ap.add_argument("--steps", required=True, help="comma separated lead times in hours")
    ap.add_argument("--nmem", type=int, default=10)
    ap.add_argument("--parameters", default="2t,2d,10ff")
    ap.add_argument("--domains", default="n.hem,tropics,s.hem,europe")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-gross-error-check", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    cache_dir = Path(args.cache_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    steps = [int(s) for s in args.steps.split(",") if s.strip()]
    parameters = [p.strip() for p in args.parameters.split(",") if p.strip()]
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]

    orography_path = ensure_orography(cache_dir, args.grid)

    valid_dates = sorted({
        (dt.datetime.strptime(d, "%Y%m%d") + dt.timedelta(hours=s)).strftime("%Y%m%d")
        for d in dates for s in steps
    })

    for parameter in parameters:
        ensure_forecast_fields(
            cache_dir, parameter, dates, steps, args.nmem, args.expver,
            args.class_, args.stream, args.type_, args.database,
        )
        if not args.no_gross_error_check and parameter in ANALYSIS_DEPARTURE_MAX:
            ensure_analysis_fields(cache_dir, parameter, valid_dates, args.grid)

    rows = score_window(
        cache_dir, args.expver, parameters, domains, dates, steps, args.nmem,
        args.grid, orography_path,
        gross_error_check=not args.no_gross_error_check,
    )
    if rows.empty:
        LOG.error("no scores produced")
        return 1

    by_lead = summarise_by_lead(rows)
    rows.to_csv(out_dir / "scores_by_date_and_lead.csv", index=False)
    by_lead.to_csv(out_dir / "summary_by_lead.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps({
        "expver": args.expver,
        "parameters": parameters,
        "domains": domains,
        "dates": dates,
        "steps": steps,
        "nmem": args.nmem,
        "grid": args.grid,
        "n_rows": int(len(rows)),
        "summary": by_lead.to_dict(orient="records"),
    }, indent=2))
    (out_dir / "compute.done").write_text(dt.datetime.utcnow().isoformat() + "Z\n")
    LOG.info("wrote %s rows to %s", len(rows), out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
