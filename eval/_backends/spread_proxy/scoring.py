"""Local ML-vs-ENFO ensemble-spread comparison for eval.cli prediction NetCDFs.

Motivation (spec: docs epics/training-diagnostics/metric-skill-gap/in-progress/
20260826_spread_proxy_overspread_spec.md): the champion class measured 3-25%
over-dispersed vs the calibrated ENFO target on quaver; this backend answers the
same question locally, without FDB, by comparing the spread of ``y_pred`` (ML
members) against the spread of ``y`` (the ENFO target members carried in the
same file) on identical support.

Three readouts:
  1. per-field area-weighted mean spread vs lead, ML and ENFO (curves);
  2. per-gridpoint spread-ratio maps, binned to a coarse lat-lon grid and
     aggregated over dates+steps (qualitative: local vs global);
  3. band-resolved spread spectra via HEALPix binning + anafast of member
     deviations (is the excess in the fine band).

Fairness: both ensembles come from the same file with the same member count, so
the finite-ensemble bias cancels. When an external ENFO ensemble with more
members is used (Phase 2, months from MARS), pass ``enfo_n_members`` to
subsample and ``enfo_exclude_members`` to drop the verifying member.

Unlike the ``probabilistic`` backend, ``y`` is NOT collapsed to member 0: the
full target ensemble is the reference. No truth field enters any metric here,
so the ENFO-is-the-truth guard does not apply to the ratio itself; exclusions
are still recorded in the summary for auditability.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import xarray as xr

from eval._backends.probabilistic.scoring import (
    _as_list,
    _domain_mask,
    _lat_lon,
    _weighted_mean,
)
from eval._backends.scoreboard._surface_compute import (
    _area_weights,
    _to_member_point_weather,
    _weather_state_index,
)
from eval.discovery.predictions import PREDICTION_RE, PredictionFile, find_predictions

LOG = logging.getLogger(__name__)

DEFAULT_WEATHER_STATES = [
    "2t", "10ff", "10u", "10v", "2d", "msl", "sp", "skt", "tcw", "t_850", "z_500",
]
DEFAULT_DOMAINS = ["global", "n.hem", "tropics", "s.hem", "europe"]
DEFAULT_SPECTRA_FIELDS = ["2t", "10u", "msl", "z_500"]
METRICS = ("spread_ml", "spread_enfo", "spread_ratio", "spread_input", "spread_ratio_input")


def find_predictions_recursive(predictions_dir: Path) -> list[PredictionFile]:
    """find_predictions, falling back to a recursive scan for multi-date roots.

    Manual campaign roots lay files out as arms/<arm>/date_<d>/predictions/, so
    the flat scan of the canonical discovery finds nothing at the root.
    """
    direct = find_predictions(predictions_dir)
    if direct:
        return direct
    results: list[PredictionFile] = []
    for path in sorted(predictions_dir.rglob("predictions_*.nc")):
        match = PREDICTION_RE.match(path.name)
        if not match:
            continue
        results.append(
            PredictionFile(path=path, date=match.group(1), step=int(match.group(2)), member=0)
        )
    results.sort(key=lambda p: (p.date, p.step))
    return results


def _load_member_point_weather(ds: xr.Dataset, var: str) -> np.ndarray:
    """Load one ensemble variable ONCE as float32 (member, point, weather).

    A per-field ``isel(weather_state=...)`` re-reads the whole variable from
    disk for every field (the weather_state stride touches every chunk), which
    multiplies I/O by the field count; loading once avoids that.
    """
    da = _to_member_point_weather(ds[var], ds, label=var)
    da = da.transpose("member", "grid_point_hres", "weather_state")
    return np.asarray(da.values, dtype=np.float32)


def _field_members(arr: np.ndarray, ws_index: dict[str, int], field: str) -> np.ndarray | None:
    """Return (member, point) float32 for a field, deriving 10ff from 10u/10v."""
    if field in ws_index:
        return arr[:, :, ws_index[field]]
    if field == "10ff":
        if "10u" not in ws_index or "10v" not in ws_index:
            return None
        return np.hypot(arr[:, :, ws_index["10u"]], arr[:, :, ws_index["10v"]])
    return None


def _pointwise_spread(members: np.ndarray, ddof: int) -> np.ndarray:
    """Pointwise ensemble standard deviation, float64, NaN where any member is."""
    n = members.shape[0]
    ddof = min(max(int(ddof), 0), max(n - 1, 0))
    out = np.std(members, axis=0, ddof=ddof, dtype=np.float64)
    bad = ~np.all(np.isfinite(members), axis=0)
    if np.any(bad):
        out[bad] = np.nan
    return out


def _select_enfo_members(
    n_available: int,
    *,
    exclude: list[int],
    n_members: int | None,
    seed: int,
) -> np.ndarray:
    keep = np.array([i for i in range(n_available) if i not in set(exclude)], dtype=int)
    if n_members is not None and n_members < keep.size:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(keep, size=int(n_members), replace=False))
    return keep


class _MapAccumulator:
    """Aggregate pointwise ensemble variance onto a coarse lat-lon grid.

    Accumulates the SUM of pointwise variance per bin over every (date, step)
    file, separately for ML and ENFO; the plotted ratio is
    sqrt(sum_var_ml / sum_var_enfo), i.e. the ratio of RMS spreads.
    """

    def __init__(self, bin_deg: float):
        self.bin_deg = float(bin_deg)
        self.n_lat = int(round(180.0 / self.bin_deg))
        self.n_lon = int(round(360.0 / self.bin_deg))
        self.ml: dict[str, np.ndarray] = {}
        self.enfo: dict[str, np.ndarray] = {}
        self.count: dict[str, np.ndarray] = {}
        self._bin_idx: np.ndarray | None = None
        self._sig: tuple[int, float, float] | None = None

    def _bins(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        sig = (lat.size, float(lat[0]), float(lon[0]))
        if self._bin_idx is None or self._sig != sig:
            ilat = np.clip(((lat + 90.0) / self.bin_deg).astype(np.int64), 0, self.n_lat - 1)
            ilon = np.clip(((lon + 180.0) / self.bin_deg).astype(np.int64), 0, self.n_lon - 1)
            self._bin_idx = ilat * self.n_lon + ilon
            self._sig = sig
        return self._bin_idx

    def add(self, field: str, lat: np.ndarray, lon: np.ndarray,
            var_ml: np.ndarray, var_enfo: np.ndarray) -> None:
        nbins = self.n_lat * self.n_lon
        idx = self._bins(lat, lon)
        finite = np.isfinite(var_ml) & np.isfinite(var_enfo)
        if field not in self.ml:
            self.ml[field] = np.zeros(nbins)
            self.enfo[field] = np.zeros(nbins)
            self.count[field] = np.zeros(nbins)
        self.ml[field] += np.bincount(idx[finite], weights=var_ml[finite], minlength=nbins)
        self.enfo[field] += np.bincount(idx[finite], weights=var_enfo[finite], minlength=nbins)
        self.count[field] += np.bincount(idx[finite], minlength=nbins)

    def save(self, path: Path) -> None:
        lat_centers = -90.0 + self.bin_deg * (np.arange(self.n_lat) + 0.5)
        lon_centers = -180.0 + self.bin_deg * (np.arange(self.n_lon) + 0.5)
        payload: dict[str, np.ndarray] = {
            "lat_centers": lat_centers,
            "lon_centers": lon_centers,
        }
        for field in self.ml:
            shape = (self.n_lat, self.n_lon)
            payload[f"{field}__ml_var"] = self.ml[field].reshape(shape)
            payload[f"{field}__enfo_var"] = self.enfo[field].reshape(shape)
            payload[f"{field}__count"] = self.count[field].reshape(shape)
        np.savez_compressed(path, **payload)


class _SpectraAccumulator:
    """Mean angular power spectrum of member deviations, per (field, step).

    Reuses the fast-spectra-proxy route: bin the unstructured grid onto a
    HEALPix map, then hp.anafast on each member's deviation from the ensemble
    mean. The spread spectrum is the member-mean C_ell of those deviations,
    computed identically for ML and ENFO so the comparison is fair even though
    the HEALPix binning itself smooths both.
    """

    def __init__(self, nside: int):
        import healpy as hp  # lazy: optional dependency of this readout

        self._hp = hp
        self.nside = int(nside)
        self.lmax = 2 * self.nside
        self.sum_ml: dict[tuple[str, int], np.ndarray] = {}
        self.sum_enfo: dict[tuple[str, int], np.ndarray] = {}
        self.sum_input: dict[tuple[str, int], np.ndarray] = {}
        self.n_dates: dict[tuple[str, int], int] = {}
        self.skipped_low_coverage = False
        self._pix: np.ndarray | None = None
        self._sig: tuple[int, float, float] | None = None

    def _pixels(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        sig = (lat.size, float(lat[0]), float(lon[0]))
        if self._pix is None or self._sig != sig:
            theta = np.deg2rad(90.0 - lat)
            phi = np.deg2rad(np.mod(lon, 360.0))
            self._pix = self._hp.ang2pix(self.nside, theta, phi, nest=False)
            self._sig = sig
        return self._pix

    def _mean_deviation_cl(self, members: np.ndarray, pix: np.ndarray) -> np.ndarray:
        hp = self._hp
        npix = hp.nside2npix(self.nside)
        counts = np.bincount(pix, minlength=npix).astype(np.float64)
        occupied = counts > 0
        deviations = members - members.mean(axis=0, keepdims=True)
        cls = []
        for m in range(members.shape[0]):
            sums = np.bincount(pix, weights=deviations[m].astype(np.float64), minlength=npix)
            hmap = np.zeros(npix)
            hmap[occupied] = sums[occupied] / counts[occupied]
            cls.append(hp.anafast(hmap, lmax=self.lmax))
        return np.mean(np.asarray(cls), axis=0)

    def add(self, field: str, step: int, lat: np.ndarray, lon: np.ndarray,
            ml_members: np.ndarray, enfo_members: np.ndarray,
            input_members: np.ndarray | None = None) -> None:
        finite = (np.all(np.isfinite(ml_members), axis=0)
                  & np.all(np.isfinite(enfo_members), axis=0))
        if input_members is not None:
            finite &= np.all(np.isfinite(input_members), axis=0)
        pix = self._pixels(lat, lon)[finite]
        # anafast needs near-global coverage; a regional box fills a few % of
        # the sphere and the zero-filled remainder turns C_ell into a window
        # artifact. Warn once and skip rather than emit a biased spectrum.
        occupancy = np.unique(pix).size / self._hp.nside2npix(self.nside)
        if occupancy < 0.5:
            if not self.skipped_low_coverage:
                LOG.warning(
                    "spread_proxy spectra: grid covers %.1f%% of the sphere "
                    "(< 50%%) — skipping the spectral readout for this run.",
                    100.0 * occupancy,
                )
            self.skipped_low_coverage = True
            return
        key = (field, int(step))
        cl_ml = self._mean_deviation_cl(ml_members[:, finite], pix)
        cl_enfo = self._mean_deviation_cl(enfo_members[:, finite], pix)
        if key not in self.sum_ml:
            self.sum_ml[key] = np.zeros_like(cl_ml)
            self.sum_enfo[key] = np.zeros_like(cl_enfo)
            self.n_dates[key] = 0
        self.sum_ml[key] += cl_ml
        self.sum_enfo[key] += cl_enfo
        if input_members is not None:
            if key not in self.sum_input:
                self.sum_input[key] = np.zeros_like(cl_ml)
            self.sum_input[key] += self._mean_deviation_cl(input_members[:, finite], pix)
        self.n_dates[key] += 1

    def save(self, path: Path) -> None:
        payload: dict[str, Any] = {"ell": np.arange(self.lmax + 1), "nside": self.nside}
        for (field, step), total in self.sum_ml.items():
            n = self.n_dates[(field, step)]
            payload[f"{field}__step{step:03d}__ml"] = total / n
            payload[f"{field}__step{step:03d}__enfo"] = self.sum_enfo[(field, step)] / n
            if (field, step) in self.sum_input:
                payload[f"{field}__step{step:03d}__input"] = self.sum_input[(field, step)] / n
            payload[f"{field}__step{step:03d}__n_dates"] = np.array(n)
        np.savez_compressed(path, **payload)


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        key = (row["step"], row["weather_state"], row["domain"], row["metric"])
        value = float(row["value"])
        if math.isfinite(value):
            grouped[key].append(value)
    summaries: list[dict[str, Any]] = []
    for (step, weather_state, domain, metric), values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=np.float64)
        std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        summaries.append({
            "step": int(step),
            "weather_state": weather_state,
            "domain": domain,
            "metric": metric,
            "mean": float(np.mean(arr)),
            "std": std,
            "stderr": float(std / math.sqrt(arr.size)) if arr.size > 1 else 0.0,
            "n_dates": int(arr.size),
        })
    return summaries


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _headline_metrics(summary_rows: list[dict[str, Any]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in summary_rows:
        if row["metric"] != "spread_ratio":
            continue
        grouped[(str(row["weather_state"]), str(row["domain"]))].append(float(row["mean"]))
    for (weather_state, domain), values in sorted(grouped.items()):
        metrics[f"spread_proxy_{weather_state}_{domain}_ratio_mean"] = float(np.mean(values))
    return metrics


def compute_spread_proxy(
    predictions_dir: str | Path,
    output_dir: str | Path,
    *,
    weather_states: Iterable[str] | str | None = None,
    domains: Iterable[str] | str | None = None,
    steps: Iterable[int] | str | None = None,
    dates: Iterable[str] | str | None = None,
    spread_ddof: int = 1,
    map_bin_deg: float = 0.5,
    spectra: bool = True,
    spectra_fields: Iterable[str] | str | None = None,
    spectra_steps: Iterable[int] | str | None = None,
    spectra_nside: int = 256,
    enfo_exclude_members: Iterable[int] | str | None = None,
    enfo_n_members: int | None = None,
    enfo_subsample_seed: int = 0,
    include_input: bool = True,
) -> dict[str, Any]:
    """Compare ML (y_pred) vs ENFO (y) ensemble spread; write CSV/JSON/NPZ."""
    predictions_dir = Path(predictions_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fields = _as_list(weather_states, cast=str) or list(DEFAULT_WEATHER_STATES)
    domain_names = _as_list(domains, cast=str) or list(DEFAULT_DOMAINS)
    step_filter = set(_as_list(steps, cast=int)) if steps else None
    date_filter = set(_as_list(dates, cast=str)) if dates else None
    spec_fields = _as_list(spectra_fields, cast=str) or list(DEFAULT_SPECTRA_FIELDS)
    spec_steps = set(_as_list(spectra_steps, cast=int)) if spectra_steps else None
    exclude = _as_list(enfo_exclude_members, cast=int)

    pred_files = find_predictions_recursive(predictions_dir)
    if step_filter is not None:
        pred_files = [p for p in pred_files if int(p.step) in step_filter]
    if date_filter is not None:
        pred_files = [p for p in pred_files if str(p.date) in date_filter]
    if not pred_files:
        raise ValueError(f"No prediction files matched filters in {predictions_dir}")

    maps = _MapAccumulator(map_bin_deg)
    spectra_acc: _SpectraAccumulator | None = None
    spectra_error: str | None = None
    if spectra:
        try:
            spectra_acc = _SpectraAccumulator(spectra_nside)
        except ImportError as exc:
            spectra_error = f"healpy unavailable, spectra readout skipped: {exc}"
            LOG.warning("%s", spectra_error)

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    enfo_members_used: list[int] | None = None
    file_attrs: dict[str, Any] = {}
    grid_info: dict[str, Any] = {}

    for pred in pred_files:
        LOG.info("spread_proxy: %s", pred.path)
        with xr.open_dataset(pred.path, cache=False, decode_timedelta=False) as ds:
            for required in ("y_pred", "y", "weather_state"):
                if required not in ds:
                    raise ValueError(f"{pred.path}: missing required variable {required!r}")
            if not file_attrs:
                file_attrs = {k: str(v) for k, v in ds.attrs.items()
                              if k in ("checkpoint_id", "member_ids", "sampling_config_json")}
            ws_index = _weather_state_index(ds)
            ml = _load_member_point_weather(ds, "y_pred")
            enfo = _load_member_point_weather(ds, "y")
            inp: np.ndarray | None = None
            if include_input and "x_interp" in ds:
                inp = _load_member_point_weather(ds, "x_interp")
                if inp.shape[0] < 2:
                    inp = None
            n_points = ml.shape[1]
            weights = _area_weights(ds, n_points)
            lat, lon = _lat_lon(ds, n_points)
            if not grid_info:
                grid_info = {
                    "n_points": int(n_points),
                    "lat_min": float(lat.min()), "lat_max": float(lat.max()),
                    "lon_min": float(lon.min()), "lon_max": float(lon.max()),
                }
                if n_points < 3_000_000:
                    LOG.warning(
                        "spread_proxy: grid has %d points — this looks REGIONAL, "
                        "not global (lat %.1f..%.1f, lon %.1f..%.1f). The 'global' "
                        "domain then means the whole box.",
                        n_points, lat.min(), lat.max(), lon.min(), lon.max(),
                    )

        if enfo.shape[0] < 2 or ml.shape[0] < 2:
            # Single-member files (e.g. leftover smoke-test runs inside a
            # campaign tree) carry no spread; skip them rather than fail.
            LOG.warning(
                "spread_proxy: skipping %s — need >=2 members in both ensembles "
                "(ml=%d, enfo=%d)", pred.path, ml.shape[0], enfo.shape[0],
            )
            skipped.append({
                "path": str(pred.path), "date": pred.date, "step": pred.step,
                "reason": f"<2 members (ml={ml.shape[0]}, enfo={enfo.shape[0]})",
            })
            del ml, enfo, inp
            continue
        keep = _select_enfo_members(
            enfo.shape[0], exclude=exclude, n_members=enfo_n_members,
            seed=enfo_subsample_seed,
        )
        enfo = enfo[keep]
        enfo_members_used = [int(i) for i in keep]

        domain_masks = {name: _domain_mask(name, lat, lon) for name in domain_names}

        for field in fields:
            ml_f = _field_members(ml, ws_index, field)
            enfo_f = _field_members(enfo, ws_index, field)
            if ml_f is None or enfo_f is None:
                skipped.append({
                    "path": str(pred.path), "date": pred.date, "step": pred.step,
                    "weather_state": field, "reason": "missing field",
                })
                continue
            inp_f = _field_members(inp, ws_index, field) if inp is not None else None
            spread_ml = _pointwise_spread(ml_f, spread_ddof)
            spread_enfo = _pointwise_spread(enfo_f, spread_ddof)
            spread_inp = (_pointwise_spread(inp_f, spread_ddof)
                          if inp_f is not None else None)
            valid = np.isfinite(spread_ml) & np.isfinite(spread_enfo)

            maps.add(field, lat, lon, np.square(spread_ml), np.square(spread_enfo))
            if (spectra_acc is not None and field in spec_fields
                    and (spec_steps is None or int(pred.step) in spec_steps)):
                spectra_acc.add(field, pred.step, lat, lon, ml_f, enfo_f,
                                input_members=inp_f)

            for domain_name, domain_mask in domain_masks.items():
                mask = valid & domain_mask
                if not np.any(mask):
                    skipped.append({
                        "path": str(pred.path), "date": pred.date, "step": pred.step,
                        "weather_state": field, "domain": domain_name,
                        "reason": "no valid points",
                    })
                    continue
                domain_weights = np.where(mask, weights, 0.0)
                value_ml = _weighted_mean(spread_ml, domain_weights)
                value_enfo = _weighted_mean(spread_enfo, domain_weights)
                ratio = value_ml / value_enfo if value_enfo else math.nan
                metric_values = [("spread_ml", value_ml),
                                 ("spread_enfo", value_enfo),
                                 ("spread_ratio", ratio)]
                if spread_inp is not None:
                    value_inp = _weighted_mean(spread_inp, domain_weights)
                    metric_values.append(("spread_input", value_inp))
                    metric_values.append((
                        "spread_ratio_input",
                        value_inp / value_enfo if value_enfo else math.nan,
                    ))
                for metric, value in metric_values:
                    rows.append({
                        "date": pred.date,
                        "step": int(pred.step),
                        "weather_state": field,
                        "domain": domain_name,
                        "metric": metric,
                        "value": value,
                        "n_points": int(mask.sum()),
                        "n_members_ml": int(ml_f.shape[0]),
                        "n_members_enfo": int(enfo_f.shape[0]),
                        "source_path": str(pred.path),
                    })
        del ml, enfo, inp

    if not rows:
        raise ValueError(
            f"spread_proxy produced no rows for {predictions_dir}; skipped={skipped[:5]}"
        )

    summary_rows = _summarize(rows)
    _write_csv(output_dir / "spread_by_lead.csv", rows, [
        "date", "step", "weather_state", "domain", "metric", "value",
        "n_points", "n_members_ml", "n_members_enfo", "source_path",
    ])
    _write_csv(output_dir / "summary_by_lead.csv", summary_rows, [
        "step", "weather_state", "domain", "metric", "mean", "std", "stderr", "n_dates",
    ])
    maps.save(output_dir / "spread_maps.npz")
    if spectra_acc is not None and spectra_acc.sum_ml:
        spectra_acc.save(output_dir / "spread_spectra.npz")

    payload: dict[str, Any] = {
        "predictions_dir": str(predictions_dir),
        "n_files": len(pred_files),
        "n_rows": len(rows),
        "skipped_count": len(skipped),
        "skipped": skipped[:50],
        "weather_states": fields,
        "domains": domain_names,
        "spread_ddof": int(spread_ddof),
        "map_bin_deg": float(map_bin_deg),
        "spectra_nside": int(spectra_nside) if spectra_acc is not None else None,
        "spectra_error": spectra_error,
        "enfo_members_used": enfo_members_used,
        "enfo_exclude_members": exclude,
        "enfo_n_members": enfo_n_members,
        "include_input": bool(include_input),
        "grid": grid_info,
        "source_attrs": file_attrs,
        "headline_metrics": _headline_metrics(summary_rows),
    }
    (output_dir / "spread_proxy_summary.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare ML (y_pred) vs ENFO (y) ensemble spread from prediction NetCDFs."
    )
    parser.add_argument("--predictions-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--weather-states", default=None)
    parser.add_argument("--domains", default=None)
    parser.add_argument("--steps", default=None)
    parser.add_argument("--dates", default=None)
    parser.add_argument("--spread-ddof", type=int, default=1)
    parser.add_argument("--map-bin-deg", type=float, default=0.5)
    parser.add_argument("--no-spectra", action="store_true")
    parser.add_argument("--spectra-fields", default=None)
    parser.add_argument("--spectra-steps", default=None)
    parser.add_argument("--spectra-nside", type=int, default=256)
    parser.add_argument("--enfo-exclude-members", default=None)
    parser.add_argument("--enfo-n-members", type=int, default=None)
    parser.add_argument("--enfo-subsample-seed", type=int, default=0)
    parser.add_argument("--no-input", action="store_true",
                        help="Skip the x_interp (EEFO input) spread readout.")
    parser.add_argument("--plots", action="store_true", help="Also render the three PDFs.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    payload = compute_spread_proxy(
        args.predictions_dir,
        args.output_dir,
        weather_states=args.weather_states,
        domains=args.domains,
        steps=args.steps,
        dates=args.dates,
        spread_ddof=args.spread_ddof,
        map_bin_deg=args.map_bin_deg,
        spectra=not args.no_spectra,
        spectra_fields=args.spectra_fields,
        spectra_steps=args.spectra_steps,
        spectra_nside=args.spectra_nside,
        enfo_exclude_members=args.enfo_exclude_members,
        enfo_n_members=args.enfo_n_members,
        enfo_subsample_seed=args.enfo_subsample_seed,
        include_input=not args.no_input,
    )
    LOG.info("spread_proxy: %s rows -> %s", payload["n_rows"], args.output_dir)
    if args.plots:
        from eval._backends.spread_proxy.plotting import plot_all

        plot_all(Path(args.output_dir), Path(args.output_dir) / "plots")


if __name__ == "__main__":
    main()
