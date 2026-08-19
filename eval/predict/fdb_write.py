"""Push manual-inference prediction NetCDFs to FDB under a (usually empty) expver.

Why this exists
---------------
The ``eval.cli predict --mode prepml`` path stages MARS, runs the model, and
writes ``out_hres`` to FDB in one suite. When that path is unavailable -- e.g. a
checkpoint whose embedded metadata the prepml staging fork cannot parse
(``checkpoint.mars_requests() -> "No variables provided"``) -- the *model run*
itself still works via manual inference. This module takes the already-computed
manual prediction NetCDFs and archives them to FDB directly, so FDB-based
scorecards (quaver CRPS/spread) can run against them.

Approach
--------
Rather than hand-encode ensemble GRIB2 keys (perturbationNumber, product
definition template, paramId stability), we *clone* the real per-date ENFO
ensemble GRIB that already accompanies the run (the truth-aware bundle's
``enfo_<grid>_..._sfc_y.grib``) and overwrite only ``class -> rd``, ``expver``,
and the field *values* with ``y_pred``. Every other key (stream=enfo, type=pf,
perturbationNumber, step, paramId, grid) is inherited correct-by-construction.

Archive is via ``grib2fdb`` (the MARS ``grib2fdb5`` wrapper); the resulting
fields are keyed ``class=rd,expver=<E>,stream=enfo,type=pf,number=<member>,
date,time=0000,step,levtype=sfc,param=<paramId>`` -- identical to what the
prepml model task would have written.
"""
from __future__ import annotations

import glob
import logging
import subprocess
from pathlib import Path

import numpy as np
import xarray as xr

LOG = logging.getLogger(__name__)

# Surface paramId -> weather_state name (the y_pred ``weather_state`` coord labels).
SFC_PARAM_TO_WEATHER_STATE: dict[int, str] = {
    165: "10u", 166: "10v", 168: "2d", 167: "2t",
    151: "msl", 235: "skt", 134: "sp", 136: "tcw",
}


def _find_template_grib(input_root: Path, date: str) -> Path:
    """The per-date ENFO ensemble surface GRIB that backs the truth-aware bundle."""
    pats = [
        f"enfo_*_date{date}_*_sfc_y.grib",
        f"enfo_*{date}*sfc*.grib",
    ]
    for pat in pats:
        hits = sorted(input_root.glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"No ENFO ensemble template GRIB for date {date} under {input_root} "
        f"(looked for {pats})"
    )


def _prediction_nc(predictions_dir: Path, date: str, step: int) -> Path:
    return predictions_dir / f"predictions_{date}_step{step:03d}.nc"


def build_grib(
    predictions_dir: Path,
    input_root: Path,
    out_dir: Path,
    expver: str,
    dates: list[str],
    steps: list[int],
    param_to_weather_state: dict[int, str] | None = None,
) -> list[Path]:
    """Write one model GRIB per date by cloning the ENFO template and swapping values.

    Returns the list of written GRIB paths. Pure file IO -- no FDB side effects.
    """
    from eccodes import (
        codes_grib_new_from_file, codes_clone, codes_release, codes_get,
        codes_set, codes_set_values, codes_write,
    )

    pid2ws = param_to_weather_state or SFC_PARAM_TO_WEATHER_STATE
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for date in dates:
        ncs = {s: xr.open_dataset(_prediction_nc(predictions_dir, date, s)) for s in steps}
        weather_states = [str(w) for w in ncs[steps[0]]["weather_state"].values]
        template = _find_template_grib(input_root, date)
        out_path = out_dir / f"model_{date}.grib"

        n = 0
        with open(template, "rb") as f_in, open(out_path, "wb") as f_out:
            while True:
                gid = codes_grib_new_from_file(f_in)
                if gid is None:
                    break
                try:
                    pid = codes_get(gid, "paramId")
                    member = codes_get(gid, "perturbationNumber")
                    step = codes_get(gid, "step")
                    if pid in pid2ws and step in steps:
                        var = pid2ws[pid]
                        vi = weather_states.index(var)
                        vals = np.asarray(
                            ncs[step]["y_pred"].isel(sample=0, ensemble_member=member - 1)[..., vi].values,
                            dtype=np.float64,
                        )
                        if not np.isfinite(vals).all():
                            raise ValueError(f"non-finite y_pred {var} mem{member} {date} step{step}")
                        clone = codes_clone(gid)
                        try:
                            codes_set(clone, "class", "rd")
                            codes_set(clone, "expver", expver)
                            codes_set_values(clone, vals)
                            codes_write(clone, f_out)
                        finally:
                            codes_release(clone)
                        n += 1
                finally:
                    codes_release(gid)
        for ds in ncs.values():
            ds.close()
        LOG.info("fdb_write: %s -> %d messages (%s)", date, n, out_path)
        written.append(out_path)
    return written


def archive_to_fdb(grib_paths: list[Path], expver: str, stream: str = "enfo", type_: str = "pf") -> None:
    """Archive GRIB files to FDB via grib2fdb (the grib2fdb5 wrapper)."""
    for g in grib_paths:
        cmd = (
            f"module load fdb 2>/dev/null; "
            f"grib2fdb -c rd -e {expver} -s {stream} -T {type_} -f {g}"
        )
        LOG.info("fdb_write: archiving %s -> class=rd,expver=%s", g, expver)
        subprocess.run(["bash", "-lc", cmd], check=True)


def write_predictions_to_fdb(
    predictions_dir: Path,
    input_root: Path,
    out_dir: Path,
    expver: str,
    dates: list[str],
    steps: list[int],
    *,
    dry_run: bool = False,
) -> list[Path]:
    """Build GRIB from predictions and (unless dry_run) archive to FDB. Returns paths."""
    gribs = build_grib(predictions_dir, input_root, out_dir, expver, dates, steps)
    if dry_run:
        LOG.info("fdb_write: dry-run, skipping archive of %d files", len(gribs))
        return gribs
    archive_to_fdb(gribs, expver)
    return gribs


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description="Archive manual-inference prediction NetCDFs to FDB under an expver."
    )
    p.add_argument("--predictions-dir", required=True, type=Path)
    p.add_argument("--input-root", required=True, type=Path,
                   help="Dir holding the per-date ENFO ensemble template GRIBs (the truth-aware bundle dir).")
    p.add_argument("--expver", required=True)
    p.add_argument("--dates", required=True, help="Comma-separated YYYYMMDD.")
    p.add_argument("--steps", required=True, help="Comma-separated forecast steps, e.g. 24,48,72,96,120.")
    p.add_argument("--out-dir", default=None, type=Path,
                   help="Where to stage the model GRIB files (default: <predictions-dir>/../fdb_write_<expver>).")
    p.add_argument("--dry-run", action="store_true", help="Build GRIB but do not archive to FDB.")
    a = p.parse_args()

    dates = [d.strip() for d in a.dates.split(",") if d.strip()]
    steps = [int(s) for s in a.steps.split(",") if s.strip()]
    out_dir = a.out_dir or (a.predictions_dir.parent / f"fdb_write_{a.expver}")
    gribs = write_predictions_to_fdb(
        a.predictions_dir, a.input_root, out_dir, a.expver, dates, steps, dry_run=a.dry_run
    )
    LOG.info("fdb_write: done (%d GRIB files, expver=%s, archived=%s)",
             len(gribs), a.expver, not a.dry_run)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# o2560 (single-realization template) variant — 2026-08-19
# ---------------------------------------------------------------------------
# The o1280->o2560 lane has no ENFO ensemble GRIB on the TARGET grid: truth is
# the single-realization rd/iekm (destine i4ql) forecast, GRIB edition 1 with
# an ECMWF local section (so ``number`` is settable). ``build_grib_expand``
# clones ONE template message per paramId and expands over (member, step),
# rewriting the ensemble keys. Precision note: template packing is 16-bit
# grid_simple — quantization ~range/65k per field, same class as the prepml
# GRIB writeback.

EXPAND_PARAMS: dict[int, str] = {165: "10u", 166: "10v", 167: "2t", 151: "msl"}


def build_grib_expand(
    predictions_dir: Path,
    template_glob: str,
    out_dir: Path,
    expver: str,
    dates: list[str],
    steps: list[int],
    members: list[int],
    param_to_weather_state: dict[int, str] | None = None,
) -> list[Path]:
    """Write one model GRIB per date from a single-realization template.

    template_glob: glob with a ``{date}`` placeholder resolving to the per-date
    template GRIB on the SAME grid as y_pred (e.g. the iekm o2560 y.grib).
    """
    from eccodes import (
        codes_grib_new_from_file, codes_clone, codes_release, codes_get,
        codes_set, codes_set_values, codes_write,
    )

    pid2ws = param_to_weather_state or EXPAND_PARAMS
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for date in dates:
        hits = sorted(glob.glob(template_glob.format(date=date)))
        if not hits:
            raise FileNotFoundError(f"no template GRIB for {date}: {template_glob}")
        template = hits[0]

        # One template gid per paramId (first message wins).
        base_gids: dict[int, int] = {}
        with open(template, "rb") as f_in:
            while True:
                gid = codes_grib_new_from_file(f_in)
                if gid is None:
                    break
                pid = codes_get(gid, "paramId")
                if pid in pid2ws and pid not in base_gids:
                    base_gids[pid] = gid  # keep handle; released at end of date
                else:
                    codes_release(gid)
        # A param absent from the template (e.g. 2d in the 4-param iekm y.grib)
        # is cloned from ANY template message with paramId rewritten — the grid/
        # packing sections are param-independent here.
        fallback_gid = next(iter(base_gids.values()), None)
        missing = [pid for pid in pid2ws if pid not in base_gids]
        if missing and fallback_gid is None:
            raise KeyError(f"template {template} lacks paramIds {missing} and has no fallback message")

        ncs = {s: xr.open_dataset(_prediction_nc(predictions_dir, date, s)) for s in steps}
        weather_states = [str(w) for w in ncs[steps[0]]["weather_state"].values]
        out_path = out_dir / f"model_{date}.grib"

        n = 0
        with open(out_path, "wb") as f_out:
            for pid, ws in pid2ws.items():
                vi = weather_states.index(ws)  # by LABEL, never position (P3)
                for s in steps:
                    for m in members:
                        vals = np.asarray(
                            ncs[s]["y_pred"].isel(sample=0, ensemble_member=m - 1)[..., vi].values,
                            dtype=np.float64,
                        )
                        if not np.isfinite(vals).all():
                            raise ValueError(f"non-finite y_pred {ws} mem{m} {date} step{s}")
                        clone = codes_clone(base_gids.get(pid, fallback_gid))
                        try:
                            if pid not in base_gids:
                                codes_set(clone, "paramId", pid)
                            codes_set(clone, "class", "rd")
                            codes_set(clone, "expver", expver)
                            codes_set(clone, "stream", "enfo")
                            codes_set(clone, "type", "pf")
                            codes_set(clone, "number", m)
                            codes_set(clone, "step", s)
                            codes_set_values(clone, vals)
                            codes_write(clone, f_out)
                        finally:
                            codes_release(clone)
                        n += 1
        for gid in base_gids.values():
            codes_release(gid)
        for ds in ncs.values():
            ds.close()
        LOG.info("fdb_write(expand): %s -> %d messages (%s)", date, n, out_path)
        written.append(out_path)
    return written
