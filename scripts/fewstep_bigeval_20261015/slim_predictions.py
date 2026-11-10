#!/usr/bin/env python3
"""Copy prediction files keeping only the weather states the big eval reads (Atos scratch is short).

    python scripts/fewstep_bigeval_20261015/slim_predictions.py --src <tree or predictions dir> \
        --dst <new dir> [--extra u_850,v_850] [--keep-member-views] [--skip-existing] [--dry-run]

``--src`` is a ``predictions/`` directory or any tree above it (for example
``<root>/<arm>/step_NNNNNN/<sampler>/predictions``); every ``predictions_*.nc`` under it is written
to the same relative path under ``--dst``, and every other small file of the tree
(``predictions_manifest.csv``, ``costs.jsonl``, logs, configs; up to ``--max-sidecar-mb``) is
copied as is, so the aggregator still finds ``<arm>/step_NNNNNN/costs.jsonl``.

What a slim file keeps:
- every dimension, variable, attribute, dtype, fill value, compression and chunking of the
  source, except that the ``weather_state`` dimension holds only the kept states, in the
  source's order (a chunk size along it is clipped to the new length);
- the kept states: KEEP below plus ``--extra``; a kept state absent from a file is reported
  and skipped (the file is still written with the states it has);
- the per-member views ``x_<i>``, ``y_<i>``, ``x_interp_<i>``, ``y_pred_<i>`` that
  build_predictions_dataset adds are dropped unless ``--keep-member-views``: they repeat
  ``x``/``y``/``x_interp``/``y_pred`` member by member (double the size) and no evaluator of the
  big eval reads them (zoom_maps, box_spectra, ... read the 4-d variables);
- a global attribute ``slimmed_from`` (the source path) and ``slimmed_weather_states``.

One file at a time, one member at a time per variable, so memory stays at one member's field.
A file is written to ``<name>.part`` and renamed when complete. An existing destination file is
never overwritten: the run stops on it, or skips it with ``--skip-existing`` (for a restart).
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

LOG = logging.getLogger("slim_predictions")

# Read by the evaluators of the big eval (AUDIT.md):
#   10u 10v   every structure/spectra/score instrument, tc, tc_structure, zoom_maps (wind)
#   2t        probabilistic, spectra, texture, zoom_maps
#   2d        probabilistic's default list (kept so a default rerun works)
#   msl       probabilistic, spectra, texture, tc, tc_structure, storm-box centre of maps.sh
#   sp        spectra_ecmwf_v2
#   tp, cp    precip_dist, probabilistic (tp), spectra (tp); cp beside tp for rain reads
#   t_850, z_500   probabilistic, texture (t_850), spectra_ecmwf_v2
KEEP = ("10u", "10v", "2t", "2d", "msl", "sp", "tp", "cp", "t_850", "z_500")
MEMBER_VIEW_RE = re.compile(r"^(x|y|x_interp|y_pred)_\d+$")
FILE_GLOB = "predictions_*.nc"


def _decode(values) -> list[str]:
    out = []
    for v in np.asarray(values).reshape(-1):
        if isinstance(v, bytes):
            v = v.decode()
        out.append(str(v).strip())
    return out


def _states_of(ds) -> list[str]:
    v = ds["weather_state"]
    if v.dtype is not str and np.dtype(v.dtype).kind == "S" and v.ndim == 2:
        import netCDF4
        return [str(s).strip() for s in netCDF4.chartostring(v[:])]
    return _decode(v[:])


def _create_like(dst, name: str, src_var, new_len: dict[str, int]):
    kw: dict = {}
    filt = src_var.filters() or {}
    if filt.get("zlib"):
        kw.update(zlib=True, complevel=filt.get("complevel", 4))
    if filt.get("shuffle"):
        kw["shuffle"] = True
    if filt.get("fletcher32"):
        kw["fletcher32"] = True
    chunks = src_var.chunking()
    if isinstance(chunks, list):
        kw["chunksizes"] = tuple(min(c, new_len.get(d, len(src_var.group().dimensions[d])))
                                 for c, d in zip(chunks, src_var.dimensions))
    elif chunks == "contiguous" and src_var.dtype is not str:
        kw["contiguous"] = True
    attrs = {k: src_var.getncattr(k) for k in src_var.ncattrs()}
    fill = attrs.pop("_FillValue", None)
    dtype = str if src_var.dtype is str else src_var.datatype
    if dtype is str:
        kw = {}
    out = dst.createVariable(name, dtype, src_var.dimensions, fill_value=fill, **kw)
    out.setncatts(attrs)
    return out


def slim_file(src: Path, dst: Path, keep: list[str], *, keep_member_views: bool = False) -> dict:
    """Write the slim copy of one file; returns a small report."""
    import netCDF4

    if dst.exists():
        raise FileExistsError(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    if part.exists():
        part.unlink()
    with netCDF4.Dataset(str(src)) as s:
        if s.groups:
            raise ValueError(f"{src}: groups are not supported")
        s.set_auto_mask(False)
        s.set_auto_scale(False)
        states = _states_of(s)
        idx = [i for i, st in enumerate(states) if st in set(keep)]
        absent = [k for k in keep if k not in states]
        if not idx:
            raise ValueError(f"{src}: none of {keep} in weather_state {states[:10]}...")
        new_len = {"weather_state": len(idx)}
        dropped = []
        with netCDF4.Dataset(str(part), "w", format=s.data_model) as d:
            for name, dim in s.dimensions.items():
                n = new_len.get(name, len(dim))
                d.createDimension(name, None if dim.isunlimited() else n)
            d.setncatts({k: s.getncattr(k) for k in s.ncattrs()})
            d.setncattr("slimmed_from", str(src))
            d.setncattr("slimmed_weather_states", ",".join(states[i] for i in idx))
            for name, v in s.variables.items():
                if not keep_member_views and MEMBER_VIEW_RE.match(name):
                    dropped.append(name)
                    continue
                out = _create_like(d, name, v, new_len)
                dims = v.dimensions
                if "weather_state" not in dims:
                    out[...] = v[...]
                    continue
                ax = dims.index("weather_state")
                if v.dtype is str or v.ndim == 1:
                    vals = v[...]
                    out[...] = np.take(np.asarray(vals, dtype=object if v.dtype is str else None), idx, axis=ax)
                    continue
                # stream along the member axis (or the first axis) to bound memory
                loop_ax = dims.index("ensemble_member") if "ensemble_member" in dims else 0
                if loop_ax == ax:
                    loop_ax = next(i for i in range(v.ndim) if i != ax)
                for k in range(v.shape[loop_ax]):
                    key = [slice(None)] * v.ndim
                    key[loop_ax] = k
                    key[ax] = idx
                    block = v[tuple(key)]
                    okey = [slice(None)] * v.ndim
                    okey[loop_ax] = k
                    out[tuple(okey)] = block
    os.replace(part, dst)
    return {"file": dst.name, "kept": [states[i] for i in idx], "absent": absent,
            "dropped_vars": dropped, "bytes_in": src.stat().st_size, "bytes_out": dst.stat().st_size}


def _walk(src: Path):
    """(prediction files, sidecar files) under src, sorted."""
    if src.is_file():
        return [src], []
    nc = sorted(p for p in src.rglob(FILE_GLOB) if p.is_file())
    other = sorted(p for p in src.rglob("*") if p.is_file() and p not in set(nc)
                   and not p.name.endswith(".part"))
    return nc, other


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--dst", required=True, type=Path)
    p.add_argument("--extra", default="", help="comma list of states to keep beside the default")
    p.add_argument("--states", default=None, help="comma list replacing the default kept states")
    p.add_argument("--keep-member-views", action="store_true")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip a destination file that exists (never overwritten either way)")
    p.add_argument("--max-sidecar-mb", type=float, default=200.0,
                   help="copy non-prediction files up to this size")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    keep = [s for s in (a.states.split(",") if a.states else list(KEEP)) if s]
    keep += [s for s in a.extra.split(",") if s and s not in keep]
    src = a.src.resolve()
    dst = a.dst.resolve()
    if dst == src or src in dst.parents:
        LOG.error("--dst must not be inside --src")
        return 2
    nc, other = _walk(src)
    if not nc:
        LOG.error("no %s under %s", FILE_GLOB, src)
        return 2
    base = src.parent if src.is_file() else src
    LOG.info("%d prediction file(s), %d other file(s); keeping %s", len(nc), len(other), keep)
    if a.dry_run:
        for f in nc:
            LOG.info("[dry-run] %s -> %s", f, dst / f.relative_to(base))
        return 0
    for f in other:
        t = dst / f.relative_to(base)
        if t.exists():
            continue
        if f.stat().st_size > a.max_sidecar_mb * 1e6:
            LOG.warning("not copied (larger than %.0f MB): %s", a.max_sidecar_mb, f)
            continue
        t.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, t)
    tot_in = tot_out = 0
    for f in nc:
        t = dst / f.relative_to(base)
        if t.exists():
            if a.skip_existing:
                LOG.info("[skip] exists: %s", t)
                continue
            LOG.error("refusing to overwrite %s (use --skip-existing to resume)", t)
            return 3
        r = slim_file(f, t, keep, keep_member_views=a.keep_member_views)
        tot_in += r["bytes_in"]
        tot_out += r["bytes_out"]
        LOG.info("%s: %.2f GB -> %.2f GB%s", t, r["bytes_in"] / 1e9, r["bytes_out"] / 1e9,
                 f" (absent: {r['absent']})" if r["absent"] else "")
    LOG.info("total %.1f GB -> %.1f GB", tot_in / 1e9, tot_out / 1e9)
    return 0


if __name__ == "__main__":
    sys.exit(main())
