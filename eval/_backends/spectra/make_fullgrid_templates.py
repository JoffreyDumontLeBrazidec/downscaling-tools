#!/usr/bin/env python3
"""Build full-grid per-field GRIB templates for the spectra staging step.

The staging step writes prediction values into a per-field template.  The
templates inherited from an earlier workflow have 28 latitude rows removed near
the poles, so every staged field is transformed over an incomplete sphere.  That
was measured on 2026-08-25 to shift the spectrum by a median of 2% and up to 13%,
inside the scored band, which is the same order as the differences these spectra
are used to measure.

These templates carry the complete grid.  Because the staging step derives its
pole mask from the template's own first latitude, a full-grid template makes that
mask an identity operation, so no pipeline change is needed to use them.

Values are zeroed: staging overwrites them.  Only the grid and the field
identity matter.

Additive and idempotent: writes new files, touches nothing existing.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from eccodes import (
    codes_clone,
    codes_get,
    codes_grib_new_from_file,
    codes_release,
    codes_set,
    codes_set_values,
    codes_write,
)

# dir_name -> (paramId, typeOfLevel, level)
FIELDS: dict[str, tuple[int, str, int]] = {
    "10u_sfc": (165, "surface", 0),
    "10v_sfc": (166, "surface", 0),
    "2t_sfc": (167, "surface", 0),
    "msl_sfc": (151, "surface", 0),
    "sp_sfc": (134, "surface", 0),
    "t_850": (130, "isobaricInhPa", 850),
    "z_500": (129, "isobaricInhPa", 500),
}


def build(grid_template: Path, out_root: Path, *, apply: bool) -> None:
    with grid_template.open("rb") as handle:
        base = codes_grib_new_from_file(handle)
    if base is None:
        raise RuntimeError(f"Could not read {grid_template}")
    try:
        n_values = int(codes_get(base, "numberOfValues"))
        grid_name = str(codes_get(base, "gridName"))
        nj = int(codes_get(base, "Nj"))
        print(f"  source grid: {grid_name}  Nj={nj}  points={n_values}")
        zeros = np.zeros(n_values, dtype=np.float64)

        for dir_name, (param_id, type_of_level, level) in FIELDS.items():
            out_dir = out_root / dir_name
            out_path = out_dir / "fullgrid_template_nopoles.grb"
            print(f"    {dir_name:9s} paramId={param_id:3d} {type_of_level}/{level} -> {out_path}")
            if not apply:
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            clone = codes_clone(base)
            try:
                codes_set(clone, "paramId", param_id)
                codes_set(clone, "typeOfLevel", type_of_level)
                codes_set(clone, "level", level)
                codes_set_values(clone, zeros)
                with out_path.open("wb") as out:
                    codes_write(clone, out)
            finally:
                codes_release(clone)
    finally:
        codes_release(base)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-template", required=True, help="e.g. o1280-template.grib")
    parser.add_argument("--out-root", required=True, help="directory of per-field template dirs")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    if not args.apply:
        print("DRY RUN - nothing written. Pass --apply.\n")
    build(Path(args.grid_template), Path(args.out_root), apply=args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
