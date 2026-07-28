"""Build the EEFO-input and ENFO-target reference prediction sets.

The ladder's prediction files already carry everything needed, so neither reference costs a
single forward pass:

  y_pred     the model, 10 members            -> the curves already plotted
  x_interp   the EEFO O96 INPUT interpolated onto the O320 target grid
  y          the ENFO O320 TARGET, 10 members

`compute_probabilistic_scores` collapses the truth to `y` MEMBER 0 and scores every forecast
member against it. So, on exactly that support:

  EEFO input  = score x_interp  -> "what you get with no downscaling at all"
  ENFO target = score y members -> two ENFO members vs each other, i.e. how far apart the target
                ensemble's own members are. That is the floor a member-matched score can reach:
                truth is ONE member, so no forecast can beat the ensemble's internal distance.

For the ENFO set, member 0 is DROPPED. Leaving it in would score the truth against itself and
drag the whole reference to an unreachably optimistic value.

Only the variables the evaluators read are copied, so the reference sets stay small.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

SRC = Path(sys.argv[1])          # a rung's predictions dir
DST_ROOT = Path(sys.argv[2])     # where the two reference dirs go
# Names are lane-dependent: eefo->enfo on o96->o320 and o320->o1280, but enfo->iekm on
# o1280->o2560 and o48->o96 targets iekm. The ROLE (input / target) is what the builder
# actually branches on; the names are labels only. Defaults preserve the original paths.
INPUT_NAME = sys.argv[3] if len(sys.argv) > 3 else "eefo_input"
TARGET_NAME = sys.argv[4] if len(sys.argv) > 4 else "enfo_target"

# `x` (the low-res input) is required by the spectra proxy runner for its residual spectra --
# omitting it fails the rung with "Predictions file missing low-resolution input x".
KEEP = ["x", "date", "lon_lres", "lat_lres", "lon_hres", "lat_hres",
        "init_date", "lead_step_hours", "valid_time"]


def build(role: str, name: str) -> None:
    """role is "input" or "target"; name is only what the directory is called."""
    out = DST_ROOT / name / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted(SRC.glob("predictions_*.nc")):
        with xr.open_dataset(f, decode_timedelta=False) as ds:
            if role == "input":
                pred_v, truth_v = ds["x_interp"].values, ds["y"].values
            else:
                # ENFO target: forecast = members 1..N-1, truth = member 0 repeated to match.
                # Member 0 MUST be excluded from the forecast -- it is the truth, and leaving
                # it in scores it against itself and drags the reference to an unreachable
                # value. y_pred and y must also share the ensemble_member dim, hence the
                # repeat rather than a 1-member truth.
                yv = ds["y"].values
                pred_v = yv[:, 1:]
                truth_v = np.repeat(yv[:, 0:1], pred_v.shape[1], axis=1)
            # built from raw arrays with explicit dims: handing xarray two DataArrays whose
            # ensemble_member coords differ makes it ALIGN them, which NaNs the overlap and
            # the scorer then reports "no valid points".
            dims = ds["y"].dims
            new = xr.Dataset(
                {"y_pred": (dims, pred_v), "y": (dims, truth_v)},
                coords={c: ds[c] for c in ds.coords
                        if c not in ("ensemble_member",) and (c in dims or c == "grid_point_lres")},
            )
            n_mem = new.sizes["ensemble_member"]
            for k in KEEP:
                if k not in ds:
                    continue
                v = ds[k]
                # a kept variable that carries the member axis must be cut to the same length,
                # or adding it re-triggers the alignment conflict (`x` is 10-member).
                if "ensemble_member" in v.dims and v.sizes["ensemble_member"] != n_mem:
                    v = v.isel(ensemble_member=slice(-n_mem, None))
                    new[k] = (v.dims, v.values)
                else:
                    new[k] = v
            new.attrs = dict(ds.attrs)
            new.attrs["reference_kind"] = name
            new.attrs["reference_role"] = role
            new.attrs["reference_note"] = (
                "the lane INPUT, interpolated onto the target grid and scored as if it "
                "were the forecast -- what you get with no downscaling at all"
                if role == "input" else
                "the lane TARGET ensemble, members 1..N scored against member 0; member 0 "
                "is the verifying truth and MUST stay out of the forecast")
            new.to_netcdf(out / f.name)
        print("  %s <- %s" % (name, f.name), flush=True)


for _role, _name in (("input", INPUT_NAME), ("target", TARGET_NAME)):
    build(_role, _name)
print("done ->", DST_ROOT)
