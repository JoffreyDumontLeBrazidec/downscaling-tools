"""Write an external target into prediction files that carry none (``eval.cli suite --attach-target DIR``).

Some rows have no embedded ``y`` (AIFS-ENS rows: the analysis store ended before the dates). The target is then
given as one file per valid time, ``truth_<YYYYMMDDHH>.nc`` holding ``y(grid_point, weather_state)`` on the
prediction file's output points in the same order, with the same weather_state names (NaN allowed, for example tp
of an analysis). Each prediction file without ``y`` gets a variable ``y(sample, grid_point_hres, weather_state)``:
one field, no member axis (every reader of the framework takes a field without a member axis as serving every
member). A file whose truth file is missing gets an all-NaN ``y``, so its target row is empty and labelled,
not invented. Files that already carry ``y`` are left alone. This WRITES INTO the prediction files (append only).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from eval.suite.cases import prediction_index

LOG = logging.getLogger(__name__)


def _states(var) -> list[str]:
    return [str(s) for s in np.asarray(var[:]).reshape(-1)]


def attach_target(run_dir: Path, truth_dir: Path) -> dict:
    """Append y to every prediction file of run_dir that has none. Returns counts."""
    import netCDF4

    done = {"written": 0, "missing_truth": 0, "already": 0}
    for (date, step), path in sorted(prediction_index(run_dir).items()):
        valid = datetime.strptime(date, "%Y%m%d") + timedelta(hours=step)
        with netCDF4.Dataset(str(path), "a") as d:
            if "y" in d.variables:
                done["already"] += 1
                continue
            states = _states(d["weather_state"])
            hdim = d["y_pred"].dimensions[-2]
            n = len(d.dimensions[hdim])
            arr = np.full((n, len(states)), np.nan, dtype=np.float32)
            tf = Path(truth_dir) / f"truth_{valid:%Y%m%d%H}.nc"
            if tf.exists():
                with netCDF4.Dataset(str(tf)) as t:
                    t.set_auto_mask(False)
                    t_states = _states(t["weather_state"])
                    ty = np.asarray(t["y"][:], dtype=np.float32)
                    if ty.shape[0] != n:
                        raise ValueError(f"{tf}: {ty.shape[0]} points, {path.name} has {n}")
                    for j, s in enumerate(states):
                        if s in t_states:
                            arr[:, j] = ty[:, t_states.index(s)]
                done["written"] += 1
            else:
                done["missing_truth"] += 1
                LOG.warning("attach_target: no %s; %s gets an all-NaN target", tf.name, path.name)
            dims = ("sample", hdim, "weather_state") if "sample" in d.dimensions else (hdim, "weather_state")
            v = d.createVariable("y", "f4", dims, zlib=True, complevel=1, fill_value=None)
            v[:] = arr[None] if dims[0] == "sample" else arr
            v.setncattr("source", f"eval.cli suite --attach-target {tf}" if tf.exists() else "none (all NaN)")
    return done
