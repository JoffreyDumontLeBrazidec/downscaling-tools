"""The fixed case list of a lane (``eval/config/suite/<name>.yaml``) and the automatic picks.

A case list fixes what every suite page shows, so that pages of two runs of the same lane can be compared
by eye. ``date: auto`` (rain, areas) picks from the predictions themselves, the same way every time:

* rain: the (date) at the case step whose TARGET member-mean tp maximum inside the extent is highest
  (the model is not looked at, so the pick cannot favour a run); without a target, the input;
* areas, stills: the first date common to every run.

Extents are [lon_min, lon_max, lat_min, lat_max] (the zoom_maps / map_video order).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "suite"
_FILE_RE = re.compile(r"predictions_(\d{8})_step(\d{3})\.nc$")


@dataclass
class Case:
    """One map case: date (YYYYMMDD or 'auto'), lead in hours, extent or None (whole file), variable."""

    name: str
    date: str
    step: int
    extent: list[float] | None = None
    variable: str = "wind10m"
    event: str | None = None
    vmax: float | None = None


@dataclass
class CaseList:
    name: str
    description: str = ""
    spectra_steps: list[int] = field(default_factory=lambda: [24, 72])
    spectra_box: str | None = None
    spectra_variables: list[str] = field(default_factory=lambda: ["tp", "ws", "2t", "msl", "10u", "10v"])
    tc: Case | None = None
    rain: Case | None = None
    areas: list[Case] = field(default_factory=list)
    stills: list[Case] = field(default_factory=list)
    stills_grid: str | None = None
    dist_steps: list[int] = field(default_factory=lambda: [24, 48, 72])
    score_variables: list[list[str]] = field(default_factory=lambda: [["tp", "raw"], ["10ff", "raw"],
                                                                     ["2t", "lapse"], ["msl", "raw"]])
    score_region: str = "all"
    members: str = "1-10"


def _case(name: str, d: dict | None, default_var: str) -> Case | None:
    if not d:
        return None
    return Case(name=str(d.get("name", name)), date=str(d.get("date", "auto")), step=int(d.get("step", 24)),
                extent=[float(v) for v in d["extent"]] if d.get("extent") else None,
                variable=str(d.get("variable", default_var)), event=d.get("event"),
                vmax=float(d["vmax"]) if d.get("vmax") is not None else
                (70.0 if name == "tc" and d.get("variable", default_var) == "wind10m" else None))


def available_case_lists() -> list[str]:
    return sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))


def load_case_list(spec: str) -> CaseList:
    """A case list by name (a file of eval/config/suite) or by path."""
    import yaml

    path = Path(spec)
    if not path.suffix:
        path = CONFIG_DIR / f"{spec}.yaml"
    if not path.exists():
        raise SystemExit(f"eval.cli suite: no case list {spec!r}; known: {', '.join(available_case_lists())}")
    d = yaml.safe_load(path.read_text()) or {}
    sp = d.get("spectra", {}) or {}
    st = d.get("stills", {}) or {}
    sc = d.get("scores", {}) or {}
    cl = CaseList(name=str(d.get("name", path.stem)), description=str(d.get("description", "")))
    cl.spectra_steps = [int(s) for s in sp.get("steps", cl.spectra_steps)]
    cl.spectra_box = sp.get("box")
    cl.spectra_variables = list(sp.get("variables", cl.spectra_variables))
    cl.tc = _case("tc", d.get("tc"), "wind10m")
    cl.rain = _case("rain", d.get("rain"), "tp")
    cl.areas = [_case(a.get("name", f"area{i + 1}"), a, "wind10m") for i, a in enumerate(d.get("areas", []) or [])]
    cl.stills = [_case(a.get("name", f"still{i + 1}"), a, "tp") for i, a in enumerate(st.get("cases", []) or [])]
    cl.stills_grid = st.get("grid")
    cl.dist_steps = [int(s) for s in (d.get("tp_distribution", {}) or {}).get("steps", cl.dist_steps)]
    cl.score_variables = [list(v) for v in sc.get("variables", cl.score_variables)]
    cl.score_region = str(sc.get("region", cl.score_region))
    cl.members = str(d.get("members", cl.members))
    return cl


def prediction_index(run_dir: Path) -> dict[tuple[str, int], Path]:
    """{(date, step): file} of a predictions folder."""
    out = {}
    for p in Path(run_dir).glob("predictions_*_step*.nc"):
        m = _FILE_RE.search(p.name)
        if m:
            out[(m.group(1), int(m.group(2)))] = p
    return out


def common_keys(run_dirs: list[Path]) -> set[tuple[str, int]]:
    keys = None
    for d in run_dirs:
        k = set(prediction_index(d))
        keys = k if keys is None else keys & k
    return keys or set()


def _in_extent(lat, lon, extent):
    if extent is None:
        return np.ones(lat.shape, bool)
    w, e, s, n = extent
    lon = ((np.asarray(lon) + 180.0) % 360.0) - 180.0
    return (lat >= s) & (lat <= n) & (lon >= w) & (lon <= e)


def field_max(path: Path, variable: str, name: str, extent) -> float:
    """Member-mean maximum of one state of ``name`` (y, x) inside the extent; NaN when absent or empty."""
    import xarray as xr

    with xr.open_dataset(path, decode_timedelta=False) as ds:
        states = [str(s) for s in ds["weather_state"].values]
        if variable not in states or name not in ds:
            return float("nan")
        da = ds[name]
        if "sample" in da.dims:
            da = da.isel(sample=0)
        da = da.isel(weather_state=states.index(variable))
        grid = "lres" if name == "x" else "hres"
        lat, lon = ds[f"lat_{grid}"].values, ds[f"lon_{grid}"].values
        mean = np.asarray(da.mean("ensemble_member").values if "ensemble_member" in da.dims else da.values, float)
    sel = _in_extent(lat, lon, extent)
    v = mean[sel]
    if not np.isfinite(v).any() or np.nanmax(np.abs(v)) == 0.0:
        return float("nan")
    return float(np.nanmax(v))


def resolve_date(case: Case, run_dirs: list[Path], *, pick: str) -> tuple[str, str]:
    """(date, how it was chosen) for a case; ``pick`` = 'rain' or 'first'."""
    keys = sorted(k for k in common_keys(run_dirs) if k[1] == case.step)
    if not keys:
        raise SystemExit(f"eval.cli suite: no date at +{case.step} h common to every run (case {case.name})")
    dates = [d for d, _ in keys]
    if case.date != "auto":
        if case.date not in dates:
            raise SystemExit(f"eval.cli suite: case {case.name} date {case.date} +{case.step} h is not in every run; "
                             f"common dates: {', '.join(dates)}")
        return case.date, "fixed in the case list"
    if pick == "first":
        return dates[0], "first common date"
    first = prediction_index(run_dirs[0])
    for name, label in (("y", "target"), ("x", "input")):
        vals = [field_max(first[(d, case.step)], "tp", name, case.extent) for d in dates]
        if np.isfinite(vals).any():
            i = int(np.nanargmax(vals))
            return dates[i], f"wettest {label} member-mean tp max in the box ({vals[i] * 1000:.1f} mm) of {len(dates)} dates"
    return dates[0], "first common date (no tp in target or input)"


def tc_auto(case: Case, cases_csv: Path, half_lat: float = 7.0) -> tuple[Case, str]:
    """The deepest storm case at the case step of a tc_structure read: (case with date and extent, how).

    Depth = member-mean central pressure of the target (else of the input) over the found members of the
    storm_ok cases; the model is not looked at. Extent: +-``half_lat`` degrees of latitude around the
    member-mean centre, the longitude span widened by 1 / cos(latitude).
    """
    import pandas as pd

    c = pd.read_csv(cases_csv, dtype={"date": str})
    c = c[(c.storm_ok == 1) & (c.found == 1) & (c.lead.astype(int) == case.step)]
    for field_name, label in (("truth", "target"), ("input", "input")):
        f = c[c.field == field_name]
        if len(f):
            g = f.groupby(["event", "date"]).agg(p=("pmin_hpa", "mean"), lat=("centre_lat", "mean"),
                                                 lon=("centre_lon", "mean")).reset_index()
            r = g.loc[g.p.idxmin()]
            half_lon = half_lat / max(0.3, np.cos(np.deg2rad(float(r.lat))))
            lon = ((float(r.lon) + 180.0) % 360.0) - 180.0
            out = Case(name=str(r.event), date=str(r.date), step=case.step, variable=case.variable, vmax=case.vmax,
                       extent=[lon - half_lon, lon + half_lon, float(r.lat) - half_lat, float(r.lat) + half_lat],
                       event=str(r.event))
            return out, f"deepest {label} storm at +{case.step} h ({float(r.p):.0f} hPa member mean, {len(g)} storm cases)"
    raise SystemExit(f"eval.cli suite: no storm case at +{case.step} h in {cases_csv}")
