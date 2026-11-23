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
    vmin: float | None = None
    box_deg: float | None = None      # map box of box_deg degrees of latitude (longitude widened by 1/cos(lat))


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
    zoom_levels: list[int] = field(default_factory=lambda: [1, 2, 4])
    zoom_close_points: int = 230


# Map box of the suite (owner 23 Nov 2026 ~10:00 UTC: 3 degrees far too tight, 17 degrees too blurry): about
# 10 degrees of latitude, with the colour range fitted to the box.
BOX_DEG = 10.0


def box_around(lat: float, lon: float, box_deg: float = BOX_DEG) -> list[float]:
    """[lon_min, lon_max, lat_min, lat_max] of box_deg degrees of latitude centred on (lat, lon)."""
    half = 0.5 * box_deg
    half_lon = half / max(0.3, float(np.cos(np.deg2rad(lat))))
    return [lon - half_lon, lon + half_lon, lat - half, lat + half]


def _case(name: str, d: dict | None, default_var: str) -> Case | None:
    if not d:
        return None
    if d.get("center") and not d.get("extent"):
        d = {**d, "extent": box_around(float(d["center"][0]), float(d["center"][1]), float(d.get("box_deg", BOX_DEG)))}
    return Case(name=str(d.get("name", name)), date=str(d.get("date", "auto")), step=int(d.get("step", 24)),
                extent=[float(v) for v in d["extent"]] if d.get("extent") else None,
                variable=str(d.get("variable", default_var)), event=d.get("event"),
                # no fixed TC 0-70 m/s any more: the top is fitted to the box (owner 23 Nov ~10:00)
                vmax=float(d["vmax"]) if d.get("vmax") is not None else None,
                vmin=float(d["vmin"]) if d.get("vmin") is not None else None,
                box_deg=float(d["box_deg"]) if d.get("box_deg") is not None else None)


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
    zm = d.get("zoom", {}) or {}
    cl.zoom_levels = [int(v) for v in zm.get("levels", cl.zoom_levels)]
    cl.zoom_close_points = int(zm.get("close_points", cl.zoom_close_points))
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


def field_stats(path: Path, variable: str, name: str, extent) -> dict | None:
    """Member-mean field of one state of ``name`` (y, x) inside the extent: max, its position, 1st and 99th
    percentiles. None when absent, all-NaN or all zero."""
    import xarray as xr

    with xr.open_dataset(path, decode_timedelta=False) as ds:
        states = [str(s) for s in ds["weather_state"].values]
        if variable not in states or name not in ds:
            return None
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
        return None
    i = int(np.nanargmax(v))
    lon_s = ((np.asarray(lon)[sel] + 180.0) % 360.0) - 180.0
    return {"max": float(v[i]), "lat": float(np.asarray(lat)[sel][i]), "lon": float(lon_s[i]),
            "p01": float(np.nanpercentile(v, 1)), "p99": float(np.nanpercentile(v, 99))}


def speed_max(path: Path, extent, names=("y", "y_pred")) -> float:
    """Largest 10 m wind speed (m/s) of any member of ``names`` inside the extent; NaN when none is there."""
    import xarray as xr

    best = float("nan")
    with xr.open_dataset(path, decode_timedelta=False) as ds:
        states = [str(s) for s in ds["weather_state"].values]
        if "10u" not in states or "10v" not in states:
            return best
        sel = _in_extent(ds["lat_hres"].values, ds["lon_hres"].values, extent)
        for name in names:
            if name not in ds:
                continue
            da = ds[name].isel(sample=0) if "sample" in ds[name].dims else ds[name]
            u = np.asarray(da.isel(weather_state=states.index("10u")).values, float)[..., sel]
            v = np.asarray(da.isel(weather_state=states.index("10v")).values, float)[..., sel]
            sp = np.hypot(u, v)
            if np.isfinite(sp).any():
                best = float(np.nanmax([best, np.nanmax(sp)]))
    return best


def field_max(path: Path, variable: str, name: str, extent) -> float:
    """Member-mean maximum of one state of ``name`` (y, x) inside the extent; NaN when absent or empty."""
    st = field_stats(path, variable, name, extent)
    return float("nan") if st is None else st["max"]


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


def tc_auto(case: Case, cases_csv: Path, half_lat: float = 0.5 * BOX_DEG) -> tuple[Case, str]:
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


def best_track_fixes(cache_dir: Path | None = None, ibtracs_csv: Path | None = None):
    """Best-track fixes as a frame (event, time YYYYMMDDHH, lat, lon, pmin_hpa): a tc_obs cache's best_track.csv
    or a NOAA IBTrACS csv (v04, two header rows; pressure = WMO_PRES, else USA_PRES)."""
    import pandas as pd

    if cache_dir is not None and (Path(cache_dir) / "best_track.csv").exists():
        bt = pd.read_csv(Path(cache_dir) / "best_track.csv", dtype={"time": str})
        return bt[["event", "time", "lat", "lon", "pmin_hpa"]].dropna()
    if ibtracs_csv is None:
        return None
    df = pd.read_csv(ibtracs_csv, skiprows=[1], low_memory=False, keep_default_na=False)
    pres = pd.to_numeric(df.get("WMO_PRES"), errors="coerce")
    if "USA_PRES" in df:
        pres = pres.fillna(pd.to_numeric(df["USA_PRES"], errors="coerce"))
    t = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    out = pd.DataFrame({"event": df["NAME"].astype(str).str.lower() + "_" + df["SID"].astype(str),
                        "time": t.dt.strftime("%Y%m%d%H"), "lat": pd.to_numeric(df["LAT"], errors="coerce"),
                        "lon": ((pd.to_numeric(df["LON"], errors="coerce") + 180.0) % 360.0) - 180.0,
                        "pmin_hpa": pres})
    return out.dropna()


def tc_from_best_track(case: Case, run_dirs: list[Path], fixes, half_lat: float = 0.5 * BOX_DEG,
                       edge_margin_deg: float = 6.0) -> tuple[Case, str]:
    """The deepest best-track fix at the valid time of a common (date, case step), inside the files' domain and at
    least ``edge_margin_deg`` from a regional box edge (a storm at the edge loses its core, Melissa 23 Nov).
    No event yaml and no tc_structure read needed. Extent: +-half_lat around the fix."""
    from datetime import datetime, timedelta

    import xarray as xr

    keys = sorted(k for k in common_keys(run_dirs) if k[1] == case.step)
    if not keys:
        raise SystemExit(f"eval.cli suite: no date at +{case.step} h common to every run")
    first = prediction_index(run_dirs[0])[keys[0]]
    with xr.open_dataset(first) as ds:
        lat, lon = ds["lat_hres"].values, ((ds["lon_hres"].values + 180.0) % 360.0) - 180.0
    global_file = lat.max() - lat.min() > 170.0
    s, n, w, e = lat.min(), lat.max(), lon.min(), lon.max()
    m = 0.0 if global_file else edge_margin_deg
    valid = {(datetime.strptime(d, "%Y%m%d") + timedelta(hours=st)).strftime("%Y%m%d%H"): d for d, st in keys}
    f = fixes[fixes.time.astype(str).isin(valid)]
    if not global_file:
        f = f[(f.lat >= s + m) & (f.lat <= n - m) & (f.lon >= w + m) & (f.lon <= e - m)]
    if f.empty:
        raise SystemExit(f"eval.cli suite: no best-track fix at +{case.step} h valid times inside the files")
    r = f.loc[f.pmin_hpa.idxmin()]
    half_lon = half_lat / max(0.3, np.cos(np.deg2rad(float(r.lat))))
    out = Case(name=str(r.event).split("_")[0], date=valid[str(r.time)], step=case.step, variable=case.variable,
               vmax=case.vmax, event=str(r.event),
               extent=[float(r.lon) - half_lon, float(r.lon) + half_lon, float(r.lat) - half_lat, float(r.lat) + half_lat])
    return out, (f"deepest best-track fix at +{case.step} h ({float(r.pmin_hpa):.0f} hPa, {r.event}, "
                 f"{f.event.nunique()} storms in the files)")
