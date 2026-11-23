"""Build the classic suite pages of one or more runs (see ``eval.suite`` for the page list).

Light pages are computed from the prediction folders (spectra reads, maps, tp distribution). Pages that need
an observation cache or a storm read take an existing read: ``obs`` = obs_points results folders (page 06),
``tc`` = tc_structure results folders plus a tc_obs cache (page 09). A page whose inputs are missing is
skipped and listed as such in ``SUITE.md``; the suite never invents a stand-in.
"""
from __future__ import annotations

import json
import logging
import traceback
from argparse import Namespace
from pathlib import Path

import numpy as np

from eval.suite.cases import CaseList, prediction_index, resolve_date
from eval.suite.stitch import stitch

LOG = logging.getLogger(__name__)
# variables whose colour range is fitted to the map box (1st-99th percentile); wind keeps 0 at the bottom and fits
# its top to the box; tp keeps its fixed rain classes
FITTED_RANGE = ("2t", "msl", "t_850", "z_500")
PAGES = ("spectra", "tc_members", "rain_members", "areas", "zooms", "scores", "stills", "tp_dist", "tc_dist")


def _zoom(runs: dict[str, Path], case, date: str, out: Path, *, members: str | None, cols: int = 5,
          cells: bool = False) -> list[Path]:
    """zoom_maps of every source (member grids, or member 1 alone with members=None); [input, target, model...]."""
    from eval.evaluators.zoom_maps.core.plot_member_wind_maps import build_arg_parser
    from eval.evaluators.zoom_maps.core.plot_member_wind_maps import run as zoom_run

    argv = ["--date", date, "--step", str(case.step), "--grid-cols", str(cols),
            "--output-dir", str(out), "--variable", case.variable, "--region-tag", case.name,
            "--title", "eefo=Input ({lres})", "--title", "enfo=Target ({hres})"]
    for k, d in runs.items():
        argv += ["--run", f"{k}={d}", "--title", f"{k}={k} ({{hres}})"]
    if case.extent:
        argv += ["--extent", *[str(v) for v in case.extent]]
    if members:
        argv += ["--members", members]
    if case.vmax is not None:
        argv += ["--vmax", str(case.vmax)]
    if case.vmin is not None:
        argv += ["--vmin", str(case.vmin)]
    argv += ["--dpi", "200"]
    if cells:   # opt-in only: the owner rejected grid cells as the look of the pages (23 Nov 09:45, "awful")
        argv += ["--cells"]
    from eval.evaluators.zoom_maps.core.plot_member_wind_maps import VARIABLES as _ZV
    from eval.suite.cases import field_max

    state = _ZV[case.variable]["states"][0]
    first = prediction_index(next(iter(runs.values())))[(date, case.step)]
    if not np.isfinite(field_max(first, state, "y", case.extent)):
        argv += ["--no-truth"]   # the target carries no such field here (e.g. tp of an analysis): no target row
        LOG.warning("suite: no target %s in %s; the target row is left out", state, first.name)
    if case.variable in FITTED_RANGE and case.vmin is None and case.vmax is None and case.extent:
        # colour range fitted to the box (owner 23 Nov): 1st-99th percentile of the target (else input) member mean
        from eval.suite.cases import field_stats

        st = field_stats(first, state, "y", case.extent) or field_stats(first, state, "x", case.extent)
        if st is not None:
            sc, off = _ZV[case.variable]["scale"], _ZV[case.variable]["offset"]
            lo, hi = st["p01"] * sc + off, st["p99"] * sc + off
            pad = 0.05 * max(hi - lo, 1e-6)
            argv += ["--vmin", f"{lo - pad:.2f}", "--vmax", f"{hi + pad:.2f}"]
    elif case.variable == "wind10m" and case.vmax is None and case.extent:
        # wind keeps 0 at the bottom; the top is fitted to the strongest target or model member in the box
        from eval.suite.cases import speed_max

        top = speed_max(first, case.extent)
        if np.isfinite(top):
            argv += ["--vmax", f"{max(15.0, 5.0 * np.ceil(1.02 * top / 5.0)):g}"]
    args = build_arg_parser().parse_args(argv)
    w, e, s_, n_ = args.extent
    if args.proj_lon is None:
        args.proj_lon = 0.5 * (w + e)
    if args.proj_lat is None:
        args.proj_lat = 0.5 * (s_ + n_)
    zoom_run(args)
    from eval.evaluators.zoom_maps.core.plot_member_wind_maps import VARIABLES

    token = VARIABLES[case.variable]["token"]
    found = []
    for key in ["eefo", "enfo", *runs]:
        tag = "members*" if members else "n001"
        found += sorted(out.glob(f"{key}_{token}_init{date}_{tag}_{case.name}_f{case.step:03d}.png"))[:1]
    return found


def _not_tighter_than_close(case, runs: dict[str, Path], date: str, close_points: int):
    """The case with its extent widened to the close zoom level when it is tighter (owner 23 Nov 10:19, rule 53:
    nothing more zoomed in than the close level; fixed in output grid points, so it depends on the lane)."""
    import dataclasses

    import xarray as xr

    from eval.suite.simple_maps import spacing_km, widen_to_close

    if not case.extent:
        return case
    f = prediction_index(next(iter(runs.values())))[(date, case.step)]
    with xr.open_dataset(f) as ds:
        dx = spacing_km(ds["lat_hres"].values, ((ds["lon_hres"].values + 180.0) % 360.0) - 180.0)
    ext, widened = widen_to_close(case.extent, close_points * dx)
    if widened:
        LOG.info("suite: %s box %s widened to the close zoom level %s (%d x %.1f km)", case.name, case.extent,
                 [round(v, 2) for v in ext], close_points, dx)
    return dataclasses.replace(case, extent=ext)


def _stills(runs: dict[str, Path], case, date: str, out: Path, grid: str | None) -> list[Path]:
    """map_video stills: input | target | first model, then the other models' predictions."""
    from eval.tools.map_video.__main__ import build_arg_parser, render

    pngs = []
    for i, (label, d) in enumerate(runs.items()):
        f = prediction_index(d)[(date, case.step)]
        sd = out / label
        argv = [str(f), "--stills-dir", str(sd), "--variable", case.variable, "--label", label,
                "--sources", "input,truth,pred" if i == 0 else "pred", "--native-input"]
        if case.extent:
            argv += ["--extent", *[str(v) for v in case.extent]]
        if grid:
            argv += ["--grid", grid]
        render(build_arg_parser().parse_args(argv))
        pngs += sorted(sd.glob("*.png"))[:1]
    return pngs


def build_suite(runs: dict[str, Path], cases: CaseList, out: Path, *, obs: dict[str, Path] | None = None,
                tc: dict[str, Path] | None = None, tc_cache: Path | None = None, only=None,
                maps_all_runs: bool = False, cells: bool = False, ibtracs_csv: Path | None = None) -> dict:
    out = Path(out)
    work = out / "work"
    out.mkdir(parents=True, exist_ok=True)
    pages = list(only or PAGES)
    names = list(runs)
    # map pages (03, 04, 05, 07) show ONE model, the first run (owner 23 Nov 09:07: "just 300"); curves pages all runs
    map_runs = dict(runs) if maps_all_runs else dict(list(runs.items())[:1])
    record: dict = {"case_list": cases.name, "runs": {k: str(v) for k, v in runs.items()}, "pages": []}

    def page(num: str, title: str, fn):
        if title.split(":")[0] not in pages:
            return
        try:
            paths, note = fn()
            record["pages"].append({"page": num, "title": title, "files": [str(p) for p in paths], "note": note})
            LOG.info("suite page %s done: %s", num, note)
        except Exception as exc:  # one page failing must not lose the others
            LOG.error("suite page %s failed: %s", num, exc)
            record["pages"].append({"page": num, "title": title, "files": [], "note": f"SKIPPED: {exc}",
                                    "trace": traceback.format_exc(limit=3)})

    # 01-02 spectra: exactly the leads of the case list (24 and 72 h), one page each, every run on it
    def spectra(step):
        def go():
            from eval.evaluators.box_spectra.runner import run as box_run
            from eval.suite.pages import spectra_page

            reads = {}
            for k, d in runs.items():
                cfg = {"steps": [step], "variables": list(cases.spectra_variables), "box": cases.spectra_box}
                res = work / "spectra" / f"{k}_L{step:03d}"
                reads[k] = res if (res / "bins.csv").exists() else box_run(d, {}, cfg, output_dir=res, run_label=k)
            files = spectra_page(reads, out / f"{'01' if step == cases.spectra_steps[0] else '02'}_spectra_L{step:03d}",
                                 step=step, box=cases.spectra_box)
            return files, f"{len(reads)} runs at +{step} h"
        return go

    for step in cases.spectra_steps[:2]:
        page("01" if step == cases.spectra_steps[0] else "02", f"spectra: +{step} h", spectra(step))

    def members_page(num, case_in, pick, label):
        def go():
            case = case_in
            if case is None:
                raise ValueError(f"no {label} case in case list {cases.name}")
            if label == "tc_members" and (case.date == "auto" or case.extent is None):
                from eval.suite.cases import best_track_fixes, tc_auto, tc_from_best_track

                if tc:
                    case, how = tc_auto(case, Path(next(iter(tc.values()))) / "cases.csv")
                else:
                    fixes = best_track_fixes(tc_cache, ibtracs_csv)
                    if fixes is None:
                        raise ValueError("the TC case is picked from a tc_structure read (--tc) or from best-track "
                                         "fixes (--tc-cache or --ibtracs-csv)")
                    case, how = tc_from_best_track(case, list(runs.values()), fixes)
                date, _ = resolve_date(case, list(runs.values()), pick="first")
            else:
                date, how = resolve_date(case, list(runs.values()), pick=pick)
                if label == "rain_members" and case.box_deg and case.extent:
                    # a box_deg box centred on the wettest target (else input) point of the search extent
                    from dataclasses import replace

                    from eval.suite.cases import box_around, field_stats

                    f0 = prediction_index(next(iter(runs.values())))[(date, case.step)]
                    st = field_stats(f0, "tp", "y", case.extent) or field_stats(f0, "tp", "x", case.extent)
                    if st is not None:
                        case = replace(case, extent=box_around(st["lat"], st["lon"], case.box_deg))
                        how += f"; {case.box_deg:g}-degree box on its maximum at {st['lat']:.1f}N {st['lon']:.1f}E"
            case = _not_tighter_than_close(case, map_runs, date, cases.zoom_close_points)
            # side by side (owner 23 Nov 09:07): input | target | model, each 5 rows x 2 members
            pngs = _zoom(map_runs, case, date, work / f"zoom_{case.name}", members=cases.members, cols=2, cells=cells)
            dst = stitch([pngs], top_align=True, out=out / f"{num}_{label}_{case.name}_{case.variable}_{date}_L{case.step:03d}.png")
            return [dst], (f"{case.name} {date} +{case.step} h ({how}); side by side: input, target, "
                           f"{', '.join(map_runs)}")
        return go

    page("03", "tc_members: 10 members around a storm", members_page("03", cases.tc, "first", "tc_members"))
    page("04", "rain_members: 10 members on intense rain", members_page("04", cases.rain, "rain", "rain_members"))

    def areas():
        rows, notes = [], []
        for case in cases.areas:
            date, how = resolve_date(case, list(runs.values()), pick="first")
            rows.append(_zoom(map_runs, case, date, work / f"area_{case.name}", members=None, cells=cells))
            notes.append(f"{case.name} {case.variable} {date} +{case.step} h ({how})")
        if not rows:
            raise ValueError(f"no areas in case list {cases.name}")
        return [stitch(rows, out / "05_areas_member1.png")], (f"member 1, input | target | {', '.join(map_runs)}; "
                                                              + "; ".join(notes))

    page("05", "areas: input / target / model by area", areas)

    def zooms():
        # zoom ladder (owner 23 Nov 10:19, rule 53): one page per area, one row per zoom level, simple smooth maps;
        # the close level is the tightest allowed, fixed in output grid points (eval/suite/simple_maps.py)
        from eval.suite.simple_maps import zoom_ladder

        files, notes = [], []
        label, d = next(iter(map_runs.items()))
        for case in cases.areas:
            if not case.extent:
                continue
            date, how = resolve_date(case, list(runs.values()), pick="first")
            f = prediction_index(d)[(date, case.step)]
            rows, info = zoom_ladder(f, case, date, work / f"zooms_{case.name}", model_label=label,
                                     levels=cases.zoom_levels, close_points=cases.zoom_close_points)
            files.append(stitch([[r] for r in rows],
                                out / f"05z_zooms_{case.name}_{case.variable}_{date}_L{case.step:03d}.png"))
            notes.append(f"{case.name} {case.variable} {date} +{case.step} h ({how}); {info}")
        if not files:
            raise ValueError(f"no area with a box in case list {cases.name}")
        return files, f"member 1, input | target | {label}; " + "; ".join(notes)

    page("05z", "zooms: zoom ladder by area", zooms)

    def scores():
        if not obs:
            raise ValueError("no obs_points read given (--obs LABEL=DIR)")
        from eval.suite.pages import scores_page

        files = scores_page(obs, out / "06_scores_stations", variables=cases.score_variables, region=cases.score_region)
        return files, f"obs_points reads of {', '.join(obs)}; region {cases.score_region}"

    page("06", "scores: fair CRPS and spread at stations", scores)

    def stills():
        rows, notes = [], []
        for case in cases.stills:
            date, how = resolve_date(case, list(runs.values()), pick="rain" if case.date == "auto" else "first")
            rows.append(_stills(map_runs, case, date, work / f"stills_{case.name}_{date}", cases.stills_grid))
            notes.append(f"{case.name} {date} +{case.step} h ({how})")
        if not rows:
            raise ValueError(f"no stills in case list {cases.name}")
        return [stitch(rows, out / "07_tp_stills.png")], "member 1; " + "; ".join(notes)

    page("07", "stills: tp maps", stills)

    def tp_dist():
        from eval.suite.pages import tp_distribution_page

        files, info = tp_distribution_page(runs, out / "08_tp_distribution", steps=cases.dist_steps)
        return files, f"leads {cases.dist_steps}; sources {json.dumps(info)}"

    page("08", "tp_dist: tp PDF against input and target", tp_dist)

    def tc_dist():
        if not tc:
            raise ValueError("no tc_structure read given (--tc LABEL=DIR)")
        from eval.suite.pages import tc_distribution_page

        files, info = tc_distribution_page(tc, tc_cache, out / "09_tc_distribution")
        return files, json.dumps(info)

    page("09", "tc_dist: storm pressure and wind against IBTrACS", tc_dist)

    (out / "suite_manifest.json").write_text(json.dumps(record, indent=2))
    lines = [f"# Classic suite: {', '.join(names)} (case list {cases.name})", "",
             "Pages 10 and 11 are the agent's two picks for the case (eval.cli figures), added by hand.", ""]
    for p in record["pages"]:
        files = ", ".join(Path(f).name for f in p["files"] if str(f).endswith(".png")) or "-"
        lines.append(f"- {p['page']} {p['title']}: {files}. {p['note']}")
    (out / "SUITE.md").write_text("\n".join(lines) + "\n")
    return record
