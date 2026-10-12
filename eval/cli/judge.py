"""``eval.cli judge``: run the run-judging protocol on a run and its reference, write the read.

The protocol is docs ``skills/downscaling-evaluation/references/run-judging-protocol.md``
(standing rule 18): one fixed instrument, the run and its reference on the same cases, and a
read in the owner's order 1) physical structure 2) spectra 3) extremes 4) scores. This command:

1. runs the protocol's evaluators on both prediction directories, each into
   ``<output-dir>/<label>/evaluators/<name>/`` (finished evaluators are skipped on a rerun):
   ``shape`` (on one box both runs hold), ``tc_structure``, ``texture``, ``wind_extremes``,
   ``box_spectra``, ``probabilistic``, and ``precip_dist`` when the run predicts ``tp``;
2. draws the fixed ``zoom_maps``: the Alps box (43-49 N, 4-16 E) on the Alps case and the storm
   box (12 degrees around the truth's msl minimum) on the storm case, the run and the reference
   side by side, wind10m and 2t, value and fine part;
3. writes ``read.md``, a ``## Run read:`` section with the header (reference, instrument,
   sampler read from the files) and the four parts filled with the numbers above, or
   "Not available: <reason>" where an evaluator could not run. The verdict lines are left for
   the reader (marked TODO). When the docs repo is found (``$DOCS_ROOT`` or ``~/dev/docs``),
   ``scripts/check_run_read.py`` is run on it.

It predicts nothing: both directories must already hold ``predictions_<date>_step<NNN>.nc``.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from eval.cli._common import Command, add_common_args

LOG = logging.getLogger(__name__)

SUMMARY = "Judge a run against its reference with the run-judging protocol and write the read."

PROTOCOL_EVALUATORS = ["shape", "tc_structure", "texture", "wind_extremes", "box_spectra",
                       "probabilistic"]
MAP_VARIABLES = ["wind10m", "2t"]
MAP_FIELDS = ["value", "fine"]
SPECTRA_TABLE_VARS = ["10u", "2t"]
SCORE_VARS = ["10u", "10v", "10ff", "2t", "msl", "tp"]
STORM_HALF_DEG = 6.0


def register(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser("judge", help=SUMMARY, description=__doc__.split("\n\n")[0],
                              formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--run", required=True, metavar="LABEL=DIR",
                   help="The run judged: a label and its predictions directory.")
    p.add_argument("--reference", required=True, metavar="LABEL=DIR",
                   help="Its reference on the same instrument (donor, parent or control).")
    p.add_argument("--output-dir", required=True, help="Where the read and every output go.")
    p.add_argument("--step", type=int, default=24, help="Lead time in hours (protocol: 24).")
    p.add_argument("--dates", default=None,
                   help="Comma-separated case dates (default: every date both directories hold).")
    p.add_argument("--member", type=int, default=1, help="Member drawn on the maps (default 1).")
    p.add_argument("--alps-date", default="20230226", help="Case of the Alps maps (default 20230226).")
    p.add_argument("--storm", default="franklin:20230829", metavar="EVENT:DATE",
                   help="Storm box of the maps: a tc event and its date (default franklin:20230829).")
    p.add_argument("--shape-box", default="auto",
                   help="Box of the shape statistics: auto (north_atlantic when the files hold it, "
                        "else tc_atlantic), a name of eval/evaluators/shape/boxes.py, or lat_min,lat_max,lon_min,lon_max.")
    p.add_argument("--score-domain", default="europe",
                   help="Domain of the probabilistic scores quoted in the read (default europe).")
    p.add_argument("--skip", default="",
                   help="Comma-separated evaluators (or 'zoom_maps') not to run; the read says so.")
    p.add_argument("--weights", default=None,
                   help="Weights of the two runs, e.g. 'EMA-10000 / raw' (not recorded in the files).")
    p.add_argument("--overwrite", action="store_true", default=False,
                   help="Re-run evaluators that already finished.")
    p.add_argument("--opt", action="append", default=None, metavar="BLOCK.KEY=VALUE",
                   help="Set one key of a lane block, as in `evaluate` (repeatable).")
    return p


# --------------------------------------------------------------------------- helpers

def parse_label(text: str, flag: str) -> tuple[str, Path]:
    label, sep, path = text.partition("=")
    if not sep or not label or not path:
        raise SystemExit(f"{flag} {text!r}: expected LABEL=DIR")
    return label.strip(), Path(path).expanduser()


def file_facts(pred_dir: Path, step: int) -> dict:
    """Dates, members, grid size and the recorded settings of a predictions directory."""
    import netCDF4

    from eval.discovery.predictions import find_predictions
    from eval.evaluators.box_spectra.runner import _decode_states

    files = [p for p in find_predictions(pred_dir) if p.step == step]
    facts = {"dates": sorted({p.date for p in files}), "n_files": len(files)}
    if not files:
        return facts
    with netCDF4.Dataset(str(files[0].path)) as d:
        facts["n_points"] = len(d.dimensions["grid_point_hres"])
        facts["n_members"] = len(d.dimensions["ensemble_member"])
        facts["states"] = _decode_states(d["weather_state"]) if "weather_state" in d.variables else []
        lat = d["lat_hres"][:]
        lon = d["lon_hres"][:]
        facts["lat_range"] = [float(lat.min()), float(lat.max())]
        facts["lon_range"] = [float(lon.min()), float(lon.max())]
        for key in ("checkpoint_id", "sampling_config_json", "member_ids"):
            if key in d.ncattrs():
                facts[key] = str(d.getncattr(key))
    return facts


def choose_shape_box(spec: str, facts: dict) -> str:
    """The shape box: north_atlantic when the files hold it, else the epic's tc_atlantic."""
    if spec != "auto":
        return spec
    from eval.shared.regions import NORTH_ATLANTIC

    la0, la1, lo0, lo1 = NORTH_ATLANTIC
    lat = facts.get("lat_range", [0, 0])
    lon = facts.get("lon_range", [0, 0])
    lon_n = sorted(((v + 180.0) % 360.0) - 180.0 for v in lon)
    holds_lat = lat[0] <= la0 and lat[1] >= la1
    holds_lon = (lon[1] - lon[0] > 300.0) or (lon_n[0] <= lo0 and lon_n[1] >= lo1)
    return "north_atlantic" if holds_lat and holds_lon else "tc_atlantic"


def sampler_line(facts: dict) -> str:
    raw = facts.get("sampling_config_json")
    if not raw:
        return "not recorded in the files"
    try:
        cfg = json.loads(raw)
    except ValueError:
        return raw[:200]
    keys = ("num_steps", "steps", "S_churn", "churn", "S_noise", "sigma_max", "sigma_min", "rho", "sampler")
    picked = {k: cfg[k] for k in keys if k in cfg}
    return json.dumps(picked or cfg, sort_keys=True)[:300]


def storm_centre(pred_dir: Path, date: str, step: int, member: int, event: str):
    """(lat, lon, pmin hPa) of the truth's msl minimum inside the event box, or None."""
    import netCDF4
    import numpy as np

    from eval.evaluators.tc.core.events import EVENTS

    path = pred_dir / f"predictions_{date}_step{step:03d}.nc"
    if not path.exists() or event not in EVENTS:
        return None
    bb = EVENTS[event].bbox
    from eval.evaluators.box_spectra.runner import _decode_states

    with netCDF4.Dataset(str(path)) as d:
        states = _decode_states(d["weather_state"])
        if "msl" not in states:
            return None
        lat = np.asarray(d["lat_hres"][:], dtype=float)
        lon = ((np.asarray(d["lon_hres"][:], dtype=float) + 180.0) % 360.0) - 180.0
        m = (lat >= bb.south) & (lat <= bb.north) & (lon >= bb.west) & (lon <= bb.east)
        if not m.any():
            return None
        v = d["y"]
        key = []
        for dim in v.dimensions:
            if dim == "sample":
                key.append(0)
            elif dim == "ensemble_member":
                ids = [int(x) for x in d["ensemble_member"][:]] if "ensemble_member" in d.variables \
                    else list(range(1, len(d.dimensions["ensemble_member"]) + 1))
                key.append(ids.index(member) if member in ids else 0)
            elif dim == "weather_state":
                key.append(states.index("msl"))
            else:
                key.append(slice(None))
        msl = np.asarray(v[tuple(key)], dtype=float).reshape(-1)
    i = np.flatnonzero(m)[int(np.argmin(msl[m]))]
    p = msl[i] / 100.0 if msl[i] > 2000.0 else msl[i]
    return float(lat[i]), float(lon[i]), float(p)


# --------------------------------------------------------------------------- running

def lane_for_arm(lane_config: dict, *, step: int, dates: list[str], shape_box: str,
                 states: list[str]) -> dict:
    """The lane configuration with the protocol's settings in the evaluator blocks."""
    cfg = copy.deepcopy(lane_config)

    def block(name):
        cfg[name] = dict(cfg.get(name) or {})
        return cfg[name]

    for name in PROTOCOL_EVALUATORS + ["precip_dist"]:
        b = block(name)
        b["steps"] = [step]
        if dates:
            b["dates"] = list(dates)
    block("shape")["box"] = shape_box
    score_vars = [v for v in SCORE_VARS if v in states or (v == "10ff" and "10u" in states)]
    block("probabilistic").setdefault("weather_states", score_vars)
    probabilistic = block("probabilistic")
    domains = list(probabilistic.get("domains") or ["n.hem", "tropics", "s.hem", "europe"])
    if "europe" not in domains:
        domains.append("europe")
    probabilistic["domains"] = domains
    return cfg


def run_evaluators(pred_dir: Path, lane_config: dict, names: list[str], out_dir: Path,
                   label: str, overwrite: bool) -> dict[str, str]:
    """Run each evaluator on its own, so one failure leaves the others; returns name -> status."""
    from eval.cli.evaluate import _run_evaluators

    status = {}
    for name in names:
        try:
            _run_evaluators(pred_dir, lane_config, [name], out_dir, overwrite=overwrite,
                            run_label=label)
            status[name] = "ran"
        except Exception as exc:  # noqa: BLE001 - reported in the read
            LOG.error("judge: %s on %s failed: %s", name, label, exc)
            causes = [ln[2:] for ln in str(exc).splitlines() if ln.startswith("- ")]
            status[name] = f"failed: {(causes[0] if causes else str(exc) or type(exc).__name__)[:300]}"
    return status


def draw_maps(runs: dict[str, Path], out_dir: Path, *, date: str, step: int, member: int,
              extent: list[float], tag: str) -> tuple[list[str], str | None]:
    """zoom_maps for one box: every variable and field; returns (png paths, failure)."""
    from eval.evaluators.zoom_maps.core.plot_member_wind_maps import build_arg_parser, run as zm_run

    missing = [lab for lab, d in runs.items() if not (d / f"predictions_{date}_step{step:03d}.nc").exists()]
    if missing:
        return [], f"no predictions_{date}_step{step:03d}.nc for {', '.join(missing)}"
    lat_min, lat_max, lon_min, lon_max = extent
    box_dir = out_dir / tag
    box_dir.mkdir(parents=True, exist_ok=True)
    before = set(box_dir.glob("*.png"))
    try:
        for var in MAP_VARIABLES:
            for field in MAP_FIELDS:
                argv = [a for lab, d in runs.items() for a in ("--run", f"{lab}={d}")]
                argv += ["--date", date, "--step", str(step), "--member", str(member),
                         "--extent", str(lon_min), str(lon_max), str(lat_min), str(lat_max),
                         "--region-tag", tag, "--variable", var, "--field", field,
                         "--output-dir", str(box_dir)]
                zm_run(build_arg_parser().parse_args(argv))
    except Exception as exc:  # noqa: BLE001 - reported in the read
        return [], f"zoom_maps failed: {exc}"
    pngs = sorted(set(box_dir.glob("*.png")) - before) or sorted(box_dir.glob("*.png"))
    return [str(p) for p in pngs], None


# --------------------------------------------------------------------------- the read

def _fmt(v, nd=3):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if f != f else f"{f:.{nd}f}"


def _shape_rows(results: Path, step: int) -> dict:
    """{(var, field, statistic): (mean, se)} for the mid band, pooled window."""
    path = results / "summary.json"
    if not path.exists():
        return {}
    rows = json.loads(path.read_text())["rows"]
    windows = {r["window"] for r in rows}
    pooled = "all" if "all" in windows else ("both" if "both" in windows else sorted(windows)[0])
    return {(r["var"], r["field"], r["statistic"]): (r["mean"], r["se_date"]) for r in rows
            if r["window"] == pooled and r["band"] == "mid" and int(r["step"]) == step}


def part_structure(arms: dict, maps: dict, shape_box: str, step: int) -> list[str]:
    run, ref = arms["run"], arms["reference"]
    lines = ["### 1. Physical structure"]
    for tag, (pngs, why) in maps.items():
        if why:
            lines.append(f"- zoom_maps {tag}: Not available: {why}")
        else:
            folder = Path(pngs[0]).parent
            lines.append(f"- zoom_maps {tag}: `{folder}`, {len(pngs)} panels (input, truth, {run['label']}, "
                         f"{ref['label']}; wind10m and 2t; value and fine part; member {arms['member']}; one "
                         "colour scale per variable and field). By eye: TODO")
    a, b = _shape_rows(run["eval"] / "shape", step), _shape_rows(ref["eval"] / "shape", step)
    if a and b:
        lines += [f"- Shape, elongation of the residual's features (box {shape_box}, 40-109 km band, date-clustered "
                  "standard error in brackets):", "",
                  f"| var | statistic | {run['label']} | {ref['label']} | truth |", "|---|---|---|---|---|"]
        for var in ("10u", "10v"):
            for st, name in (("elong_frac_gt3", "elongated fraction (> 3:1)"),
                             ("ell_ratio_median", "autocorrelation axis ratio")):
                if (var, "draw", st) not in a:
                    continue
                cell = lambda d, f: f"{_fmt(d[(var, f, st)][0])} ({_fmt(d[(var, f, st)][1])})" \
                    if (var, f, st) in d else "n/a"  # noqa: E731
                lines.append(f"| {var} | {name} | {cell(a, 'draw')} | {cell(b, 'draw')} | {cell(a, 'truth')} |")
        lines.append("")
    else:
        lines.append(f"- Shape (elongation): Not available: {run['status'].get('shape', 'not run')} / "
                     f"{ref['status'].get('shape', 'not run')}")
    lines += part_rotation(run, ref, step)
    for name in ("texture", "wind_extremes"):
        st = run["status"].get(name, "not run")
        lines.append(f"- {name}: `{run['eval'] / name}` and `{ref['eval'] / name}` ({st})")
    lines.append("- Verdict on structure: TODO")
    return lines


def _tc_summary(results: Path) -> dict:
    path = results / "summary.json"
    return json.loads(path.read_text()).get("events", {}) if path.exists() else {}


def storms_found(a: dict, b: dict, band: str) -> list[str]:
    """Events where tc_structure found the storm in the run, the reference and the truth."""
    def found(s, e, field):
        return bool(s.get(e, {}).get(band, {}).get("fields", {}).get(field))
    return sorted(e for e in a if found(a, e, "model") and found(a, e, "truth") and found(b, e, "model"))


def part_rotation(run: dict, ref: dict, step: int) -> list[str]:
    a, b = _tc_summary(run["eval"] / "tc_structure"), _tc_summary(ref["eval"] / "tc_structure")
    band = str(step)
    events = storms_found(a, b, band)
    if not events:
        why = run["status"].get("tc_structure", "not run")
        return [f"- Rotation (tc_structure vorticity): Not available: no storm found at +{step} h in both runs ({why})"]
    lines = ["- Rotation (tc_structure: mean vorticity inside 50 / 100 / 200 km in 1e-5 s-1, radius of maximum "
             "wind in km):", "", f"| storm | {run['label']} | {ref['label']} | truth |", "|---|---|---|---|"]
    for e in events:
        def cell(summary, field):
            f = summary[e][band]["fields"].get(field) or {}
            z = [_fmt(1e5 * float(f[k]["value"]), 1) if k in f and f[k]["value"] is not None else "n/a"
                 for k in ("zeta50_s", "zeta100_s", "zeta200_s")]
            rmw = _fmt(f["rmw_km"]["value"], 0) if "rmw_km" in f else "n/a"
            return f"{' / '.join(z)}; RMW {rmw}"
        lines.append(f"| {e} | {cell(a, 'model')} | {cell(b, 'model')} | {cell(a, 'truth')} |")
    lines.append("")
    return lines


def part_spectra(arms: dict) -> list[str]:
    from eval.evaluators.box_spectra.runner import read_bins

    run, ref = arms["run"], arms["reference"]
    pa, pb = run["eval"] / "box_spectra" / "bins.csv", ref["eval"] / "box_spectra" / "bins.csv"
    lines = ["### 2. Spectra"]
    if not (pa.exists() and pb.exists()):
        return lines + [f"- Amplitude per bin, full field and residual: Not available: "
                        f"{run['status'].get('box_spectra', 'not run')} / {ref['status'].get('box_spectra', 'not run')}"]
    meta = json.loads((run["eval"] / "box_spectra" / "run_meta.json").read_text())
    rows_a = [r for r in read_bins(pa) if r["scope"] == "all"]
    rows_b = {(r["variable"], r["bin_centre_km"]): r for r in read_bins(pb) if r["scope"] == "all"}
    lines.append(f"Amplitude ratio to the truth per bin (rule 14), full field and residual, box {meta['box']}, "
                 f"{meta['method']} sampling; aligned part = coherence x full-field ratio. Target 1 from 125 to 31 km; "
                 f"below 31 km model against model only. All variables and per-date rows: `{pa}`, `{pb}`.")
    for var in SPECTRA_TABLE_VARS:
        sub = [r for r in rows_a if r["variable"] == var and 15.0 <= float(r["bin_centre_km"]) <= 600.0]
        if not sub:
            continue
        lines += ["", f"{var} (draws {sub[0]['n_draws']}):", "",
                  f"| km | {run['label']} full field | {run['label']} residual | {ref['label']} full field | "
                  f"{ref['label']} residual | {run['label']} aligned | {ref['label']} aligned | input/truth |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in sub:
            o = rows_b.get((var, r["bin_centre_km"]), {})
            lines.append(f"| {float(r['bin_centre_km']):.1f} | {_fmt(r['ratio_full'])} | {_fmt(r['ratio_resid'])} | "
                         f"{_fmt(o.get('ratio_full'))} | {_fmt(o.get('ratio_resid'))} | {_fmt(r['aligned'])} | "
                         f"{_fmt(o.get('aligned'))} | {_fmt(r['ratio_interp_to_truth'])} |")
    lines += ["", f"Figures: `{run['eval'] / 'box_spectra' / 'box_spectra_ladder.png'}` and the reference's.",
              "- Verdict on spectra: TODO"]
    return lines


def part_extremes(arms: dict, step: int) -> list[str]:
    run, ref = arms["run"], arms["reference"]
    a, b = _tc_summary(run["eval"] / "tc_structure"), _tc_summary(ref["eval"] / "tc_structure")
    band = str(step)
    lines = ["### 3. Extremes"]
    events = storms_found(a, b, band)
    if events:
        lines += ["Peak wind (maximum 10 m wind within 300 km, m/s) and minimum msl (hPa) per storm, mean over "
                  "members (standard error):", "",
                  f"| storm | {run['label']} | {ref['label']} | truth |", "|---|---|---|---|"]
        for e in events:
            def cell(s, field):
                f = s[e][band]["fields"].get(field) or {}
                w, p = f.get("maxwind300_ms", {}), f.get("pmin_hpa", {})
                return (f"{_fmt(w.get('value'), 1)} ({_fmt(w.get('se'), 1)}) m/s, "
                        f"{_fmt(p.get('value'), 1)} ({_fmt(p.get('se'), 1)}) hPa")
            lines.append(f"| {e} | {cell(a, 'model')} | {cell(b, 'model')} | {cell(a, 'truth')} |")
        lines.append("")
    else:
        lines.append(f"- Peak wind and minimum msl per storm: Not available: no storm found at +{step} h in both "
                     f"runs ({run['status'].get('tc_structure', 'not run')})")
    if "precip_dist" in run["status"]:
        lines.append(f"- Rain tail (tp p99, p99.9, maximum): `{run['eval'] / 'precip_dist'}` "
                     f"({run['status']['precip_dist']})")
    lines.append("- Verdict on extremes: TODO")
    return lines


def _scores(results: Path, step: int, domain: str) -> dict:
    import csv

    path = results / "summary_by_lead.csv"
    if not path.exists():
        return {}
    with open(path) as fh:
        return {(r["weather_state"], r["metric"]): r["mean"] for r in csv.DictReader(fh)
                if int(r["step"]) == step and r["domain"] == domain}


def part_scores(arms: dict, step: int, domain: str) -> list[str]:
    run, ref = arms["run"], arms["reference"]
    a = _scores(run["eval"] / "probabilistic", step, domain)
    b = _scores(ref["eval"] / "probabilistic", step, domain)
    lines = ["### 4. Scores"]
    if not (a and b):
        return lines + [f"- Fair CRPS, spread, ensemble-mean RMSE: Not available: "
                        f"{run['status'].get('probabilistic', 'not run')} / {ref['status'].get('probabilistic', 'not run')}"
                        f" (domain {domain})"]
    lines += [f"Domain {domain}, +{step} h; spread = member standard deviation (ddof 1); change = run against "
              "reference (negative CRPS and RMSE changes are better):", "",
              f"| var | metric | {run['label']} | {ref['label']} | change |", "|---|---|---|---|---|"]
    names = {"fcrps": "fair CRPS", "spread": "spread", "rmse_ens_mean": "ensemble-mean RMSE"}
    for var in sorted({v for v, _ in a}):
        for m, name in names.items():
            if (var, m) not in a or (var, m) not in b:
                continue
            x, y = float(a[(var, m)]), float(b[(var, m)])
            ch = f"{100 * (x - y) / y:+.1f} %" if y else "n/a"
            lines.append(f"| {var} | {name} | {_fmt(x, 4)} | {_fmt(y, 4)} | {ch} |")
    lines += ["", "- Verdict on scores (never outranks a loss in parts 1 or 2): TODO"]
    return lines


def compose_read(arms: dict, *, step: int, dates: list[str], shape_box: str, maps: dict,
                 domain: str, weights: str | None, out_dir: Path, commit: str) -> str:
    run, ref = arms["run"], arms["reference"]
    fr, ff = run["facts"], ref["facts"]
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ckpt = fr.get("checkpoint_id", "checkpoint not recorded")
    head = [f"## Run read: {run['label']} ({ckpt}) +{step} h ({today})", "",
            f"Reference: {ref['label']} (`{ref['dir']}`, {ff.get('checkpoint_id', 'checkpoint not recorded')}). "
            f"Instrument: cases {', '.join(dates)} at +{step} h, {fr.get('n_members', '?')} members per case "
            f"(reference {ff.get('n_members', '?')}), {fr.get('n_points', '?')} points per file "
            f"(reference {ff.get('n_points', '?')}). Run: `{run['dir']}`.",
            f"Sampler from the files: run {sampler_line(fr)}; reference {sampler_line(ff)}. "
            f"Weights: {weights or 'TODO: state EMA or raw for both (not recorded in the files)'}."]
    warnings = []
    if fr.get("n_points") != ff.get("n_points"):
        warnings.append("the two runs have different point sets (for example global against Europe box): "
                        "evaluators that pool over the whole file are not on one instrument")
    if fr.get("sampling_config_json") and ff.get("sampling_config_json") \
            and fr["sampling_config_json"] != ff["sampling_config_json"]:
        warnings.append("the sampler settings recorded in the files differ (never compare unequal samplers)")
    if fr.get("n_members") != ff.get("n_members"):
        warnings.append("the member counts differ")
    for w in warnings:
        head.append(f"WARNING: {w}.")
    head += [f"Written by `eval.cli judge` (downscaling-tools {commit}); every output is under `{out_dir}`.", ""]
    body = (part_structure(arms, maps, shape_box, step) + [""] + part_spectra(arms) + [""]
            + part_extremes(arms, step) + [""] + part_scores(arms, step, domain))
    return "\n".join(head + body) + "\n"


def run_checker(read_path: Path) -> str | None:
    for root in (os.environ.get("DOCS_ROOT"), str(Path.home() / "dev" / "docs")):
        if root and (Path(root) / "scripts" / "check_run_read.py").exists():
            res = subprocess.run([sys.executable, str(Path(root) / "scripts" / "check_run_read.py"),
                                  str(read_path)], capture_output=True, text=True)
            return (res.stdout + res.stderr).strip()
    return None


# --------------------------------------------------------------------------- command

def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def run(args: argparse.Namespace, session) -> None:
    run_label, run_dir = parse_label(args.run, "--run")
    ref_label, ref_dir = parse_label(args.reference, "--reference")
    if run_label == ref_label:
        raise SystemExit("--run and --reference need different labels")
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    step = int(args.step)
    skip = {s.strip() for s in (args.skip or "").split(",") if s.strip()}

    facts = {lab: file_facts(d, step) for lab, d in ((run_label, run_dir), (ref_label, ref_dir))}
    for lab, d in ((run_label, run_dir), (ref_label, ref_dir)):
        if not facts[lab]["n_files"]:
            raise SystemExit(f"judge: no predictions_*_step{step:03d}.nc under {d} ({lab})")
    common = sorted(set(facts[run_label]["dates"]) & set(facts[ref_label]["dates"]))
    dates = [d.strip() for d in args.dates.split(",")] if args.dates else common
    if not set(dates) <= set(common):
        raise SystemExit(f"judge: dates {sorted(set(dates) - set(common))} are not in both directories")
    shape_box = choose_shape_box(args.shape_box, facts[run_label])
    if args.shape_box == "auto" and shape_box != choose_shape_box("auto", facts[ref_label]):
        LOG.warning("judge: the reference does not hold the %s box", shape_box)

    names = [n for n in PROTOCOL_EVALUATORS if n not in skip]
    if "tp" in facts[run_label].get("states", []) and "precip_dist" not in skip:
        names.append("precip_dist")
    arms = {"member": args.member}
    for role, lab, d in (("run", run_label, run_dir), ("reference", ref_label, ref_dir)):
        lane = lane_for_arm(session.lane_config, step=step, dates=dates, shape_box=shape_box,
                            states=facts[lab].get("states", []))
        arm_out = out_dir / lab
        status = run_evaluators(d, lane, names, arm_out, lab, args.overwrite)
        for n in skip & set(PROTOCOL_EVALUATORS):
            status[n] = "skipped on the command line"
        arms[role] = {"label": lab, "dir": d, "facts": facts[lab], "status": status,
                      "eval": arm_out / "evaluators"}

    maps: dict[str, tuple[list[str], str | None]] = {}
    pair = {run_label: run_dir, ref_label: ref_dir}
    if "zoom_maps" in skip:
        maps["Alps"] = ([], "skipped on the command line")
    else:
        from eval.shared.regions import ALPS

        maps["Alps 43-49 N, 4-16 E"] = draw_maps(pair, out_dir / "zoom_maps", date=args.alps_date, step=step,
                                                 member=args.member, extent=ALPS, tag="alps")
        event, _, sdate = args.storm.partition(":")
        c = storm_centre(run_dir, sdate, step, args.member, event)
        if c is None:
            maps[f"storm {event} {sdate}"] = ([], f"no truth msl minimum for {event} in "
                                                  f"predictions_{sdate}_step{step:03d}.nc of {run_label}")
        else:
            ext = [c[0] - STORM_HALF_DEG, c[0] + STORM_HALF_DEG, c[1] - STORM_HALF_DEG, c[1] + STORM_HALF_DEG]
            maps[f"storm {event} {sdate} (truth centre {c[0]:.1f} N, {c[1]:.1f} E, {c[2]:.0f} hPa)"] = draw_maps(
                pair, out_dir / "zoom_maps", date=sdate, step=step, member=args.member, extent=ext,
                tag=f"storm_{event}")

    text = compose_read(arms, step=step, dates=dates, shape_box=shape_box, maps=maps,
                        domain=args.score_domain, weights=args.weights, out_dir=out_dir,
                        commit=_git_commit())
    read_path = out_dir / "read.md"
    read_path.write_text(text)
    (out_dir / "judge_status.json").write_text(json.dumps(
        {"run": arms["run"]["status"], "reference": arms["reference"]["status"],
         "maps": {k: (v[1] or "ok") for k, v in maps.items()}, "shape_box": shape_box, "dates": dates},
        indent=1) + "\n")
    print(f"read written to {read_path}")
    checked = run_checker(read_path)
    print(checked if checked is not None else
          "docs repo not found ($DOCS_ROOT or ~/dev/docs): run "
          f"`python3 scripts/check_run_read.py {read_path}` from the docs root")


COMMANDS = (Command("judge", "pipeline", SUMMARY, register, run, needs_lane=True),)
