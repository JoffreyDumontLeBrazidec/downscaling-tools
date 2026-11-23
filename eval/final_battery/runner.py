"""Run the final-model battery: the three 17 Nov realism kits, in order, on one member row, then the summary page.

The kits under ``kits/`` are the reference implementations (docs master, support/20261117_realism_{t2m,consistency,wind});
each runs as its own process with ``FB_*`` environment variables that point it at this row. A step that finished
leaves ``<out>/.done/<step>``, so a second call resumes where the first stopped. Every step logs to
``<out>/logs/<step>.log``. A failed step stops the steps after it in the same part; the other parts go on.

Prediction files: ``<pred_root>/pred/<row>_L<LLL>/predictions_<YYYYMMDD>_step<LLL>.nc`` with ``y_pred`` (one draw per
input member), ``x_interp_members``, ``y``, ``x_interp``, ``weather_state`` (the box member-mode predictor layout).
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from eval.final_battery import layout

KITS = Path(__file__).resolve().parent / "kits"
PARTS = ("terrain", "consistency", "wind")
JUPITER_USER_ROOT = "/e/data1/jureap-data/ecmwf/users/jdumont"
ALL_OUTPUTS = ("10u", "10v", "2t", "msl", "sp", "tp", "cp", "2d", "skt", "tcw")
NEEDED = {"terrain": ("2t", "msl", "sp", "10u", "10v"), "consistency": ("10u", "10v", "2t", "msl", "sp", "tp", "cp"),
          "wind": ("10u", "10v", "msl")}

# Zoom sheets of the terrain note (m01-m11) and the consistency member maps (run_jupiter.sbatch), as the 17 Nov runs drew them.
TERRAIN_MAPS = (
    "m01_alps_2t_20230226_L024:alps:20230226:24:2t",
    "m02a_alps_2t_jan_20230101_L024:alps:20230101:24:2t",
    "m02b_alps_2t_inversion_20230212_L048:alps:20230212:48:2t",
    "m03_norway_2t_msl_20230122_L024:norway:20230122:24:2t,msl",
    "m04_iceland_2t_sp_20230212_L024:iceland:20230212:24:2t,sp",
    "m05_sgreenland_2t_20230304_L024:s_greenland:20230304:24:2t",
    "m06_alps_sp_msl_20230226_L024:alps:20230226:24:sp,msl",
    "m07_alps_2t_20230718_L012:alps:20230718:12:2t",
    "m08_corsica_2t_20230718_L012:corsica_sardinia:20230718:12:2t",
    "m09_corsica_2t_20230718_L024:corsica_sardinia:20230718:24:2t",
)
TERRAIN_MAPS_10 = (
    "m10_alps_2d_skt_20230226_L024:alps:20230226:24:2d,skt",
    "m11_alps_skt_2d_20230718_L012:alps:20230718:12:skt,2d",
)
CONSISTENCY_MAPS = (
    "--case", "franklin,20230829,024,ws,low", "--case", "franklin,20230829,024,tp,low", "--case", "franklin,20230829,024,msl,low",
    "--case", "alps,20230226,024,ws,43,49,4,16", "--case", "alps,20230226,024,fine_ws,43,49,4,16",
    "--case", "alps,20230226,024,2t,43,49,4,16", "--case", "front,20231031,024,tp,42,58,-14,10",
    "--case", "front,20231031,024,ws,42,58,-14,10",
    "--seq", "franklin,20230829,ws,1,24,44,-78,-52", "--seq", "front,20231031,tp,1,42,58,-14,10",
)


@dataclass
class Config:
    final_model: str                     # what was declared finished, e.g. "FS-245 100k EMA-2000"
    row: str                             # prediction folder stem, <pred_root>/pred/<row>_L<LLL>
    pred_root: Path
    out: Path
    label: str = ""
    user_root: str = JUPITER_USER_ROOT
    member_stores: str = ""              # default <user_root>/agent-work/20261104-ens-member-box
    datasets: str = ""                   # default <user_root>/datasets
    orog320: str = ""                    # default <user_root>/station_caches/orography_O320_eefh.grib
    leads: tuple[int, ...] = (12, 24, 36, 48, 60, 72)
    wind_leads: tuple[int, ...] = (24, 72)
    parts: tuple[str, ...] = PARTS
    workers: int = 6
    consistency_workers: int = 3         # about 5 GB per worker (17 Nov: 3 workers on the login node, 42 min)
    edge5: bool = True
    maps: bool = True
    xset_rows: str = ""                  # XSET-1 row root (Atos) for W9; empty = W9 skipped
    xset_cases: str = ""
    python: str = sys.executable
    only_steps: tuple[str, ...] = ()
    force: bool = False
    extra_env: dict = field(default_factory=dict)

    def resolved(self) -> "Config":
        u = self.user_root
        self.member_stores = self.member_stores or f"{u}/agent-work/20261104-ens-member-box"
        self.datasets = self.datasets or f"{u}/datasets"
        self.orog320 = self.orog320 or f"{u}/station_caches/orography_O320_eefh.grib"
        self.label = self.label or self.final_model
        self.out = Path(self.out).resolve()
        self.pred_root = Path(self.pred_root)
        return self


@dataclass
class Step:
    name: str
    part: str
    argv: list[str]
    cwd: Path | None = None
    stdout_to: Path | None = None   # also copy the log here (printouts that are results, e.g. repeat_check.txt)
    note: str = ""
    optional: bool = False          # a figure or printout: its failure is reported but does not stop the part


def env_for(cfg: Config) -> dict[str, str]:
    t = layout.terrain_paths(cfg.out)
    e = {
        "FB_USER_ROOT": cfg.user_root, "FB_PRED_ROOT": str(cfg.pred_root), "FB_MEMBER_STORES": cfg.member_stores,
        "FB_DATASETS": cfg.datasets, "FB_OROG320": cfg.orog320, "FB_ROW": cfg.row, "FB_ROW10": cfg.row,
        "FB_LABEL": cfg.label, "FB_LEADS": " ".join(str(x) for x in cfg.leads),
        "FB_LEADS10": " ".join(str(x) for x in cfg.leads), "FB_WORK": str(cfg.out), "FB_TERRAIN_OUT": str(t["root"]),
        "FB_WIND_WORK": str(layout.wind_paths(cfg.out)["root"]),
        "FB_FORCINGS": f"{cfg.datasets}/o320_o1280_europe_e2/in_hres_o1280_forcings.zarr",
        "RC_FORCINGS": f"{cfg.datasets}/o320_o1280_europe_e2/in_hres_o1280_forcings.zarr",
        "MPLBACKEND": "Agg", "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "2"),
    }
    e.update({k: str(v) for k, v in cfg.extra_env.items()})
    return e


def pred_dir(cfg: Config, lead: int) -> Path:
    return cfg.pred_root / "pred" / f"{cfg.row}_L{lead:03d}"


def dates_of(cfg: Config, lead: int) -> list[str]:
    return sorted(p.name.split("_")[1] for p in pred_dir(cfg, lead).glob(f"predictions_*_step{lead:03d}.nc"))


def file_states(path: Path) -> list[str]:
    import netCDF4
    with netCDF4.Dataset(path) as d:
        v = d["weather_state"][:]
        try:
            return [str(x) for x in v]
        except Exception:  # noqa: char array
            return [str(x) for x in netCDF4.chartostring(v)]


def preflight(cfg: Config) -> dict:
    """What exists, what each part can run. Raises SystemExit when nothing can run."""
    rep: dict = {"leads": {}, "states": None, "has_10_outputs": False, "problems": []}
    for lead in cfg.leads:
        rep["leads"][lead] = len(dates_of(cfg, lead))
    if not any(rep["leads"].values()):
        raise SystemExit(f"final_battery: no prediction files under {cfg.pred_root}/pred/{cfg.row}_L*/predictions_*.nc")
    first = next(p for lead in cfg.leads for p in sorted(pred_dir(cfg, lead).glob("predictions_*.nc")))
    try:
        st = file_states(first)
        rep["states"] = st
        rep["has_10_outputs"] = all(v in st for v in ("2d", "skt", "tcw"))
        for part in cfg.parts:
            miss = [v for v in NEEDED[part] if v not in st]
            if miss:
                rep["problems"].append(f"{part}: file lacks {miss}")
    except Exception as e:  # noqa: netCDF4 missing or odd file
        rep["problems"].append(f"could not read weather_state of {first}: {e!r}")
    counts = set(v for v in rep["leads"].values() if v)
    if len(counts) > 1:
        rep["problems"].append(f"unequal date counts across leads: {rep['leads']}")
    for lead, n in rep["leads"].items():
        if n and n < 30:
            rep["problems"].append(f"+{lead} h has {n} dates (reference: 30); dates are the sample, read the SEs")
    return rep


def build_steps(cfg: Config, rep: dict | None = None) -> list[Step]:
    py = cfg.python
    T, C, W = KITS / "terrain", KITS / "consistency", KITS / "wind"
    tp, cp, wp = layout.terrain_paths(cfg.out), layout.consistency_paths(cfg.out), layout.wind_paths(cfg.out)
    has10 = bool(rep and rep.get("has_10_outputs"))
    steps: list[Step] = []
    if "terrain" in cfg.parts:
        w = str(cfg.workers)
        steps += [
            Step("terrain.statics", "terrain", [py, str(T / "realism_t2m.py"), "statics"]),
            Step("terrain.t", "terrain", [py, str(T / "realism_t2m.py"), "run", "--workers", w]),
            Step("terrain.t.aggregate", "terrain", [py, str(T / "realism_t2m.py"), "aggregate"]),
            Step("terrain.z", "terrain", [py, str(T / "realism_z.py"), "run", "--workers", w]),
            Step("terrain.z.aggregate", "terrain", [py, str(T / "realism_z.py"), "aggregate"]),
            Step("terrain.z6", "terrain", [py, str(T / "realism_z6.py"), "run", "--workers", w]),
            Step("terrain.z6.aggregate", "terrain", [py, str(T / "realism_z6.py"), "aggregate"]),
        ]
        if has10:
            steps += [Step("terrain.d2", "terrain", [py, str(T / "realism_2d.py"), "run", "--workers", w]),
                      Step("terrain.d2.aggregate", "terrain", [py, str(T / "realism_2d.py"), "aggregate"]),
                      Step("terrain.d2.print", "terrain", [py, str(T / "danalyze.py"), str(tp["d2"])],
                           stdout_to=tp["d2"] / "danalyze_stdout.txt", optional=True)]
        steps += [
            Step("terrain.print", "terrain", [py, str(T / "analyze.py"), str(tp["tables"])], stdout_to=tp["tables"] / "analyze_stdout.txt", optional=True),
            Step("terrain.inversions", "terrain", [py, str(T / "check_inv.py"), str(tp["tables"] / "T2_lapse_slopes.csv")],
                 stdout_to=tp["tables"] / "check_inv_stdout.txt", optional=True),
            Step("terrain.repeat", "terrain", [py, str(T / "repeat_check.py")], stdout_to=tp["root"] / "repeat_check.txt", optional=True),
            Step("terrain.figs", "terrain", [py, str(T / "plot_summary.py"), str(tp["tables"]), str(tp["figs"] / "summary")], optional=True),
            Step("terrain.figs.lead", "terrain", [py, str(T / "plot_spread_by_lead.py"), str(tp["tables"]), str(tp["figs"] / "summary")], optional=True),
            Step("terrain.figs.z", "terrain", [py, str(T / "zplots.py"), str(tp["z"]), str(tp["figs"] / "z")],
                 stdout_to=tp["z"] / "zplots_stdout.txt", optional=True),
        ]
        if cfg.maps:
            specs = list(TERRAIN_MAPS) + (list(TERRAIN_MAPS_10) if has10 else [])
            steps.append(Step("terrain.maps", "terrain", [py, str(T / "maps_t2m.py"), str(tp["figs"] / "maps"), *specs],
                              note="needs cartopy and cmcrameri", optional=True))
    if "consistency" in cfg.parts:
        a = cfg.member_stores
        pattern = str(cfg.pred_root / "pred" / f"{cfg.row}_L{{lead}}" / "predictions_{date}_step{lead}.nc")
        common = ["--pred", pattern, "--truth-stores", f"{a}/stores/emb_out_hres_o1280_pf.zarr",
                  f"{a}/stores_leads/emb_out_hres_o1280_pf.zarr", "--input-stores", f"{a}/stores/emb_in_lres_o320_halo1p5_pf.zarr",
                  f"{a}/stores_leads/emb_in_lres_o320_halo1p5_pf.zarr", "--leads", *[f"{x:03d}" for x in cfg.leads]]
        for tag, edge in (("edge1", "1"), ("edge5", "5")):
            if tag == "edge5" and not cfg.edge5:
                continue
            o = str(cp[tag])
            steps += [
                Step(f"consistency.{tag}", "consistency", [py, str(C / "realism_consistency.py"), *common, "--outdir", o,
                                                           "--workers", str(cfg.consistency_workers)], note=f"RC_EDGE={edge}"),
                Step(f"consistency.{tag}.aggregate", "consistency", [py, str(C / "realism_consistency.py"), "--aggregate", "--outdir", o],
                     note=f"RC_EDGE={edge}"),
            ]
        steps.append(Step("consistency.figs", "consistency", [py, str(C / "plots.py"), str(cp["edge1"]), str(cp["edge1"] / "figs"), cfg.label], optional=True))
        if cfg.maps:
            steps.append(Step("consistency.maps", "consistency",
                              [py, str(C / "member_maps.py"), "--pred", pattern, "--truth-stores", f"{a}/stores/emb_out_hres_o1280_pf.zarr",
                               f"{a}/stores_leads/emb_out_hres_o1280_pf.zarr", "--cache", str(cp["edge1"]),
                               "--out", str(cp["edge1"] / "maps"), "--label", cfg.label, *CONSISTENCY_MAPS], optional=True))
    if "wind" in cfg.parts:
        root = wp["root"]
        steps.append(Step("wind.orog", "wind", [py, str(W / "extract.py"), "--orog-only", "--out", str(root / "fields")]))
        for lead in cfg.wind_leads:
            ds = ",".join(dates_of(cfg, lead)) if cfg.pred_root.exists() else "<dates>"
            fdir, odir = root / f"fields{lead:03d}", root / f"out{lead:03d}"
            steps += [
                Step(f"wind.extract.{lead}", "wind", [py, str(W / "extract.py"), "--lead", str(lead), "--dates", ds, "--out", str(fdir)]),
                Step(f"wind.accumulate.{lead}", "wind", [py, str(W / "accumulate.py"), "--fields", str(fdir), "--orog",
                                                          str(root / "fields" / "orog.npz"), "--out", str(odir)]),
                Step(f"wind.reduce.{lead}", "wind", [py, str(W / "reduce.py"), str(odir)]),
                Step(f"wind.spread.{lead}", "wind", [py, str(W / "spread.py"), "--fields", str(fdir), "--orog",
                                                      str(root / "fields" / "orog.npz"), "--out", str(odir / "spread.csv")], cwd=W),
            ]
        if cfg.xset_rows and cfg.xset_cases:
            steps.append(Step("wind.xset", "wind", [py, str(W / "xset_tc.py"), "--cases", cfg.xset_cases, "--rows", cfg.xset_rows,
                                                     "--row", "ENS9", "--row", cfg.row, "--out", str(wp["xset"])], cwd=W))
    if cfg.only_steps:
        steps = [s for s in steps if any(s.name == o or s.name.startswith(o + ".") for o in cfg.only_steps)]
    return steps


def _step_env(cfg: Config, step: Step) -> dict[str, str]:
    e = dict(os.environ)
    e.update(env_for(cfg))
    if step.part == "consistency":
        e["RC_EDGE"] = "5" if ".edge5" in step.name else "1"
        e["RC_LEADS"] = " ".join(f"{x:03d}" for x in cfg.leads)
    return e


def describe(cfg: Config, steps: list[Step]) -> str:
    lines = ["# environment"] + [f"export {k}={shlex.quote(v)}" for k, v in env_for(cfg).items()]
    for s in steps:
        pre = ""
        if s.part == "consistency":
            pre = f"RC_EDGE={'5' if '.edge5' in s.name else '1'} "
        cd = f"cd {shlex.quote(str(s.cwd))} && " if s.cwd else ""
        lines.append(f"# {s.name}{' (' + s.note + ')' if s.note else ''}")
        lines.append(cd + pre + " ".join(shlex.quote(a) for a in s.argv))
    return "\n".join(lines)


def run(cfg: Config, dry_run: bool = False) -> int:
    cfg = cfg.resolved()
    rep = preflight(cfg) if cfg.pred_root.exists() else {"problems": [f"{cfg.pred_root} not found"], "leads": {}}
    steps = build_steps(cfg, rep)
    for p in rep.get("problems", []):
        print(f"PREFLIGHT {p}", flush=True)
    if dry_run:
        print(describe(cfg, steps))
        return 0
    cfg.out.mkdir(parents=True, exist_ok=True)
    (cfg.out / "logs").mkdir(exist_ok=True)
    (cfg.out / ".done").mkdir(exist_ok=True)
    meta = {**{k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items() if k not in ("extra_env",)},
            "when": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "preflight": rep,
            "n_dates": max(rep.get("leads", {}).values() or [0]), "kits": _kit_versions()}
    (cfg.out / "battery_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    failed_parts: set[str] = set()
    n_fail = 0
    for s in steps:
        done = cfg.out / ".done" / s.name
        if s.part in failed_parts:
            print(f"SKIP {s.name} (earlier step of {s.part} failed)", flush=True)
            continue
        if done.exists() and not cfg.force:
            print(f"DONE {s.name} (marker)", flush=True)
            continue
        log = cfg.out / "logs" / f"{s.name}.log"
        t0 = time.time()
        print(f"RUN  {s.name}", flush=True)
        with open(log, "w") as fh:
            fh.write("$ " + " ".join(shlex.quote(a) for a in s.argv) + "\n")
            fh.flush()
            rc = subprocess.call(s.argv, cwd=s.cwd, env=_step_env(cfg, s), stdout=fh, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        bad = rc != 0 or _kit_reported_failure(log)
        if bad:
            n_fail += 1
            if not s.optional:
                failed_parts.add(s.part)
            print(f"FAIL {s.name} rc={rc} {dt:.0f}s, log {log}", flush=True)
            continue
        if s.stdout_to:
            s.stdout_to.parent.mkdir(parents=True, exist_ok=True)
            s.stdout_to.write_text("".join(log.read_text().splitlines(keepends=True)[1:]))
        done.write_text(f"{dt:.0f}s\n")
        print(f"OK   {s.name} {dt:.0f}s", flush=True)
    from eval.final_battery import summary
    page = summary.write(cfg.out)
    print(f"SUMMARY {page}", flush=True)
    print(f"BATTERY_{'DONE' if n_fail == 0 else 'FAIL'} failed_steps={n_fail}", flush=True)
    return 0 if n_fail == 0 else 1


def _kit_reported_failure(log: Path) -> bool:
    """The kits print RUN_FAIL / CASE_FAIL / CHECK_FAIL lines instead of exiting non-zero."""
    try:
        txt = log.read_text(errors="replace")
    except OSError:
        return False
    import re
    return bool(re.search(r"RUN_FAIL|CASE_FAIL|CHECK_FAIL\(|^FAIL \d{8}|Traceback \(most recent call last\)", txt, re.M))


def _kit_versions() -> dict[str, str]:
    import hashlib
    return {str(p.relative_to(KITS)): hashlib.md5(p.read_bytes()).hexdigest()[:8] for p in sorted(KITS.glob("*/*.py"))}
