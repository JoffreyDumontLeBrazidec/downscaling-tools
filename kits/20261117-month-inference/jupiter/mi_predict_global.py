#!/usr/bin/env python3
"""Global 36 -> 9 km month inference of the set MI-S25 (17 Nov 2026): full O1280 fields of every output, 10 members.

Copy of kits/20261116-extremes-set/xset_predict_global.py (trainer build, strict EMA load, GlobalNoise, rule-31 seeds,
PW_SCHED / PW_SAMPLER, anchor proof of the two input paths and the rank checks unchanged). What differs:
  * in_lres of draw k = operational eefo pf k+1 of MI_IN/<date>.zarr (mi_build_in.py, the XSET-1 store layout, 20 leads);
  * no template: every model output is written on the whole O1280 grid, 16-bit packed per (member, variable):
    OUTDIR/<date>/pred_<date>_step<LLL>.npz with q (members, outputs, points) uint16, lo and hi (members, outputs) float64,
    value = lo + q / 65535 * (hi - lo) (max error (hi - lo) / 131070), states, seeds, attrs (json); OUTDIR/grid_o1280.npz
    holds lat / lon of the output points (the out_hres store order). tp is the raw model value (12 h window ending at the
    lead, metres; negatives kept here, clipped at the GRIB encoding);
  * SEED_BASE from env, default 1234 = the base of every XSET-1 / S25 paper row (derive_draw_seed(1234, date, lead, member
    0-based)), so every generative paper row is seed-paired (rule 31); prepml jdno / jdyb use 756 (not paired with them);
  * global tail histograms as XSET-1, to OUTDIR/hist/<ROW>_<date>_step<LLL>.npz.
MODE=xcheck (the gate before a month): cases of XSET-1 dates, input from XCHECK_IN (the XSET stores), SEED_BASE must be 1234;
the raw draws at the point_index of XCHECK_ROWS/predictions_<date>_step<LLL>.nc must equal that file's y_pred bit for bit
for the 5 XSET variables, and the packed values must round-trip within the packing bound; writes OUTDIR/MI_XCHECK_PASS.

Env: MODE (predict | xcheck), CKPT_STEP, RUN_DIR, CFG_DIR, CFG_NAME, EXPECT_CODE, MODEL_LABEL, ROW_LABEL, EMA_TAG, OUTDIR, CASES
("YYYYMMDD:lead,..."), MEMBERS (10), PART, MI_IN, SEED_BASE, ANCHOR; xcheck: XCHECK_IN, XCHECK_ROWS.
Nothing here trains, writes a checkpoint, touches a chain or writes FDB.
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "20261116-extremes-set"))
from xset_common import SAVE_VARS, tail_histograms, wrap180  # noqa: E402

LOG = logging.getLogger("mi_predict")
DEV = "/e/home/jusers/dumontlebrazidec1/jupiter/dev"
RUN_DIR = os.environ.get("RUN_DIR", "/e/data1/jureap-data/ecmwf/users/jdumont/outputs/checkpoint_nm50_global/db_global_512_300k")
EXPECT_CODE = os.environ.get("EXPECT_CODE", "db_global_20261027/code/anemoi-core")
MODEL_LABEL = os.environ.get("MODEL_LABEL", "db_global_512_300k")
CFG_DIR = os.environ.get("CFG_DIR", f"{DEV}/db_global_20261027/configs")
CFG_NAME = os.environ.get("CFG_NAME", "db_global_512_300k")
SEED_BASE = int(os.environ.get("SEED_BASE", "1234"))
SEED_FORMULA = "numpy SeedSequence([base, yyyymmdd, lead_h, member]).generate_state(1, uint32)"
TIME_VARS = ("cos_julian_day", "sin_julian_day", "cos_local_time", "sin_local_time", "insolation")


def derive_draw_seed(base, date, lead, member):
    """Sampler seed for one draw. date as yyyymmdd int, lead in hours, member 0-based."""
    return int(np.random.SeedSequence([int(base), int(date), int(lead), int(member)]).generate_state(1, dtype=np.uint32)[0])


def case_seeds(hdate: str, lead: int, n: int) -> list[int]:
    """Rule 31: one seed per (date, lead, member), identical across arms and case lists."""
    return [derive_draw_seed(SEED_BASE, int(hdate), int(lead), m) for m in range(n)]


PW_SCHED = {
    "schedule_type": "experimental_piecewise", "num_steps": 30, "sigma_max": 100000.0,
    "sigma_transition": 10.0, "sigma_min": 0.03, "high_schedule_type": "exponential",
    "low_schedule_type": "karras", "num_steps_high": 10, "num_steps_low": 20, "rho": 7.0,
}
PW_SAMPLER = {"sampler": "heun", "S_churn": 0.0, "S_min": 0.75, "S_max": 100000.0, "S_noise": 1.05}
# SAMPLER_MODE=det (26 Nov 2026, one-call rows of the paper, same rule as kits/20261116-extremes-set/xset_predict_global.py):
# ONE denoiser call at sigma MI_DET_SIGMA (default 5e5 = REG_GL512_*): linear schedule [MI_DET_SIGMA, 1e-12], so
# y = y_init + (1e-12 - s) (y_init - D) / s = D(x; s) up to 1e-12/s y_init. On a diffusion model this is its
# conditional-mean estimate at noise level s. The unit noise of that call still comes from the rule-31 seed of the draw.
# XSET_SCHED_JSON / XSET_SAMPLER_JSON replace the whole dictionaries (never merged); what was used is in every file.
if os.environ.get("SAMPLER_MODE", "edm") == "det":
    PW_SCHED = {"schedule_type": "linear", "num_steps": 2, "sigma_max": float(os.environ.get("MI_DET_SIGMA", "500000")),
                "sigma_min": 1e-12}
    PW_SAMPLER = {"sampler": "heun", "S_churn": 0.0}
elif os.environ.get("SAMPLER_MODE", "edm") != "edm":
    raise SystemExit(f"unknown SAMPLER_MODE {os.environ.get('SAMPLER_MODE')!r} (edm | det)")
if os.environ.get("XSET_SCHED_JSON"):
    PW_SCHED = json.loads(os.environ["XSET_SCHED_JSON"])
if os.environ.get("XSET_SAMPLER_JSON"):
    PW_SAMPLER = json.loads(os.environ["XSET_SAMPLER_JSON"])


def one_file(pattern: str, what: str) -> str:
    hits = sorted(glob.glob(pattern))
    if len(hits) != 1:
        raise SystemExit(f"FATAL: {len(hits)} {what} files match {pattern}")
    return hits[0]


def ema_file(step: int) -> str:
    if os.environ.get("EMA_FILE"):
        return os.environ["EMA_FILE"]
    return one_file(f"{RUN_DIR}/*/inference-{os.environ.get('EMA_TAG', 'ema2000')}-anemoi-by_step-epoch_*-step_{step:06d}.ckpt", "EMA")


def raw_file(step: int) -> str:
    if os.environ.get("RAW_FILE"):
        return os.environ["RAW_FILE"]
    return one_file(f"{RUN_DIR}/*/anemoi-by_step-epoch_*-step_{step:06d}.ckpt", "raw by-step")


def init_distributed(use_cuda: bool):
    """One model_comm_group over the job's SLURM tasks (None for one task)."""
    world_size = int(os.environ.get("SLURM_NTASKS", "1")) if use_cuda else 1
    rank = int(os.environ.get("SLURM_PROCID", "0")) if use_cuda else 0
    local_rank = int(os.environ.get("SLURM_LOCALID", "0")) if use_cuda else 0
    if world_size == 1:
        return None, 0, 0, 1
    import torch.distributed as dist
    os.environ.setdefault("MASTER_PORT", "29591")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    return dist.new_group(ranks=list(range(world_size))), rank, local_rank, world_size


class GlobalNoise:
    """While active, a 5-D torch.randn whose grid axis (-2) has this rank's shard size draws the FULL global field
    (n_grid points) and returns this rank's contiguous slice (torch.split order of shard_tensor). Records the float64
    sum and sum of squares of each full field drawn."""

    def __init__(self, n_grid: int, world: int, rank: int):
        from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
        self.n_grid, self.world, self.rank = n_grid, world, rank
        self.sizes = list(get_balanced_partition_sizes(n_grid, world))
        self.off = int(sum(self.sizes[:rank]))
        self.full, self.other = [], 0

    def _randn(self, *size, **kw):
        shp = tuple(size[0]) if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)) else tuple(size)
        if self.world > 1 and len(shp) == 5 and int(shp[-2]) == self.sizes[self.rank]:
            z = self._orig(shp[:-2] + (self.n_grid,) + shp[-1:], **kw)
            zd = z.double()
            self.full.append((float(zd.sum().item()), float((zd * zd).sum().item())))
            return z.narrow(-2, self.off, int(shp[-2])).contiguous()
        self.other += 1
        return self._orig(*size, **kw)

    def __enter__(self):
        self.full, self.other = [], 0
        self._orig = torch.randn
        torch.randn = self._randn
        return self

    def __exit__(self, *exc):
        torch.randn = self._orig
        return False


def resolve_rows(reader, lead: int):
    """All (index, hindcast date 'YYYYMMDD', fake date) rows of the split at one lead, through the fake-hindcast map."""
    ds = reader.data
    for _ in range(32):
        try:
            ds.fake_hindcasts
            break
        except AttributeError:
            ds = ds.forward
    else:
        raise SystemExit("FATAL: no fake_hindcasts mapping under the reader")
    rows = []
    for i, d in enumerate(np.asarray(reader.dates)):
        _, hd, st = ds.fake_date_to_date_step(d)
        if int(st) == lead:
            rows.append((i, str(np.datetime64(hd, "D")).replace("-", ""), str(np.datetime64(d, "s"))))
    return rows


def time_forcings(valid, lat, lon):
    """The five time-dependent in_hres fields at one valid time (formulas of earthkit.data.sources.forcings, as
    make_forcing_rows.py of the 4 Nov member kit)."""
    from earthkit.meteo.solar import cos_solar_zenith_angle
    d = valid - dt.datetime(valid.year, 1, 1)
    jd = (d.days + d.seconds / 86400.0) / 365.25 * np.pi * 2
    s = valid - dt.datetime(valid.year, valid.month, valid.day)
    lt = ((lon / 360.0 * 24.0 + (s.days + s.seconds / 86400.0) * 24) % 24) / 24 * np.pi * 2
    return dict(cos_julian_day=np.full(lat.shape, np.cos(jd)), sin_julian_day=np.full(lat.shape, np.sin(jd)),
                cos_local_time=np.cos(lt), sin_local_time=np.sin(lt),
                insolation=np.asarray(cos_solar_zenith_angle(valid, lat, lon)).ravel())


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s %(message)s", stream=sys.stdout)
    mode = os.environ.get("MODE", "predict")
    assert mode in ("xcheck", "predict"), mode
    if mode == "xcheck":
        assert SEED_BASE == 1234, "xcheck reproduces XSET-1 rows, which use seed base 1234"
    step = int(os.environ["CKPT_STEP"])
    ema, raw = ema_file(step), raw_file(step)
    outdir = Path(os.environ["OUTDIR"])
    row_label = os.environ.get("ROW_LABEL", outdir.name)
    cases = [(c.split(":")[0], int(c.split(":")[1])) for c in os.environ.get("CASES", "").split(",") if c.strip()]
    n_mem = int(os.environ.get("MEMBERS", "10"))
    anchor_date, anchor_lead = os.environ.get("ANCHOR", "20230101:24").split(":")
    anchor_lead = int(anchor_lead)
    torch.set_float32_matmul_precision("high")
    mcg, rank, local_rank, world = init_distributed(True)
    is_main = rank == 0
    if is_main:
        outdir.mkdir(parents=True, exist_ok=True)
    LOG.info("MODE %s step %d ema %s raw %s %d cases %s members %d label %s row %s code %s rank %d of %d", mode, step, ema, raw,
             len(cases), cases[:3], n_mem, MODEL_LABEL, row_label, EXPECT_CODE, rank, world)
    LOG.info("SEED base %d; check derive_draw_seed(1234, 20230101, 12, 0..1) = %s", SEED_BASE,
             [derive_draw_seed(1234, 20230101, 12, m) for m in (0, 1)])
    LOG.info("PW_SCHED %s", json.dumps(PW_SCHED))
    LOG.info("PW_SAMPLER %s", json.dumps(PW_SAMPLER))
    import anemoi.datasets, anemoi.models, anemoi.training  # noqa: E401
    LOG.info("anemoi.datasets from %s; anemoi.models from %s; anemoi.training from %s", anemoi.datasets.__file__,
             anemoi.models.__file__, anemoi.training.__file__)
    assert f"/{EXPECT_CODE}/" in anemoi.models.__file__, anemoi.models.__file__
    assert f"/{EXPECT_CODE}/" in anemoi.training.__file__, anemoi.training.__file__

    import hashlib
    import zarr
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from anemoi.datasets import open_dataset
    from anemoi.training.train.train import AnemoiTrainer
    overrides = [
        "config_validation=False", "diagnostics.log.mlflow.enabled=False", "diagnostics.log.tensorboard.enabled=False",
        "diagnostics.log.wandb.enabled=False", "diagnostics.enable_checkpointing=False",
        f"system.input.warm_start={raw}", "training.load_weights_only=True", "training.transfer_learning=False",
        "training.run_id=null", "training.fork_run_id=null", "training.ema.enabled=false",
        "system.hardware.num_gpus_per_model=1", "system.hardware.num_nodes=1", "system.hardware.num_gpus_per_node=1",
        "dataloader.read_group_size=1",
    ] + [o for o in os.environ.get("EXTRA_OVERRIDES", "").split(";") if o.strip()]
    cfg_file = Path(CFG_DIR) / f"{CFG_NAME}.yaml"
    LOG.info("config %s sha256 %s overrides %s", cfg_file, hashlib.sha256(cfg_file.read_bytes()).hexdigest(), overrides)
    with initialize_config_dir(version_base=None, config_dir=CFG_DIR):
        cfg = compose(config_name=CFG_NAME, overrides=overrides)

    class DiagTrainer(AnemoiTrainer):
        @property
        def profiler(self):  # noqa: D102
            return None

    t0 = time.time()
    trainer = DiagTrainer(cfg)
    trainer.__dict__["initial_seed"] = int(os.environ.get("ANEMOI_BASE_SEED", "756"))
    dm = trainer.datamodule
    lm = trainer.model
    device = torch.device(f"cuda:{local_rank}")
    LOG.info("trainer built from the raw warm start %s in %.0fs", raw, time.time() - t0)

    obj = torch.load(ema, map_location="cpu", weights_only=False)
    sd = obj.state_dict() if hasattr(obj, "state_dict") else obj["state_dict"]
    missing, unexpected = lm.model.load_state_dict(sd, strict=False)
    LOG.info("EMA load: %d missing %s, %d unexpected %s", len(missing), list(missing), len(unexpected), list(unexpected))
    bad = [k for k in missing if not ("node_attributes" in k and "latlons" in k)]
    if bad or unexpected:
        raise SystemExit(f"FATAL: EMA state dict mismatch: missing {bad[:5]} unexpected {list(unexpected)[:5]}")
    del obj, sd
    lm.to(device); lm.eval()
    lm.model_comm_group = mcg
    lm.grid_shard_sizes = {}
    fp = sum(float(p.double().sum().item()) for i, (_, p) in enumerate(sorted(lm.state_dict().items()))
             if torch.is_tensor(p) and p.is_floating_point() and i % 37 == 0)
    LOG.info("weights loaded (%d buffers not in the file), fingerprint=%.6e, training-mode modules: %d, model class %s",
             len(missing), fp, sum(1 for m in lm.modules() if m.training), type(lm.model.model).__name__)
    emit_lead = bool(cfg.dataloader.get("emit_lead_hours", True))
    ds_cfg = OmegaConf.to_container(cfg.dataloader.test.datasets, resolve=True)

    def raw_ds(name, drop_none=False):
        g = {k: x for k, x in ds_cfg[name].items() if not (k == "reorder" and x is None)}
        if drop_none:
            g.pop("drop", None)
        return open_dataset(g)

    out_raw = raw_ds("out_hres")
    lat_all = np.asarray(out_raw.latitudes, dtype=np.float64)
    lon_all = wrap180(np.asarray(out_raw.longitudes, dtype=np.float64))
    n_grid = lat_all.size
    in_full = raw_ds("in_lres", drop_none=True)        # nothing dropped: the 70 variables
    in_idx = in_full.name_to_index
    full_vars = [v for v, _ in sorted(in_idx.items(), key=lambda kv: kv[1])]
    hres_raw = raw_ds("in_hres")
    h_idx = hres_raw.name_to_index
    h_vars = [v for v, _ in sorted(h_idx.items(), key=lambda kv: kv[1])]
    h_lat = np.asarray(hres_raw.latitudes, np.float64)
    h_lon = np.asarray(hres_raw.longitudes, np.float64)
    assert h_lat.size == n_grid and all(v in h_idx for v in TIME_VARS), (h_lat.size, n_grid, h_vars)
    LOG.info("grids: out_hres %d points, in_lres %d points x %d variables, in_hres variables %s", n_grid, in_full.shape[-1],
             len(full_vars), h_vars)

    # ---- anchor: the dataset proves the two input paths once
    ds = dm.ds_test
    shard_sizes = {n: [int(s.data.shape[-1])] for n, s in ds.datasets.items()}
    ds.set_comm_group_info(0, 0, 0, 1, 0, 1, shard_sizes)
    ds.per_worker_init(1, 0)
    n2i_out_model = dm.data_indices["out_hres"].model.output.name_to_index
    states = [v for v, _ in sorted(n2i_out_model.items(), key=lambda kv: kv[1])]
    assert all(v in states for v in SAVE_VARS), (states, SAVE_VARS)
    out_cols = [n2i_out_model[v] for v in states]
    LOG.info("outputs written (model order): %s", states)
    hits = [r for r in resolve_rows(ds.datasets["out_hres"], anchor_lead) if r[1] == anchor_date]
    if len(hits) != 1:
        raise SystemExit(f"FATAL: {len(hits)} rows for the anchor {anchor_date} +{anchor_lead} h in the test split")
    a_idx = hits[0][0]
    sample = ds.get_sample(a_idx)
    rd_in, rd_h = ds.datasets["in_lres"], ds.datasets["in_hres"]
    sel_i = [in_idx[v] for v, _ in sorted(rd_in.name_to_index.items(), key=lambda kv: kv[1])]
    sel_h = [h_idx[v] for v, _ in sorted(rd_h.name_to_index.items(), key=lambda kv: kv[1])]

    def member_in(a70):   # raw (70 vars, points) -> the dataset's sample layout (dates, ensemble, points, selected vars)
        return torch.from_numpy(np.ascontiguousarray(a70[sel_i].T[None, None]))

    def hres_in(a11):     # raw (in_hres vars, points) -> the dataset's sample layout
        return torch.from_numpy(np.ascontiguousarray(a11[sel_h].T[None, None]))

    def hres_row(valid):  # the anchor's constants + the time forcings of this valid time, raw layout
        x = h_anchor.copy()
        f = time_forcings(valid, h_lat, h_lon)
        for v in TIME_VARS:
            x[h_idx[v]] = f[v].astype(np.float32)
        return x

    ctl70 = np.asarray(in_full[a_idx][:, 0, :])
    chk = member_in(ctl70)
    if not (chk.shape == sample["in_lres"].shape and chk.dtype == sample["in_lres"].dtype and torch.equal(chk, sample["in_lres"])):
        raise SystemExit("FATAL: member_in differs from the dataset's in_lres sample on the anchor row")
    h_anchor = np.asarray(hres_raw[a_idx][:, 0, :]).astype(np.float32)
    chk = hres_in(h_anchor)
    if not (chk.shape == sample["in_hres"].shape and chk.dtype == sample["in_hres"].dtype and torch.equal(chk, sample["in_hres"])):
        raise SystemExit("FATAL: hres_in differs from the dataset's in_hres sample on the anchor row")
    a_valid = dt.datetime.strptime(anchor_date, "%Y%m%d") + dt.timedelta(hours=anchor_lead)
    h_calc = hres_row(a_valid)
    dmax = {v: float(np.abs(h_calc[h_idx[v]].astype(np.float64) - h_anchor[h_idx[v]].astype(np.float64)).max()) for v in TIME_VARS}
    if is_main:
        LOG.info("ANCHOR %s +%d h row %d: member_in and hres_in equal the dataset sample; computed forcings max |diff| %s",
                 anchor_date, anchor_lead, a_idx, dmax)
    if not torch.equal(hres_in(h_calc), sample["in_hres"]):
        raise SystemExit(f"FATAL: computed forcings differ from the dataset's in_hres sample on the anchor row: {dmax}")
    del sample

    # ---- per-case inputs
    in_dir = Path(os.environ["XCHECK_IN"] if mode == "xcheck" else os.environ["MI_IN"])

    def members_of(date, lead):
        g = zarr.open_group(str(in_dir / f"{date}.zarr"), mode="r")
        assert list(g.attrs["variables"]) == full_vars, "input store variables differ from the global in_lres store"
        assert np.array_equal(np.asarray(g["latitudes"][:]), np.asarray(in_full.latitudes))
        assert np.array_equal(np.asarray(g["longitudes"][:]), np.asarray(in_full.longitudes))
        return np.asarray(g["data"][list(g.attrs["leads"]).index(lead)])

    import netCDF4
    part = os.environ.get("PART", "x")
    hdir = outdir / "hist"
    if is_main:
        hdir.mkdir(parents=True, exist_ok=True)
        gfile = outdir / "grid_o1280.npz"
        if not gfile.exists():   # parallel jobs share OUTDIR: a private temporary name per process, then an atomic rename
            gtmp = outdir / f"grid_o1280.part.{os.environ.get('SLURM_JOB_ID', 'x')}.{os.getpid()}.npz"
            np.savez(gtmp, lat=lat_all, lon=lon_all)
            os.replace(gtmp, gfile)
    records, xres = [], []
    n_draws = n_finite = n_skip = 0
    for date, lead in cases:
        ddir = outdir / date
        name = f"pred_{date}_step{lead:03d}.npz"
        if mode == "predict" and (ddir / name).exists():
            n_skip += 1
            LOG.info("SKIP %s exists", ddir / name)
            continue
        valid = dt.datetime.strptime(date, "%Y%m%d") + dt.timedelta(hours=lead)
        mem70 = members_of(date, lead)
        assert mem70.shape == (len(full_vars), mem70.shape[1], in_full.shape[-1]) and mem70.shape[1] >= n_mem, mem70.shape
        assert np.isfinite(mem70[:, :n_mem]).all(), f"non-finite member input {date} +{lead}"
        in_hres = hres_in(hres_row(valid)).unsqueeze(0).to(device)
        cseeds = case_seeds(date, lead, n_mem)
        if is_main:
            LOG.info("CASE %s +%d h valid %s: seeds %s", date, lead, valid, cseeds)
        q = np.empty((n_mem, len(states), n_grid), np.uint16) if is_main else None
        lo = np.empty((n_mem, len(states)), np.float64)
        hi = np.empty((n_mem, len(states)), np.float64)
        raw5, glob_tp, glob_u, glob_v = [], [], [], []
        for km, seed in enumerate(cseeds):
            batch = {"in_lres": member_in(mem70[:, km, :]).unsqueeze(0).to(device), "in_hres": in_hres}
            s = seed   # the same seed on every rank (one global noise field, see GlobalNoise)
            torch.manual_seed(s); np.random.seed(s % (2**32)); random.seed(s)
            torch.cuda.manual_seed_all(s)
            t1 = time.time()
            gn = GlobalNoise(n_grid, world, rank)
            with torch.inference_mode(), gn:
                pred = lm.model.predict_step(batch, model_comm_group=mcg, gather_out=True,
                                             **({"lead_hours": torch.tensor([float(lead)], device=device)} if emit_lead else {}),
                                             noise_scheduler_params=PW_SCHED, sampler_params=PW_SAMPLER)
            pt = pred["out_hres"] if isinstance(pred, dict) else pred
            if world > 1:
                import torch.distributed as dist
                if len(gn.full) != 1:
                    raise SystemExit(f"FATAL: rank {rank}: {len(gn.full)} full-field noise draws in one member (want 1), "
                                     f"{gn.other} other randn calls")
                pd_ = pt[0, 0, 0].double()
                mine = torch.tensor([gn.full[0][0], gn.full[0][1], float(pd_.sum().item()), float((pd_ * pd_).sum().item()),
                                     float(gn.other)], dtype=torch.float64, device=device)
                allr = [torch.zeros_like(mine) for _ in range(world)]
                dist.all_gather(allr, mine, group=mcg)
                allr = torch.stack(allr).cpu().numpy()
                same = bool((allr == allr[0]).all())
                if is_main:
                    LOG.info("RANKCHECK case=%s seed=%d noise sum/sumsq %s output sum/sumsq %s identical on %d ranks: %s",
                             date, seed, allr[0, :2].tolist(), allr[0, 2:4].tolist(), world, same)
                if not same:
                    raise SystemExit(f"FATAL: noise or gathered output differ across ranks for {date} seed {seed}")
            p = pt[0, 0, 0].float().cpu().numpy()
            if p.shape[0] != n_grid:
                raise SystemExit(f"FATAL: gathered output grid {p.shape} vs {n_grid}")
            n_draws += 1; n_finite += int(bool(np.isfinite(p).all()))
            if is_main:
                for j, c in enumerate(out_cols):
                    a = p[:, c].astype(np.float64)
                    lo[km, j], hi[km, j] = float(a.min()), float(a.max())
                    span = hi[km, j] - lo[km, j]
                    q[km, j] = np.rint((a - lo[km, j]) / span * 65535.0).astype(np.uint16) if span > 0 else 0
                if mode == "xcheck":
                    raw5.append(p[:, [n2i_out_model[v] for v in SAVE_VARS]].copy())
                glob_tp.append(p[:, n2i_out_model["tp"]].copy()); glob_u.append(p[:, n2i_out_model["10u"]].copy())
                glob_v.append(p[:, n2i_out_model["10v"]].copy())
                LOG.info("DRAW case=%s lead=%d seed=%d %.0fs finite=%s peak_alloc_GiB=%.2f", date, lead, seed, time.time() - t1,
                         bool(np.isfinite(p).all()), torch.cuda.max_memory_allocated(device) / 2**30)
            del pred, pt, p
        if not is_main:
            continue
        attrs = dict(model=MODEL_LABEL, row=row_label, ckpt_step=step, weights=ema, fingerprint=fp, date=date, lead=lead,
                     sampler={**PW_SCHED, **PW_SAMPLER}, seeds=cseeds, seed_base=SEED_BASE, seed_formula=SEED_FORMULA,
                     matmul=torch.get_float32_matmul_precision(), set="MI-S25", member_mode="pf", member_store=str(in_dir),
                     member_rule=f"draw k conditioned on input member eefo pf k+1 (k 0-based), seed derive_draw_seed({SEED_BASE}, date, lead, k)",
                     in_hres="6 constants of the dataset anchor row + 5 time forcings computed at date + lead (anchor-checked)",
                     anchor=f"{anchor_date}:{anchor_lead}", tp_units="m per 12 h window ending at the lead (raw, not clipped)",
                     packing="value = lo + q / 65535 * (hi - lo), per (member, output)", grid="O1280, out_hres store order")
        if mode == "xcheck":
            xf = Path(os.environ["XCHECK_ROWS"]) / f"predictions_{date}_step{lead:03d}.nc"
            with netCDF4.Dataset(str(xf)) as d:
                d.set_auto_mask(False)
                rst = [str(v) for v in d["weather_state"][:]]
                ref = np.asarray(d["y_pred"][0, :n_mem])
                pidx = np.asarray(d["point_index"][:]).astype(np.int64)
            r5 = np.stack(raw5)                                            # (members, points, 5)
            diff = {v: float(np.abs(r5[:, pidx, j] - ref[:, :, rst.index(v)]).max()) for j, v in enumerate(SAVE_VARS)}
            unp = lo[:, :, None] + q.astype(np.float64) / 65535.0 * (hi - lo)[:, :, None]
            perr = {v: float(np.abs(unp[:, states.index(v), :] - r5[:, :, j]).max() / max((hi - lo)[:, states.index(v)].max(), 1e-30))
                    for j, v in enumerate(SAVE_VARS)}
            ok = max(diff.values()) == 0.0 and max(perr.values()) <= 1.0 / 131070 * 1.0001
            LOG.info("MI_XCHECK %s +%d h, %d members, %d points vs %s: max |raw - XSET y_pred| %s; packing error / span %s -> %s",
                     date, lead, n_mem, pidx.size, xf, diff, perr, "PASS" if ok else "FAIL")
            xres.append(dict(case=f"{date}:{lead}", diff=diff, pack_err_over_span=perr, ok=ok))
            if not ok:
                raise SystemExit("FATAL: the month path does not reproduce the XSET-1 row")
            continue
        hist = tail_histograms(lat_all, np.stack(glob_tp), np.stack(glob_u), np.stack(glob_v))
        np.savez(hdir / f"{row_label}_{date}_step{lead:03d}.npz", **hist)
        ddir.mkdir(parents=True, exist_ok=True)
        tw = time.time()
        tmp = ddir / (name[:-4] + f".part.{os.environ.get('SLURM_JOB_ID', 'x')}.npz")
        np.savez(tmp, q=q, lo=lo, hi=hi, states=np.array(states), seeds=np.array(cseeds, np.int64),
                 members=np.arange(1, n_mem + 1), attrs=json.dumps(attrs))
        os.replace(tmp, ddir / name)
        records.append(dict(date=date, lead=lead, file=f"{date}/{name}", bytes=(ddir / name).stat().st_size))
        (outdir / f"records_{part}.json").write_text(json.dumps(records, indent=1))
        LOG.info("WROTE %s (%d bytes, %.1fs)", ddir / name, (ddir / name).stat().st_size, time.time() - tw)
    if mcg is not None:
        import torch.distributed as dist
        dist.barrier(); dist.destroy_process_group()
    if is_main:
        LOG.info("FINITE_DRAWS %d of %d", n_finite, n_draws)
        if mode == "xcheck":
            (outdir / "MI_XCHECK_PASS").write_text(json.dumps(dict(cases=xres, step=step, ema=ema,
                                                                    utc=time.strftime("%FT%TZ", time.gmtime())), indent=1))
        LOG.info("MI_ALL_DONE mode=%s %d files written, %d skipped", mode, len(records), n_skip)
    if n_finite != n_draws:
        raise SystemExit(f"FATAL: {n_draws - n_finite} of {n_draws} draws are not finite")

if __name__ == "__main__":
    main()
