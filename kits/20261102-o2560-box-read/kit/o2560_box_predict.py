#!/usr/bin/env python3
"""o2560 Europe-box read, 2 Nov 2026: EMA predictions of one row (checkpoint) on the o1280 -> o2560 box E2, rule-31 seeds.

Copy of the E2 seed-fix box predictor (agent-work/20261030-seedfix-wide/kit/e2_rain_predict.py, o320 -> o1280) adapted to the
o2560 box stores (datasets/o1280_o2560_europe_e2: in_lres_o1280.zarr, hres_o2560.zarr) and the r9 graph trio both chains
trained with. What changed against that predictor, nothing else:
  * the checkpoint is a file path (CKPT_FILE), the code root is checked against CODE_ROOT and its pin (CODE_PIN_SHA), the
    config is the kit's copy of the Jupiter chain's rendered config (CFG_DIR/CFG_NAME, same model and graph for both rows);
  * all 11 outputs are saved (10u 10v 2d 2t msl skt tcw cp tp z_500 t_850);
  * the input reference x_interp: the 9 outputs present in in_lres (o1280 operational IFS, box with a 1.5-degree halo) are
    sent through the config's own o1280 -> o2560 matrix; tp and cp, which the box store does not hold, are read from the
    global od o1280 source store at the same (date, lead) row, cut to the same 876,159 points and sent through the same matrix;
  * tp and cp of y_pred and x_interp are clipped at 0 before writing; the count and minimum of the clipped values are file
    attributes. tp and cp are 6 h windows ending at the lead in both stores (no de-accumulation; dry mode prints the per-lead
    box means as evidence);
  * the file layout is the one eval.cli reads without a convert step: dimension grid_point_hres; y and x_interp
    (sample, grid_point_hres, weather_state), y_pred (sample, ensemble_member, grid_point_hres, weather_state); init_date and
    lead_step_hours variables; written to a temporary name and renamed when complete; an existing complete file is skipped;
  * ANEMOI_INFERENCE_NUM_CHUNKS (env.sh, 8) chunks the mappers and processor in inference (8 and 16 are bit-identical,
    smoke 2645931; 8 is faster). The model is sharded over the job's ranks (one GH200 is not enough, smoke 2645083).
Env: MODE (dry | predict | smoke), ROW, CKPT_FILE, CKPT_STEP, RAW_FILE (raw inference file of the same step, dry mode
compares it with the EMA file), CODE_ROOT, CODE_PIN_SHA, CFG_DIR, CFG_NAME, OUTDIR, CASES ("YYYYMMDD:lead,..."), MEMBERS.
Nothing here trains, writes a checkpoint, touches a chain or writes FDB.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from o2560_common import (KIT_ROOT, OUTPUT_VARS, PW_SAMPLER, PW_SCHED, SEED_BASE, SEED_FORMULA, case_seeds,  # noqa: E402
                          clip_precip, derive_draw_seed, od_area_mask)

LOG = logging.getLogger("o2560_box_predict")
OD_STORE = "/e/data1/jureap-data/ecmwf/ai-ml/datasets/downscaling-od-fc-oper-0001-mars-o1280-2023-2024-24h-v1.zarr"
OD_INDEX_CACHE = f"{KIT_ROOT}/cache/od_box_index.npy"
SANE = {"10u": (-90, 90), "10v": (-90, 90), "2d": (170, 320), "2t": (180, 340), "msl": (87000, 109000), "skt": (170, 350),
        "tcw": (0, 130), "cp": (0, 0.5), "tp": (0, 0.5), "z_500": (40000, 62000), "t_850": (210, 320)}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def fake_map(obj) -> dict:
    """The store's fake-forecast map {fake datetime: (real datetime, step)} under an opened dataset. The o2560 box stores
    carry the attribute `fake_forecasts`; the E2 stores of the o320 -> o1280 kit carried `fake_hindcasts`, empty here."""
    import datetime
    ds = obj
    for _ in range(32):
        try:
            m = ds.fake_forecasts
        except AttributeError:
            m = None
        if isinstance(m, dict) and m and isinstance(next(iter(m)), datetime.datetime):
            return m
        ds = getattr(ds, "forward", None)
        if ds is None:
            break
    raise SystemExit("FATAL: no fake_forecasts mapping under the dataset")


def resolve_rows(reader, lead: int):
    """All (index, forecast date 'YYYYMMDD', fake date) rows of the split at one lead, through the fake-forecast map."""
    from earthkit.data.utils.dates import to_datetime
    m = fake_map(reader.data)
    rows = []
    for i, d in enumerate(np.asarray(reader.dates)):
        real, st = m[to_datetime(d)]
        if int(st) == lead:
            rows.append((i, str(np.datetime64(real, "D")).replace("-", ""), str(np.datetime64(d, "s"))))
    return rows


def lead_of(reader, i: int) -> int:
    """Step (h) of row i of an opened dataset through its fake-forecast map."""
    from earthkit.data.utils.dates import to_datetime
    return int(fake_map(reader)[to_datetime(np.asarray(reader.dates)[i])][1])


def find_row(reader, hdate: str, lead: int):
    hits = [r for r in resolve_rows(reader, lead) if r[1] == hdate]
    if len(hits) != 1:
        raise SystemExit(f"FATAL: {len(hits)} rows for {hdate} +{lead} h in this split")
    return hits[0][0], hits[0][2]


class OdPrecip:
    """tp and cp of the global od o1280 source store at one (date, lead), on the 876,159 points of the in_lres box store."""

    def __init__(self, box_lat: np.ndarray, box_lon360: np.ndarray):
        import zarr
        self.z = zarr.open(OD_STORE, "r")
        a = dict(self.z.attrs)
        self.vars = list(a["variables"])
        dates = [str(np.datetime64(d, "s")) for d in np.asarray(self.z["dates"][:])]
        pos = {d: i for i, d in enumerate(dates)}
        self.row = {}
        for fake, (hd, st) in a["fake_forecasts"].items():
            self.row[(str(hd)[:10].replace("-", ""), int(st))] = pos[str(np.datetime64(fake, "s"))]
        lat = np.asarray(self.z["latitudes"][:], dtype=np.float64)
        lon = np.asarray(self.z["longitudes"][:], dtype=np.float64) % 360.0
        if os.path.exists(OD_INDEX_CACHE):
            idx = np.load(OD_INDEX_CACHE)
            how = f"cache {OD_INDEX_CACHE}"
        else:
            idx = np.flatnonzero(od_area_mask(lat, lon))
            how = "area mask [71.5, -81.5, 8.5, 41.5]"
            if idx.size != box_lat.size or np.abs(lat[idx] - box_lat).max() > 1e-6 or np.abs(lon[idx] - box_lon360 % 360.0).max() > 1e-6:
                from scipy.spatial import cKDTree
                xyz = lambda la, lo: np.column_stack([np.cos(np.deg2rad(la)) * np.cos(np.deg2rad(lo)),
                                                      np.cos(np.deg2rad(la)) * np.sin(np.deg2rad(lo)), np.sin(np.deg2rad(la))])
                d, idx = cKDTree(xyz(lat, lon)).query(xyz(box_lat, box_lon360), workers=-1)
                how = f"nearest neighbour (max chord {d.max():.2e})"
            Path(OD_INDEX_CACHE).parent.mkdir(parents=True, exist_ok=True)
            np.save(OD_INDEX_CACHE, idx)
        dlat = float(np.abs(lat[idx] - box_lat).max())
        dlon = float(np.abs(((lon[idx] - box_lon360 % 360.0) + 180.0) % 360.0 - 180.0).max())
        LOG.info("od store %s: %d rows, %d points; box index by %s, %d points, max |dlat| %.2e |dlon| %.2e", OD_STORE,
                 self.z["data"].shape[0], lat.size, how, idx.size, dlat, dlon)
        if dlat > 1e-6 or dlon > 1e-6:
            raise SystemExit("FATAL: od box points do not match the in_lres box points")
        self.idx = idx

    def get(self, hdate: str, lead: int, var: str) -> np.ndarray:
        r = self.row[(hdate, int(lead))]
        return np.asarray(self.z["data"][r, self.vars.index(var), 0, :], dtype=np.float64)[self.idx]


def write_nc(path: Path, lat, lon, states, y, xi, yp, init_ns: int, lead: int, attrs: dict):
    import netCDF4
    n_m, n_p, n_s = yp.shape
    tmp = path.with_suffix(".nc.tmp")
    with netCDF4.Dataset(str(tmp), "w") as d:
        d.createDimension("sample", 1)
        d.createDimension("ensemble_member", n_m)
        d.createDimension("grid_point_hres", n_p)
        d.createDimension("weather_state", n_s)
        v = d.createVariable("lat_hres", "f8", ("grid_point_hres",)); v[:] = lat
        v = d.createVariable("lon_hres", "f8", ("grid_point_hres",)); v[:] = lon
        v = d.createVariable("weather_state", str, ("weather_state",))
        for i, s in enumerate(states):
            v[i] = s
        v = d.createVariable("ensemble_member", "i4", ("ensemble_member",)); v[:] = np.arange(1, n_m + 1)
        v = d.createVariable("init_date", "i8", ("sample",)); v[:] = [init_ns]
        v.long_name = "forecast initialization date"; v.units = "nanoseconds since 1970-01-01"
        v = d.createVariable("lead_step_hours", "i4", ("sample",)); v[:] = [lead]; v.units = "hours"
        v = d.createVariable("y", "f4", ("sample", "grid_point_hres", "weather_state")); v[:] = y[None]
        v = d.createVariable("x_interp", "f4", ("sample", "grid_point_hres", "weather_state")); v[:] = xi[None]
        v = d.createVariable("y_pred", "f4", ("sample", "ensemble_member", "grid_point_hres", "weather_state")); v[:] = yp[None]
        for k, val in attrs.items():
            d.setncattr(k, val if isinstance(val, (int, float)) else str(val))
    os.replace(tmp, path)


def file_complete(path: Path, n_members: int) -> bool:
    if not path.exists():
        return False
    try:
        import netCDF4
        with netCDF4.Dataset(str(path)) as d:
            return len(d.dimensions["ensemble_member"]) == n_members and d.getncattr("complete") == "yes"
    except Exception:  # noqa: BLE001 - an unreadable file is rewritten
        return False


def init_distributed(use_cuda: bool):
    """From the GL512 kit (agent-work/20261030-gl512-100k-wide/kit/gl512_predict.py, itself from h9p1_box_predict.py): one
    model_comm_group over the job's SLURM tasks. The o2560 full box does not fit one GH200 in inference (smoke 2645083:
    out of memory at 88 GiB in the decoder's node MLP), so the model is sharded over the ranks, as in training. A one-task
    run gets a one-rank group (the sharding paths need a group object)."""
    import torch
    import torch.distributed as dist
    world = int(os.environ.get("SLURM_NTASKS", "1")) if use_cuda else 1
    rank = int(os.environ.get("SLURM_PROCID", "0")) if use_cuda else 0
    local_rank = int(os.environ.get("SLURM_LOCALID", "0")) if use_cuda else 0
    if not use_cuda:
        return None, 0, 0, 1
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29594")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world, rank=rank)
    return dist.new_group(ranks=list(range(world))), rank, local_rank, world


class GlobalNoise:
    """From the GL512 kit, unchanged in logic. While active, a 5-D torch.randn whose grid axis (-2) has this rank's shard size
    draws the FULL box field (n_grid points) and returns this rank's contiguous slice (the torch.split order of
    shard_tensor). The code draws the initial noise per grid shard, so without this the same seed on several ranks would
    repeat one noise block; with it, the noise of a draw is the field a one-GPU run draws with that seed, whatever the
    number of ranks. Records the float64 sum and sum of squares of each full field drawn."""

    def __init__(self, n_grid: int, world: int, rank: int):
        from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
        self.n_grid, self.world, self.rank = n_grid, world, rank
        self.sizes = list(get_balanced_partition_sizes(n_grid, world))
        self.off = int(sum(self.sizes[:rank]))
        self.full, self.other = [], 0

    def _randn(self, *size, **kw):
        import torch
        shp = tuple(size[0]) if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)) else tuple(size)
        if self.world > 1 and len(shp) == 5 and int(shp[-2]) == self.sizes[self.rank]:
            z = self._orig(shp[:-2] + (self.n_grid,) + shp[-1:], **kw)
            zd = z.double()
            self.full.append((float(zd.sum().item()), float((zd * zd).sum().item())))
            return z.narrow(-2, self.off, int(shp[-2])).contiguous()
        self.other += 1
        return self._orig(*size, **kw)

    def __enter__(self):
        import torch
        self.full, self.other = [], 0
        self._orig = torch.randn
        torch.randn = self._randn
        return self

    def __exit__(self, *exc):
        import torch
        torch.randn = self._orig
        return False


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s %(message)s", stream=sys.stdout)
    mode = os.environ.get("MODE", "dry")
    assert mode in ("dry", "predict", "smoke"), mode
    row = os.environ["ROW"]
    step = int(os.environ["CKPT_STEP"])
    ckpt = os.environ["CKPT_FILE"]
    code_root = os.environ["CODE_ROOT"]
    outdir = Path(os.environ["OUTDIR"])
    outdir.mkdir(parents=True, exist_ok=True)
    n_mem = int(os.environ.get("MEMBERS", "10"))
    cases = [(c.split(":")[0], int(c.split(":")[1])) for c in os.environ["CASES"].split(",") if c.strip()]
    for k in ("/home/", "/e/home/"):
        assert not str(outdir).startswith(k), f"output under home: {outdir}"
    import torch
    torch.set_float32_matmul_precision("high")
    LOG.info("MODE %s row %s step %d ckpt %s members %d cases %d %s", mode, row, step, ckpt, n_mem, len(cases), cases[:3])
    LOG.info("SEED_RULE per_case base %d; check derive_draw_seed(1234, 20230101, 12, 0..1) = %s (expect [4097385291, 2762430685])",
             SEED_BASE, [derive_draw_seed(1234, 20230101, 12, m) for m in (0, 1)])
    LOG.info("ANEMOI_INFERENCE_NUM_CHUNKS=%s TORCH_COMPILE_DISABLE=%s", os.environ.get("ANEMOI_INFERENCE_NUM_CHUNKS"),
             os.environ.get("TORCH_COMPILE_DISABLE"))
    import anemoi.datasets, anemoi.models, anemoi.training  # noqa: E401
    LOG.info("anemoi.datasets %s; anemoi.models %s; anemoi.training %s", anemoi.datasets.__file__, anemoi.models.__file__,
             anemoi.training.__file__)
    assert "anemoi-datasets-hres-precip-ru" in anemoi.datasets.__file__
    for m in (anemoi.models, anemoi.training):
        assert os.path.realpath(m.__file__).startswith(os.path.realpath(code_root) + "/"), (m.__file__, code_root)

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from anemoi.datasets import open_dataset
    from anemoi.training.train.train import AnemoiTrainer
    overrides = [
        "config_validation=False", "diagnostics.log.mlflow.enabled=False", "diagnostics.log.tensorboard.enabled=False",
        "diagnostics.log.wandb.enabled=False", "diagnostics.enable_checkpointing=False",
        "training.load_weights_only=False", "training.transfer_learning=False", "training.run_id=null",
        "training.fork_run_id=null", "training.ema.enabled=false",
        "system.hardware.num_gpus_per_model=1", "system.hardware.num_nodes=1", "system.hardware.num_gpus_per_node=1",
        "dataloader.read_group_size=1",
    ]
    cfg_dir, cfg_name = os.environ["CFG_DIR"], os.environ["CFG_NAME"]
    cfg_file = Path(cfg_dir) / f"{cfg_name}.yaml"
    cfg_sha = hashlib.sha256(cfg_file.read_bytes()).hexdigest()
    LOG.info("config %s sha256 %s overrides %s", cfg_file, cfg_sha, overrides)
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(config_name=cfg_name, overrides=overrides)
    _sel = lambda k: OmegaConf.select(cfg, k, default=None)
    LOG.info("MODEL_FLAGS class %s preconditioning %s freeze_qk_gains %s/%s/%s shard enc/dec %s/%s crop.enabled %s graph %s",
             _sel("model.model._target_"), _sel("model.model.diffusion.preconditioning"), _sel("model.encoder.freeze_qk_gains"),
             _sel("model.processor.freeze_qk_gains"), _sel("model.decoder.freeze_qk_gains"), _sel("model.encoder.shard_strategy"),
             _sel("model.decoder.shard_strategy"), _sel("training.crop.enabled"), _sel("system.input.graph"))
    assert not _sel("training.crop.enabled"), "crop training must be off in inference"

    class DiagTrainer(AnemoiTrainer):
        @property
        def profiler(self):  # noqa: D102
            return None

    trainer = DiagTrainer(cfg)
    trainer.__dict__["initial_seed"] = int(os.environ.get("ANEMOI_BASE_SEED", "756"))
    dm = trainer.datamodule
    lm = trainer.model
    use_cuda = mode in ("predict", "smoke")
    mcg, rank, local_rank, world = init_distributed(use_cuda)
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}") if use_cuda else torch.device("cpu")
    LOG.info("rank %d of %d (local %d), device %s; the model is sharded over the %d ranks of the model_comm_group", rank, world,
             local_rank, device, world)

    t0 = time.time()
    ckpt_sha = sha256_file(ckpt)
    LOG.info("weights %s sha256 %s (%.0f s)", ckpt, ckpt_sha, time.time() - t0)
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = obj.state_dict() if hasattr(obj, "state_dict") else obj["state_dict"]
    missing, unexpected = lm.model.load_state_dict(sd, strict=False)
    LOG.info("EMA load: %d missing %s, %d unexpected %s", len(missing), list(missing)[:8], len(unexpected), list(unexpected)[:8])
    if missing or unexpected:
        raise SystemExit(f"FATAL: EMA state dict mismatch: missing {list(missing)[:5]} unexpected {list(unexpected)[:5]}")
    if mode == "dry" and os.environ.get("RAW_FILE"):
        raw_obj = torch.load(os.environ["RAW_FILE"], map_location="cpu", weights_only=False)
        rsd = raw_obj.state_dict() if hasattr(raw_obj, "state_dict") else raw_obj["state_dict"]
        diffs = [float((sd[k].double() - rsd[k].double()).abs().max()) for k in sd if torch.is_tensor(sd[k]) and sd[k].is_floating_point()
                 and k in rsd and "processor" in k and "weight" in k]
        LOG.info("EMA_VS_RAW %d processor weight tensors, max |EMA - raw| over tensors: median %.3e max %.3e (0 would mean the "
                 "EMA file holds raw weights)", len(diffs), float(np.median(diffs)), float(np.max(diffs)))
        if float(np.max(diffs)) == 0.0:
            raise SystemExit("FATAL: the EMA file equals the raw weights")
        del raw_obj, rsd
    lm.to(device); lm.eval()
    lm.model_comm_group = mcg
    lm.grid_shard_sizes = {}
    fp = sum(float(p.double().sum().item()) for i, (_, p) in enumerate(sorted(lm.state_dict().items()))
             if torch.is_tensor(p) and p.is_floating_point() and i % 37 == 0)
    LOG.info("weights loaded, fingerprint=%.6e, training-mode modules: %d, model class %s", fp,
             sum(1 for m in lm.modules() if m.training), type(lm.model.model).__name__)

    emit_lead = bool(cfg.dataloader.get("emit_lead_hours", True))
    ds_cfg = OmegaConf.to_container(cfg.dataloader.test.datasets, resolve=True)

    def raw(name):
        g = {k: x for k, x in ds_cfg[name].items() if not (k == "reorder" and x is None)}
        return open_dataset(g)

    out_raw = raw("out_hres")
    lat = np.asarray(out_raw.latitudes, dtype=np.float64)
    lon = ((np.asarray(out_raw.longitudes, dtype=np.float64) + 180.0) % 360.0) - 180.0
    LOG.info("out_hres: %d points, lat %.2f..%.2f lon %.2f..%.2f", lat.size, lat.min(), lat.max(), lon.min(), lon.max())
    import scipy.sparse as sp
    M = sp.load_npz(str(cfg.system.input.truncation)).tocsr()
    assert M.shape[0] == lat.size, f"truncation matrix rows {M.shape} vs grid {lat.size}"
    in_full = raw("in_lres")
    in_idx = in_full.name_to_index
    assert in_full.shape[-1] == M.shape[1], (in_full.shape, M.shape)
    in_lat = np.asarray(in_full.latitudes, dtype=np.float64)
    in_lon = np.asarray(in_full.longitudes, dtype=np.float64)
    od = OdPrecip(in_lat, in_lon)
    LOG.info("in_lres: %d variables, %d points (matrix %s, sha256 %s)", len(in_idx), in_full.shape[-1], M.shape,
             hashlib.sha256(Path(str(cfg.system.input.truncation)).read_bytes()).hexdigest()[:16])

    split = "test"
    ds = getattr(dm, f"ds_{split}")
    shard_sizes = {n: [int(sub.data.shape[-1])] for n, sub in ds.datasets.items()}
    ds.set_comm_group_info(0, 0, 0, 1, 0, 1, shard_sizes)
    ds.per_worker_init(1, 0)
    n2i_out_data = ds.name_to_index["out_hres"]
    n2i_out_model = dm.data_indices["out_hres"].model.output.name_to_index
    states = [v for v in OUTPUT_VARS if v in n2i_out_model]
    model_order = sorted(n2i_out_model, key=n2i_out_model.get)
    LOG.info("model outputs (model order) %s; saved %s", model_order, states)
    if tuple(states) != OUTPUT_VARS or len(n2i_out_model) != len(OUTPUT_VARS) or "sp" in n2i_out_model:
        raise SystemExit(f"FATAL: outputs {model_order} are not the 11 expected {OUTPUT_VARS}")
    in_dates = np.asarray(in_full.dates)

    records = []
    n_draws = n_finite = 0
    sanity_bad = []
    for hdate, lead in cases:
        idx, fake = find_row(ds.datasets["out_hres"], hdate, lead)
        assert str(np.datetime64(in_dates[idx], "s")) == fake, (in_dates[idx], fake)
        name = f"predictions_{hdate}_step{lead:03d}.nc"
        if mode == "predict" and file_complete(outdir / name, n_mem):
            LOG.info("SKIP %s complete", name)
            records.append(dict(hdate=hdate, lead=lead, idx=idx, fake=fake, file=name, skipped=True))
            continue
        sample = ds.get_sample(idx)
        # emit_lead_hours is false in these configs and lead_hours_of() reads the empty fake_hindcasts map: the lead is the
        # step of the store's fake_forecasts map at this row (find_row matched it), re-read here for the in_lres row too
        got_lead = float(sample["lead_hours"]) if "lead_hours" in sample else float(lead_of(in_full, idx))
        if got_lead != lead:
            raise SystemExit(f"FATAL: sample {idx} lead {got_lead} != {lead}")
        batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items() if k in ("in_lres", "in_hres", "out_hres")}
        truth = batch["out_hres"][0, 0, 0].float().cpu().numpy()
        y = np.stack([truth[:, n2i_out_data[v]] for v in states], axis=-1).astype(np.float64)
        if is_main:   # the input reference: rank 0 only (it writes the file)
            xfull = np.asarray(in_full[idx][:, 0, :], dtype=np.float64)
            xi = np.full_like(y, np.nan)
            for j, v in enumerate(states):
                if v in in_idx:
                    xi[:, j] = M @ xfull[in_idx[v]]
                elif v in ("tp", "cp"):
                    xi[:, j] = M @ od.get(hdate, lead, v)
            xi, xi_neg = clip_precip(xi, states)
            if not np.isfinite(xi).all():
                raise SystemExit(f"FATAL: x_interp not finite for {hdate} +{lead}: {[s for j, s in enumerate(states) if not np.isfinite(xi[:, j]).all()]}")
        cseeds = case_seeds(hdate, lead, n_mem)
        if mode == "dry":
            ti, tpo = states.index("tp"), states.index("tp")
            LOG.info("DRY case %s +%d: row %d fake %s; box mean tp truth %.3f mm, input %.3f mm; box max tp truth %.1f mm input "
                     "%.1f mm; 2t mean truth %.2f input %.2f; seeds %s; batch %s", hdate, lead, idx, fake,
                     1000 * y[:, ti].mean(), 1000 * xi[:, tpo].mean(), 1000 * y[:, ti].max(), 1000 * xi[:, tpo].max(),
                     y[:, states.index("2t")].mean(), xi[:, states.index("2t")].mean(), cseeds[:2],
                     {k: tuple(v.shape) for k, v in batch.items()})
            records.append(dict(hdate=hdate, lead=lead, idx=idx, fake=fake, tp_mean_truth_mm=1000 * float(y[:, ti].mean()),
                                tp_mean_input_mm=1000 * float(xi[:, tpo].mean()), seeds=cseeds))
            continue
        preds = []
        if is_main:
            LOG.info("SEEDS case=%s lead=%d %s", hdate, lead, cseeds)
        for s in cseeds:   # the same seed on every rank; GlobalNoise makes one box-wide noise field of it
            torch.manual_seed(s); np.random.seed(s % (2**32)); random.seed(s)
            torch.cuda.manual_seed_all(s)
            t1 = time.time()
            gn = GlobalNoise(lat.size, world, rank)
            with torch.inference_mode(), gn:
                pred = lm.model.predict_step({k: batch[k] for k in ("in_lres", "in_hres")}, model_comm_group=mcg, gather_out=True,
                                             **({"lead_hours": torch.tensor([float(lead)], device=device)} if emit_lead else {}),
                                             noise_scheduler_params=PW_SCHED, sampler_params=PW_SAMPLER)
            pt = pred["out_hres"] if isinstance(pred, dict) else pred
            if world > 1:
                import torch.distributed as dist
                if len(gn.full) != 1:
                    raise SystemExit(f"FATAL: rank {rank}: {len(gn.full)} full-field noise draws in one member (want 1), "
                                     f"{gn.other} other randn calls")
                pd = pt[0, 0, 0].double()
                mine = torch.tensor([gn.full[0][0], gn.full[0][1], float(pd.sum().item()), float((pd * pd).sum().item()),
                                     float(gn.other)], dtype=torch.float64, device=device)
                allr = [torch.zeros_like(mine) for _ in range(world)]
                dist.all_gather(allr, mine, group=mcg)
                allr = torch.stack(allr).cpu().numpy()
                same = bool((allr == allr[0]).all())
                if is_main:
                    LOG.info("RANKCHECK case=%s lead=%d seed=%d noise sum/sumsq %s output sum/sumsq %s other randn %s identical on "
                             "%d ranks: %s", hdate, lead, s, allr[0, :2].tolist(), allr[0, 2:4].tolist(), allr[:, 4].tolist(), world, same)
                if not same:
                    raise SystemExit(f"FATAL: noise or gathered output differ across ranks for {hdate} +{lead} seed {s}")
            p = pt[0, 0, 0].float().cpu().numpy()
            if p.shape[0] != lat.size:
                raise SystemExit(f"FATAL: gathered output grid {p.shape} vs {lat.size}")
            fin = bool(np.isfinite(p).all())
            n_draws += 1; n_finite += int(fin)
            if is_main:
                preds.append(np.stack([p[:, n2i_out_model[v]] for v in states], axis=-1))
            LOG.info("DRAW case=%s lead=%d seed=%d %.1fs finite=%s rank=%d peak_alloc_GiB=%.2f peak_reserved_GiB=%.2f", hdate, lead, s,
                     time.time() - t1, fin, rank, torch.cuda.max_memory_allocated(device) / 2**30, torch.cuda.max_memory_reserved(device) / 2**30)
        if not is_main:
            continue
        yp, yp_neg = clip_precip(np.stack(preds), states)
        for j, v in enumerate(states):
            lo, hi = SANE[v]
            for lab, a in (("truth", y[:, j]), ("input", xi[:, j]), ("pred", yp[..., j])):
                amin, amax = float(np.nanmin(a)), float(np.nanmax(a))
                if amin < lo or amax > hi:
                    sanity_bad.append(f"{hdate}+{lead} {v} {lab} [{amin:.4g}, {amax:.4g}] outside [{lo}, {hi}]")
            LOG.info("RANGE %s+%d %-5s truth [%.4g, %.4g] mean %.4g std %.4g | pred [%.4g, %.4g] mean %.4g std %.4g | input std %.4g",
                     hdate, lead, v, y[:, j].min(), y[:, j].max(), y[:, j].mean(), y[:, j].std(), yp[..., j].min(), yp[..., j].max(),
                     yp[..., j].mean(), yp[..., j].std(axis=1).mean(), xi[:, j].std())
        init_ns = int(np.datetime64(f"{hdate[:4]}-{hdate[4:6]}-{hdate[6:]}T00:00:00", "ns").astype(np.int64))
        write_nc(outdir / name, lat, lon, states, y.astype(np.float32), xi.astype(np.float32), yp.astype(np.float32), init_ns, lead,
                 dict(model=row, ckpt_step=step, weights=ckpt, weights_sha256=ckpt_sha, fingerprint=fp, code_root=code_root,
                      code_pin=os.environ.get("CODE_PIN_SHA", ""), config=str(cfg_file), config_sha256=cfg_sha, split=split,
                      fake_date=fake, idx=idx, sampler=json.dumps({**PW_SCHED, **PW_SAMPLER}), seeds=json.dumps(cseeds),
                      seed_rule="per_case", seed_base=SEED_BASE, seed_formula=SEED_FORMULA,
                      matmul=torch.get_float32_matmul_precision(), num_chunks=os.environ.get("ANEMOI_INFERENCE_NUM_CHUNKS", "1"),
                      precip_clipped_pred=json.dumps(yp_neg), precip_clipped_input=json.dumps(xi_neg),
                      precip_note="tp and cp in m per 6 h window ending at the lead (store accumulations; no de-accumulation); "
                                  "negative values clipped to 0 before writing", init_date=f"{hdate}T00",
                      lead_step_hours=lead, box="E2 o2560 10-70N 80W-40E (no crop)", model_ranks=world, complete="yes"))
        records.append(dict(hdate=hdate, lead=lead, idx=idx, fake=fake, file=name, seeds=cseeds))
        (outdir / f"records_{os.environ.get('SLURM_JOB_ID', 'local')}.json").write_text(json.dumps(records, indent=1))
        LOG.info("WROTE %s", outdir / name)
    if mode == "dry":
        (outdir / "dry_records.json").write_text(json.dumps(records, indent=1))
        by_lead = {}
        for r in records:
            by_lead.setdefault(r["lead"], []).append((r["tp_mean_truth_mm"], r["tp_mean_input_mm"]))
        for L in sorted(by_lead):
            a = np.asarray(by_lead[L])
            LOG.info("TP_BY_LEAD +%02d h: box-mean tp over %d dates, truth %.3f mm, input %.3f mm (a 6 h window stays flat "
                     "with lead; a total since the start grows about linearly)", L, len(a), a[:, 0].mean(), a[:, 1].mean())
        LOG.info("DRY_DONE row=%s %d cases", row, len(records))
        return
    LOG.info("FINITE_DRAWS %d of %d", n_finite, n_draws)
    for s in sanity_bad:
        LOG.warning("SANITY %s", s)
    LOG.info("O2560_PREDICT_DONE mode=%s row=%s %d files sanity_flags=%d", mode, row, len(records), len(sanity_bad))
    if n_finite != n_draws:
        raise SystemExit(f"FATAL: {n_draws - n_finite} of {n_draws} draws are not finite")


if __name__ == "__main__":
    main()
