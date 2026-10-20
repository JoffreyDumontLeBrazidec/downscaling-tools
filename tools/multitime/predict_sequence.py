#!/usr/bin/env python3
"""Downscale a whole forecast (leads 12-72 h of one hindcast date) with a multi-time model.

For a spatial downscaler trained on Vera Gahlen's `feat/spatial_downscaler` branch with
K time slots per sample (``task.input_offsets == task.output_offsets``, K entries), this
driver samples all L leads of one forecast block jointly:

- ``--stride 1`` (default): overlapping windows of K slots; at every denoiser call each
  slot's denoised estimate is the mean over the windows covering it (the owner's method:
  denoise (t, t+1) and (t+1, t+2), t+1 takes the average, for all 30 steps). Sampler
  ``heun_window`` of the fork branch.
- ``--stride K``: non-overlapping windows, one noise per slot, no averaging: the model's
  own "isolated windows" output, the within-model control of the averaging.
- ``--leads 12,24`` (any subset of consecutive leads of length K) reproduces the plain
  per-window prediction.

One noise field is drawn per slot of the trajectory (the sampled source has the time
length L), so the windows that share a slot denoise the SAME noisy state, and consecutive
leads of one member start from unrelated noise (independent noise across leads). Member m
(from 1) uses ``torch.manual_seed(seed + m - 1)``, so ``--seed 1234`` gives the E2 kit's
member seeds 1234-1243; stride 1 and stride 2 of one member start from the same noise.

Sampler (2026-10-20 read kit): the lane's standing 30-step churn-off sampler, i.e. the fork's
``experimental_piecewise`` schedule (10 exponential steps from sigma_max 1e5 to 10, then 20
Karras steps rho 7 to 0.03, final 0), registered here into the branch's ``SIGMA_SCHEDULES``
because Vera's branch only has single-segment schedules (and the checkpoint's own default,
Karras with sigma_min 0, is refused by her schedule class). ``--schedule karras`` keeps the
checkpoint's default instead. S_churn 0: 59 network calls per window.

Data: ``--config`` (the run's rendered YAML) opens the three test readers exactly as the
run's ``dataloader.test.datasets.<name>.dataset_config`` (store, frequency, select,
fake_hindcasts). ``--hdate YYYYMMDD`` finds the block: each (hindcast date, lead) is looked up
in the out_hres store's ``fake_hindcasts`` attribute, the fake dates must be consecutive rows
of the view starting at a multiple of L, and all three readers must carry the same dates.
The older ``--data-dir/--forcings/--select-file/--block`` interface is kept.

Weights: the ``inference-*.ckpt`` given (the saved ``AnemoiModelInterface``). On Vera's branch
the training callback writes the live (raw) network there; the EMA copy is made by
``make_ema_inference.py`` from the Lightning checkpoint. The model is put in eval mode.

Output: ``<out>_m<member>.npz`` per member with ``pred`` (L, grid, n_out) float32,
``variables``, ``leads_h``, ``lat``, ``lon``, ``hdate``, ``refdate``, ``fake_dates``, sampler
record; the truth (L, grid, n_out) is written once as ``<out>_truth.npz`` and each member file
names it in ``truth_file`` (``--truth-in-members`` puts it in every member file instead).
Members whose file exists are skipped (resume).

Example (Atos AC, one GPU, sandbox venv):
  python3 predict_sequence.py --ckpt <work>/checkpoints/inference-ema2000-...-step_020000.ckpt \
      --config <sandbox>/configs/mt2_e2_20k.rendered.yaml --hdate 20230226 --members 10 --seed 1234 \
      --stride 1 --out <work>/runs/mt2_s1/npz/20230226/mt2_20230226
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

STEPS_H = [12, 24, 36, 48, 60, 72]

# The lane's standing sampler (E2 kit predict_e2.yaml, DECISIONS-OPEN 2026-10-02): 30 steps, churn off.
LANE_SCHEDULE = {
    "schedule_type": "experimental_piecewise",
    "num_steps": 30,
    "sigma_max": 100000.0,
    "sigma_transition": 10.0,
    "sigma_min": 0.03,
    "high_schedule_type": "exponential",
    "low_schedule_type": "karras",
    "num_steps_high": 10,
    "num_steps_low": 20,
    "rho": 7.0,
}
# The fork's ExperimentalSamplerScheduler with LANE_SCHEDULE, computed independently (numpy, 2026-10-20).
LANE_SIGMAS_CHECK = [1.0e5, 3.981072e4, 1.584893e4, 6.309573e3, 2.511886e3, 1.0e3, 3.981072e2, 1.584893e2,
                     6.309573e1, 2.511886e1, 1.0e1, 8.098575, 6.515861, 5.206035, 4.128705, 3.248410, 2.534156,
                     1.958977, 1.499534, 1.135729, 8.503616e-1, 6.288003e-1, 4.586837e-1, 3.296458e-1,
                     2.330636e-1, 1.618278e-1, 1.101338e-1, 7.329291e-2, 4.756369e-2, 3.0e-2, 0.0]
DIAG_VARS = ["10u", "10v", "2t", "msl", "tp"]


def register_piecewise_schedule() -> None:
    """Add the fork's two-segment schedule to Vera's SIGMA_SCHEDULES as ``experimental_piecewise``."""
    from anemoi.models.transport import schedules as sch

    class PiecewiseSigmaSchedule(sch.SigmaSchedule):
        """Fork ``ExperimentalSamplerScheduler``: high segment sigma_max -> sigma_transition, low segment to sigma_min."""

        def __init__(self, sigma_max, sigma_min, num_steps, sigma_transition=10.0, high_schedule_type="exponential",
                     low_schedule_type="karras", num_steps_high=None, num_steps_low=None, rho=7.0,
                     rho_high=None, rho_low=None):
            super().__init__(sigma_max=sigma_max, sigma_min=sigma_min, num_steps=num_steps)
            self.sigma_transition = float(sigma_transition)
            self.num_steps_low = num_steps // 2 if num_steps_low is None else int(num_steps_low)
            self.num_steps_high = num_steps - self.num_steps_low if num_steps_high is None else int(num_steps_high)
            if self.num_steps_high + self.num_steps_low != self.num_steps:
                raise ValueError("num_steps_high + num_steps_low must equal num_steps")
            if not self.sigma_min < self.sigma_transition < self.sigma_max:
                raise ValueError("need sigma_min < sigma_transition < sigma_max")
            self.types = (high_schedule_type, low_schedule_type)
            self.rhos = (rho if rho_high is None else rho_high, rho if rho_low is None else rho_low)

        @staticmethod
        def _segment(kind, start, end, n, rho, device, dtype):
            if n <= 1:
                return torch.tensor([start], device=device, dtype=dtype)
            if kind == "exponential":
                return torch.exp(torch.linspace(np.log(start), np.log(end), n, device=device, dtype=dtype))
            if kind == "karras":
                i = torch.arange(n, device=device, dtype=dtype)
                return (start ** (1.0 / rho) + i / (n - 1.0) * (end ** (1.0 / rho) - start ** (1.0 / rho))) ** rho
            raise ValueError(f"unsupported segment type {kind}")

        def _build_schedule(self, device=None, dtype_compute=torch.float64):
            high = self._segment(self.types[0], self.sigma_max, self.sigma_transition, self.num_steps_high + 1,
                                 self.rhos[0], device, dtype_compute)
            low = self._segment(self.types[1], self.sigma_transition, self.sigma_min, self.num_steps_low,
                                self.rhos[1], device, dtype_compute)
            return torch.cat([high, low[1:], torch.zeros(1, device=device, dtype=dtype_compute)])

    sch.SIGMA_SCHEDULES["experimental_piecewise"] = PiecewiseSigmaSchedule
    params = {k: v for k, v in LANE_SCHEDULE.items() if k != "schedule_type"}
    sig = PiecewiseSigmaSchedule(**params).get_schedule(None, torch.float64).numpy()
    ref = np.array(LANE_SIGMAS_CHECK)
    if sig.shape != ref.shape or not np.allclose(sig, ref, rtol=2e-6, atol=0):
        raise SystemExit(f"piecewise schedule check FAILED: {sig} vs {ref}")
    print(f"piecewise schedule check OK: {len(sig) - 1} steps, sigmas {sig[0]:.0f} ... {sig[-2]:.3f}, 0")


def _open(path: str, year: int | None, select: list[str] | None, steps: list[int]):
    from anemoi.datasets import open_dataset

    kwargs = {"fake_hindcasts": {"steps": steps}, "frequency": "1h"}
    if year is not None:
        kwargs["fake_hindcasts"]["start"] = year
        kwargs["fake_hindcasts"]["end"] = year
    if select:
        kwargs["select"] = select
    return open_dataset(path, **kwargs)


def _open_from_config(config_path: str) -> dict:
    """The three test readers, opened with the run's own dataset_config."""
    import yaml
    from anemoi.datasets import open_dataset

    cfg = yaml.safe_load(open(config_path))
    out = {}
    for name in ("in_lres", "in_hres", "out_hres"):
        dc = dict(cfg["dataloader"]["test"]["datasets"][name]["dataset_config"])
        out[name] = (open_dataset(**dc), dc["dataset"])
    return out


def _norm(value) -> str:
    return str(np.datetime64(value, "s"))


def _rows_for_hdate(ds_out, store_path: str, hdate: str, steps: list[int]) -> tuple[list[int], list[str], str]:
    """(hindcast date, lead) -> rows of the view, through the store's fake_hindcasts mapping."""
    import zarr

    fh = dict(zarr.open(store_path, mode="r").attrs["fake_hindcasts"])
    by_target = {}
    for fake, (refdate, hd, step) in fh.items():
        by_target[(str(hd)[:10].replace("-", ""), int(step))] = (_norm(fake), str(refdate)[:10])
    axis = {_norm(d): i for i, d in enumerate(np.asarray(ds_out.dates))}
    rows, fakes, refdates = [], [], set()
    for lead in steps:
        if (hdate, lead) not in by_target:
            raise SystemExit(f"no fake-hindcast row for {hdate} +{lead} h in {store_path}")
        fake, refdate = by_target[(hdate, lead)]
        if fake not in axis:
            raise SystemExit(f"{hdate} +{lead} h (fake date {fake}) is not in the test view")
        rows.append(axis[fake])
        fakes.append(fake)
        refdates.add(refdate)
    if rows != list(range(rows[0], rows[0] + len(steps))) or rows[0] % len(steps) != 0:
        raise SystemExit(f"{hdate}: rows {rows} are not one aligned block of {len(steps)}")
    if len(refdates) != 1:
        raise SystemExit(f"{hdate}: leads map to several reference dates {refdates}")
    return rows, fakes, refdates.pop()


def _block_rows(block: int, k: int) -> list[int]:
    return list(range(block * k, (block + 1) * k))


def _to_batch(ds, rows: list[int]) -> torch.Tensor:
    x = np.asarray(ds[rows[0] : rows[-1] + 1])  # (time, variables, ensemble, grid)
    x = x[:, :, 0, :]  # (time, variables, grid)
    x = np.transpose(x, (0, 2, 1))  # (time, grid, variables)
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None]  # (1, time, grid, variables)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, help="inference-*.ckpt of the multi-time run (raw or EMA copy)")
    parser.add_argument("--config", default=None, help="the run's rendered YAML: open the test readers from it")
    parser.add_argument("--hdate", default=None, help="hindcast date YYYYMMDD; resolves the block (needs --config)")
    parser.add_argument("--data-dir", default=None, help="folder with in_lres_*.zarr and out_hres_*.zarr")
    parser.add_argument("--lres-name", default="in_lres_o320_halo1p5.zarr")
    parser.add_argument("--hres-name", default="out_hres_o1280.zarr")
    parser.add_argument("--forcings", default=None, help="in_hres forcings store (E2's)")
    parser.add_argument("--select-file", default=None, help='JSON {"in_lres": [...], "in_hres": [...], "out_hres": [...]}')
    parser.add_argument("--year", type=int, default=2023, help="fake_hindcasts year (test 2023)")
    parser.add_argument("--steps", default="12,24,36,48,60,72")
    parser.add_argument("--block", type=int, default=0, help="forecast block index within the year's selection")
    parser.add_argument("--leads", default=None, help="subset of consecutive leads, e.g. 12,24 (default: all)")
    parser.add_argument("--members", type=int, default=1, help="number of members (ids from --member-start)")
    parser.add_argument("--member-start", type=int, default=1, help="first member id (seed = seed + id - 1)")
    parser.add_argument("--seed", type=int, default=1234, help="seed of member 1 (E2 kit: 1234)")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--schedule", default="lane", choices=["lane", "karras"],
                        help="lane = the fork's 30-step piecewise schedule; karras = the checkpoint's default")
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--sigma-max", type=float, default=None)
    parser.add_argument("--sigma-min", type=float, default=None)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--truth-in-members", action="store_true")
    parser.add_argument("--out", required=True, help="output prefix; writes <out>_m<member>.npz and <out>_truth.npz")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    steps = [int(s) for s in args.steps.split(",")]
    k_block = len(steps)
    hdate = None
    if args.config:
        readers = _open_from_config(args.config)
        ds_lres, ds_forc, (ds_hres, hres_store) = readers["in_lres"][0], readers["in_hres"][0], readers["out_hres"]
    else:
        select = json.load(open(args.select_file)) if args.select_file else {}
        data_dir = Path(args.data_dir)
        ds_lres = _open(str(data_dir / args.lres_name), args.year, select.get("in_lres"), steps)
        ds_forc = _open(args.forcings, args.year, select.get("in_hres"), steps)
        hres_store = str(data_dir / args.hres_name)
        ds_hres = _open(hres_store, args.year, select.get("out_hres"), steps)
    n_rows = int(ds_lres.shape[0])
    assert n_rows % k_block == 0, f"{n_rows} rows not whole blocks of {k_block}"
    assert int(ds_forc.shape[0]) == n_rows == int(ds_hres.shape[0]), "the three stores must share the axis"

    if args.hdate:
        hdate = args.hdate
        rows, fakes, refdate = _rows_for_hdate(ds_hres, hres_store, hdate, steps)
        block = rows[0] // k_block
    else:
        block = args.block
        rows = _block_rows(block, k_block)
        fakes = [_norm(np.asarray(ds_hres.dates)[r]) for r in rows]
        refdate = ""
    for name, ds in (("in_lres", ds_lres), ("in_hres", ds_forc), ("out_hres", ds_hres)):
        got = [_norm(np.asarray(ds.dates)[r]) for r in rows]
        if got != fakes:
            raise SystemExit(f"reader {name} rows {rows} carry dates {got}, expected {fakes}")
    if args.leads:
        wanted = [int(s) for s in args.leads.split(",")]
        rows = [rows[steps.index(s)] for s in wanted]
        fakes = [fakes[steps.index(s)] for s in wanted]
        leads = wanted
    else:
        leads = steps
    print(f"hdate {hdate} block {block}, view rows {rows}, leads {leads} h, fake dates {fakes[0]} .. {fakes[-1]}, "
          f"refdate {refdate}")

    if args.schedule == "lane":
        register_piecewise_schedule()
        schedule_params = dict(LANE_SCHEDULE)
        schedule_params["num_steps"] = args.num_steps
    else:
        schedule_params = {"num_steps": args.num_steps}
    for key in ("sigma_max", "sigma_min", "rho"):
        value = getattr(args, key)
        if value is not None:
            schedule_params[key] = value

    interface = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = interface.model
    device = torch.device(args.device)
    interface.to(device).eval()
    n_slots_trained = int(interface.n_step_input)
    # The readers must hand the network its training column order, and the output columns must be the
    # out_hres store's order (the truth and the eval files are written in that order).
    for name, ds, part in (("in_lres", ds_lres, "input"), ("in_hres", ds_forc, "input"), ("out_hres", ds_hres, "output")):
        n2i = getattr(interface.data_indices[name].data, part).name_to_index
        trained = sorted(n2i, key=n2i.get)
        if trained != list(ds.variables):
            raise SystemExit(f"{name}: reader variables {list(ds.variables)} differ from the trained order {trained}")
    n2i = interface.data_indices["out_hres"].model.output.name_to_index
    if sorted(n2i, key=n2i.get) != list(ds_hres.variables):
        raise SystemExit(f"out_hres model output order {sorted(n2i, key=n2i.get)} differs from the store order")
    print("variable order check OK: in_lres, in_hres, out_hres readers and the model output match the training order")
    print(f"model trained on {n_slots_trained} slots; sampling {len(rows)} slots with window {n_slots_trained}, "
          f"stride {args.stride}; schedule {schedule_params}")

    batch = {
        "in_lres": _to_batch(ds_lres, rows).to(device),
        "in_hres": _to_batch(ds_forc, rows).to(device),
    }
    truth = _to_batch(ds_hres, rows)[0].numpy()  # (L, grid, vars)
    sampler_params = {"sampler": "heun_window", "window": n_slots_trained, "stride": args.stride, "S_churn": 0.0}

    lat = np.asarray(ds_hres.latitudes)
    lon = np.asarray(ds_hres.longitudes)
    variables = list(ds_hres.variables)
    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    truth_path = out_prefix.parent / f"{out_prefix.name}_truth.npz"
    common = dict(variables=np.array(variables), leads_h=np.array(leads), lat=lat, lon=lon,
                  hdate=str(hdate), refdate=refdate, fake_dates=np.array(fakes), rows=np.array(rows))
    if not args.truth_in_members and not truth_path.exists():
        np.savez(str(truth_path) + ".partial.npz", truth=truth, **common)
        Path(str(truth_path) + ".partial.npz").rename(truth_path)
    record = json.dumps({"sampler": sampler_params, "schedule": schedule_params, "ckpt": str(args.ckpt),
                         "noise_across_leads": "independent (one draw over all slots)", "precision": "fp32"})

    idx = {v: variables.index(v) for v in DIAG_VARS if v in variables}
    for member in range(args.member_start, args.member_start + args.members):
        path = out_prefix.parent / f"{out_prefix.name}_m{member:02d}.npz"
        if path.exists():
            print(f"member {member}: {path} exists, skipped")
            continue
        seed = args.seed + member - 1
        torch.manual_seed(seed)
        t0 = time.time()
        with torch.no_grad():
            pred = model.predict_step(
                batch=batch,
                pre_processors=interface.pre_processors,
                post_processors=interface.post_processors,
                n_step_input=len(rows),
                pre_processors_residual=interface.pre_processors_residual,
                post_processors_residual=interface.post_processors_residual,
                spatial_pre_processors=interface.spatial_pre_processors,
                schedule_params=dict(schedule_params),
                sampler_params=dict(sampler_params),
            )
        y = pred["out_hres"]  # (1, L, 1, grid, n_out) or (1, L, grid, n_out)
        y = y.reshape(len(rows), y.shape[-2], y.shape[-1]).float().cpu().numpy()
        if y.shape != truth.shape:
            raise SystemExit(f"prediction shape {y.shape} differs from truth {truth.shape}")
        extra = {"truth": truth} if args.truth_in_members else {"truth_file": str(truth_path)}
        tmp = Path(str(path) + ".partial.npz")
        np.savez(tmp, pred=y, stride=args.stride, window=n_slots_trained, num_steps=args.num_steps, seed=seed,
                 member=member, sampling_json=record, **common, **extra)
        tmp.rename(path)
        finite = [bool(np.isfinite(y[t]).all()) for t in range(len(rows))]
        print(f"member {member} seed {seed}: {time.time() - t0:.1f} s, wrote {path}, finite per lead {finite}")
        for v, j in idx.items():
            ratios = " ".join(f"{float(y[t, :, j].std() / (truth[t, :, j].std() + 1e-30)):.3f}" for t in range(len(rows)))
            corr = " ".join(f"{float(np.corrcoef(y[t, :, j], truth[t, :, j])[0, 1]):.3f}" for t in range(len(rows)))
            print(f"  {v}: std pred/truth per lead {ratios}; corr {corr}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
