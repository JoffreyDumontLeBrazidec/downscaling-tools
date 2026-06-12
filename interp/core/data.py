"""Event-bundle data loading + the named-event registry.

ALL interp tools load real event .nc bundles, never datamodule.val_dataloader():
after the 2026-06-11 env reorg the o96->o320 val dataloader yields lres on the
O320 (hres) grid, which makes the assembled encoder input 219-wide vs the
trained 151 and crashes ``emb_nodes_src``. Real bundles carry O96 lres, so
prepare_batch interpolates O96->O320 correctly. Each (date, member, step)
becomes one batch element, so pass several to get batch_size > 1.
"""

from __future__ import annotations

import glob
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from interp.core.model import get_variable_names

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Named events: one place for "which bundles is this experiment about".
# Add new storms / regions / lanes here instead of hardcoding paths in tools.
# ---------------------------------------------------------------------------
EVENTS: dict[str, dict] = {
    # Hurricane Franklin near its 2023-08-28 peak (init 20230826 + 48h).
    "franklin_o96_o320": {
        "bundle_dir": "/home/ecm5702/hpcperm/data/input_data/o96_o320/idalia",
        "dates": ["20230826"],
        "members": ["01"],
        "steps": ["048"],
        "label": "franklin",
        # Single bundle at the storm peak (valid 2023-08-29, +72h) for patching.
        "peak_bundle": ("/home/ecm5702/hpcperm/data/input_data/o96_o320/idalia/"
                        "eefo_o96_0001_date20230826_time0000_mem01_step072h_input_bundle.nc"),
    },
    # Same window, 4 members -> batch_size 4 (needed by permutation).
    "franklin_o96_o320_m4": {
        "bundle_dir": "/home/ecm5702/hpcperm/data/input_data/o96_o320/idalia",
        "dates": ["20230826"],
        "members": ["01", "02", "03", "04"],
        "steps": ["048"],
        "label": "franklin",
    },
}


@dataclass
class BundleBatch:
    """A stacked (x_lres, x_hres, y) batch built from real event bundles."""

    x_lres: torch.Tensor
    x_hres: torch.Tensor
    y: torch.Tensor
    paths: list[str] = field(default_factory=list)
    # (lat_lres, lon_lres, lat_hres, lon_hres) numpy arrays, or None
    coords: tuple | None = None


def find_bundles(bundle_dir, dates, members, steps) -> list[str]:
    """Resolve (date, member, step) triplets to bundle paths (one per triplet)."""
    paths = []
    for d in dates:
        for m in members:
            for s in steps:
                pat = f"*date{d}*mem{m}*step{s}h*input_bundle.nc"
                hits = sorted(glob.glob(str(Path(bundle_dir) / pat)))
                if not hits:
                    raise FileNotFoundError(f"No bundle matching {pat} in {bundle_dir}")
                paths.append(hits[0])
    return paths


def collect_event_bundles(bundle, bundle_dir, dates, members, steps) -> BundleBatch:
    """Build a BundleBatch from real event .nc bundles.

    Targets are NaN->0 sanitized (a bundle may carry fewer target channels than
    the output schema; NaNs would otherwise poison every MSE downstream).
    """
    from manual_inference.input_data_construction.bundle import (
        load_inputs_from_bundle_numpy as _load_bundle_np,
        extract_target_from_bundle as _extract_target,
    )

    vn = get_variable_names(bundle)
    n2i_lres = {name: idx for idx, name in vn["input_lres"].items()}
    n2i_hres = {name: idx for idx, name in vn["input_hres"].items()}
    out_states = [vn["output"][i] for i in sorted(vn["output"])]

    paths = find_bundles(bundle_dir, dates, members, steps)
    xl, xh, ys, coords = [], [], [], None
    for p in paths:
        x_lres_np, x_hres_np, lon_l, lat_l, lon_h, lat_h = _load_bundle_np(
            p, n2i_lres, n2i_hres)
        target_np, found = _extract_target(p, out_states)
        if target_np is None:
            raise SystemExit(f"Could not extract target from bundle {p}")
        n_nan = int(np.isnan(target_np).sum())
        if n_nan:
            LOGGER.info("bundle %s: target coverage %s/%s, %d NaN cells -> 0",
                        Path(p).name, found, len(out_states), n_nan)
        xl.append(torch.from_numpy(x_lres_np)[None, None, None, ...])
        xh.append(torch.from_numpy(x_hres_np)[None, None, None, ...])
        ys.append(torch.from_numpy(np.nan_to_num(target_np, nan=0.0))[None, None, None, ...])
        coords = (np.asarray(lat_l), np.asarray(lon_l),
                  np.asarray(lat_h), np.asarray(lon_h))
    x_lres, x_hres, y = torch.cat(xl), torch.cat(xh), torch.cat(ys)
    LOGGER.info("collect_event_bundles: %d bundle(s) from %s x_lres=%s x_hres=%s y=%s",
                len(paths), bundle_dir, tuple(x_lres.shape), tuple(x_hres.shape),
                tuple(y.shape))
    return BundleBatch(x_lres=x_lres, x_hres=x_hres, y=y, paths=paths, coords=coords)


def load_single_bundle(bundle, bundle_path: str) -> BundleBatch:
    """Load one explicit .nc bundle path (batch_size 1)."""
    p = Path(bundle_path)
    from manual_inference.input_data_construction.bundle import (
        load_inputs_from_bundle_numpy as _load_bundle_np,
        extract_target_from_bundle as _extract_target,
    )
    vn = get_variable_names(bundle)
    n2i_lres = {name: idx for idx, name in vn["input_lres"].items()}
    n2i_hres = {name: idx for idx, name in vn["input_hres"].items()}
    out_states = [vn["output"][i] for i in sorted(vn["output"])]

    x_lres_np, x_hres_np, lon_l, lat_l, lon_h, lat_h = _load_bundle_np(
        str(p), n2i_lres, n2i_hres)
    target_np, found = _extract_target(str(p), out_states)
    if target_np is None:
        raise RuntimeError(f"bundle has no target channels: {bundle_path}")
    LOGGER.info("load_single_bundle: %s target coverage %s/%s",
                p.name, found, len(out_states))
    return BundleBatch(
        x_lres=torch.from_numpy(x_lres_np)[None, None, None, ...],
        x_hres=torch.from_numpy(x_hres_np)[None, None, None, ...],
        y=torch.from_numpy(np.nan_to_num(target_np, nan=0.0))[None, None, None, ...],
        paths=[str(p)],
        coords=(np.asarray(lat_l), np.asarray(lon_l),
                np.asarray(lat_h), np.asarray(lon_h)),
    )


def resolve_event_args(args) -> tuple[str, list, list, list, str]:
    """Resolve --event vs explicit --bundle-dir/--dates/--members/--steps.

    Returns (bundle_dir, dates, members, steps, label). --event supplies the
    defaults; any explicit flag overrides the event's value.
    """
    ev = EVENTS.get(getattr(args, "event", None) or "", None)
    if getattr(args, "event", None) and ev is None:
        raise SystemExit(f"unknown --event {args.event!r}; options: {sorted(EVENTS)}")
    bundle_dir = getattr(args, "bundle_dir", None) or (ev or {}).get("bundle_dir")
    if not bundle_dir:
        raise SystemExit("need --event <name> or --bundle-dir <dir> "
                         f"(events: {sorted(EVENTS)})")
    dates = getattr(args, "dates", None) or (ev or {}).get("dates") or ["20230826"]
    members = getattr(args, "members", None) or (ev or {}).get("members") or ["01"]
    steps = getattr(args, "steps", None) or (ev or {}).get("steps") or ["048"]
    label = (ev or {}).get("label", "event")
    return bundle_dir, dates, members, steps, label
