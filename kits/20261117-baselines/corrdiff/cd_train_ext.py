"""Training additions of the CorrDiff kit (27 Nov), used by train_cd.py (= the PhysicsNeMo v2.2.2 example train.py with
train_cd_v2.2.2.patch applied). Everything here is opt-in from the training config; without `training.ema` and without
`dataset.crop` train_cd.py trains exactly as train.py.

EMA (restored from the CorrDiff paper; absent from the v2.2.2 code): EDM's update (Karras et al. 2022, training_loop.py),
  half-life h = min(halflife_nimg, rampup_ratio * cur_nimg) samples, beta = 0.5 ** (batch / max(h, 1e-8)),
  p_ema <- lerp(p_net, p_ema, beta) after every optimizer step, cur_nimg = samples before the step (EDM's order).
  Defaults = EDM's: halflife 500 kimg = 500,000 samples, rampup_ratio 0.05. Parameters only (buffers are copied once:
  the regression's sinusoidal grid is a constant). Kept on rank 0 only (it saves). Saved beside every checkpoint as
  <checkpoint_dir>_ema/<Model>.0.<cur_nimg>.mdlus (same name as the raw .mdlus, sibling folder, so the kit's
  *.mdlus globs in checkpoints_* see the raw files only). Resume: the EMA file of the resumed sample count is loaded;
  a resume (cur_nimg > 0) without it stops the run (no silent restart of the average).

Regression on random crops: the dataset (cd_dataset.py crop: N) returns (y, x, origin); the U-Net is built at the full
  box size (dataset.image_shape()), and crop_global_index(origin) gives SongUNetPosEmbd the box indices of every crop
  pixel, so its positional grid (a fixed function of the box index) is selected at the crop: the same embedding value
  at the same box point as in a full-box forward. Each crop is its own "patch" (P = batch, one sample per patch).
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

import numpy as np
import torch


def unwrap(model):
    m = model
    while hasattr(m, "module") and isinstance(m, (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)):
        m = m.module
    return getattr(m, "_orig_mod", m)


def ema_dir(checkpoint_dir: str) -> str:
    return str(checkpoint_dir).rstrip("/") + "_ema"


class EMA:
    def __init__(self, model, halflife_nimg: float = 500_000, rampup_ratio: float | None = 0.05):
        self.halflife_nimg = float(halflife_nimg)
        self.rampup_ratio = None if rampup_ratio in (None, 0, "null") else float(rampup_ratio)
        self.module = copy.deepcopy(unwrap(model)).eval().requires_grad_(False)
        self.beta = None

    def beta_at(self, cur_nimg: int, batch: int) -> float:
        h = self.halflife_nimg
        if self.rampup_ratio is not None:
            h = min(h, cur_nimg * self.rampup_ratio)
        return 0.5 ** (batch / max(h, 1e-8))

    @torch.no_grad()
    def update(self, model, cur_nimg: int, batch: int) -> float:
        b = self.beta_at(cur_nimg, batch)
        for p_ema, p_net in zip(self.module.parameters(), unwrap(model).parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, b))
        self.beta = b
        return b

    def path(self, checkpoint_dir: str, cur_nimg: int) -> str:
        name = type(self.module).__name__
        return f"{ema_dir(checkpoint_dir)}/{name}.0.{int(cur_nimg)}.mdlus"

    def save(self, checkpoint_dir: str, cur_nimg: int, keep: int = -1) -> str:
        p = self.path(checkpoint_dir, cur_nimg)
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        self.module.save(p + ".part.mdlus")
        os.replace(p + ".part.mdlus", p)
        if keep and keep > 0:
            files = sorted(Path(p).parent.glob(f"{type(self.module).__name__}.0.*.mdlus"),
                           key=lambda f: int(f.name.split(".")[-2]) if f.name.split(".")[-2].isdigit() else -1)
            for f in files[:-keep]:
                f.unlink()
        return p

    def load(self, checkpoint_dir: str, cur_nimg: int, device) -> str | None:
        """EMA weights of the checkpoint at cur_nimg, or None if the file is absent."""
        p = self.path(checkpoint_dir, cur_nimg)
        if not os.path.exists(p):
            return None
        from physicsnemo import Module
        m = Module.from_checkpoint(p)
        sd = m.state_dict()
        missing = self.module.load_state_dict(sd, strict=True)
        del m
        self.module.to(device)
        return f"{p} ({missing})"


def crop_global_index(origin: torch.Tensor, h: int, w: int, device) -> torch.Tensor:
    """origin (B, 2) [oy, ox] -> (B, 2, h, w) long: box row / column index of every crop pixel."""
    o = origin.to(device=device, dtype=torch.long)
    dy = torch.arange(h, device=device, dtype=torch.long)
    dx = torch.arange(w, device=device, dtype=torch.long)
    gy = (o[:, 0, None, None] + dy[None, :, None]).expand(-1, h, w)
    gx = (o[:, 1, None, None] + dx[None, None, :]).expand(-1, h, w)
    return torch.stack([gy, gx], dim=1).contiguous()


class CropRegressionLoss:
    """PhysicsNeMo RegressionLoss (v2.2.2, legacy_losses.py) with the crop's global_index passed to the U-Net
    (RegressionLoss itself passes none, which selects the whole positional grid)."""

    def __call__(self, net, img_clean, img_lr, global_index, augment_pipe=None):
        assert augment_pipe is None
        zero_input = torch.zeros_like(img_clean, device=img_clean.device)
        D_yn = net(zero_input, img_lr, force_fp32=False, global_index=global_index)
        return (D_yn - img_clean) ** 2


class CropVisits:
    """Histogram of the crop origins drawn on this rank, and the pixel visit counts they give (written as npz)."""

    def __init__(self, img_shape, crop: int):
        self.ny, self.nx = int(img_shape[0]), int(img_shape[1])
        self.crop = int(crop)
        self.hist = np.zeros((self.ny - self.crop + 1, self.nx - self.crop + 1), np.int64)

    def add(self, origin: torch.Tensor):
        o = origin.detach().cpu().numpy().reshape(-1, 2)
        np.add.at(self.hist, (o[:, 0], o[:, 1]), 1)

    @staticmethod
    def visits_from(hist: np.ndarray, crop: int, shape) -> np.ndarray:
        """pixel (i, j) visit count = number of drawn origins with oy <= i < oy + crop and ox <= j < ox + crop."""
        ny, nx = shape
        d = np.zeros((ny + 1, nx + 1), np.int64)
        H, W = hist.shape
        d[:H, :W] += hist
        d[crop:crop + H, :W] -= hist
        d[:H, crop:crop + W] -= hist
        d[crop:crop + H, crop:crop + W] += hist
        return d.cumsum(0).cumsum(1)[:ny, :nx]

    def expected(self) -> np.ndarray:
        """visit probability of each pixel per crop under the uniform origin law (exact)."""
        H, W = self.hist.shape
        return self.visits_from(np.ones((H, W), np.int64), self.crop, (self.ny, self.nx)) / float(H * W)

    def write(self, path: str, cur_nimg: int, world_size: int):
        v = self.visits_from(self.hist, self.crop, (self.ny, self.nx))
        n = int(self.hist.sum())
        e = self.expected()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path + ".part.npz", origin_hist=self.hist.astype(np.int32), visits=v.astype(np.int32),
                            expected_per_crop=e.astype(np.float32), n_crops=n, crop=self.crop, cur_nimg=int(cur_nimg),
                            ranks_counted=1, world_size=int(world_size))
        os.replace(path + ".part.npz", path)
        return n, float(e.min()), float(e.max())
