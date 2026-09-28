"""Pre-built ``SceneConfig`` instances. Copy-and-edit to add new ones.

Keep this file as a flat registry of dataclass values — no logic.
"""
from __future__ import annotations

from pathlib import Path

from .config import SceneConfig

# ---------------------------------------------------------------------------
# Default inputs / outputs
# ---------------------------------------------------------------------------

_PRED_DIR_2241ADE8 = Path(
    "/home/ecm5702/scratch/eval/manual_2241ade8_new_o320_o1280_20260503_manual_eval/predictions"
)
_DEFAULT_OUTPUT_DIR = Path(
    "/home/ecm5702/scratch/eval/video_o320_o1280_idalia_franklin_24h"
)

_CKPT_2241ADE8 = "ckpt 2241ade8"

_INITS_AUG_2023 = ("20230826", "20230827", "20230828", "20230829", "20230830")
_STEPS_24H = (24, 48, 72, 96, 120)

# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------

FRANKLIN_DUAL = SceneConfig(
    name="franklin_dual",
    title="Hurricane Franklin — O320 input vs O1280 prediction",
    ckpt_label=_CKPT_2241ADE8,
    predictions_dir=_PRED_DIR_2241ADE8,
    output_dir=_DEFAULT_OUTPUT_DIR,
    inits=_INITS_AUG_2023,
    steps=_STEPS_24H,
    bg_bbox=(-82.0, -45.0, 18.0, 42.0),
    inset_kind="track_msl_min",
    inset_search_bbox=(-78.0, -45.0, 20.0, 42.0),
    inset_half_deg=4.0,
    vars=("msl", "wind"),
    layout="dual_row",
)

FRANKLIN_MAGNIFY = SceneConfig(
    name="franklin",
    title="Hurricane Franklin",
    ckpt_label=_CKPT_2241ADE8,
    predictions_dir=_PRED_DIR_2241ADE8,
    output_dir=_DEFAULT_OUTPUT_DIR,
    inits=_INITS_AUG_2023,
    steps=_STEPS_24H,
    bg_bbox=(-82.0, -50.0, 18.0, 40.0),
    inset_kind="track_msl_min",
    inset_search_bbox=(-78.0, -45.0, 20.0, 42.0),
    inset_half_deg=4.0,
    vars=("msl",),
    layout="single_inset",
)

HIMALAYAS = SceneConfig(
    name="himalayas",
    title="Himalayas",
    ckpt_label=_CKPT_2241ADE8,
    predictions_dir=_PRED_DIR_2241ADE8,
    output_dir=_DEFAULT_OUTPUT_DIR,
    inits=_INITS_AUG_2023,
    steps=_STEPS_24H,
    bg_bbox=(68.0, 102.0, 22.0, 42.0),
    inset_kind="fixed",
    inset_bbox=(82.0, 90.0, 27.0, 34.0),
    vars=("2t",),
    layout="single_inset",
)


SCENES: dict[str, SceneConfig] = {
    "franklin_dual": FRANKLIN_DUAL,
    "franklin": FRANKLIN_MAGNIFY,
    "himalayas": HIMALAYAS,
}
