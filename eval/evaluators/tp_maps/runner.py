"""tp_maps runner: groups the prediction files by base date and calls the map_video renderer."""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from eval.discovery.predictions import find_predictions

LOG = logging.getLogger(__name__)


def _ffmpeg_available(explicit: str | None) -> bool:
    from eval.tools.map_video.render import find_ffmpeg

    try:
        find_ffmpeg(explicit)
        return True
    except RuntimeError:
        return False


def plan(predictions_dir: Path, eval_config: dict) -> dict[str, list[Path]]:
    """Files to draw, by base date: the lane's ``dates``/``steps`` or the first ``max_dates`` dates."""
    files = find_predictions(predictions_dir)
    steps = {int(s) for s in eval_config.get("steps") or []}
    if steps:
        files = [f for f in files if f.step in steps]
    by_date: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        by_date[f.date].append(f.path)
    dates = [str(d) for d in eval_config.get("dates") or []]
    if dates:
        return {d: by_date[d] for d in dates if by_date.get(d)}
    keep = sorted(by_date)[: int(eval_config.get("max_dates", 3))]
    return {d: by_date[d] for d in keep}


def run(predictions_dir, lane_config, eval_config, *, output_dir=None, overwrite=False, checkpoint=None,
        run_label="", **kwargs) -> Path:
    from eval.tools.map_video.__main__ import main as map_video

    predictions_dir = Path(predictions_dir).expanduser().resolve()
    output_dir = Path(output_dir) if output_dir else predictions_dir / "evaluators" / "tp_maps"
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"tp_maps output exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = plan(predictions_dir, eval_config or {})
    if not groups:
        raise FileNotFoundError(f"No prediction files to draw in {predictions_dir}")
    cfg = eval_config or {}
    variable = str(cfg.get("variable", "tp"))
    video = bool(cfg.get("video", True)) and _ffmpeg_available(cfg.get("ffmpeg"))
    if cfg.get("video", True) and not video:
        LOG.warning("tp_maps: no ffmpeg found; writing stills only (module load ffmpeg or pip install imageio-ffmpeg)")

    common = ["--variable", variable, "--sources", str(cfg.get("sources", "auto")),
              "--member", str(int(cfg.get("member", 0))), "--palette", str(cfg.get("palette", "classes")),
              "--tween", str(int(cfg.get("tween", 2))), "--fps", str(cfg.get("fps", 4)),
              "--stills-dir", str(output_dir / "stills")]
    label = str(cfg.get("run_label") or run_label or "")
    if label:
        common += ["--label", label]
    for key, flag in (("res", "--res"), ("accum_hours", "--accum-hours"), ("grid", "--grid"),
                      ("geography", "--geography"), ("ffmpeg", "--ffmpeg")):
        if cfg.get(key) is not None:
            common += [flag, str(cfg[key])]
    if cfg.get("extent"):
        common += ["--extent", *[str(v) for v in cfg["extent"]]]

    for date, paths in groups.items():
        argv = [str(p) for p in paths] + common
        if video and len(paths) > 1:
            argv += ["--out", str(output_dir / f"{variable}_{date}.mp4")]
        LOG.info("tp_maps %s: %d file(s)%s", date, len(paths), " + video" if "--out" in argv else "")
        map_video(argv)
    return output_dir
