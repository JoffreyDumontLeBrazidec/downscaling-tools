"""tp_maps runner: groups the prediction files by base date and calls the map_video renderer.

Per base date it writes the whole map (stills + video), one zoom per region (stills + video), and
small previews of each (PNG and MP4 under 1 MB). On a whole global grid the regions default to
``GLOBAL_CROPS`` (the tropical-cyclone basins of the 9 Nov demo); a crop run (the E2 box) is already
a zoom and gets none unless the lane names some.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from eval.discovery.predictions import find_predictions

LOG = logging.getLogger(__name__)

# name -> [W, E, S, N] in degrees; the West Pacific and Atlantic crops of the 9 Nov GL512 demo, plus Europe
GLOBAL_CROPS = {
    "west_pacific": [60.0, 180.0, -20.0, 50.0],
    "atlantic": [-100.0, 20.0, -15.0, 55.0],
    "europe": [-30.0, 45.0, 30.0, 72.0],
}
CROP_RES = 0.05


def _ffmpeg(explicit: str | None) -> str | None:
    from eval.tools.map_video.render import find_ffmpeg

    try:
        return find_ffmpeg(explicit)
    except RuntimeError:
        return None


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


def has_variable(path: Path, variable: str) -> bool:
    """True when the file's model output holds ``variable`` (``10ff`` needs ``10u`` and ``10v``)."""
    import xarray as xr

    with xr.open_dataset(path) as ds:
        if "y_pred" not in ds or "weather_state" not in ds["y_pred"].dims:
            return False
        names = {str(v) for v in ds["y_pred"]["weather_state"].values}
    want = [("10u", "10v")] if variable == "10ff" else [(variable,), (f"{variable}_sfc",)]
    return any(all(n in names for n in group) for group in want)


def is_whole_globe(path: Path) -> bool:
    from eval.tools.map_video.frames import read_coords
    from eval.tools.map_video.grid import octahedral_n_from_size

    lat, _ = read_coords(path)
    return octahedral_n_from_size(lat.size) is not None


def crops_for(cfg: dict, whole_globe: bool) -> dict[str, list[float]]:
    """The lane's ``crops`` ({name: [W, E, S, N]}; false or {} for none), else the global defaults."""
    crops = cfg.get("crops")
    if crops is None:
        return dict(GLOBAL_CROPS) if whole_globe else {}
    if not crops:
        return {}
    return {str(k): [float(x) for x in v] for k, v in dict(crops).items()}


def run(predictions_dir, lane_config, eval_config, *, output_dir=None, overwrite=False, checkpoint=None,
        run_label="", **kwargs) -> Path:
    from eval.tools.map_video.__main__ import main as map_video

    from . import previews as pv

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
    first = next(iter(groups.values()))[0]
    if not has_variable(first, variable):
        msg = f"tp_maps: {first.name} has no {variable} in y_pred; nothing drawn"
        LOG.warning(msg)
        (output_dir / "SKIPPED.txt").write_text(msg + "\n")
        return output_dir

    ffmpeg = _ffmpeg(cfg.get("ffmpeg")) if cfg.get("video", True) else None
    if cfg.get("video", True) and ffmpeg is None:
        LOG.warning("tp_maps: no ffmpeg found; writing stills only (module load ffmpeg or pip install imageio-ffmpeg)")
    crops = crops_for(cfg, is_whole_globe(first))
    want_previews = bool(cfg.get("previews", True))

    common = ["--variable", variable, "--sources", str(cfg.get("sources", "auto")),
              "--member", str(int(cfg.get("member", 0))), "--palette", str(cfg.get("palette", "classes")),
              "--tween", str(int(cfg.get("tween", 2))), "--fps", str(cfg.get("fps", 4))]
    label = str(cfg.get("run_label") or run_label or "")
    if label:
        common += ["--label", label]
    for key, flag in (("accum_hours", "--accum-hours"), ("grid", "--grid"), ("geography", "--geography")):
        if cfg.get(key) is not None:
            common += [flag, str(cfg[key])]
    if ffmpeg:
        common += ["--ffmpeg", ffmpeg]

    views = {"": (cfg.get("extent"), cfg.get("res"))}
    views.update({name: (ext, cfg.get("crop_res", CROP_RES)) for name, ext in crops.items()})
    for date, paths in groups.items():
        for name, (extent, res) in views.items():
            base = output_dir / "crops" / name if name else output_dir
            stills_dir = base / "stills"
            argv = [str(p) for p in paths] + common + ["--stills-dir", str(stills_dir)]
            if res is not None:
                argv += ["--res", str(res)]
            if extent:
                argv += ["--extent", *[str(v) for v in extent]]
            video = base / f"{variable}_{date}.mp4" if ffmpeg and len(paths) > 1 else None
            if video:
                argv += ["--out", str(video)]
            LOG.info("tp_maps %s %s: %d file(s)%s", date, name or "map", len(paths), " + video" if video else "")
            map_video(argv)

            if not want_previews:
                continue
            tag = f"{variable}_{date}" + (f"_{name}" if name else "")
            still = pv.poster_still(list(stills_dir.glob(f"{variable}_{date}??_+*h.png")))
            if still is not None:
                pv.png_preview(still, output_dir / "previews" / f"{tag}.png")
            if video is not None and video.exists():
                pv.mp4_preview(video, output_dir / "previews" / f"{tag}.mp4", ffmpeg=ffmpeg)
    return output_dir
