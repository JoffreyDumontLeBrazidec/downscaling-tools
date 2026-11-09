"""Map videos of downscaled fields: ``python -m eval.tools.map_video FILES... --out video.mp4``.

Each input file is one (base date, lead) prediction file; frames are sorted by valid time.
The field is interpolated from its native octahedral grid (O320, O1280, O2560, whole or a crop)
to a regular latitude-longitude grid with ring-wise bilinear weights (``grid.py``), and each
grid cell is one pixel of the video. Defaults: total precipitation, the model (``pred``),
member 0, global at 0.1 degrees (3600 x 1801, a 3840 x 2160 frame), 4 frames per second.

Examples
--------
Global 6 h precipitation of member 0, 0.1 degrees, 4K::

    python -m eval.tools.map_video run/predictions_20230829_step*.nc --out tp.mp4 --poster tp.png

Input, model and truth side by side on the Europe box at 0.05 degrees::

    python -m eval.tools.map_video pred/*.nc --sources input,pred,truth \\
        --extent -80 40 10 70 --res 0.05 --out tp_e2.mp4

Smoother playback for 6-hourly leads (two time-interpolated frames between leads)::

    ... --tween 2 --fps 6
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np

from . import frames as fr
from .grid import OctahedralGrid, RingBilinear, octahedral_n_from_size, regular_lat_lon
from .palette import continuous_palette, tp_palette
from .render import FrameComposer, Layout, Mp4Writer, load_geography

ACCUM_BY_GRID = {1280: 12, 2560: 6}  # tp window of the 9 km (12 h) and 4 km (6 h) lane stores


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m eval.tools.map_video", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", type=Path, help="prediction files, one per (base date, lead)")
    p.add_argument("--out", type=Path, required=True, help="output .mp4")
    p.add_argument("--poster", type=Path, help="also write one frame as PNG (see --poster-index)")
    p.add_argument("--poster-index", type=int, default=0, help="which real frame the poster is (default 0)")
    p.add_argument("--variable", default="tp", help="tp (default), 10ff, 2t, msl, ... (weather_state name)")
    p.add_argument("--sources", default="pred", help="comma list of pred, input, truth (one panel each)")
    p.add_argument("--member", type=int, default=0, help="ensemble member position (default 0)")
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--res", type=float, default=0.1, help="output grid spacing in degrees (default 0.1)")
    p.add_argument("--extent", type=float, nargs=4, metavar=("W", "E", "S", "N"),
                   help="box instead of the globe (degrees)")
    p.add_argument("--central-lon", type=float, default=0.0, help="centre longitude of a global map")
    p.add_argument("--grid", help="native grid, e.g. O1280 (needed only when the file is a crop)")
    p.add_argument("--palette", default="classes", choices=("classes", "rain", "reference"),
                   help="tp colours: classes = blue-green-yellow-red-purple of the 9 Nov frame (default, "
                        "the owner's choice for videos); rain = house cmocean rain (rule 25); reference = classes")
    p.add_argument("--accum-hours", type=int, help="tp accumulation window for the title (default: 12 h "
                   "for O1280, 6 h for O2560)")
    p.add_argument("--vmin", type=float)
    p.add_argument("--vmax", type=float)
    p.add_argument("--label", default="", help="what is shown, first words of the title (run name)")
    p.add_argument("--fps", type=float, default=4.0)
    p.add_argument("--tween", type=int, default=0, help="time-interpolated frames between two leads")
    p.add_argument("--hold", type=float, default=1.0, help="seconds to hold the last frame")
    p.add_argument("--crf", type=int, default=16, help="H.264 quality, lower is better (default 16)")
    p.add_argument("--chroma", default="yuv420p", choices=("yuv420p", "yuv444p"),
                   help="yuv444p keeps thin colour detail but some players refuse it")
    p.add_argument("--max-width", type=int, default=7680, help="coarsen --res so the frame fits (8K)")
    p.add_argument("--geography", type=Path, help=".npz of coastlines and borders (written from cartopy "
                   "if missing); default: cartopy Natural Earth 50m")
    p.add_argument("--no-geography", action="store_true")
    p.add_argument("--ffmpeg", help="ffmpeg executable")
    p.add_argument("--frames-dir", type=Path, help="also write every frame as PNG here")
    return p


def _title(args, f: fr.FrameFile, member: str, grid_name: str, shape, *, valid_override=None) -> str:
    valid = valid_override or f.valid
    parts = [args.label] if args.label else []
    parts.append(f"member {member}")
    lead = f"+{f.step} h" if valid_override is None else f"+{(valid - f.init).total_seconds() / 3600:.0f} h"
    parts.append(f"{f.init:%Y-%m-%d %H} UTC {lead} (valid {valid:%Y-%m-%d %H} UTC"
                 + (", time-interpolated)" if valid_override is not None else ")"))
    if args.variable == "tp":
        parts.append(f"{args.accum_hours} h precipitation")
    else:
        from eval.plotting.variables import display_name

        parts.append(display_name(args.variable))
    parts.append(f"bilinear from {grid_name} to {args.res:g} degrees ({shape[1]} x {shape[0]})")
    return ", ".join(parts)


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    bad = [s for s in sources if s not in fr.SOURCE_VARS]
    if bad:
        raise SystemExit(f"unknown source(s) {bad}; use pred, input, truth")

    files = fr.scan(args.files, sample=args.sample)
    if not files:
        raise SystemExit("no files")
    print(f"{len(files)} frames: {files[0].valid:%Y-%m-%d %H} .. {files[-1].valid:%Y-%m-%d %H} UTC", flush=True)

    lat_s, lon_s = fr.read_coords(files[0].path)
    n = octahedral_n_from_size(lat_s.size)
    if args.grid:
        n = int(args.grid.upper().lstrip("O"))
    if n is None:
        raise SystemExit(f"{lat_s.size} points is not a whole octahedral grid; pass --grid O<N> for a crop")
    grid = OctahedralGrid.build(n)
    whole = lat_s.size == grid.size
    src_index = grid.index_of(lat_s, lon_s)
    if whole and np.array_equal(src_index, np.arange(grid.size)):
        src_index = None

    if args.extent is None and not whole:
        # a crop: show its bounding box, not an almost empty globe
        lon_c = (lon_s + 180.0) % 360.0 - 180.0
        args.extent = (float(lon_c.min()), float(lon_c.max()), float(lat_s.min()), float(lat_s.max()))
        print(f"crop of {grid.name}: extent {tuple(round(v, 2) for v in args.extent)}", flush=True)

    span_lon = 360.0 if args.extent is None else (args.extent[1] - args.extent[0]) % 360.0 or 360.0
    min_res = span_lon / max(1.0, (args.max_width - 200) / len(sources))
    if args.res < min_res - 1e-9:
        print(f"--res {args.res:g} gives a frame wider than {args.max_width} px; using {min_res:.3g}", flush=True)
        args.res = float(np.ceil(min_res * 1000) / 1000)
    lat_t, lon_t = regular_lat_lon(args.res, central_longitude=args.central_lon, extent=args.extent)
    t0 = time.time()
    interp = RingBilinear(grid, lat_t, lon_t, source_index=src_index)
    print(f"weights {grid.name} -> {lat_t.size} x {lon_t.size}: {time.time() - t0:.1f} s", flush=True)

    if args.variable == "tp":
        if args.accum_hours is None:
            args.accum_hours = ACCUM_BY_GRID.get(n, 6)
            print(f"tp window assumed {args.accum_hours} h for {grid.name} (set --accum-hours)", flush=True)
        palette = tp_palette(args.palette, label=f"{args.accum_hours} h precipitation (mm)")
    else:
        palette = continuous_palette(args.variable, args.vmin, args.vmax)

    from eval.plotting.variables import convert

    def field(f: fr.FrameFile, source: str) -> np.ndarray:
        native = fr.read_field(f.path, source, args.variable, member=args.member, sample=args.sample)
        return np.asarray(convert(args.variable, interp(native)), dtype=np.float32)

    geo = None if args.no_geography else load_geography(args.geography)
    layout = Layout.build(lon_t.size, lat_t.size, len(sources), labels=len(sources) > 1)
    titles = [fr.SOURCE_TITLES[s] for s in sources] if len(sources) > 1 else []
    comp = FrameComposer(layout, lat_t, lon_t, palette, panel_titles=titles, geography=geo)
    member = fr.member_label(files[0].path, args.member)
    print(f"frame {layout.width} x {layout.height}, {len(sources)} panel(s)", flush=True)

    writer = Mp4Writer(args.out, layout.width, layout.height, args.fps, crf=args.crf, chroma=args.chroma,
                       ffmpeg=args.ffmpeg)
    if args.frames_dir:
        args.frames_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    def emit(img: np.ndarray, k: int, poster: bool):
        writer.write(img)
        if args.frames_dir:
            Image.fromarray(img).save(args.frames_dir / f"frame_{k:04d}.png")
        if poster and args.poster:
            args.poster.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img).save(args.poster)

    prev = None
    k = 0
    t0 = time.time()
    for i, f in enumerate(files):
        cur = [field(f, s) for s in sources]
        if i == 0:
            for s_name, v in zip(sources, cur):
                finite = np.isfinite(v)
                if not finite.any():
                    print(f"WARNING: {s_name} {args.variable} is all NaN in {f.path.name} (empty panel)", flush=True)
                elif np.nanmax(np.abs(v)) == 0:
                    print(f"WARNING: {s_name} {args.variable} is exactly 0 everywhere in {f.path.name} "
                          "(the input of the 9 km lane carries no tp)", flush=True)
        if prev is not None and args.tween > 0:
            pf, pfields = prev
            same_run = pf.init == f.init
            for t in range(1, args.tween + 1):
                a = t / (args.tween + 1)
                mix = [(1 - a) * p + a * c for p, c in zip(pfields, cur)]
                vt = pf.valid + timedelta(seconds=a * (f.valid - pf.valid).total_seconds())
                ref = pf if same_run else f
                emit(comp.compose(mix, _title(args, ref, member, grid.name, interp.shape, valid_override=vt)),
                     k, False)
                k += 1
        img = comp.compose(cur, _title(args, f, member, grid.name, interp.shape))
        emit(img, k, i == args.poster_index)
        k += 1
        prev = (f, cur)
        print(f"  {i + 1}/{len(files)} {f.path.name} ({time.time() - t0:.0f} s)", flush=True)
    for _ in range(int(round(args.hold * args.fps))):
        writer.write(img)
    writer.close()
    print(f"wrote {args.out} ({k} frames + hold)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
