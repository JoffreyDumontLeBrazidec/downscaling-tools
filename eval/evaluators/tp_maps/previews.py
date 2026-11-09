"""Small copies of the tp_maps outputs: a PNG and an MP4 under 1 MB each, for notes and chat.

The docs repository refuses files over 1 MB, so every hand-off of the 9 Nov videos needed downsized
copies made by hand. These are the same steps: the poster scaled to 1600 px wide (palette-quantised
when still too large) and the video scaled to 1280 px wide, with the H.264 quality lowered step by
step until it fits.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

LOG = logging.getLogger(__name__)

MAX_BYTES = 1_000_000
PNG_WIDTH = 1600
MP4_WIDTH = 1280
CRF_LADDER = (28, 32, 36, 40)


def png_preview(src: Path, dst: Path, *, width: int = PNG_WIDTH, max_bytes: int = MAX_BYTES) -> Path:
    """``src`` scaled to ``width`` px (never up), saved as PNG; quantised to 256 colours if over ``max_bytes``."""
    from PIL import Image

    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        if im.width > width:
            im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
        im.save(dst, optimize=True)
        if dst.stat().st_size > max_bytes:
            im.quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(dst, optimize=True)
    if dst.stat().st_size > max_bytes:
        LOG.warning("preview %s is %d bytes (over %d)", dst.name, dst.stat().st_size, max_bytes)
    return dst


def mp4_preview(src: Path, dst: Path, *, ffmpeg: str, width: int = MP4_WIDTH,
                max_bytes: int = MAX_BYTES) -> Path:
    """``src`` scaled to ``width`` px wide, re-encoded at the first CRF of the ladder that fits ``max_bytes``."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    for crf in CRF_LADDER:
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(src),
               "-vf", f"scale='min({width},iw)':-2:flags=lanczos", "-c:v", "libx264", "-preset", "slow",
               "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
        subprocess.run(cmd, check=True)
        if dst.stat().st_size <= max_bytes:
            return dst
    LOG.warning("preview %s is %d bytes at crf %d (over %d)", dst.name, dst.stat().st_size, CRF_LADDER[-1],
                max_bytes)
    return dst


def poster_still(stills: list[Path]) -> Path | None:
    """The still of the middle lead (the 9 Nov posters were +36 h of +12..+72 h)."""
    if not stills:
        return None
    return sorted(stills)[(len(stills) - 1) // 2]
