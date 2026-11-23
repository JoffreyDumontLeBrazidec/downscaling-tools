"""Paste existing eval.cli PNGs into one page (rows of panels), nothing redrawn."""
from __future__ import annotations

from pathlib import Path


def stitch(rows: list[list[Path]], out: Path, *, title: str | None = None, pad: int = 12) -> Path:
    """Rows of images -> one PNG. Images of a row are scaled to the row's smallest height, rows to one width."""
    from PIL import Image, ImageDraw

    built = []
    for row in rows:
        ims = [Image.open(p).convert("RGB") for p in row if p and Path(p).exists()]
        if not ims:
            continue
        h = min(im.height for im in ims)
        ims = [im.resize((max(1, round(im.width * h / im.height)), h)) for im in ims]
        w = sum(im.width for im in ims) + pad * (len(ims) - 1)
        canvas = Image.new("RGB", (w, h), "white")
        x = 0
        for im in ims:
            canvas.paste(im, (x, 0))
            x += im.width + pad
        built.append(canvas)
    if not built:
        raise FileNotFoundError(f"stitch: no image to paste for {out}")
    width = max(im.width for im in built)
    built = [im if im.width == width else im.resize((width, round(im.height * width / im.width))) for im in built]
    top = 60 if title else 0
    page = Image.new("RGB", (width, top + sum(im.height for im in built) + pad * (len(built) - 1)), "white")
    if title:
        draw = ImageDraw.Draw(page)
        try:
            from PIL import ImageFont
            font = ImageFont.load_default(size=32)
        except TypeError:  # Pillow < 10.1
            font = None
        draw.text((pad, 12), title, fill="black", font=font)
    y = top
    for im in built:
        page.paste(im, (0, y))
        y += im.height + pad
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    page.save(out, optimize=True)
    return out
