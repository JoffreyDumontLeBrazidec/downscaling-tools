"""Style context manager and figure-saving helpers.

Nothing in this package touches Matplotlib's global state when it is imported. A figure is
drawn inside ``with eval_style():`` (or a function is decorated with ``@styled``), and is
written with ``save_figure`` (a PNG at 150 dpi plus a vector PDF) or ``FigureBook`` (a
multi-page PDF, with optional PNG copies of every page).
"""
from __future__ import annotations

import contextlib
import functools
import re
from pathlib import Path

STYLE_FILE = Path(__file__).with_name("eval.mplstyle")
PNG_DPI = 150

_SUFFIX = re.compile(r"\.(png|pdf|svg|jpg|jpeg)$", re.I)


@contextlib.contextmanager
def eval_style(**rc_overrides):
    """Apply the house Matplotlib style for the duration of the ``with`` block."""
    import matplotlib.pyplot as plt

    with plt.style.context(str(STYLE_FILE)):
        if rc_overrides:
            with plt.rc_context(rc_overrides):
                yield
        else:
            yield


def styled(func):
    """Decorator: run ``func`` inside ``eval_style()``."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with eval_style():
            return func(*args, **kwargs)

    return wrapper


def _stem(path) -> Path:
    p = Path(path)
    return p.with_name(_SUFFIX.sub("", p.name))


def save_figure(fig, path_without_suffix, *, formats=("png", "pdf"), dpi: int = PNG_DPI,
                close: bool = False, tight: bool = True, **savefig_kwargs) -> list[Path]:
    """Write ``fig`` as PNG (150 dpi) and PDF next to each other; return the written paths.

    ``path_without_suffix`` may carry a ``.png`` or ``.pdf`` suffix, which is ignored.
    Rasterised artists inside the PDF (map meshes) use ``dpi`` as well, so the PDF is never
    coarser than the PNG.
    """
    stem = _stem(path_without_suffix)
    stem.parent.mkdir(parents=True, exist_ok=True)
    kw = dict(savefig_kwargs)
    if tight:
        kw.setdefault("bbox_inches", "tight")
    written: list[Path] = []
    for fmt in formats:
        out = stem.with_name(f"{stem.name}.{fmt}")
        fig.savefig(out, dpi=dpi, format=fmt, **kw)
        written.append(out)
    if close:
        import matplotlib.pyplot as plt

        plt.close(fig)
    return written


class FigureBook:
    """Multi-page PDF that can also keep a PNG of every page.

    >>> with FigureBook("out/spectra", png=True) as book:   # doctest: +SKIP
    ...     book.add(fig1, name="2t")
    ...     book.add(fig2, name="msl")

    writes ``out/spectra.pdf`` and, when ``png=True``, ``out/spectra_pages/<nn>_<name>.png``.
    Figures are closed after they are added.
    """

    def __init__(self, path_without_suffix, *, png: bool = False, dpi: int = PNG_DPI):
        self.stem = _stem(path_without_suffix)
        self.png = png
        self.dpi = dpi
        self.paths: list[Path] = []
        self._pdf = None
        self._n = 0

    def __enter__(self):
        from matplotlib.backends.backend_pdf import PdfPages

        self.stem.parent.mkdir(parents=True, exist_ok=True)
        self._pdf = PdfPages(self.stem.with_name(self.stem.name + ".pdf"))
        self.paths.append(self.stem.with_name(self.stem.name + ".pdf"))
        return self

    def add(self, fig, name: str | None = None, *, close: bool = True, tight: bool = True):
        import matplotlib.pyplot as plt

        self._n += 1
        kw = {"bbox_inches": "tight"} if tight else {}
        self._pdf.savefig(fig, dpi=self.dpi, **kw)
        if self.png:
            slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", name or f"page{self._n}").strip("_")
            pages = self.stem.with_name(self.stem.name + "_pages")
            pages.mkdir(parents=True, exist_ok=True)
            out = pages / f"{self._n:02d}_{slug}.png"
            fig.savefig(out, dpi=self.dpi, **kw)
            self.paths.append(out)
        if close:
            plt.close(fig)

    def __exit__(self, *exc):
        self._pdf.close()
        return False
