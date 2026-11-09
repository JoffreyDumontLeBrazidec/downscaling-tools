"""The standard spectra figure: one column per variable, amplitude spectrum above, ratio below.

This is the layout of the ECMWF / anemoi spectra figure the owner chose on 9 Nov 2026 as the
one standard spectra plot of ``eval.cli``:

* one panel per variable, at most four per row, in the order of ``STANDARD_ORDER``
  (total cloud cover, 6 h precipitation, 10 m wind speed, 2 m temperature, mean sea level
  pressure, 500 hPa geopotential, total column water, 850 hPa v-wind) and then the rest;
* top: spectral amplitude against total wavenumber on log-log axes, wavelength in km on a top axis;
* below: each curve divided by the truth curve, log axis from 0.3 to 3, a line at one;
* few curves per panel: truth, input ensemble and model (rule: at most three).

Curves keep the house role colours of ``eval.plotting.roles`` (truth black, model red,
input blue dashed). Values are spectral AMPLITUDES in the variable's display unit (the
quantity of standing rule 14 and of every amplitude ratio quoted in the notes), and the ratio
is an amplitude ratio; feeders average over curves as power and take the square root. Both evaluators feed it: ``spectra_ecmwf_v2`` (spherical harmonics of global fields,
total wavenumber up to the truncation) and ``box_spectra`` (FFT of a box, its wavelength bins
placed at the equivalent total wavenumber 40030 km / wavelength).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .roles import role_style
from .spec_helpers import EARTH_CIRCUMFERENCE_KM, add_wavelength_axis
from .style import eval_style, save_figure
from .variables import variable_spec

# Panel order of the reference figure; variables not listed follow in their given order.
STANDARD_ORDER = ("tcc", "tp", "10ff", "2t", "msl", "z_500", "tcw", "v_850",
                  "10u", "10v", "sp", "t_850", "q_850", "u_850")
RATIO_LIM = (0.3, 3.0)
RATIO_TICKS = (0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0)
NCOLS = 4

@dataclass
class Curve:
    """One mean spectrum: ``amplitude`` at ``wavenumber`` (both 1-D, same length)."""

    label: str
    wavenumber: np.ndarray
    amplitude: np.ndarray
    role: str = "model"             # truth | model | input | baseline | reference
    style: dict = field(default_factory=dict)

    def line_kwargs(self) -> dict:
        kw = role_style(self.role, linewidth=1.5)
        kw.update(self.style)
        return kw


_TITLES = {"tcc": "Total cloud cover"}


def _canonical(name: str) -> str:
    return variable_spec(name).key or str(name)


def panel_title(name: str) -> str:
    return _TITLES.get(_canonical(name), variable_spec(name).name)


def order_variables(names: Sequence[str]) -> list[str]:
    """Names sorted into ``STANDARD_ORDER`` (by any spelling), unknown names last, stable."""
    rank = {k: i for i, k in enumerate(STANDARD_ORDER)}

    def key(item):
        i, name = item
        canon = _canonical(name)
        return (rank.get(canon, rank.get(str(name), len(rank))), i)

    return [n for _, n in sorted(enumerate(names), key=key)]


def ratio_to_reference(num_wvn, num_val, den_wvn, den_val):
    """Divide one spectrum by another over the wavenumbers they share.

    Both curves are restricted to strictly positive, finite wavenumbers and values (the ratio
    is drawn on log axes); the denominator is interpolated onto the numerator wavenumbers in
    log-log space. When the two grids coincide (integer total wavenumbers, or the same box
    bins) the interpolation returns the stored values unchanged. Returns (wavenumbers, ratio)
    or None when there is no usable overlap.
    """
    num_wvn = np.asarray(num_wvn, dtype=float)
    num_val = np.asarray(num_val, dtype=float)
    den_wvn = np.asarray(den_wvn, dtype=float)
    den_val = np.asarray(den_val, dtype=float)
    if num_wvn.shape != num_val.shape or den_wvn.shape != den_val.shape:
        return None
    n_ok = np.isfinite(num_wvn) & np.isfinite(num_val) & (num_wvn > 0) & (num_val > 0)
    d_ok = np.isfinite(den_wvn) & np.isfinite(den_val) & (den_wvn > 0) & (den_val > 0)
    if int(n_ok.sum()) < 2 or int(d_ok.sum()) < 2:
        return None
    nwvn, nval = num_wvn[n_ok], num_val[n_ok]
    dwvn, dval = den_wvn[d_ok], den_val[d_ok]
    order = np.argsort(dwvn)
    dwvn, dval = dwvn[order], dval[order]
    lo = max(float(nwvn.min()), float(dwvn.min()))
    hi = min(float(nwvn.max()), float(dwvn.max()))
    keep = (nwvn >= lo) & (nwvn <= hi)
    if int(keep.sum()) < 2:
        return None
    nwvn, nval = nwvn[keep], nval[keep]
    den_here = np.exp(np.interp(np.log(nwvn), np.log(dwvn), np.log(dval)))
    return nwvn, nval / den_here


def wavelength_to_wavenumber(wavelength_km) -> np.ndarray:
    """Equivalent total wavenumber of a wavelength, ``40030 km / wavelength`` (the top axis rule)."""
    return EARTH_CIRCUMFERENCE_KM / np.asarray(wavelength_km, dtype=float)


def rms_amplitude(power_mean) -> np.ndarray:
    """Amplitude of a mean power: ``sqrt(mean over curves of amplitude²)``."""
    return np.sqrt(np.asarray(power_mean, dtype=float))


def plot_spectra_grid(
    panels: Mapping[str, Sequence[Curve]],
    out_stem: str | Path,
    *,
    title: str | None = None,
    footnote: str | None = None,
    ncols: int = NCOLS,
    ratio_lim: tuple[float, float] = RATIO_LIM,
    formats: Sequence[str] = ("png", "pdf"),
) -> list[Path]:
    """Draw the standard spectra figure and write it as ``<out_stem>.png`` and ``.pdf``.

    ``panels`` maps a variable name to its curves. The first curve with role ``truth`` is the
    reference of the ratio panel; without one the ratio panel is left empty with a note.
    Returns the written paths (empty when no panel has a curve).
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator

    names = [v for v in order_variables(list(panels)) if panels[v]]
    if not names:
        return []
    ncols = max(1, min(int(ncols), len(names)))
    nrows = math.ceil(len(names) / ncols)

    with eval_style():
        fig = plt.figure(figsize=(3.9 * ncols, 4.7 * nrows + (0.6 if title else 0.2)))
        outer = fig.add_gridspec(nrows, ncols, hspace=0.42, wspace=0.32)
        handles: dict[str, object] = {}
        for k, var in enumerate(names):
            r, c = divmod(k, ncols)
            inner = outer[r, c].subgridspec(2, 1, height_ratios=[2.4, 1.0], hspace=0.06)
            ax = fig.add_subplot(inner[0])
            axr = fig.add_subplot(inner[1], sharex=ax)
            curves = list(panels[var])
            truth = next((cv for cv in curves if cv.role == "truth"), None)

            for cv in curves:
                m = (np.asarray(cv.wavenumber) > 0) & np.isfinite(cv.amplitude) & (np.asarray(cv.amplitude) > 0)
                (line,) = ax.plot(np.asarray(cv.wavenumber)[m], np.asarray(cv.amplitude)[m],
                                  label=cv.label, **cv.line_kwargs())
                handles.setdefault(cv.label, line)
                if truth is not None and cv is not truth:
                    got = ratio_to_reference(cv.wavenumber, cv.amplitude, truth.wavenumber, truth.amplitude)
                    if got is not None:
                        axr.plot(got[0], got[1], **cv.line_kwargs())

            unit = variable_spec(var).unit or ""
            ax.set_title(panel_title(var), fontsize=11)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylabel(f"amplitude ({unit})" if unit else "amplitude", fontsize=9)
            ax.tick_params(labelbottom=False, labelsize=8)
            ax.grid(True, which="major", color="0.9", linewidth=0.5)
            sec = add_wavelength_axis(ax)
            sec.xaxis.label.set_size(8)
            sec.tick_params(labelsize=7)

            axr.axhline(1.0, color="0.25", linewidth=0.9, zorder=1)
            axr.set_xscale("log")
            axr.set_yscale("log")
            axr.set_ylim(*ratio_lim)
            ticks = [t for t in RATIO_TICKS if ratio_lim[0] <= t <= ratio_lim[1]]
            axr.yaxis.set_major_locator(FixedLocator(ticks))
            axr.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
            axr.yaxis.set_minor_locator(NullLocator())
            axr.yaxis.set_minor_formatter(NullFormatter())
            axr.tick_params(labelsize=8)
            axr.set_ylabel("ratio to truth" if c == 0 else "", fontsize=9)
            axr.set_xlabel("total wavenumber", fontsize=9)
            axr.grid(True, which="major", color="0.9", linewidth=0.5)
            if truth is None:
                axr.text(0.5, 0.5, "no truth curve", transform=axr.transAxes, ha="center",
                         va="center", fontsize=8, color="0.4")
            lo = min(float(np.min(np.asarray(cv.wavenumber)[np.asarray(cv.wavenumber) > 0]))
                     for cv in curves)
            hi = max(float(np.max(cv.wavenumber)) for cv in curves)
            ax.set_xlim(max(lo, 0.8) * 0.9, hi * 1.1)

        fig.legend(list(handles.values()), list(handles.keys()), loc="upper center",
                   ncol=min(len(handles), 4), frameon=False, fontsize=10,
                   bbox_to_anchor=(0.5, 1.0 if not title else 0.975))
        if title:
            fig.suptitle(title, y=1.02, fontsize=12)
        if footnote:
            fig.text(0.01, -0.01, footnote, fontsize=8, color="0.35", ha="left", va="top")
        return save_figure(fig, out_stem, formats=tuple(formats), close=True)
