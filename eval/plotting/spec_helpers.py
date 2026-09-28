"""Small drawing helpers shared by the spectra, training-curve, ladder and evolution figures.

Written by the "spec" plotting group; kept separate from the core ``eval.plotting`` modules so
that it can be reviewed and merged on its own. Nothing here touches Matplotlib's global state.

* ``power_unit`` / ``spectral_power_label``  unit of a power spectrum (the square of the
  variable's display unit) and the matching axis label, e.g. "Spectral power (hPa²)".
* ``amplitude_to_power``  squared spectral amplitudes in display units.
* ``add_wavelength_axis``  secondary top axis giving the wavelength in km for a total
  wavenumber axis.
* ``count_phrase``  "5 dates" / "1 date": sample counts for legends.
* ``smooth_series``  trailing exponential moving average used for faint-raw + bold-smoothed
  training curves.
* ``tint``  a light tint of a colour, for panel backgrounds.
"""
from __future__ import annotations

import numpy as np

from .labels import AXIS
from .variables import convert_difference, variable_spec

EARTH_CIRCUMFERENCE_KM = 40030.0

# Square of a display unit, written the way the variable table writes units.
_SQUARED_UNITS = {
    "hPa": "hPa²",
    "Pa": "Pa²",
    "K": "K²",
    "dam": "dam²",
    "mm": "mm²",
    "m s⁻¹": "m² s⁻²",
    "g kg⁻¹": "g² kg⁻²",
    "kg m⁻²": "kg² m⁻⁴",
    "Pa s⁻¹": "Pa² s⁻²",
    "%": "%²",
}


def power_unit(display_unit: str | None) -> str:
    """Unit of a power spectrum for a variable displayed in ``display_unit`` ("" if unknown)."""
    if not display_unit:
        return ""
    if display_unit in _SQUARED_UNITS:
        return _SQUARED_UNITS[display_unit]
    return f"({display_unit})²"


def spectral_power_label(variable: str | None = None) -> str:
    """"Spectral power (hPa²)" for ``msl``; plain "Spectral power" for an unknown variable."""
    unit = power_unit(variable_spec(variable).unit) if variable else ""
    return f"{AXIS['power']} ({unit})" if unit else AXIS["power"]


def amplitude_to_power(variable: str, amplitudes, native_unit: str | None = None) -> np.ndarray:
    """Square spectral amplitudes after converting them to the variable's display unit.

    Amplitudes are stored as ``sqrt(sum_m |X_nm|^2)`` in the native unit of the field (Pa for
    pressure, m² s⁻² for geopotential). A spectral amplitude transforms like a difference
    (no offset), so it is scaled with ``convert_difference`` before squaring.
    """
    scaled = np.asarray(convert_difference(variable, np.asarray(amplitudes, dtype=float),
                                           native_unit=native_unit), dtype=float)
    return scaled * scaled


def _wavelength(ell):
    ell = np.asarray(ell, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ell > 0, EARTH_CIRCUMFERENCE_KM / np.where(ell > 0, ell, 1.0), np.inf)


def add_wavelength_axis(ax, *, label: str | None = None):
    """Add a top axis that reads the wavenumber axis as wavelength in km (``40030 km / ℓ``)."""
    sec = ax.secondary_xaxis("top", functions=(_wavelength, _wavelength))
    sec.set_xlabel(label or AXIS["wavelength"])
    return sec


def count_phrase(n: int, singular: str, plural: str | None = None) -> str:
    """``count_phrase(5, "date")`` -> "5 dates"; ``count_phrase(1, "date")`` -> "1 date"."""
    word = singular if int(n) == 1 else (plural or f"{singular}s")
    return f"{int(n)} {word}"


def smooth_series(values, span: int) -> np.ndarray:
    """Trailing exponential moving average with smoothing ``span`` (in samples).

    NaN samples are skipped (they keep the previous average) so a gap does not poison the rest
    of the curve. ``span <= 1`` returns the values unchanged.
    """
    v = np.asarray(values, dtype=float)
    if span <= 1 or v.size == 0:
        return v.copy()
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(v)
    acc = np.nan
    for i, x in enumerate(v):
        if np.isfinite(x):
            acc = x if not np.isfinite(acc) else acc + alpha * (x - acc)
        out[i] = acc
    return out


def tint(color: str, amount: float = 0.85) -> tuple[float, float, float]:
    """Mix ``color`` with white; ``amount`` = 0 keeps the colour, 1 gives white."""
    from matplotlib.colors import to_rgb

    r, g, b = to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)
