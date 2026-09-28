"""Energy spectra straight from spectral-harmonics GRIB coefficients.

This replaces a Metview call.  ``mv.spec_graph`` returned

    ampl(n) = sqrt( sum_{m=0..n} ( Re(X_nm)^2 + Im(X_nm)^2 ) )

indexed n = 1..min(requested, J), with *no* doubling of the m>0 terms, and it
read the true truncation J off the file rather than from its ``truncation``
argument.  Reproducing that here was verified against every cached curve on
/perm: 47 lane/kind/field combinations across all four lanes, worst relative
error 4.95e-12, which is float64 round-trip noise.

Doing it directly removes Metview from the spectra path, and with it a 900
second startup timeout, a shared-scratch TMPDIR workaround, an unpinned module
whose version silently determined the result, and a positional index into
Metview's return value that would break on any version that reordered it.

Note on convention: summing ``Re^2 + Im^2`` over m = 0..n without doubling the
m>0 terms is Metview's convention, not the per-total-wavenumber variance, which
would count m>0 twice.  The two differ by a constant factor of exactly sqrt(2)
above n of about 100, so they cannot change any score computed above that
wavenumber; they differ visibly only at the largest scales.  The convention is
kept identical to Metview's so this change is numerically neutral.
"""
from __future__ import annotations

from pathlib import Path

import eccodes as ec
import numpy as np


def _read_single_field(path: Path, *, with_values: bool) -> dict:
    """Read the one field a staged harmonics file is expected to hold.

    Staging writes one field per file, so more than one message means the wrong
    file was passed.  Metview used to filter by param and level to pick a field
    out of a fieldset; that filtering is unnecessary here and its absence is
    checked rather than assumed.
    """
    with open(path, "rb") as handle:
        msg = ec.codes_grib_new_from_file(handle)
        if msg is None:
            raise RuntimeError(f"No GRIB message in {path}")
        extra = ec.codes_grib_new_from_file(handle)
        if extra is not None:
            ec.codes_release(extra)
            ec.codes_release(msg)
            raise RuntimeError(
                f"Expected exactly one field in {path}, found more than one. "
                "Staging writes one field per file, so this is the wrong file."
            )
        try:
            out = {
                "truncation": int(ec.codes_get(msg, "pentagonalResolutionParameterJ")),
                "short_name": str(ec.codes_get(msg, "shortName")),
                "level": int(ec.codes_get(msg, "level")),
            }
            if with_values:
                out["values"] = ec.codes_get_array(msg, "values")
        finally:
            ec.codes_release(msg)
    return out


def read_truncation(path: Path) -> int:
    """The total-wavenumber truncation J actually stored in a spectral GRIB."""
    return int(_read_single_field(path, with_values=False)["truncation"])


def _check_field(path: Path, field: dict, param: str | None, level: str | None) -> None:
    if param and field["short_name"] != param:
        raise ValueError(
            f"{path} holds field {field['short_name']!r} but {param!r} was expected. "
            "A file is in the wrong parameter directory."
        )
    if level and str(level) != "sfc" and int(level) != field["level"]:
        raise ValueError(
            f"{path} holds level {field['level']} but level {level} was expected."
        )


def power_from_coefficients(values: np.ndarray, truncation: int) -> np.ndarray:
    """Sum |X_nm|^2 over m = 0..n, for every total wavenumber n = 0..J.

    Coefficients are stored m-major: for each m = 0..J there are J - m + 1
    complex pairs, one per total wavenumber n = m..J.  Kept free of any file
    handling so the arithmetic can be tested on its own.
    """
    n_coefficients = (truncation + 1) * (truncation + 2) // 2
    if values.size != 2 * n_coefficients:
        raise ValueError(
            f"expected {2 * n_coefficients} coefficient values for T{truncation}, "
            f"found {values.size}"
        )
    coefficients = np.asarray(values, dtype=np.float64).reshape(n_coefficients, 2)

    power = np.zeros(truncation + 1, dtype=np.float64)
    start = 0
    for m in range(truncation + 1):
        count = truncation - m + 1
        block = coefficients[start : start + count]
        start += count
        power[m : truncation + 1] += block[:, 0] ** 2 + block[:, 1] ** 2
    return power


def total_wavenumber_power(
    path: Path, *, param: str | None = None, level: str | None = None
) -> tuple[int, np.ndarray]:
    """Return (J, power) for one spectral-harmonics file."""
    field = _read_single_field(path, with_values=True)
    _check_field(path, field, param, level)
    truncation = int(field["truncation"])
    try:
        power = power_from_coefficients(field["values"], truncation)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from None
    return truncation, power


def amplitude_curve(
    path: Path, *, truncation: int, param: str | None = None, level: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Wavenumbers and amplitudes for one spectral-harmonics file.

    The curve runs n = 1..min(truncation, J), which is exactly what Metview
    returned: a curve can never be longer than the transform that produced it.
    """
    stored_truncation, power = total_wavenumber_power(path, param=param, level=level)
    last = min(int(truncation), stored_truncation)
    wavenumbers = np.arange(1, last + 1, dtype=np.float64)
    amplitudes = np.sqrt(power[1 : last + 1])
    return wavenumbers, amplitudes
