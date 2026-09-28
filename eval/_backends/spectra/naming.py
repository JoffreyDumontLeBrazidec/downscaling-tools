"""One place that knows what a spectra curve file is called.

Two conventions exist on disk, and they did not agree, which is why no
spectra_ecmwf result ever reached a scoreboard.

The six-field form ``ampl_<date>_<step>_<param>_<level>_<expid>_n<member>.npy``
(for example ``ampl_20230826_120_msl_sfc_1_n1.npy``) is what
``eval/_backends/scoreboard/spectra.py`` and the analysis scripts under
``eval/_backends/spectra`` read, and it matches the staged GRIB naming that
``_grib_stager.py`` already follows.  It is therefore the canonical form.

The short form ``ampl_<date>_<step>_<weather_state>_n<member>.npy`` (for example
``ampl_20230826_120_msl_n1.npy``) is what ``_amplitude_computer.py`` wrote.

New files are written in the canonical form.  Both forms are accepted when
reading, so the caches already on /perm keep working and nothing has to be
renamed.
"""
from __future__ import annotations

import re
from pathlib import Path

# Longest first, so a field directory can never be shadowed by a shorter one
# that happens to be a prefix of it.
FIELD_DIRS: tuple[str, ...] = (
    "10u_sfc",
    "10v_sfc",
    "msl_sfc",
    "2t_sfc",
    "sp_sfc",
    "t_850",
    "z_500",
)

DEFAULT_TOKEN = "1"

_NAME_RE = re.compile(r"^(?P<prefix>ampl|wvn)_(?P<date>\d{8})_(?P<step>\d+)_(?P<rest>.+)_n(?P<member>\d+)\.npy$")


def weather_state_for(field_dir: str) -> str:
    """The short weather-state label matching a field directory.

    ``msl_sfc`` -> ``msl``, ``2t_sfc`` -> ``2t``, ``t_850`` -> ``t_850``.
    """
    return field_dir[: -len("_sfc")] if field_dir.endswith("_sfc") else field_dir


def canonical_name(
    prefix: str, *, date: int | str, step: int | str, field_dir: str, token: str, member: int | str
) -> str:
    """The six-field name that the scoreboard reads."""
    return f"{prefix}_{date}_{step}_{field_dir}_{token}_n{member}.npy"


def short_name(
    prefix: str, *, date: int | str, step: int | str, weather_state: str, member: int | str
) -> str:
    """The legacy short name that _amplitude_computer.py used to write."""
    return f"{prefix}_{date}_{step}_{weather_state}_n{member}.npy"


def candidate_names(
    prefix: str,
    *,
    date: int | str,
    step: int | str,
    field_dir: str,
    member: int | str,
    token: str = DEFAULT_TOKEN,
) -> list[str]:
    """Names to try when reading, canonical first."""
    return [
        canonical_name(prefix, date=date, step=step, field_dir=field_dir, token=token, member=member),
        short_name(
            prefix, date=date, step=step, weather_state=weather_state_for(field_dir), member=member
        ),
    ]


def find(
    directory: Path,
    prefix: str,
    *,
    date: int | str,
    step: int | str,
    field_dir: str,
    member: int | str,
    token: str = DEFAULT_TOKEN,
) -> Path | None:
    """Locate one curve file under either naming convention."""
    for name in candidate_names(
        prefix, date=date, step=step, field_dir=field_dir, member=member, token=token
    ):
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None


def parse(name: str) -> dict[str, object] | None:
    """Decompose a curve filename, or return None if it is neither convention.

    ``t_850`` and ``z_500`` contain an underscore, so the middle of the name is
    matched against the known field directories rather than split blindly.
    """
    match = _NAME_RE.match(name)
    if not match:
        return None
    rest = match.group("rest")
    for field_dir in FIELD_DIRS:
        if rest == weather_state_for(field_dir):
            form, token = "short", None
        elif rest.startswith(f"{field_dir}_"):
            form, token = "canonical", rest[len(field_dir) + 1 :]
            if "_" in token:
                continue
        else:
            continue
        return {
            "prefix": match.group("prefix"),
            "date": int(match.group("date")),
            "step": int(match.group("step")),
            "field_dir": field_dir,
            "weather_state": weather_state_for(field_dir),
            "token": token,
            "member": int(match.group("member")),
            "form": form,
        }
    return None
