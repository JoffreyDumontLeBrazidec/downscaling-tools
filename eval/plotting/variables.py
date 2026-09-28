"""One table of the framework's variables: names, units, unit conversion and colour maps.

The framework stores fields in the units of the source data (Pa for pressure, m² s⁻² for
geopotential, K for temperature, m s⁻¹ for wind, ...). Figures show them in the units below:

=================  ===========================  ====================================
Variable           Native unit                  Displayed unit
=================  ===========================  ====================================
msl, sp            Pa                           hPa
z_<level>          m² s⁻²                       dam (decametres of geopotential height)
2t, 2d, skt, t_*   K                            K (temperatures are not converted)
10u, 10v, 10ff     m s⁻¹                        m s⁻¹
tp                 m accumulated                mm per accumulation period
q_<level>          kg kg⁻¹                      g kg⁻¹
tcw                kg m⁻²                       kg m⁻²
=================  ===========================  ====================================

Colour maps: fields use perceptually uniform sequential maps (never jet, rainbow or turbo);
signed fields (u and v wind components) use a colour-blind-safe diverging map centred on zero;
errors (prediction minus truth) always use ``RdBu_r`` centred on zero with symmetric limits
that are shared by the panels being compared (precipitation errors use ``BrBG`` so that
wetter-than-truth is blue-green and drier is brown).

Use ``variable_spec("10u_sfc")`` to look a variable up by any of its spellings, ``convert``
to change native values to display units, ``axis_label`` for a label with the unit in
parentheses.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace

import numpy as np

G0 = 9.80665  # m s-2, standard gravity

# Unit conversions to the displayed unit: (factor, offset), value_display = value * factor + offset.
_TO_DISPLAY: dict[tuple[str, str], tuple[float, float]] = {
    ("Pa", "hPa"): (0.01, 0.0),
    ("hPa", "hPa"): (1.0, 0.0),
    ("m2 s-2", "dam"): (1.0 / (G0 * 10.0), 0.0),
    ("m2/s2", "dam"): (1.0 / (G0 * 10.0), 0.0),
    ("gpm", "dam"): (0.1, 0.0),
    ("m", "dam"): (0.1, 0.0),
    ("dam", "dam"): (1.0, 0.0),
    ("K", "K"): (1.0, 0.0),
    ("degC", "K"): (1.0, 273.15),
    ("m s-1", "m s⁻¹"): (1.0, 0.0),
    ("m/s", "m s⁻¹"): (1.0, 0.0),
    ("m", "mm"): (1000.0, 0.0),
    ("mm", "mm"): (1.0, 0.0),
    ("kg kg-1", "g kg⁻¹"): (1000.0, 0.0),
    ("g/kg", "g kg⁻¹"): (1.0, 0.0),
}

# Accepted spellings of units passed as ``native_unit``.
_UNIT_ALIASES = {
    "m s**-1": "m s-1", "m s^-1": "m s-1", "m s⁻¹": "m s-1", "m/s": "m s-1",
    "m**2 s**-2": "m2 s-2", "m2 s**-2": "m2 s-2", "m^2/s^2": "m2 s-2", "m2/s2": "m2 s-2",
    "m² s⁻²": "m2 s-2", "j/kg": "m2 s-2", "j kg-1": "m2 s-2",
    "k": "K", "kelvin": "K", "c": "degC", "°c": "degC", "celsius": "degC",
    "pa": "Pa", "hpa": "hPa", "mb": "hPa", "mm": "mm", "m": "m", "gpm": "gpm", "dam": "dam",
    "kg kg**-1": "kg kg-1", "kg/kg": "kg kg-1", "g/kg": "g/kg",
}


def _cmap(name: str, fallback: str):
    """A Matplotlib colour map by name; ``cmcrameri`` names are tried first when prefixed."""
    if name.startswith("cmc."):
        try:
            import cmcrameri.cm as cmc

            return getattr(cmc, name[4:])
        except Exception:  # cmcrameri missing or name unknown
            return fallback
    return name


@dataclass(frozen=True)
class VariableSpec:
    """Display information for one variable."""

    key: str
    name: str            # "Mean sea level pressure"
    unit: str            # displayed unit, e.g. "hPa"
    native_unit: str     # unit of the values stored by the framework
    cmap: str            # field colour map (sequential, or diverging when ``signed``)
    err_cmap: str = "RdBu_r"
    signed: bool = False  # field takes both signs and is drawn with a centred colour scale
    accumulated: bool = False  # per-accumulation quantity (tp)
    short: str = ""      # short axis name, e.g. "MSLP"
    default_range: tuple[float, float] | None = None  # sensible fixed field range, display units
    extend: str = "both"

    @property
    def scale_offset(self) -> tuple[float, float]:
        return _TO_DISPLAY.get((self.native_unit, self.unit), (1.0, 0.0))

    @property
    def title(self) -> str:
        return self.name

    @property
    def label(self) -> str:
        """Display name with the unit in parentheses."""
        return f"{self.name} ({self.unit})" if self.unit else self.name

    def field_cmap(self):
        return _cmap(self.cmap, "viridis")

    def error_cmap(self):
        return _cmap(self.err_cmap, "RdBu_r")


def _v(key, name, unit, native, cmap, **kw) -> VariableSpec:
    return VariableSpec(key=key, name=name, unit=unit, native_unit=native, cmap=cmap, **kw)


_WIND_MS = "m s⁻¹"

VARIABLES: dict[str, VariableSpec] = {
    "10u": _v("10u", "10 m zonal wind", _WIND_MS, "m s-1", "PuOr_r", signed=True, short="10u"),
    "10v": _v("10v", "10 m meridional wind", _WIND_MS, "m s-1", "PuOr_r", signed=True, short="10v"),
    "10ff": _v("10ff", "10 m wind speed", _WIND_MS, "m s-1", "viridis", short="10 m wind",
               default_range=(0.0, 25.0), extend="max"),
    "2t": _v("2t", "2 m temperature", "K", "K", "magma", short="2t", default_range=(250.0, 310.0)),
    "2d": _v("2d", "2 m dew point temperature", "K", "K", "magma", short="2d"),
    "skt": _v("skt", "Skin temperature", "K", "K", "magma", short="skt"),
    "sst": _v("sst", "Sea surface temperature", "K", "K", "magma", short="sst"),
    "msl": _v("msl", "Mean sea level pressure", "hPa", "Pa", "cividis", short="MSLP",
              default_range=(960.0, 1040.0)),
    "sp": _v("sp", "Surface pressure", "hPa", "Pa", "cividis", short="sp"),
    "tcw": _v("tcw", "Total column water", "kg m⁻²", "kg m-2", "cmc.davos_r", short="tcw",
              default_range=(0.0, 70.0), extend="max"),
    "tp": _v("tp", "Total precipitation", "mm", "m", "cmc.lapaz_r", err_cmap="BrBG",
             accumulated=True, short="tp", extend="max"),
    "cp": _v("cp", "Convective precipitation", "mm", "m", "cmc.lapaz_r", err_cmap="BrBG",
             accumulated=True, short="cp", extend="max"),
    "lsm": _v("lsm", "Land-sea mask", "", "", "Greys", short="lsm", extend="neither"),
}

# Per-level families, keyed by the leading name; the level (hPa) is filled in at lookup.
_LEVEL_FAMILIES: dict[str, dict] = {
    "z": dict(name="geopotential height", unit="dam", native="m2 s-2", cmap="cividis"),
    "t": dict(name="temperature", unit="K", native="K", cmap="magma"),
    "u": dict(name="zonal wind", unit=_WIND_MS, native="m s-1", cmap="PuOr_r", signed=True),
    "v": dict(name="meridional wind", unit=_WIND_MS, native="m s-1", cmap="PuOr_r", signed=True),
    "w": dict(name="vertical velocity", unit="Pa s⁻¹", native="Pa s-1", cmap="PuOr_r", signed=True),
    "q": dict(name="specific humidity", unit="g kg⁻¹", native="kg kg-1", cmap="cmc.davos_r"),
    "r": dict(name="relative humidity", unit="%", native="%", cmap="cmc.davos_r"),
}

_ALIASES = {
    "wind": "10ff", "wind10m": "10ff", "ws": "10ff", "ws10": "10ff", "10si": "10ff",
    "10m_wind": "10ff", "10m_wind_speed": "10ff", "wind_speed": "10ff", "10ws": "10ff",
    "t2m": "2t", "d2m": "2d", "u10": "10u", "v10": "10v", "mslp": "msl",
    "total_precipitation": "tp", "precip": "tp", "precipitation": "tp",
}

_LEVEL_RE = re.compile(r"^(?P<fam>[ztuvwqr])_(?P<lev>\d{1,4})$")
_STRIP_RE = re.compile(r"(_sfc|_pl|_ml)$")


def variable_spec(name: str) -> VariableSpec:
    """Look a variable up by any spelling the framework uses.

    Accepts ``10u``, ``10u_sfc``, ``wind``, ``z_500``, ``t_850_pl`` and so on. Unknown names
    return a neutral spec (display name = the key, no unit, viridis) instead of raising, so
    that a figure for a new variable still renders.
    """
    key = str(name).strip()
    low = _STRIP_RE.sub("", key.lower())
    low = _ALIASES.get(low, low)
    if low in VARIABLES:
        return VARIABLES[low]
    m = _LEVEL_RE.match(low)
    if m and m.group("fam") in _LEVEL_FAMILIES:
        fam = _LEVEL_FAMILIES[m.group("fam")]
        lev = int(m.group("lev"))
        return VariableSpec(
            key=low, name=f"{lev} hPa {fam['name']}", unit=fam["unit"], native_unit=fam["native"],
            cmap=fam["cmap"], signed=fam.get("signed", False), short=low,
        )
    return VariableSpec(key=key, name=key, unit="", native_unit="", cmap="viridis", short=key)


def is_known(name: str) -> bool:
    """True when ``name`` resolves to a variable in the table (or a level family)."""
    return variable_spec(name).unit != "" or variable_spec(name).key in VARIABLES


def _canonical_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    u = str(unit).strip()
    return _UNIT_ALIASES.get(u.lower(), _UNIT_ALIASES.get(u, u))


def convert(name: str, values, native_unit: str | None = None):
    """Convert ``values`` of variable ``name`` from native units to display units.

    ``native_unit`` overrides the framework default for callers that already converted
    (for example precipitation that is stored in mm: ``native_unit="mm"``). Values whose
    unit pair is not in the table are returned unchanged.
    """
    spec = variable_spec(name)
    src = _canonical_unit(native_unit) or spec.native_unit
    factor, offset = _TO_DISPLAY.get((src, spec.unit), (1.0, 0.0))
    if factor == 1.0 and offset == 0.0:
        return values
    return np.asarray(values, dtype=float) * factor + offset


def convert_difference(name: str, values, native_unit: str | None = None):
    """Convert a *difference* (error, bias, spread) to display units: scale only, no offset."""
    spec = variable_spec(name)
    src = _canonical_unit(native_unit) or spec.native_unit
    factor, _ = _TO_DISPLAY.get((src, spec.unit), (1.0, 0.0))
    if factor == 1.0:
        return values
    return np.asarray(values, dtype=float) * factor


def axis_label(name: str, *, difference: bool = False, prefix: str = "") -> str:
    """Axis or colour-bar label such as ``"Mean sea level pressure (hPa)"``.

    ``difference=True`` gives ``"Mean sea level pressure error (hPa)"``-style wording via
    ``prefix``; the unit is unchanged because errors carry the unit of the variable.
    """
    spec = variable_spec(name)
    base = f"{prefix}{spec.name}" if prefix else spec.name
    if difference:
        base = f"{base} error"
    return f"{base} ({spec.unit})" if spec.unit else base


def display_name(name: str) -> str:
    return variable_spec(name).name


def unit_of(name: str) -> str:
    return variable_spec(name).unit


def with_unit(spec_or_name, unit: str | None) -> VariableSpec:
    """A copy of a spec with a different displayed unit (used when a caller pre-converted)."""
    spec = variable_spec(spec_or_name) if isinstance(spec_or_name, str) else spec_or_name
    return replace(spec, unit=unit or spec.unit)
