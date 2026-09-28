"""Standard axis wording and readable legend labels.

Axis wording is defined once here so that the same quantity carries the same words on every
figure: ``AXIS["wavenumber"]`` is always "Total wavenumber ℓ", lead time is always
"Lead time (h)", and units are always in parentheses.

``readable_label`` turns the raw keys that leak out of the data layer (``od_enfo_0001``,
``eval_inputs``, ``y_pred_0``, ``rmse_ens_mean``, stream and grid identifiers such as
``ENFO_O1280_0001``) into wording a reader can follow, for example "truth (ENFO O1280)".
"""
from __future__ import annotations

import re
from dataclasses import dataclass

AXIS: dict[str, str] = {
    "wavenumber": "Total wavenumber ℓ",
    "wavelength": "Wavelength (km)",
    "power": "Spectral power",
    "amplitude": "Spectral amplitude",
    "amplitude_ratio": "Spectral amplitude ratio (model / truth)",
    "power_ratio": "Spectral power ratio (model / truth)",
    "pdf": "Probability density",
    "lead": "Lead time (h)",
    "count": "Number of samples",
    "cdf": "Cumulative probability",
    "lat": "Latitude (°N)",
    "lon": "Longitude (°E)",
    "sigma": "Noise level σ",
    "step": "Training step",
    "loss": "Loss",
}


def power_label(unit: str | None = None) -> str:
    """"Spectral power" with the unit in parentheses when one is known."""
    return f"{AXIS['power']} ({unit})" if unit else AXIS["power"]


def pdf_label(unit: str | None = None) -> str:
    """"Probability density (per hPa)" style label; ``unit`` is the variable's display unit."""
    return f"{AXIS['pdf']} (per {unit})" if unit else AXIS["pdf"]


def lead_label() -> str:
    return AXIS["lead"]


@dataclass(frozen=True)
class LabelContext:
    """What the raw ``truth`` and ``input`` keys stand for in the lane being plotted."""

    truth: str = "ENFO O1280"
    input: str = "interpolated input"
    model: str = "model"


_STREAM_GRID = re.compile(
    r"^(?P<stream>enfo|eefo|iekm|oper|od_enfo|od_eefo)_?(?:o(?P<grid>\d+))?(?:_(?P<mem>\d{4}|[a-z0-9]{4}|target))?$",
    re.I,
)
_RUN_LABEL = re.compile(r"^(?:manual|anemoi)_(?P<ckpt>[0-9a-f]{8})_\w+?_o\d+_o\d+_\d{8}_(?P<rest>.+)$")

_SIMPLE = {
    "eval_inputs": "input",
    "eval_input": "input",
    "inputs": "input",
    "x": "input",
    "x_0": "input",
    "x_interp": "input interpolated to the target grid",
    "x_interp_0": "input interpolated to the target grid",
    "y": "truth",
    "y_0": "truth",
    "y_pred": "model",
    "y_pred_0": "model",
    "residuals_0": "true residual",
    "residuals_pred_0": "predicted residual",
    "rmse_ens_mean": "RMSE of the ensemble mean",
    "rmse": "RMSE",
    "crps": "CRPS",
    "fcrps": "Fair CRPS",
    "rmsef": "RMSE of the ensemble mean",
    "spread": "Ensemble spread",
    "sdaf": "Standard deviation of the error",
    "ssr": "Spread-skill ratio",
    "seeps": "SEEPS",
}


def shorten_run_label(label: str) -> str:
    """Shorten ``manual_<ckpt8>_..._<date>_<sampler>`` to ``<ckpt6> <sampler>``."""
    m = _RUN_LABEL.match(label)
    if m:
        return f"{m.group('ckpt')[:6]} {m.group('rest')}"
    return label if len(label) <= 28 else label[:27] + "…"


def readable_label(key: str, context: LabelContext | None = None, *, with_id: bool = False) -> str:
    """Readable legend or title text for a raw key.

    With ``with_id=True`` the original key is appended in parentheses so the exact source stays
    traceable (put it in the small text of the legend or a footer, not in the main label).
    """
    ctx = context or LabelContext()
    raw = str(key)
    low = raw.strip().lower()
    text: str
    if low in ("y", "y_0", "truth", "target", "od_enfo_0001"):
        text = f"truth ({ctx.truth})"
    elif low in ("y_pred", "y_pred_0", "model", "prediction", "pred"):
        text = ctx.model
    elif low in ("x", "x_0", "input", "inputs", "eval_inputs", "eval_input"):
        text = ctx.input if ctx.input != "interpolated input" else "input"
    elif low in _SIMPLE:
        text = _SIMPLE[low]
    else:
        m = _STREAM_GRID.match(low)
        if m:
            stream = m.group("stream").replace("od_", "").upper()
            grid = f" O{m.group('grid')}" if m.group("grid") else ""
            mem = m.group("mem")
            if stream == "OPER":
                text = f"operational analysis (OPER-AN{grid})"
            elif mem and mem.isdigit() and mem not in ("0001",):
                text = f"{stream}{grid} member {int(mem)}"
            elif mem and mem.isdigit():
                text = f"{stream}{grid}"
            elif mem:
                text = f"{stream}{grid} ({mem})"
            else:
                text = f"{stream}{grid}"
        elif _RUN_LABEL.match(raw):
            text = shorten_run_label(raw)
        else:
            text = re.sub(r"_+", " ", raw).strip()
            text = text[:1].upper() + text[1:] if text and text.islower() and " " in text else text
    return f"{text} ({raw})" if with_id and text != raw else text


def label_and_id(key: str, context: LabelContext | None = None) -> tuple[str, str]:
    """``(readable label, original key)`` for legends that print the id in small text."""
    return readable_label(key, context), str(key)
