"""Fixed line styles for the roles a curve can play in an evaluation figure.

Every figure in the framework draws the same few kinds of things, and each kind
always looks the same:

* ``truth``      the field or statistic everything is compared with: black, solid, thick;
* ``model``      the downscaling model under evaluation: red, solid;
* ``input``      the coarse driving forecast the model starts from: blue, dashed;
* ``baseline``   a simple reference method (for example plain interpolation of the input):
                 dark grey, dash-dot;
* ``reference``  any other anchor curve (operational forecasts on other grids, older
                 systems, ...): a muted colour or grey with its own dash pattern, taken from
                 ``REFERENCE_STYLES``. No two entries share a colour and a dash pattern.

When several checkpoints or experiment arms are drawn in one figure, they are all
"model" curves and must be told apart from each other. ``SEQUENCE`` is a colour list built
from the Okabe-Ito colour-blind-safe palette that avoids the role colours (no black, no red,
no blue), and ``sequence_style(i)`` combines it with a dash pattern once the colours run out.

A curve is styled with ``role_style("model")`` (a dict of matplotlib keyword arguments) or,
when only a raw key such as ``od_enfo_0001`` or ``y_pred_0`` is at hand, with
``style_for_key(key)``.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

TRUTH_COLOR = "#000000"
MODEL_COLOR = "#d62728"
INPUT_COLOR = "#1f77b4"
BASELINE_COLOR = "#4d4d4d"

# Okabe-Ito colours (Okabe and Ito 2008) minus black, vermillion (too close to the model red),
# blue (the input blue) and yellow (unreadable on white), plus two extra muted hues.
SEQUENCE: tuple[str, ...] = (
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#8C6D31",  # brown
    "#6A3D9A",  # violet
)

# Dash patterns as (offset, on-off sequence); used with SEQUENCE once colours repeat.
_DASHES: tuple[object, ...] = ("-", (0, (6, 2)), (0, (1.5, 1.5)), (0, (4, 1.5, 1, 1.5)))

# Encoding of "better" versus "worse" that stays readable with red-green colour blindness
# and in greyscale: blue against orange (Okabe-Ito), never red against green.
BETTER_COLOR = "#0072B2"
WORSE_COLOR = "#E69F00"
NEUTRAL_COLOR = "#BDBDBD"


@dataclass(frozen=True)
class LineStyle:
    """Keyword arguments for ``Axes.plot`` describing one role."""

    color: str
    linestyle: object = "-"
    linewidth: float = 2.0
    zorder: float = 3.0
    alpha: float = 1.0

    def kwargs(self, **overrides) -> dict:
        out = asdict(self)
        out.update(overrides)
        return out


ROLES: dict[str, LineStyle] = {
    "truth": LineStyle(TRUTH_COLOR, "-", 2.6, zorder=4),
    "model": LineStyle(MODEL_COLOR, "-", 2.2, zorder=5),
    "input": LineStyle(INPUT_COLOR, (0, (5, 2)), 2.0, zorder=3),
    "baseline": LineStyle(BASELINE_COLOR, (0, (4, 1.5, 1, 1.5)), 1.8, zorder=2),
}

# Anchor curves that are neither truth, model, input nor baseline. Colour and dash pair up
# so that every entry is distinguishable even in greyscale. Keep this list free of duplicates
# (a unit test checks it).
REFERENCE_STYLES: tuple[LineStyle, ...] = (
    LineStyle("#7F7F7F", (0, (6, 2)), 1.8, zorder=2),      # mid grey, dashed
    LineStyle("#8C564B", (0, (1.5, 1.5)), 2.0, zorder=2),  # brown, dotted
    LineStyle("#9467BD", (0, (6, 2, 1.5, 2)), 1.8, zorder=2),  # purple, dash-dot
    LineStyle("#2A9D8F", (0, (8, 2, 1.5, 2, 1.5, 2)), 1.8, zorder=2),  # teal, dash-dot-dot
    LineStyle("#B0B0B0", "-", 1.6, zorder=2),              # light grey, solid
    LineStyle("#A6761D", (0, (3, 1.5)), 1.8, zorder=2),    # ochre, short dash
    LineStyle("#5F5F5F", (0, (1, 2)), 1.8, zorder=2),      # dark grey, sparse dots
    LineStyle("#17BECF", (0, (9, 3)), 1.8, zorder=2),      # cyan, long dash
)

# Named anchors that appear in many figures always get the same reference style, so that
# "ENFO O320" is the same line in every plot. Keys are lower-case names without member ids.
_NAMED_REFERENCES: dict[str, int] = {
    "enfo_o320": 0,
    "enfo_o96": 1,
    "eefo_o96": 2,
    "iekm": 3,
    "enfo_o48": 5,
    "eefo_o320": 6,
    "oper": 4,
}


def role_style(role: str, **overrides) -> dict:
    """Matplotlib line keyword arguments for a role (``truth``, ``model``, ...)."""
    if role == "reference":
        return REFERENCE_STYLES[0].kwargs(**overrides)
    try:
        return ROLES[role].kwargs(**overrides)
    except KeyError as exc:
        raise KeyError(f"unknown role {role!r}; known: {sorted(ROLES) + ['reference']}") from exc


def reference_style(index: int, **overrides) -> dict:
    """Style of the ``index``-th reference curve (cycles after the list is exhausted)."""
    return REFERENCE_STYLES[index % len(REFERENCE_STYLES)].kwargs(**overrides)


def reference_styles(keys, **overrides) -> dict[str, dict]:
    """Assign a distinct reference style to every key in ``keys``.

    Well-known anchors (``enfo_o320``, ``eefo_o96``, ...) keep a fixed style everywhere;
    the rest take the remaining styles in order of appearance.
    """
    out: dict[str, dict] = {}
    used: set[int] = set()
    unknown: list[str] = []
    for key in keys:
        idx = _NAMED_REFERENCES.get(_reference_name(key))
        if idx is not None and idx not in used:
            out[key] = REFERENCE_STYLES[idx].kwargs(**overrides)
            used.add(idx)
        else:
            unknown.append(key)
    free = [i for i in range(len(REFERENCE_STYLES)) if i not in used]
    for n, key in enumerate(unknown):
        idx = free[n % len(free)] if free else n % len(REFERENCE_STYLES)
        out[key] = REFERENCE_STYLES[idx].kwargs(**overrides)
    return out


def sequence_style(index: int, **overrides) -> dict:
    """Style of the ``index``-th checkpoint or arm in a multi-model figure.

    The first six curves differ in colour; after that the colours repeat with a different
    dash pattern. All of them are solid-weight lines that stay clear of the role colours.
    """
    color = SEQUENCE[index % len(SEQUENCE)]
    dash = _DASHES[(index // len(SEQUENCE)) % len(_DASHES)]
    out = {"color": color, "linestyle": dash, "linewidth": 2.2, "zorder": 4.0}
    out.update(overrides)
    return out


def sequence_colors(n: int) -> list[str]:
    """``n`` colours from ``SEQUENCE`` (repeating when ``n`` exceeds its length)."""
    return [SEQUENCE[i % len(SEQUENCE)] for i in range(n)]


_TRUTH_KEYS = re.compile(r"^(truth|target|y|y_?\d*|od_enfo_\d+|oper_?an|oper_o\d+_\d+)$", re.I)
_MODEL_KEYS = re.compile(r"^(model|pred|prediction|y_?pred(_\d+)?|residuals_pred(_\d+)?|ml)$", re.I)
_INPUT_KEYS = re.compile(r"^(input|inputs|eval_inputs|x|x_?\d*|x_?interp(_\d+)?|driver)$", re.I)
_BASELINE_KEYS = re.compile(r"^(baseline|interp|interpolated|interp_input)$", re.I)


def _reference_name(key: str) -> str:
    k = str(key).lower()
    k = re.sub(r"_(\d{4}|target|[a-z0-9]{4})$", "", k)
    return k


def role_of(key: str) -> str | None:
    """Return the role a raw source key stands for, or ``None`` if it is not a role key."""
    k = str(key).strip()
    if _MODEL_KEYS.match(k):
        return "model"
    if _TRUTH_KEYS.match(k):
        return "truth"
    if _INPUT_KEYS.match(k):
        return "input"
    if _BASELINE_KEYS.match(k):
        return "baseline"
    return None


def style_for_key(key: str, **overrides) -> dict:
    """Style for a raw key: the role style if it names a role, else a reference style."""
    role = role_of(key)
    if role is not None:
        return role_style(role, **overrides)
    name = _reference_name(key)
    idx = _NAMED_REFERENCES.get(name, sum(map(ord, name)) % len(REFERENCE_STYLES))
    return REFERENCE_STYLES[idx].kwargs(**overrides)


def better_worse_cmap():
    """Diverging colour map, orange (worse) through white to blue (better)."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "eval_better_worse", ["#B85400", "#F2A93B", "#FFFFFF", "#56A0D3", BETTER_COLOR]
    )
