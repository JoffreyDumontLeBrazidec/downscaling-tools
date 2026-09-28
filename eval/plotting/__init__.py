"""House plotting style for the evaluation framework.

Import from here::

    from eval.plotting import (
        eval_style, save_figure, FigureBook,
        role_style, sequence_style, reference_style,
        variable_spec, convert, axis_label,
        select_projection, add_geography, shared_norm, symmetric_norm,
        readable_label, AXIS,
    )

Design rules (details in the module docstrings):

* ``roles``      truth black solid thick; model red solid; input blue dashed; baseline dark grey
                 dash-dot; other anchors distinct muted colour + dash; a colour-blind-safe
                 sequence for several checkpoints or arms.
* ``variables``  one table of names, display units, native-to-display conversion and colour maps.
* ``labels``     one wording for axes; readable legend labels instead of raw keys.
* ``maps``       Cartopy projection rule, coastlines, borders, labelled gridlines, shared scales.
* ``style``      ``eval_style()`` context (no global side effects at import), ``save_figure``
                 (PNG 150 dpi plus PDF), ``FigureBook`` (multi-page PDF).
* ``probabilistic``  the single probabilistic-score figure used by both the local
                 ``probabilistic`` evaluator and ``quaver``.

Importing this package never changes Matplotlib's global settings and never imports Cartopy
(the ``maps`` helpers import it when called).
"""
from .labels import AXIS, LabelContext, label_and_id, lead_label, pdf_label, power_label, readable_label, shorten_run_label
from .maps import (
    add_geography,
    add_row_colorbar,
    extend_for,
    map_grid,
    new_map_axes,
    select_projection,
    select_projection_bbox,
    shared_norm,
    symmetric_limit,
    symmetric_norm,
)
from .roles import (
    BASELINE_COLOR,
    BETTER_COLOR,
    INPUT_COLOR,
    MODEL_COLOR,
    NEUTRAL_COLOR,
    REFERENCE_STYLES,
    ROLES,
    SEQUENCE,
    TRUTH_COLOR,
    WORSE_COLOR,
    better_worse_cmap,
    reference_style,
    reference_styles,
    role_of,
    role_style,
    sequence_colors,
    sequence_style,
    style_for_key,
)
from .style import FigureBook, eval_style, save_figure, styled
from .variables import VARIABLES, axis_label, convert, convert_difference, display_name, unit_of, variable_spec

__all__ = [n for n in dir() if not n.startswith("_")]
