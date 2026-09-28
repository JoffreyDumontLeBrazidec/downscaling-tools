"""The one probabilistic-score figure, shared by every source of ensemble scores.

``plot_probabilistic_scores(curves, source, out)`` takes a tidy table and draws the same
layout whatever produced the numbers: the local ``probabilistic`` evaluator (scored on the
prediction files) or ``quaver`` (scored from FDB against observations and analyses).

Table columns (one row per curve point)
---------------------------------------
``metric``        ``crps``, ``fcrps``, ``spread``, ``rmse_ens_mean`` (alias ``rmsef``), ``ssr``, ...
``variable``      framework variable name: ``2t``, ``10ff``, ``msl``, ``z_500``, ``t_850``, ...
``domain``        ``n.hem``, ``tropics``, ``s.hem``, ``europe``, ``global``, ...
``lead_h``        lead time in hours
``series_role``   ``model``, ``input``, ``truth``, ``baseline`` or ``reference``
``series_label``  text for the legend (readable, not a raw key)
``value``         the score, in the variable's native unit (converted here for display: hPa for
                  pressure, dam for geopotential height). An optional ``native_unit`` column
                  names a different native unit (quaver stores z in metres, so ``"gpm"``). If a
                  ``unit`` column is present the value is used as given and ``unit`` is printed
optional:  ``ci_low``, ``ci_high`` (confidence band), ``n`` (number of samples, added to the
           legend), ``unit`` (unit of ``value``; also used for dimensionless scores)

Layout
------
One page per variable. Rows are metrics, columns are domains, x is lead time in hours, the
y axis carries the unit in parentheses. Line styles come from the fixed role table
(``eval.plotting.roles``), so "model" is always the red solid line and "input" the blue dashed
one. The page title names the source, e.g. "Source: quaver, FDB, surface vs station
observations, upper air vs 1.5° analysis".
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .labels import AXIS
from .roles import reference_style, role_style, sequence_style
from .style import FigureBook, eval_style
from .variables import axis_label, convert, convert_difference, variable_spec

METRIC_ORDER = ("fcrps", "crps", "spread", "rmse_ens_mean", "rmsef", "sdaf", "ssr")
METRIC_NAMES = {
    "crps": "CRPS",
    "fcrps": "Fair CRPS",
    "spread": "Ensemble spread",
    "rmse_ens_mean": "RMSE of the ensemble mean",
    "rmsef": "RMSE of the ensemble mean",
    "sdaf": "Standard deviation of the error",
    "ssr": "Spread-skill ratio",
}
_DIMENSIONLESS = {"ssr", "crpss", "fcrpss", "ratio"}
DOMAIN_NAMES = {
    "n.hem": "Northern Hemisphere extratropics",
    "s.hem": "Southern Hemisphere extratropics",
    "tropics": "Tropics",
    "europe": "Europe",
    "global": "Global",
    "all": "Global",
}
_ROLE_ORDER = {"truth": 0, "model": 1, "input": 2, "baseline": 3, "reference": 4}

SOURCE_QUAVER = ("Source: quaver — FDB, surface vs station observations, "
                 "upper air vs 1.5° analysis")
# Text of the local evaluator when the lane is not known. The truth of the local evaluator is
# member 0 of the target ensemble stored in the bundle (``y``); which forecast system that is
# depends on the lane, so ``source_local`` names it when the caller can say (see
# ``eval.evaluators.probabilistic.core.plotting.truth_from_lane``).
SOURCE_LOCAL = "Source: local probabilistic evaluator — truth = member 0 of the lane's target ensemble"


def source_local(truth: str | None = None) -> str:
    """Figure source line of the local evaluator, naming the truth (for example "ENFO O1280")."""
    if not truth:
        return SOURCE_LOCAL
    return f"Source: local probabilistic evaluator — truth = {truth} member 0"


def _as_frame(curves):
    import pandas as pd

    df = curves.copy() if isinstance(curves, pd.DataFrame) else pd.DataFrame(list(curves))
    need = {"metric", "variable", "domain", "lead_h", "series_role", "series_label", "value"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"probabilistic table is missing columns: {sorted(missing)}")
    return df


def _canonical_metric(m: str) -> str:
    return "rmse_ens_mean" if m == "rmsef" else m


def _series_styles(df) -> dict[tuple[str, str], dict]:
    """One style per (role, label). Several series of one role get distinct styles."""
    styles: dict[tuple[str, str], dict] = {}
    by_role: dict[str, list[str]] = {}
    for role, label in df[["series_role", "series_label"]].drop_duplicates().itertuples(index=False):
        by_role.setdefault(role, []).append(label)
    for role, labels in by_role.items():
        for i, label in enumerate(labels):
            if len(labels) == 1 and role != "reference":
                styles[(role, label)] = role_style(role)
            elif role == "reference":
                styles[(role, label)] = reference_style(i)
            else:
                styles[(role, label)] = sequence_style(i)
    return styles


def _panel(ax, sub, metric: str, variable: str, styles, unit_col: bool):
    dimensional = metric not in _DIMENSIONLESS
    for (role, label), g in sub.groupby(["series_role", "series_label"], sort=False):
        g = g.sort_values("lead_h")
        x = g["lead_h"].to_numpy(float)
        y = g["value"].to_numpy(float)
        lo = g["ci_low"].to_numpy(float) if "ci_low" in g else None
        hi = g["ci_high"].to_numpy(float) if "ci_high" in g else None
        if dimensional and not unit_col:
            nu = g["native_unit"].dropna().iloc[0] if "native_unit" in g and g["native_unit"].notna().any() else None
            y = np.asarray(convert_difference(variable, y, native_unit=nu), dtype=float)
            if lo is not None:
                lo = np.asarray(convert_difference(variable, lo, native_unit=nu), dtype=float)
                hi = np.asarray(convert_difference(variable, hi, native_unit=nu), dtype=float)
        st = styles[(role, label)]
        marker = "o" if x.size <= 12 else None
        ax.plot(x, y, marker=marker, markersize=4, **st)
        if lo is not None and np.isfinite(lo).any() and np.isfinite(hi).any():
            ax.fill_between(x, lo, hi, color=st["color"], alpha=0.16, linewidth=0, zorder=1)


def _y_label(metric: str, variable: str, unit: str | None) -> str:
    name = METRIC_NAMES.get(metric, metric)
    if metric in _DIMENSIONLESS:
        return name
    u = unit if unit is not None else variable_spec(variable).unit
    return f"{name} ({u})" if u else name


def plot_probabilistic_scores(curves, source: str, out, *, title: str | None = None,
                              png: bool = True, n_noun: str = "samples", metrics=None, variables=None,
                              domains=None, footnote: str | None = None,
                              band_label: str | None = None) -> list[Path]:
    """Draw the probabilistic-score figure and return the files written.

    Parameters
    ----------
    curves
        Tidy table (``pandas.DataFrame`` or iterable of dict), columns described in the module
        docstring.
    source
        Text naming where the numbers come from; shown as the figure title, for example
        ``SOURCE_QUAVER`` or ``SOURCE_LOCAL``.
    out
        Output path, with or without suffix. A multi-page PDF is written to ``<out>.pdf`` and,
        when ``png`` is true, one PNG per page to ``<out>_pages/``.
    title
        Optional second line (for example the run name and date window).
    n_noun
        What the optional ``n`` column counts, for the legend ("dates", "samples").
    metrics, variables, domains
        Optional lists that restrict and order what is drawn.
    band_label
        Legend text for the shaded band (the ``ci_low`` to ``ci_high`` interval), for example
        "95 % confidence interval of the mean over dates". When the table has a band and this
        is given, the legend gets a shaded entry with that text.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    df = _as_frame(curves)
    df = df.assign(metric=df["metric"].map(_canonical_metric))
    if metrics is not None:
        df = df[df["metric"].isin([_canonical_metric(m) for m in metrics])]
    if variables is not None:
        df = df[df["variable"].isin(list(variables))]
    if domains is not None:
        df = df[df["domain"].isin(list(domains))]
    if df.empty:
        raise ValueError("no probabilistic curves to plot")
    unit_col = "unit" in df.columns

    styles = _series_styles(df)
    order = sorted(styles, key=lambda k: (_ROLE_ORDER.get(k[0], 9), k[1]))
    present_metrics = set(df["metric"])
    ordered = [_canonical_metric(m) for m in METRIC_ORDER]
    metric_rows = [m for m in dict.fromkeys(ordered) if m in present_metrics] + \
        sorted(present_metrics - set(ordered))
    dom_cols = list(dict.fromkeys(domains)) if domains else \
        [d for d in DOMAIN_NAMES if d in set(df["domain"])] + \
        sorted(set(df["domain"]) - set(DOMAIN_NAMES))
    var_order = list(dict.fromkeys(variables)) if variables else list(dict.fromkeys(df["variable"]))

    counts = {}
    if "n" in df.columns:
        for key, g in df.groupby(["series_role", "series_label"]):
            n = g["n"].dropna()
            if len(n):
                counts[key] = int(n.max())

    out = Path(out)
    written: list[Path] = []
    with eval_style():
        with FigureBook(out, png=png) as book:
            for var in var_order:
                dv = df[df["variable"] == var]
                if dv.empty:
                    continue
                cols = [d for d in dom_cols if d in set(dv["domain"])]
                rows = [m for m in metric_rows if m in set(dv["metric"])]
                fig, axes = plt.subplots(len(rows), len(cols), squeeze=False,
                                         figsize=(3.9 * len(cols) + 0.6, 2.7 * len(rows) + 1.6))
                for i, m in enumerate(rows):
                    for j, d in enumerate(cols):
                        ax = axes[i, j]
                        sub = dv[(dv["metric"] == m) & (dv["domain"] == d)]
                        if sub.empty:
                            ax.set_axis_off()
                            continue
                        _panel(ax, sub, m, var, styles, unit_col)
                        unit = sub["unit"].iloc[0] if unit_col and sub["unit"].notna().any() else None
                        if j == 0:
                            ax.set_ylabel(_y_label(m, var, unit))
                        if i == 0:
                            ax.set_title(DOMAIN_NAMES.get(d, d))
                        if i == len(rows) - 1:
                            ax.set_xlabel(AXIS["lead"])
                        ax.margins(x=0.04)
                present = set(zip(dv["series_role"], dv["series_label"]))
                handles = []
                for key in order:
                    if key in present:
                        st = styles[key]
                        lab = key[1] + (f" (n = {counts[key]} {n_noun})" if key in counts else "")
                        handles.append(plt.Line2D([], [], color=st["color"], linestyle=st["linestyle"],
                                                  linewidth=st["linewidth"], label=lab))
                has_band = "ci_low" in dv and "ci_high" in dv and \
                    bool(np.isfinite(dv["ci_low"].to_numpy(float)).any())
                if band_label and has_band:
                    handles.append(Patch(facecolor="0.45", alpha=0.3, linewidth=0, label=band_label))
                fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4),
                           bbox_to_anchor=(0.5, 0.0))
                head = variable_spec(var).name if variable_spec(var).unit else str(var)
                n_title_lines = 2 + (1 if title else 0)
                fig_h = fig.get_size_inches()[1]
                fig.suptitle(f"{head}\n{source}" + (f"\n{title}" if title else ""),
                             fontsize=11, y=1.0 - 0.05 / fig_h, va="top")
                if footnote:
                    fig.text(0.99, 0.005, footnote, ha="right", va="bottom", fontsize=7, color="0.4")
                legend_in = 0.35 * (1 + (len(handles) - 1) // 4)
                fig.tight_layout(rect=(0, legend_in / fig_h, 1, 1.0))
                book.add(fig, name=str(var))
            written = list(book.paths)
    return written
