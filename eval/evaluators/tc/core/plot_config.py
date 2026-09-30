"""Per-event visualization defaults and reference styles."""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class TCPlotConfig:
    mslp_bin_range: tuple[float, float, float] = (980, 1022, 5)
    wind_bin_range: tuple[float, float, float] = (0, 36.01, 4)
    mslp_ylim: tuple[float, float] = (0, 4)
    wind_ylim: tuple[float, float] = (0, 2)
    regrid_resolution: float = 0.25
    plot_title: str = ""
    # Read only by the per-member maps, retired 2026-09-30. Kept so that a lane
    # tc.plot_config override naming them still loads.
    member_map_msl_range: tuple[float, float] | None = None
    member_map_wind_range: tuple[float, float] | None = None


PLOT_CONFIGS: dict[str, TCPlotConfig] = {
    "franklin": TCPlotConfig(
        mslp_bin_range=(930, 1026, 5),
        wind_bin_range=(0, 56.01, 4),
        mslp_ylim=(0, 4),
        wind_ylim=(0, 2),
        plot_title="Franklin normed pdfs",
    ),
    "idalia": TCPlotConfig(
        mslp_bin_range=(955, 1026, 5),
        wind_bin_range=(0, 46.01, 4),
        mslp_ylim=(0, 4),
        wind_ylim=(0, 2),
        plot_title="Idalia normed pdfs",
    ),
    "franklin_idalia": TCPlotConfig(
        mslp_bin_range=(930, 1026, 5),
        wind_bin_range=(0, 56.01, 4),
        plot_title="Franklin + Idalia normed pdfs",
    ),
    "hilary": TCPlotConfig(
        mslp_bin_range=(960, 1022, 5),
        wind_bin_range=(0, 30.01, 4),
        mslp_ylim=(0, 2),
        plot_title="Hilary normed pdfs",
    ),
    "dora": TCPlotConfig(
        mslp_bin_range=(970, 1022, 5),
        plot_title="Dora normed pdfs",
    ),
    "fernanda": TCPlotConfig(
        wind_bin_range=(0, 30.01, 4),
        mslp_ylim=(0, 10),
        wind_ylim=(0, 5),
        plot_title="Fernanda normed pdfs",
    ),
    "humberto": TCPlotConfig(
        mslp_bin_range=(910, 1026, 5),
        wind_bin_range=(0, 60.01, 4),
        plot_title="TC Humberto 2025-09 | norm. PDFs (MSLP & 10m Wind)",
        member_map_msl_range=(950, 1025),
        member_map_wind_range=(0, 45),
    ),
}


_TUPLE_FIELDS = {"mslp_bin_range", "wind_bin_range", "mslp_ylim", "wind_ylim", "member_map_msl_range", "member_map_wind_range"}


def resolve_plot_config(event_name: str, eval_config: dict | None = None) -> TCPlotConfig:
    """Return plot config for *event_name*, with optional per-lane overrides.

    *eval_config* is the ``tc:`` section of a lane YAML.  If it contains a
    ``plot_config:<event_name>`` mapping, those fields override the base
    PLOT_CONFIGS entry via ``dataclasses.replace()``.

    Example lane YAML::

        tc:
          plot_config:
            humberto:
              mslp_bin_range: [910, 1025, 4]
              wind_bin_range: [0, 60.01, 3]
    """
    base = PLOT_CONFIGS.get(event_name, TCPlotConfig())
    if not eval_config:
        return base
    overrides = (eval_config.get("plot_config") or {}).get(event_name)
    if not overrides:
        return base
    # YAML lists → tuples for frozen-dataclass fields
    coerced = {k: tuple(v) if k in _TUPLE_FIELDS and isinstance(v, list) else v for k, v in overrides.items()}
    return replace(base, **coerced)


def _reference_entry(label: str, index: int) -> dict:
    """Readable legend label plus the ``index``-th house reference style (eval.plotting)."""
    from eval.plotting import reference_style

    style = reference_style(index)
    return {"label": label, "color": style["color"], "linestyle": style["linestyle"],
            "linewidth": style["linewidth"]}


# Anchor curves drawn next to the model on the TC distribution figures. Each entry has its own
# colour and dash pattern taken from ``eval.plotting.REFERENCE_STYLES`` (none of them uses the
# black, red or blue that mean truth, model and input). The indices follow the house names
# where one exists (ENFO O320 = 0, ENFO O96 = 1, EEFO O96 = 2, IEKM = 3, ENFO O48 = 5).
REFERENCE_STYLES: dict[str, dict] = {
    "ENFO_O320_0001": _reference_entry("ENFO O320", 0),
    "ENFO_O48_0001": _reference_entry("ENFO O48", 5),
    "EEFO_O96_0001": _reference_entry("EEFO O96", 2),
    "ENFO_O96_0001": _reference_entry("ENFO O96", 1),
    "ENFO_O320_ip6y": _reference_entry("ENFO O320 (ip6y)", 7),
    "IEKM_O96_TARGET": _reference_entry("IEKM O96", 3),
}
