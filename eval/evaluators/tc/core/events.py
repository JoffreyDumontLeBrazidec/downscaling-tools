"""TC event identity — geographic/temporal only.

SINGLE SOURCE OF TRUTH for TC boxes/dates is ``eval/config/events/*.yaml``
(see ARCHITECTURE.md: lane/host/event configuration lives in ``eval/config/``).
This module does NOT hardcode coordinates: it loads those YAML configs into the
``EVENTS`` registry so existing imports (``from .events import EVENTS, TCEvent``)
keep working. To add or change an event, edit its YAML — never this file.
"""
from __future__ import annotations

from dataclasses import dataclass

from eval.config import loader as _loader
from eval.config.loader import load_event

from .data_types import BoundingBox


@dataclass(frozen=True)
class TCEvent:
    name: str
    year: str
    month: str
    dates: list[str]  # DD strings
    analysis_dates: list[str]  # YYYYMMDD strings
    bbox: BoundingBox  # Evaluation crop box — must NOT overlap with other scoring events
    scoring_eligible: bool = True  # False for combined events like franklin_idalia


def _event_from_cfg(cfg: dict) -> TCEvent:
    """Build a TCEvent from a validated event-config dict (see loader.load_event)."""
    analysis_dates = [str(d) for d in cfg.get("analysis_dates", [])]
    year = str(cfg.get("year") or (analysis_dates[0][:4] if analysis_dates else ""))
    month = str(cfg.get("month") or (analysis_dates[0][4:6] if analysis_dates else ""))
    return TCEvent(
        name=cfg["name"],
        year=year,
        month=month,
        dates=[str(d) for d in cfg["dates"]],
        analysis_dates=analysis_dates,
        bbox=BoundingBox(
            north=float(cfg["lat_max"]),
            south=float(cfg["lat_min"]),
            east=float(cfg["lon_max"]),
            west=float(cfg["lon_min"]),
        ),
        scoring_eligible=bool(cfg.get("scoring_eligible", True)),
    )


def _load_events() -> dict[str, TCEvent]:
    """Load every event YAML in eval/config/events/ into the registry."""
    events_dir = _loader._CONFIG_DIR / "events"
    out: dict[str, TCEvent] = {}
    for path in sorted(events_dir.glob("*.yaml")):
        cfg = load_event(path.stem)
        out[cfg["name"]] = _event_from_cfg(cfg)
    return out


# Registry built from the YAML source of truth at import time.
EVENTS: dict[str, TCEvent] = _load_events()


def get_event(name: str) -> TCEvent:
    return EVENTS[name]


def get_scoring_events() -> list[TCEvent]:
    return [e for e in EVENTS.values() if e.scoring_eligible]


def events_for_period(year: str, month: str) -> list[TCEvent]:
    return [e for e in EVENTS.values() if e.year == year and e.month == month]
