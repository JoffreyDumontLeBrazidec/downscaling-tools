"""Named boxes shared by the evaluators, so one name means one box everywhere.

Boxes are [lat_min, lat_max, lon_min, lon_max], longitudes in -180..180, the lane YAML order.

``alps`` is the box of the run-judging protocol (docs ``skills/downscaling-evaluation/
references/run-judging-protocol.md`` section 3): the texture evaluator's Alps region, the
default Alps of ``zoom_maps`` and of ``region_plot``. Other "Alps" boxes in notes
(43-49 N 6-12 E, 41-52 N 2-18 E, 40-50 N 0-10 E) are not this one; ``alps_innsbruck``
(46-49 N, 10-14 E) is a separate showcase box of the o1280->o2560 lane.
"""
from __future__ import annotations

ALPS: list[float] = [43.0, 49.0, 4.0, 16.0]
# the Europe-box (E2) model's domain; a global prediction is cut to it for a box read
E2: list[float] = [10.0, 70.0, -80.0, 40.0]
# open ocean inside both E2 and global files (the texture evaluator's open_north_atlantic)
NORTH_ATLANTIC: list[float] = [35.0, 55.0, -45.0, -20.0]
# the regional TC files of the o320->o1280 lane (Franklin/Idalia), the shape epic's box
TC_ATLANTIC: list[float] = [10.0, 40.0, -100.0, -58.0]

# zoom_maps --region presets: the fixed boxes of the protocol
PROTOCOL_REGIONS: dict[str, list[float]] = {
    "alps": ALPS,
    # the quiet box of the regional TC lane, where the Alps is outside the files
    "tc_quiet": [12.0, 22.0, -72.0, -60.0],
}
