"""Gate ids, margins and the gates csv format of ``eval.cli verdict``.

Margins are PROPOSALS until the calibration of note s7 (build step B3) has run; after B3 they are frozen here and
changed only with a dated owner line (note s7). One place for every number so the calibration edits one file.
"""
from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

HARD_GATES = {
    "G-A1": "fair CRPS vs GL512 450k, wind and tp, each lead band",
    "G-A2": "fair CRPS vs GL512 450k, 2t and msl, each lead band",
    "G-A3": "fair CRPS vs ENS9, wind tp 2t msl, leads pooled",
    "G-A4": "fair CRPS vs INPUT, wind tp 2t, each lead band (better)",
    "G-A5": "spread/RMSE vs ENS9 and spread ratio vs GL512 450k, each lead band",
    "G-B1": "spectra amplitude ratio per bin vs GL512 450k (125-31 km), ws and tp, residual and full",
    "G-B2": "no spectral overshoot above 1.15 (125-17.6 km), ws and tp",
    "G-B3": "final_battery OUT lines not more than GL512 450k and none new",
    "G-B4": "texture grain index and shape elongated fraction vs GL512 450k",
    "G-B5": "lead-to-lead change ratio inside 0.9-1.1",
    "G-C1": "cyclone central pressure and max wind fair CRPS vs GL512 450k (IBTrACS)",
    "G-C2": "storm centre distance from the INPUT centre vs GL512 450k",
    "G-C3": "station twCRPS 20 and 50 mm / 12 h vs GL512 450k",
    "G-C4": "tail shares: log-ratio distance to ENS9 not worse than GL512 450k by the margin",
}
READER_OF = {g: g[2] for g in HARD_GATES}
PREFLIGHT = ("P1", "P2", "P3", "P4", "P5")

# margins in % unless said otherwise (s4, proposals)
MARGIN = {
    "G-A1": {"10ff": 1.0, "tp": 2.0},
    "G-A2": {"2t": 1.0, "msl": 1.0},
    "G-A3": {"10ff": 2.0, "tp": 2.0, "2t": 2.0, "msl": 2.0},
    "G-A5_ssr_abs": 0.10,          # |SSR run - SSR ENS9| at most this
    "G-A5_ratio_se": 2.0,          # spread ratio: |r-1| at most |r_ref-1| + this x SE
    "G-B1_se": 2.0,                # a bin fails when |r-1| exceeds the reference's by more than this x SE
    "G-B1_adjacent": 2,            # the gate fails on this many adjacent failing bins
    "G-B1_km": (31.0, 125.0),
    "G-B1_info_band": (0.9, 1.1),  # printed, information only (owner D7, "refine later")
    "G-B2_km": (17.6, 125.0),
    "G-B2_max": 1.15,
    "G-B5_band": (0.9, 1.1),
    "G-C1": 3.0,
    "G-C2_se": 2.0,
    "G-C3": 3.0,                   # s4: "MDE of the set (2-4 %)"; 3 % until B3
    # G-C4 (B3, 19 Nov 2026): excess = |ln(share_run/share_ENS9)| - |ln(share_ref/share_ENS9)|, in ln units, non-inferiority.
    # Margins ~1.2-2x the largest move between adjacent checkpoints seen on XSET-1 (GL512 450k->500k: 0.08 / 0.10 / 0.12 for
    # rain >= 50/100/200 mm; FS-242 100k->150k: 0.04 / 0.06 / 0.16); n = 2 checkpoint pairs, to tighten with a seed floor.
    "G-C4": {"ws25": 0.10, "tp50": 0.10, "tp100": 0.15, "tp200": 0.20},
}

# station score selection of obs_points scores_per_date.csv (same as xset_table.py)
VARIANT = {"2t": "lapse", "10ff": "raw", "msl": "raw", "tp": "raw", "2d": "raw", "sp": "raw"}


@dataclass
class GateLine:
    gate: str                # G-A1 ... or P1 ...
    sub: str                 # what this line covers (variable, band, bin ...)
    hard: bool               # True for a sub-line of one of the 14 hard gates or a preflight check
    state: str               # PASS FAIL INCONCLUSIVE MISSING (or INFO for reported lines)
    value: float = math.nan  # the run's number (or its difference, see unit)
    ref_value: float = math.nan
    diff: float = math.nan
    unit: str = ""
    se: float = math.nan
    lo: float = math.nan
    hi: float = math.nan
    margin: float = math.nan
    mde: float = math.nan
    n_dates: int = 0
    n_blocks: int = 0
    note: str = ""


COLUMNS = [f.name for f in fields(GateLine)]


def _fmt(v):
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        return "" if not math.isfinite(v) else f"{v:.6g}"
    return str(v)


def write_csv(lines: list[GateLine], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for ln in lines:
            w.writerow({k: _fmt(v) for k, v in asdict(ln).items()})
    return path


def read_csv(path: Path) -> list[GateLine]:
    out = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            kw = {}
            for f in fields(GateLine):
                v = r.get(f.name, "")
                if f.type in ("float", float):
                    kw[f.name] = float(v) if v not in ("", None) else math.nan
                elif f.type in ("int", int):
                    kw[f.name] = int(float(v)) if v not in ("", None) else 0
                elif f.type in ("bool", bool):
                    kw[f.name] = v in ("1", "True", "true")
                else:
                    kw[f.name] = v or ""
            out.append(GateLine(**kw))
    return out
