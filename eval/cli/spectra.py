"""``eval.cli spectra``: the standard spectra figure of a read, one command.

The figure is the quick-look grid the owner chose on 9 Nov 2026
(``eval.plotting.spectra_grid``): one panel per variable, spectral amplitude against total
wavenumber with the wavelength on top, the amplitude ratio to the truth underneath on 0.3-3,
truth, input ensemble and model. It reads no lane or host configuration.

Each SOURCE may be:

* a ``box_spectra`` output (a folder holding ``bins.csv``): drawn as it is, nothing recomputed;
* a ``spectra_ecmwf_v2`` output (a folder holding ``spectra_summary.json``): drawn from its
  curve files (global spherical harmonics);
* a predictions folder (``predictions_*.nc``): ``box_spectra`` is run on it, one read per lead
  in ``--steps``, then drawn. Rows with one draw per input member (``_m``) give the input
  ensemble as the input curve.

One figure per source and lead, written as ``<out>/<label>_L<step>/box_spectra_grid.png`` (and
``.pdf``), or ``spectra_ecmwf_grid.png`` for a ``spectra_ecmwf_v2`` source.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from eval.cli._common import Command

LOG = logging.getLogger(__name__)

SUMMARY = "Draw the standard spectra figure (amplitude and ratio to truth, every variable) of a read."
DESCRIPTION = (
    "Draw the standard spectra quick look: one panel per variable (tp, 10 m wind speed, 2t, msl, "
    "z500, ... in that order), amplitude against total wavenumber with the wavelength on top and "
    "the amplitude ratio to the truth underneath (0.3-3), with truth, input ensemble and model. "
    "SOURCE is a box_spectra output (bins.csv), a spectra_ecmwf_v2 output (spectra_summary.json) "
    "or a predictions folder, on which box_spectra is run first, one read per lead. The per-bin "
    "ladder of rule 14 (box_spectra_ladder.png) is written next to it for box reads."
)
# tp first; variables absent from the files are skipped by box_spectra
DEFAULT_VARIABLES = ["tp", "ws", "2t", "msl", "10u", "10v"]


def register(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser("spectra", help=SUMMARY, description=DESCRIPTION)
    p.add_argument("sources", nargs="+", metavar="SOURCE",
                   help="box_spectra output, spectra_ecmwf_v2 output or predictions folder.")
    p.add_argument("--out", required=True, type=Path, help="Folder for the figures (and new reads).")
    p.add_argument("--label", action="append", default=None,
                   help="Row name per SOURCE, in order (default: the folder name).")
    p.add_argument("--steps", type=int, nargs="+", default=[24],
                   help="Leads in hours for predictions folders, one figure each (default 24).")
    p.add_argument("--dates", nargs="+", default=None, help="YYYYMMDD dates (default: all).")
    p.add_argument("--members", type=int, nargs="+", default=None, help="Member indices (default: all).")
    p.add_argument("--variables", nargs="+", default=DEFAULT_VARIABLES,
                   help=f"Variables for new box reads (default {' '.join(DEFAULT_VARIABLES)}).")
    p.add_argument("--box", default=None,
                   help="Box for new reads: a region name (e2, north_atlantic, tc_atlantic, alps). "
                        "Default: the file's own bounds, E2 for global files.")
    p.add_argument("--title", default=None, help="Figure title (default: label, lead, draws).")
    return p


def _kind(src: Path) -> str:
    if (src / "bins.csv").exists():
        return "box"
    if (src / "spectra_summary.json").exists():
        return "ecmwf"
    if any(src.glob("predictions_*.nc")):
        return "predictions"
    raise SystemExit(f"eval.cli spectra: {src} holds no bins.csv, spectra_summary.json or predictions_*.nc")


def _box_title(label: str, results: Path, step: int | None) -> str:
    meta = {}
    try:
        meta = json.loads((results / "run_meta.json").read_text())
    except (OSError, ValueError):
        pass
    steps = step if step is not None else ",".join(str(s) for s in meta.get("steps", []))
    lead = f"+{steps} h" if steps != "" else "lead unknown"
    box = meta.get("box")
    box_txt = f", box {box}" if box else ""
    return f"{label}: spectra vs truth and input, {lead}{box_txt}"


def _draw_box(results: Path, out: Path, title: str) -> list[Path]:
    from eval.evaluators.box_spectra.plotter import plot as box_plot

    written = box_plot(results, {}, {"title": title}, output_dir=out) or []
    return [p for p in written if "grid" in Path(p).name] or list(written)


def run(args: argparse.Namespace) -> list[Path]:
    labels = list(args.label or [])
    out_root = Path(args.out).expanduser()
    written: list[Path] = []
    for i, source in enumerate(args.sources):
        src = Path(source).expanduser()
        label = labels[i] if i < len(labels) else src.name
        kind = _kind(src)
        if kind == "box":
            out = out_root / label
            written += _draw_box(src, out, args.title or _box_title(label, src, None))
        elif kind == "ecmwf":
            from eval.evaluators.spectra_ecmwf_v2.plotter import plot as ecmwf_plot

            out = out_root / label
            ecmwf_plot(src, {}, {"model_label": label,
                                 "title": args.title or f"{label}: spectra vs truth and input"},
                       output_dir=out)
            written += sorted(out.glob("spectra_ecmwf_grid.*"))
        else:
            from eval.evaluators.box_spectra.runner import run as box_run

            for step in args.steps:
                out = out_root / f"{label}_L{step:03d}"
                cfg = {"steps": [step], "variables": list(args.variables), "box": args.box}
                if args.dates:
                    cfg["dates"] = list(args.dates)
                if args.members:
                    cfg["members"] = list(args.members)
                results = box_run(src, {}, cfg, output_dir=out / "box_spectra", run_label=label)
                written += _draw_box(results, out, args.title or _box_title(label, results, step))
    for p in written:
        print(p)
    return written


COMMANDS = (Command("spectra", "figures", SUMMARY, register, run),)
