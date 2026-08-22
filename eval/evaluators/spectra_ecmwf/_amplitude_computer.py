#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import eccodes as ec
import metview as mv
import numpy as np


FILE_RE = re.compile(r".*_(?P<date>\d{8})_(?P<step>\d{2,3})_(?P<member>\d+)_nopoles\.grb_sh$")


@dataclass(frozen=True)
class SpectraConfig:
    weather_state: str
    param: str
    level: str
    dir_name: str


CONFIGS: dict[str, SpectraConfig] = {
    "2t": SpectraConfig(weather_state="2t", param="2t", level="sfc", dir_name="2t_sfc"),
    "10u": SpectraConfig(weather_state="10u", param="10u", level="sfc", dir_name="10u_sfc"),
    "10v": SpectraConfig(weather_state="10v", param="10v", level="sfc", dir_name="10v_sfc"),
    "sp": SpectraConfig(weather_state="sp", param="sp", level="sfc", dir_name="sp_sfc"),
    "msl": SpectraConfig(weather_state="msl", param="msl", level="sfc", dir_name="msl_sfc"),
    "t_850": SpectraConfig(weather_state="t_850", param="t", level="850", dir_name="t_850"),
    "z_500": SpectraConfig(weather_state="z_500", param="z", level="500", dir_name="z_500"),
}


def positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {raw!r}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute spectra amplitudes from actual spectral_harmonics outputs."
    )
    parser.add_argument("--spectral-harmonics-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--weather-states", default="10u,10v,2t,sp,t_850,z_500")
    parser.add_argument(
        "--truncation",
        type=positive_int,
        required=True,
        help=(
            "Total wavenumber truncation this run must produce (for example 1279 for "
            "O1280). This is asserted against the truncation actually stored in each "
            "spectral-harmonics file, not used as an upper clamp."
        ),
    )
    parser.add_argument("--summary-path", default="")
    return parser.parse_args()


def parse_weather_states(raw: str) -> list[str]:
    states = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [state for state in states if state not in CONFIGS]
    if unknown:
        raise ValueError(f"Unsupported weather states: {unknown}")
    return states


def parse_components(path: Path) -> tuple[int, int, int]:
    match = FILE_RE.match(path.name)
    if not match:
        raise ValueError(f"Unrecognized spectral harmonic filename: {path}")
    return (
        int(match.group("date")),
        int(match.group("step")),
        int(match.group("member")),
    )


def read_truncation(path: Path) -> int:
    """Total wavenumber truncation actually stored in a spectral GRIB."""
    with open(path, "rb") as handle:
        msg = ec.codes_grib_new_from_file(handle)
    if msg is None:
        raise RuntimeError(f"No GRIB message in {path}")
    try:
        return int(ec.codes_get(msg, "pentagonalResolutionParameterJ"))
    finally:
        ec.codes_release(msg)


def read_curve(path: Path, cfg: SpectraConfig, *, truncation: int) -> tuple[np.ndarray, np.ndarray]:
    fs_in = mv.Fieldset()
    if cfg.level == "sfc":
        fs_in.append(mv.read(data=mv.read(str(path)), param=cfg.param))
    else:
        fs_in.append(mv.read(data=mv.read(str(path)), levelist=cfg.level, param=cfg.param))
    if len(fs_in) != 1:
        raise RuntimeError(f"Expected exactly one field in {path}, got {len(fs_in)}")
    # The two axis_type values below are misspelled, and deliberately left as they
    # are: mv.spec_graph reads the true truncation from the file and returns raw
    # INPUT_*_VALUES, so the axis keywords never touch the numbers.  This was
    # confirmed by reproducing every cached curve from the GRIB coefficients with
    # eccodes to ~5e-12.  Correcting the spelling would activate a plot definition
    # that is never rendered, so it is a behaviour change for no gain.  This whole
    # call is replaced by the eccodes path in a later step.
    sp = mv.spec_graph(
        data=fs_in,
        truncation=truncation,
        x_axis_type="logartihmic",
        y_axis_type="logartihmic",
    )
    wvn = np.array(sp[1]["INPUT_X_VALUES"])
    ampl = np.array(sp[1]["INPUT_Y_VALUES"])
    return wvn, ampl


def main() -> None:
    args = parse_args()
    sh_root = Path(args.spectral_harmonics_dir).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    states = parse_weather_states(args.weather_states)

    written = []
    achieved_truncations: set[int] = set()
    for state in states:
        cfg = CONFIGS[state]
        in_dir = sh_root / cfg.dir_name
        out_dir = out_root / cfg.dir_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(in_dir.glob("*_nopoles.grb_sh")):
            date_ymd, step_hours, member = parse_components(path)
            achieved = read_truncation(path)
            if achieved != args.truncation:
                raise RuntimeError(
                    f"{path} carries T{achieved} but T{args.truncation} was requested. "
                    "Stage 2 (gptosp -T) and stage 3 must agree; refusing to write a "
                    "spectrum whose truncation differs from the one recorded in the "
                    "summary."
                )
            achieved_truncations.add(achieved)
            wvn, ampl = read_curve(path, cfg, truncation=achieved)
            stem = f"{date_ymd}_{step_hours}_{cfg.weather_state}_n{member}"
            wvn_path = out_dir / f"wvn_{stem}.npy"
            ampl_path = out_dir / f"ampl_{stem}.npy"
            np.save(wvn_path, wvn)
            np.save(ampl_path, ampl)
            written.append(
                {
                    "weather_state": cfg.weather_state,
                    "input": str(path),
                    "wavenumbers": str(wvn_path),
                    "amplitudes": str(ampl_path),
                    "date": date_ymd,
                    "step_hours": step_hours,
                    "member": member,
                    "truncation": args.truncation,
                    "achieved_truncation": achieved,
                }
            )

    if not written:
        raise RuntimeError(f"No spectra amplitudes were written from {sh_root}")

    summary = {
        "spectral_harmonics_dir": str(sh_root),
        "out_dir": str(out_root),
        "weather_states": states,
        # "truncation" is kept under its original name because the runner cache
        # validation already reads it; the two explicit keys below say which is which.
        "truncation": args.truncation,
        "requested_truncation": args.truncation,
        "achieved_truncation": sorted(achieved_truncations),
        "truncation_convention": "cubic_octahedral_TCo",
        "written_count": len(written),
        "files": written,
    }
    summary_path = (
        Path(args.summary_path).expanduser().resolve()
        if args.summary_path
        else (out_root / "spectra_summary.json")
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote spectra summary: {summary_path}")


if __name__ == "__main__":
    main()
