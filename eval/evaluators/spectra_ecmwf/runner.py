"""ECMWF spectra evaluator runner — 3-stage gptosp pipeline.

Stage 1: stage prediction NetCDF files as nopoles GRIBs  (_grib_stager.py)
Stage 2: convert GRIBs to spectral harmonics via gptosp.ser (shell, AC-only)
Stage 3: compute spectra amplitudes from harmonics              (_amplitude_computer.py, needs metview)

AC-only: gptosp.ser and the eclib/pifsenv/ifs modules are not available on AG.
Resumable: stage 2 skips existing *_sh files on resubmission.

Reference spectra (truth/input) are computed once and saved to reference_dir
for reuse across runs.  Only prediction spectra are recomputed each time.
"""
from __future__ import annotations

import json
import logging
import re
import socket
import subprocess
import sys
from pathlib import Path

LOG = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent

_WEATHER_STATE_TO_DIR = {
    "10u": "10u_sfc",
    "10v": "10v_sfc",
    "2t":  "2t_sfc",
    "sp":  "sp_sfc",
    "msl": "msl_sfc",
    "t_850": "t_850",
    "z_500": "z_500",
}

_DEFAULT_WEATHER_STATES = ["10u", "10v", "2t", "sp", "t_850", "z_500"]

# ECMWF cubic-octahedral pairings: an O<N> target grid is conventionally
# analysed at TCo<N-1>.  The lane config records the input grid, so keep the
# complete downscaling-family mapping here and allow an explicit evaluator
# override for non-standard experiments.
_NOMINAL_SPECTRAL_GRID_BY_INPUT_GRID = {
    "O48": ("O96", 95),
    "O96": ("O320", 319),
    "O320": ("O1280", 1279),
    "O1280": ("O2560", 2559),
}


def _positive_truncation(value: object, *, source: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{source} must be a positive integer, got {value!r}")
    if isinstance(value, int):
        truncation = value
    elif isinstance(value, str) and re.fullmatch(r"[1-9][0-9]*", value.strip()):
        truncation = int(value)
    else:
        raise ValueError(f"{source} must be a positive integer, got {value!r}")
    if truncation <= 0:
        raise ValueError(f"{source} must be a positive integer, got {value!r}")
    return truncation


def _normalise_octahedral_grid(value: object) -> str:
    match = re.fullmatch(r"[oO]([1-9][0-9]*)", str(value).strip())
    return f"O{int(match.group(1))}" if match else ""


def _resolve_truncation(lane_config: dict, eval_config: dict) -> int:
    """Resolve the requested total-wavenumber truncation for this lane."""
    explicit = eval_config.get("truncation")
    if explicit is not None:
        return _positive_truncation(explicit, source="spectra_ecmwf.truncation")

    input_grid = _normalise_octahedral_grid(
        (lane_config.get("prepml", {}).get("input", {}) or {}).get("grid", "")
    )
    pairing = _NOMINAL_SPECTRAL_GRID_BY_INPUT_GRID.get(input_grid)
    if pairing is None:
        supported = ", ".join(sorted(_NOMINAL_SPECTRAL_GRID_BY_INPUT_GRID))
        raise ValueError(
            "Could not infer spectra_ecmwf truncation from "
            f"prepml.input.grid={input_grid or '<missing>'!r}. "
            f"Supported input grids: {supported}. Set spectra_ecmwf.truncation explicitly."
        )

    output_grid, truncation = pairing
    LOG.info(
        "spectra_ecmwf: inferred nominal TCo%d truncation for %s -> %s",
        truncation,
        input_grid,
        output_grid,
    )
    return truncation


def run(
    predictions_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    **kwargs,
) -> Path:
    predictions_dir = Path(predictions_dir).expanduser().resolve()
    output_dir = Path(output_dir) if output_dir else predictions_dir.parent / "evaluators" / "spectra_ecmwf"

    hostname = socket.gethostname()
    if not hostname.startswith("ac"):
        raise RuntimeError(
            f"spectra_ecmwf requires gptosp.ser (AC-only). Current host: {hostname}"
        )

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"spectra_ecmwf output already exists: {output_dir}")

    weather_states: list[str] = eval_config.get("weather_states", _DEFAULT_WEATHER_STATES)
    weather_states_str = ",".join(weather_states) if isinstance(weather_states, list) else str(weather_states)
    template_root: str = eval_config.get("template_root", "")
    template_grib_root: str = eval_config.get("template_grib_root", "")
    predict_cfg = lane_config.get("predict", {})
    date_list = ",".join(str(d) for d in eval_config.get("dates", predict_cfg.get("dates", [])))
    step_list = ",".join(str(s) for s in eval_config.get("steps", predict_cfg.get("steps", [120])))
    member_list = ",".join(str(m) for m in eval_config.get("members", [])) or "ALL"
    reference_dir: str = eval_config.get("reference_dir", "")
    truncation = _resolve_truncation(lane_config, eval_config)

    # --- Prediction spectra (always recomputed) ---
    _run_pipeline(
        label="prediction",
        predictions_dir=predictions_dir,
        output_dir=output_dir,
        prediction_var="y_pred",
        weather_states=weather_states,
        weather_states_str=weather_states_str,
        template_root=template_root,
        template_grib_root=template_grib_root,
        date_list=date_list,
        step_list=step_list,
        member_list=member_list,
        truncation=truncation,
    )

    # --- Reference spectra (truth + input): compute once, save to reference_dir ---
    if reference_dir:
        ref_path = Path(reference_dir).expanduser().resolve()
        for var_name, var_label in [("y", "truth"), ("x_interp", "input")]:
            var_amp_dir = ref_path / var_label / "spectra"
            if _has_amplitudes(var_amp_dir, weather_states, truncation=truncation):
                LOG.info("spectra_ecmwf: %s reference cached at %s — skipping", var_label, var_amp_dir)
                continue
            if not _has_var_in_predictions(predictions_dir, var_name):
                if var_label == "input":
                    bundles_dir = eval_config.get("input_bundles_dir", "")
                    template_grib = eval_config.get("input_template_grib", "")
                    if not template_grib:
                        grid = lane_config.get("prepml", {}).get("input", {}).get("grid", "").lower()
                        if grid:
                            candidate = Path(f"/home/ecm5702/hpcperm/data/{grid}-template.grib")
                            if candidate.exists():
                                template_grib = str(candidate)
                            else:
                                LOG.warning("spectra_ecmwf: no template GRIB for grid %s at %s — skipping input spectra", grid, candidate)
                    if bundles_dir:
                        if not template_grib:
                            LOG.warning("spectra_ecmwf: input_bundles_dir set but no template_grib resolved — skipping input spectra")
                        else:
                            LOG.info("spectra_ecmwf: computing input reference from native bundles at %s", bundles_dir)
                            _run_input_bundle_pipeline(
                                bundles_dir=Path(bundles_dir).expanduser().resolve(),
                                output_dir=ref_path / "input",
                                template_grib=template_grib,
                                weather_states=weather_states,
                                weather_states_str=weather_states_str,
                                date_list=date_list,
                                step_list=step_list,
                                member_list=member_list,
                                truncation=truncation,
                            )
                            continue
                LOG.warning(
                    "spectra_ecmwf: variable '%s' not in predictions — skipping %s reference",
                    var_name, var_label,
                )
                continue
            LOG.info("spectra_ecmwf: computing %s reference spectra → %s", var_label, ref_path / var_label)
            _run_pipeline(
                label=var_label,
                predictions_dir=predictions_dir,
                output_dir=ref_path / var_label,
                prediction_var=var_name,
                weather_states=weather_states,
                weather_states_str=weather_states_str,
                template_root=template_root,
                template_grib_root=template_grib_root,
                date_list=date_list,
                step_list=step_list,
                member_list=member_list,
                truncation=truncation,
            )

    return output_dir



def _has_var_in_predictions(predictions_dir: Path, var_name: str) -> bool:
    """Return True if the first prediction file in predictions_dir contains var_name."""
    pred_files = sorted(predictions_dir.glob("predictions_*.nc"))
    if not pred_files:
        return False
    import xarray as xr
    with xr.open_dataset(pred_files[0]) as ds:
        return var_name in ds


def _has_amplitudes(amp_dir: Path, weather_states: list[str], *, truncation: int) -> bool:
    """Check that a complete amplitude cache uses the requested truncation."""
    if not amp_dir.exists():
        return False
    for ws in weather_states:
        ws_dir = amp_dir / _WEATHER_STATE_TO_DIR.get(ws, ws)
        if not ws_dir.exists() or not list(ws_dir.glob("ampl_*.npy")):
            return False

    summary_path = amp_dir.parent / "spectra_summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        cached_truncation = int(summary["truncation"])
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        LOG.info(
            "spectra_ecmwf: cache at %s has no valid truncation metadata; recomputing",
            amp_dir,
        )
        return False
    if cached_truncation != truncation:
        LOG.info(
            "spectra_ecmwf: cache at %s uses T%d, requested T%d; recomputing",
            amp_dir,
            cached_truncation,
            truncation,
        )
        return False
    return True


def _run_pipeline(
    *,
    label: str,
    predictions_dir: Path,
    output_dir: Path,
    prediction_var: str,
    weather_states: list[str],
    weather_states_str: str,
    template_root: str,
    template_grib_root: str,
    date_list: str,
    step_list: str,
    member_list: str = "ALL",
    truncation: int,
) -> None:
    """Run the full 3-stage pipeline for a single variable."""
    grb_dir = output_dir / "grb"
    sh_dir  = output_dir / "spectral_harmonics"
    amp_dir = output_dir / "spectra"
    for d in (grb_dir, sh_dir, amp_dir):
        d.mkdir(parents=True, exist_ok=True)

    LOG.info("=== spectra_ecmwf [%s] 1/3: stage GRIBs ===", label)
    _stage_gribs(
        predictions_dir=predictions_dir,
        grb_dir=grb_dir,
        weather_states=weather_states_str,
        template_root=template_root,
        template_grib_root=template_grib_root,
        date_list=date_list,
        step_list=step_list,
        prediction_var=prediction_var,
        member_list=member_list,
        summary_path=output_dir / "staging_summary.json",
    )

    LOG.info("=== spectra_ecmwf [%s] 2/3: gptosp transforms ===", label)
    _run_gptosp(grb_dir=grb_dir, sh_dir=sh_dir, weather_states=weather_states)

    LOG.info("=== spectra_ecmwf [%s] 3/3: compute amplitudes ===", label)
    _compute_amplitudes(
        sh_dir=sh_dir,
        amp_dir=amp_dir,
        weather_states=weather_states_str,
        summary_path=output_dir / "spectra_summary.json",
        truncation=truncation,
    )



def _run_input_bundle_pipeline(
    *,
    bundles_dir: Path,
    output_dir: Path,
    template_grib: str,
    weather_states: list[str],
    weather_states_str: str,
    date_list: str,
    step_list: str,
    member_list: str = "1",
    truncation: int,
) -> None:
    """3-stage pipeline for O96 input bundles: stage GRIBs → gptosp → amplitudes."""
    grb_dir = output_dir / "grb"
    sh_dir  = output_dir / "spectral_harmonics"
    amp_dir = output_dir / "spectra"
    for d in (grb_dir, sh_dir, amp_dir):
        d.mkdir(parents=True, exist_ok=True)

    LOG.info("=== spectra_ecmwf [input] 1/3: stage O96 GRIBs from bundles ===")
    cmd = [
        sys.executable, str(_HERE / "_input_bundle_stager.py"),
        "--bundles-dir",    str(bundles_dir),
        "--out-dir",        str(grb_dir),
        "--weather-states", weather_states_str,
        "--date-list",      date_list,
        "--step-list",      step_list,
        "--member-list",    member_list,
        "--summary-path",   str(output_dir / "staging_summary.json"),
    ]
    if template_grib:
        cmd += ["--template-grib", template_grib]
    else:
        # Fall back to default o96-template.grib location
        default_tmpl = Path("/home/ecm5702/hpcperm/data/o96-template.grib")
        if default_tmpl.exists():
            cmd += ["--template-grib", str(default_tmpl)]
        else:
            raise FileNotFoundError("No input_template_grib specified and default o96-template.grib not found")
    subprocess.run(cmd, check=True)

    LOG.info("=== spectra_ecmwf [input] 2/3: gptosp transforms ===")
    _run_gptosp(grb_dir=grb_dir, sh_dir=sh_dir, weather_states=weather_states)

    LOG.info("=== spectra_ecmwf [input] 3/3: compute amplitudes ===")
    _compute_amplitudes(
        sh_dir=sh_dir,
        amp_dir=amp_dir,
        weather_states=weather_states_str,
        summary_path=output_dir / "spectra_summary.json",
        truncation=truncation,
    )


def _stage_gribs(
    *,
    predictions_dir: Path,
    grb_dir: Path,
    weather_states: str,
    template_root: str,
    template_grib_root: str,
    date_list: str,
    step_list: str,
    prediction_var: str = "y_pred",
    member_list: str = "ALL",
    summary_path: Path,
) -> None:
    cmd = [
        sys.executable, str(_HERE / "_grib_stager.py"),
        "--predictions-dir", str(predictions_dir),
        "--out-dir", str(grb_dir),
        "--weather-states", weather_states,
        "--date-list", date_list,
        "--step-list", step_list,
        "--member-list", member_list,
        "--prediction-var", prediction_var,
        "--summary-path", str(summary_path),
    ]
    if template_root:
        cmd += ["--template-root", template_root]
    if template_grib_root:
        cmd += ["--template-grib-root", template_grib_root]
    subprocess.run(cmd, check=True)


def _run_gptosp(*, grb_dir: Path, sh_dir: Path, weather_states: list[str]) -> None:
    """Run gptosp.ser for all GRIBs; skip existing harmonics (resumable).

    gptosp.ser has a ~128-char path limit (Fortran CHARACTER buffer).
    We work around this by creating a short /tmp symlink to sh_dir and
    writing output there.
    """
    param_dirs = [
        _WEATHER_STATE_TO_DIR[ws]
        for ws in weather_states
        if ws in _WEATHER_STATE_TO_DIR
    ]

    lines = [
        "set -euo pipefail",
        "module unload ecmwf-toolbox 2>/dev/null || true",
        "module load eclib   2>/dev/null || true",
        "module load pifsenv 2>/dev/null || true",
        "module load ifs     2>/dev/null || true",
        "export DR_HOOK_ASSERT_MPI_INITIALIZED=0",
        # Short symlinks to avoid gptosp ~128-char path-length truncation
        'SHORT_TMP="$(mktemp -d)"',
        f'ln -s "{grb_dir}" "$SHORT_TMP/g"',
        f'ln -s "{sh_dir}" "$SHORT_TMP/s"',
    ]
    for pd in param_dirs:
        lines += [
            f'mkdir -p "$SHORT_TMP/s/{pd}"',
            f'for grb_file in "$SHORT_TMP/g/{pd}"/*.grb; do',
            '  [[ -f "$grb_file" ]] || continue',
            '  grb_base="$(basename "$grb_file")"',
            f'  sh_out="$SHORT_TMP/s/{pd}/${{grb_base}}_sh"',
            '  [[ -f "$sh_out" ]] && [[ -s "$sh_out" ]] && continue',
            '  echo "[gptosp] $grb_base"',
            '  gptosp.ser -l -g "$grb_file" -S "$sh_out"',
            'done',
        ]
    lines += ['rm "$SHORT_TMP/g" "$SHORT_TMP/s"', 'rmdir "$SHORT_TMP"']

    subprocess.run(
        ["bash", "--login", "-c", "\n".join(lines)],
        check=True,
    )


def _compute_amplitudes(
    *,
    sh_dir: Path,
    amp_dir: Path,
    weather_states: str,
    summary_path: Path,
    truncation: int,
) -> None:
    venv_activate = Path(sys.prefix) / "bin" / "activate"
    script = "\n".join([
        "set -euo pipefail",
        "module unload ifs         2>/dev/null || true",
        "module load ecmwf-toolbox 2>/dev/null || true",
        # metview startup needs a writable shared-scratch TMPDIR (node-local /tmp hangs) +
        # a generous start timeout; else _amplitude_computer.py times out on `import metview`.
        'export TMPDIR="${SCRATCH:-/tmp}/mvamp_$$"; mkdir -p "$TMPDIR"',
        'export METVIEW_TMPDIR="$TMPDIR"',
        'export METVIEW_PYTHON_START_TIMEOUT="${METVIEW_PYTHON_START_TIMEOUT:-900}"',
        f'source "{venv_activate}"',
        f'python "{_HERE / "_amplitude_computer.py"}"'
        f' --spectral-harmonics-dir "{sh_dir}"'
        f' --out-dir "{amp_dir}"'
        f' --weather-states "{weather_states}"'
        f' --truncation "{truncation}"'
        f' --summary-path "{summary_path}"',
    ])
    subprocess.run(["bash", "-c", script], check=True)
