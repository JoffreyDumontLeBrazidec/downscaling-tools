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

import logging
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
    reference_dir: str = eval_config.get("reference_dir", "")

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
    )

    # --- Reference spectra (truth + input): compute once, save to reference_dir ---
    if reference_dir:
        ref_path = Path(reference_dir).expanduser().resolve()
        for var_name, var_label in [("y", "truth"), ("x_interp", "input")]:
            var_amp_dir = ref_path / var_label / "spectra"
            if _has_amplitudes(var_amp_dir, weather_states):
                LOG.info("spectra_ecmwf: %s reference cached at %s — skipping", var_label, var_amp_dir)
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
            )

    return output_dir


def _has_amplitudes(amp_dir: Path, weather_states: list[str]) -> bool:
    """Check if amplitude directory already has npy files for all weather states."""
    if not amp_dir.exists():
        return False
    for ws in weather_states:
        ws_dir = amp_dir / _WEATHER_STATE_TO_DIR.get(ws, ws)
        if not ws_dir.exists() or not list(ws_dir.glob("ampl_*.npy")):
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
    summary_path: Path,
) -> None:
    cmd = [
        sys.executable, str(_HERE / "_grib_stager.py"),
        "--predictions-dir", str(predictions_dir),
        "--out-dir", str(grb_dir),
        "--weather-states", weather_states,
        "--date-list", date_list,
        "--step-list", step_list,
        "--member-list", "ALL",
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
        # Short symlink to avoid gptosp path-length truncation
        f'SHORT_SH="$(mktemp -d)/sh"',
        f'ln -s "{sh_dir}" "$SHORT_SH"',
    ]
    for pd in param_dirs:
        in_dir  = grb_dir / pd
        lines += [
            f'mkdir -p "$SHORT_SH/{pd}"',
            f'for grb_file in "{in_dir}"/*.grb; do',
            '  [[ -f "$grb_file" ]] || continue',
            '  grb_base="$(basename "$grb_file")"',
            f'  sh_out="$SHORT_SH/{pd}/${{grb_base}}_sh"',
            '  [[ -f "$sh_out" ]] && [[ -s "$sh_out" ]] && continue',
            '  echo "[gptosp] $grb_base"',
            '  gptosp.ser -l -g "$grb_file" -S "$sh_out"',
            'done',
        ]
    lines.append('rm "$SHORT_SH"')

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
) -> None:
    venv_activate = Path(sys.prefix) / "bin" / "activate"
    script = "\n".join([
        "set -euo pipefail",
        "module unload ifs         2>/dev/null || true",
        "module load ecmwf-toolbox 2>/dev/null || true",
        f'source "{venv_activate}"',
        'METVIEW_BIN="$(command -v metview 2>/dev/null || true)"',
        '[[ -n "$METVIEW_BIN" ]] && export PATH="$(dirname "$METVIEW_BIN"):$PATH"',
        f'python "{_HERE / "_amplitude_computer.py"}"'
        f' --spectral-harmonics-dir "{sh_dir}"'
        f' --out-dir "{amp_dir}"'
        f' --weather-states "{weather_states}"'
        f' --summary-path "{summary_path}"',
    ])
    subprocess.run(["bash", "--login", "-c", script], check=True)
