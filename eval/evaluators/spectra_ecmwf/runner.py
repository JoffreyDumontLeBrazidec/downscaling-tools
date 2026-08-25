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

import hashlib
import json
import logging
import re
import socket
import subprocess
import sys
from pathlib import Path

from eval._backends.env.toolchain import render_module_block

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
    dates = [str(d) for d in eval_config.get("dates", predict_cfg.get("dates", []))]
    steps = [int(s) for s in eval_config.get("steps", predict_cfg.get("steps", [120]))]
    members = [int(m) for m in eval_config.get("members", [])]
    date_list = ",".join(dates)
    step_list = ",".join(str(s) for s in steps)
    member_list = ",".join(str(m) for m in members) or "ALL"
    reference_dir: str = eval_config.get("reference_dir", "")
    truncation = _resolve_truncation(lane_config, eval_config)

    # Resolve the reference location up front so the prediction summary can
    # record it. Without this the scoreboard falls back to guessing from
    # template_root, which for this evaluator is the GRIB template directory,
    # not the reference spectra directory.
    ref_path: Path | None = None
    window_key = ""
    reference_spectra_dir = ""
    if reference_dir:
        ref_path = Path(reference_dir).expanduser().resolve()
        window_key = _window_key(
            dates=dates, steps=steps, members=members, truncation=truncation
        )
        reference_spectra_dir = str(ref_path / "truth" / window_key / "spectra")
        LOG.info("spectra_ecmwf: reference window key = %s", window_key)

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
        reference_spectra_dir=reference_spectra_dir,
    )

    # --- Reference spectra (truth + input): compute once, save to reference_dir ---
    if ref_path is not None:
        for var_name, var_label in [("y", "truth"), ("x_interp", "input")]:
            var_ref_dir = ref_path / var_label / window_key
            var_amp_dir = var_ref_dir / "spectra"
            status = _validated_cache(
                var_amp_dir,
                weather_states,
                truncation=truncation,
                dates=dates,
                steps=steps,
                members=members,
            )
            if status == CACHE_VALID:
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
                                output_dir=var_ref_dir,
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
            LOG.info("spectra_ecmwf: computing %s reference spectra → %s", var_label, var_ref_dir)
            _run_pipeline(
                label=var_label,
                predictions_dir=predictions_dir,
                output_dir=var_ref_dir,
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


# A reference cache is only reusable for the window it was actually computed
# from.  Before window addressing, one directory per lane held whichever window
# happened to be computed first, and every later evaluation silently reused it:
# the o320_o1280 reference holds 2023-08-26..30 (Hurricane Idalia), so a
# September-2025 run was scored against 2023 truth.  Recomputing into the same
# directory would be no better, because the two windows would then be mixed and
# the next run would reuse the mixture.  So the window is part of the path.
CACHE_VALID = "VALID"
CACHE_STALE = "STALE"
CACHE_UNVERIFIABLE = "UNVERIFIABLE"


def _window_key(
    *,
    dates: list[str],
    steps: list[int],
    members: list[int],
    truncation: int,
) -> str:
    """Deterministic directory name for one reference window.

    The readable prefix is for eyeballing a directory listing; the hash suffix
    is what actually makes the key injective, so two different windows can never
    land in the same directory and be mixed.
    """
    d = sorted(str(x) for x in dates)
    s = sorted(int(x) for x in steps)
    m = sorted(int(x) for x in members)

    date_part = f"d{d[0]}-{d[-1]}" if len(d) > 1 else (f"d{d[0]}" if d else "dNONE")
    step_part = f"s{s[0]}-{s[-1]}" if len(s) > 1 else (f"s{s[0]}" if s else "sNONE")
    member_part = (f"m{m[0]}-{m[-1]}" if len(m) > 1 else f"m{m[0]}") if m else "mALL"

    canonical = repr((d, s, m, int(truncation)))
    digest = hashlib.blake2b(canonical.encode("utf-8"), digest_size=4).hexdigest()
    return f"{date_part}_{step_part}_{member_part}_T{truncation}_{digest}"


def _cache_coverage(summary: dict) -> tuple[set[str], set[int], set[int]]:
    """Dates, steps and members a spectra_summary.json says it actually holds."""
    dates: set[str] = set()
    steps: set[int] = set()
    members: set[int] = set()
    for entry in summary.get("files") or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("date") is not None:
            dates.add(str(entry["date"]))
        if entry.get("step_hours") is not None:
            steps.add(int(entry["step_hours"]))
        if entry.get("member") is not None:
            members.add(int(entry["member"]))
    return dates, steps, members


def _validated_cache(
    amp_dir: Path,
    weather_states: list[str],
    *,
    truncation: int,
    dates: list[str] | None = None,
    steps: list[int] | None = None,
    members: list[int] | None = None,
) -> str:
    """Decide whether a cached reference may be used as truth for this request.

    Returns CACHE_VALID, CACHE_STALE or CACHE_UNVERIFIABLE.  Both of the latter
    mean "recompute": this is a correct default rather than a validator that
    refuses to run.  The distinction matters because an UNVERIFIABLE cache must
    never be presented as truth, whereas a STALE one is simply the wrong window.

    Passing dates/steps/members as None skips the window check, which is what
    callers that genuinely do not know the window want.
    """
    if not amp_dir.exists():
        return CACHE_STALE
    for ws in weather_states:
        ws_dir = amp_dir / _WEATHER_STATE_TO_DIR.get(ws, ws)
        if not ws_dir.exists() or not list(ws_dir.glob("ampl_*.npy")):
            return CACHE_STALE

    summary_path = amp_dir.parent / "spectra_summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        cached_truncation = int(summary["truncation"])
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        LOG.warning(
            "spectra_ecmwf: cache at %s holds amplitude files but no readable truncation "
            "metadata, so what it represents cannot be established. Not usable as truth; "
            "recomputing.",
            amp_dir,
        )
        return CACHE_UNVERIFIABLE

    if cached_truncation != truncation:
        LOG.info(
            "spectra_ecmwf: cache at %s uses T%d, requested T%d; recomputing",
            amp_dir,
            cached_truncation,
            truncation,
        )
        return CACHE_STALE

    if dates is None and steps is None and members is None:
        return CACHE_VALID

    have_dates, have_steps, have_members = _cache_coverage(summary)
    if not have_dates and not have_steps:
        LOG.warning(
            "spectra_ecmwf: cache at %s records no per-file date or step metadata, so it "
            "cannot be matched against the requested window. Not usable as truth; "
            "recomputing.",
            amp_dir,
        )
        return CACHE_UNVERIFIABLE

    if dates is not None:
        missing = sorted({str(d) for d in dates} - have_dates)
        if missing:
            LOG.info(
                "spectra_ecmwf: cache at %s covers dates %s but %s were requested; recomputing",
                amp_dir,
                ",".join(sorted(have_dates)) or "(none)",
                ",".join(missing),
            )
            return CACHE_STALE
    if steps is not None:
        missing_steps = sorted({int(s) for s in steps} - have_steps)
        if missing_steps:
            LOG.info(
                "spectra_ecmwf: cache at %s covers steps %s but %s were requested; recomputing",
                amp_dir,
                ",".join(str(x) for x in sorted(have_steps)) or "(none)",
                ",".join(str(x) for x in missing_steps),
            )
            return CACHE_STALE
    if members:
        missing_members = sorted({int(m) for m in members} - have_members)
        if missing_members:
            LOG.info(
                "spectra_ecmwf: cache at %s covers members %s but %s were requested; recomputing",
                amp_dir,
                ",".join(str(x) for x in sorted(have_members)) or "(none)",
                ",".join(str(x) for x in missing_members),
            )
            return CACHE_STALE
    else:
        LOG.info(
            "spectra_ecmwf: no member list requested, so member coverage is unchecked; "
            "cache at %s holds members %s",
            amp_dir,
            ",".join(str(x) for x in sorted(have_members)) or "(none)",
        )
    return CACHE_VALID


def _has_amplitudes(amp_dir: Path, weather_states: list[str], *, truncation: int) -> bool:
    """Truncation-only cache check, kept for callers that have no window."""
    return (
        _validated_cache(amp_dir, weather_states, truncation=truncation) == CACHE_VALID
    )


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
    reference_spectra_dir: str = "",
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
    _run_gptosp(
        grb_dir=grb_dir, sh_dir=sh_dir, weather_states=weather_states, truncation=truncation
    )

    LOG.info("=== spectra_ecmwf [%s] 3/3: compute amplitudes ===", label)
    _compute_amplitudes(
        sh_dir=sh_dir,
        amp_dir=amp_dir,
        weather_states=weather_states_str,
        summary_path=output_dir / "spectra_summary.json",
        truncation=truncation,
        reference_spectra_dir=reference_spectra_dir,
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
    _run_gptosp(
        grb_dir=grb_dir, sh_dir=sh_dir, weather_states=weather_states, truncation=truncation
    )

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


def _read_achieved_truncation(sh_path: Path) -> int:
    """Return the spectral truncation actually stored in a harmonics GRIB."""
    import eccodes as ec

    with open(sh_path, "rb") as handle:
        msg = ec.codes_grib_new_from_file(handle)
    if msg is None:
        raise RuntimeError(f"spectra_ecmwf: no GRIB message in {sh_path}")
    try:
        return int(ec.codes_get(msg, "pentagonalResolutionParameterJ"))
    finally:
        ec.codes_release(msg)


def _verify_truncation(sh_dir: Path, param_dirs: list[str], *, truncation: int) -> None:
    """Fail loudly if any produced harmonics file is not at the requested T.

    Stage 2 (gptosp) and stage 3 (amplitudes) must agree on the truncation.
    A silent disagreement produces a wrong scientific number rather than a
    missing one, so this earns a hard stop rather than a warning.
    """
    checked = 0
    for pd in param_dirs:
        for sh_path in sorted((sh_dir / pd).glob("*.grb_sh")):
            achieved = _read_achieved_truncation(sh_path)
            if achieved != truncation:
                raise RuntimeError(
                    "spectra_ecmwf: gptosp produced the wrong truncation for "
                    f"{sh_path}: requested T{truncation}, file carries T{achieved}. "
                    "Stage 2 and stage 3 would disagree; refusing to continue."
                )
            checked += 1
    if not checked:
        raise RuntimeError(
            f"spectra_ecmwf: gptosp produced no harmonics under {sh_dir}. "
            "Check that staging wrote GRIBs and that gptosp.ser was on PATH."
        )
    LOG.info("spectra_ecmwf: verified T%d on %d harmonics files", truncation, checked)


def _run_gptosp(
    *, grb_dir: Path, sh_dir: Path, weather_states: list[str], truncation: int
) -> None:
    """Run gptosp.ser for all GRIBs; skip existing harmonics (resumable).

    The truncation is passed explicitly with -T so stage 2 and stage 3 use the
    same number by construction.  Without -T, gptosp derives its own truncation
    from the staged grid's latitude count, which on a pole-masked grid is
    neither the cubic truncation nor anything the amplitude stage knows about.
    -l is deliberately not passed: it only selects the derivation rule used when
    -T is absent, and both of its options are wrong for an octahedral grid.

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
        # Rendered from eval/config/toolchains.yaml so the repo and the
        # hand-written job scripts cannot drift apart, and so a missing
        # gptosp.ser is reported here rather than as an empty stage 2 later.
        render_module_block("gptosp"),
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
            f'  gptosp.ser -T {truncation} -g "$grb_file" -S "$sh_out"',
            'done',
        ]
    lines += ['rm "$SHORT_TMP/g" "$SHORT_TMP/s"', 'rmdir "$SHORT_TMP"']

    subprocess.run(
        ["bash", "--login", "-c", "\n".join(lines)],
        check=True,
    )

    _verify_truncation(sh_dir, param_dirs, truncation=truncation)


def _compute_amplitudes(
    *,
    sh_dir: Path,
    amp_dir: Path,
    weather_states: str,
    summary_path: Path,
    truncation: int,
    reference_spectra_dir: str = "",
) -> None:
    venv_activate = Path(sys.prefix) / "bin" / "activate"
    script = "\n".join([
        "set -euo pipefail",
        # No module block: amplitudes are computed from the GRIB coefficients
        # with the venv's eccodes. This stage used to load metview, which cost a
        # 900 second startup timeout and a shared-scratch TMPDIR workaround, and
        # left the result at the mercy of whichever metview version resolved.
        f'source "{venv_activate}"',
        f'python "{_HERE / "_amplitude_computer.py"}"'
        f' --spectral-harmonics-dir "{sh_dir}"'
        f' --out-dir "{amp_dir}"'
        f' --weather-states "{weather_states}"'
        f' --truncation "{truncation}"'
        + (f' --reference-spectra-dir "{reference_spectra_dir}"' if reference_spectra_dir else "")
        + f' --summary-path "{summary_path}"',
    ])
    subprocess.run(["bash", "-c", script], check=True)
