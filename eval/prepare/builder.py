"""Truth-aware bundle builder.

Reads the lane config ``prepare:`` section, which specifies per-lane GRIB
filename templates, and calls ``manual_inference.prediction.predict build-bundle``
for each (date, step, member) combination.

Lane config schema (under ``prepare:``)::

    bundle_filename_tpl: "prefix_date{date}_mem{member:02d}_step{step:03d}h_input_bundle.nc"
    args:
      lres_sfc_grib: "{source_grib_root}/prefix_date{date}_sfc.grib"
      lres_pl_grib:  "{source_grib_root}/prefix_date{date}_pl.grib"
      hres_grib:     "{source_grib_root}/prefix_hres_date{date}_sfc.grib"
      target_sfc_grib: "{source_grib_root}/prefix_target_date{date}_sfc.grib"
      # optional channel overrides (used by o1280_o2560)
      lres_sfc_channels: "10u,10v,2t,msl"
      lres_pl_channels:  "NONE"
    optional_sidecars:
      - file: "{source_grib_root}/prefix_date{date}_tp.grib"
        arg:  lres_sfc_extra_grib

Template variables available in all string values:
  {date}               YYYYMMDD
  {step}               integer forecast step (hours)
  {member}             integer member index
  {source_grib_root}   from --source-grib-root CLI arg
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)
TRUTH_MARKERS = {"yes", "true", "1"}


def build_bundles(
    lane_config: dict,
    bundle_dir: Path,
    source_grib_root: str,
    *,
    dates: list[str] | None = None,
    steps: list[int] | None = None,
    members: list[int] | None = None,
    bundle_pairs: list[str] | None = None,
    verification_path: Path | None = None,
) -> Path:
    """Build truth-aware bundles for all (date, step, member) combinations.

    Resumable: skips bundles whose output NC already exists.
    Verifies all expected bundles after building and writes a verification JSON.
    Returns ``bundle_dir``.
    """
    prepare_cfg = lane_config.get("prepare")
    if not prepare_cfg:
        raise ValueError("Lane config has no 'prepare:' section.")

    predict_cfg = lane_config.get("predict", {})
    dates = dates or predict_cfg.get("dates", [])
    steps = steps or [int(s) for s in predict_cfg.get("steps", [])]
    members = members or [int(m) for m in predict_cfg.get("members", [])]
    bundle_pairs = bundle_pairs or []

    bundle_dir.mkdir(parents=True, exist_ok=True)

    source_grib_root_original = str(source_grib_root)
    source_grib_root_resolved = _resolve_existing_path(
        source_grib_root,
        label="source_grib_root",
    )

    ctx_base = {
        "source_grib_root": str(source_grib_root_resolved),
    }

    filename_tpl: str = prepare_cfg["bundle_filename_tpl"]
    fixed_args: dict[str, str] = prepare_cfg.get("args", {})
    sidecars: list[dict] = prepare_cfg.get("optional_sidecars", [])

    combos = _expand_combos(dates, steps, members, bundle_pairs)
    required_input_paths = _resolve_required_input_paths(
        fixed_args=fixed_args,
        combos=combos,
        ctx_base=ctx_base,
        source_grib_root_original=source_grib_root_original,
    )
    LOG.info("bundle-build: %d bundle(s) → %s", len(combos), bundle_dir)

    for i, (date, step, member) in enumerate(combos, 1):
        ctx = {**ctx_base, "date": date, "step": step, "member": member}
        bundle_out = bundle_dir / filename_tpl.format(**ctx)

        if bundle_out.exists():
            try:
                validate_truth_bundle(bundle_out)
            except Exception as exc:
                LOG.warning(
                    "[%d/%d] rebuild invalid existing bundle %s: %s",
                    i, len(combos), bundle_out.name, exc,
                )
                bundle_out.unlink(missing_ok=True)
            else:
                LOG.info("[%d/%d] skip (valid): %s", i, len(combos), bundle_out.name)
                continue

        LOG.info("[%d/%d] building: %s", i, len(combos), bundle_out.name)
        cmd = [sys.executable, "-m", "manual_inference.prediction.predict", "build-bundle"]

        for arg_name, arg_val in fixed_args.items():
            rendered = arg_val.format(**ctx)
            if _is_required_grib_arg(arg_name):
                rendered = str(required_input_paths[(arg_name, rendered)])
            cmd += [f"--{arg_name.replace('_', '-')}", rendered]
        for sidecar in sidecars:
            sidecar_path = _resolve_optional_path(sidecar["file"].format(**ctx))
            if sidecar_path is not None:
                cmd += [f"--{sidecar['arg'].replace('_', '-')}", str(sidecar_path)]
        tmp_out = _tmp_bundle_path(bundle_out)
        tmp_out.unlink(missing_ok=True)
        cmd += ["--step-hours", str(step), "--member", str(member), "--out", str(tmp_out)]
        try:
            subprocess.run(cmd, check=True)
            validate_truth_bundle(tmp_out)
            tmp_out.replace(bundle_out)
        except Exception:
            tmp_out.unlink(missing_ok=True)
            raise

    verify_bundles(
        lane_config,
        bundle_dir=bundle_dir,
        dates=dates,
        steps=steps,
        members=members,
        bundle_pairs=bundle_pairs,
        verification_path=verification_path or bundle_dir.parent / "bundle_build_verification.json",
        source_grib_root_original=source_grib_root_original,
        source_grib_root_resolved=str(source_grib_root_resolved),
    )
    return bundle_dir


def validate_truth_bundle(path: Path) -> list[str]:
    """Open a bundle header and require target_hres_* plus the truth marker."""
    import xarray as xr

    try:
        with xr.open_dataset(path) as ds:
            target_vars = sorted(v for v in ds.variables if v.startswith("target_hres_"))
            if not target_vars:
                raise RuntimeError(f"Bundle missing target_hres_* variables: {path}")
            marker = str(ds.attrs.get("has_target_hres_fields", "")).strip().lower()
            if marker not in TRUTH_MARKERS:
                raise RuntimeError(
                    f"Bundle has invalid has_target_hres_fields={marker!r}: {path}"
                )
            return target_vars
    except Exception as exc:
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(f"Cannot open bundle as truth-aware NetCDF: {path}: {exc}") from exc


def verify_bundles(
    lane_config: dict,
    bundle_dir: Path,
    *,
    dates: list[str] | None = None,
    steps: list[int] | None = None,
    members: list[int] | None = None,
    bundle_pairs: list[str] | None = None,
    verification_path: Path | None = None,
    source_grib_root_original: str | None = None,
    source_grib_root_resolved: str | None = None,
) -> dict[str, Any]:
    """Verify every expected truth-aware bundle and optionally write JSON."""
    prepare_cfg = lane_config.get("prepare")
    if not prepare_cfg:
        raise ValueError("Lane config has no 'prepare:' section.")

    predict_cfg = lane_config.get("predict", {})
    dates = dates or predict_cfg.get("dates", [])
    steps = steps or [int(s) for s in predict_cfg.get("steps", [])]
    members = members or [int(m) for m in predict_cfg.get("members", [])]
    combos = _expand_combos(dates, steps, members, bundle_pairs or [])

    payload = _verify(
        bundle_dir=bundle_dir,
        filename_tpl=prepare_cfg["bundle_filename_tpl"],
        combos=combos,
        ctx_base={},
        source_grib_root_original=source_grib_root_original,
        source_grib_root_resolved=source_grib_root_resolved,
    )
    if verification_path is not None:
        verification_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        LOG.info("Bundle verification passed (%d bundles): %s", len(combos), verification_path)
    else:
        LOG.info("Bundle verification passed (%d bundles): %s", len(combos), bundle_dir)
    return payload


def _expand_combos(
    dates: list[str],
    steps: list[int],
    members: list[int],
    bundle_pairs: list[str],
) -> list[tuple[str, int, int]]:
    if bundle_pairs:
        combos: list[tuple[str, int, int]] = []
        for token in bundle_pairs:
            date, step_s = token.split(":")
            for member in members:
                combos.append((date.strip(), int(step_s), member))
        return combos
    return [
        (date, step, member)
        for date in dates
        for step in steps
        for member in members
    ]


def _is_required_grib_arg(arg_name: str) -> bool:
    return arg_name.endswith("_grib")


def _resolve_existing_path(path: str | Path, *, label: str) -> Path:
    try:
        return Path(path).expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing {label}: {path}") from exc
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(f"Cannot resolve {label}: {path}: {exc}") from exc


def _resolve_optional_path(path: str | Path) -> Path | None:
    try:
        return Path(path).expanduser().resolve(strict=True)
    except FileNotFoundError:
        return None
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(f"Cannot resolve optional input path: {path}: {exc}") from exc


def _resolve_required_input_paths(
    *,
    fixed_args: dict[str, str],
    combos: list[tuple[str, int, int]],
    ctx_base: dict,
    source_grib_root_original: str,
) -> dict[tuple[str, str], Path]:
    resolved: dict[tuple[str, str], Path] = {}
    for date, step, member in combos:
        ctx = {**ctx_base, "date": date, "step": step, "member": member}
        for arg_name, arg_val in fixed_args.items():
            if not _is_required_grib_arg(arg_name):
                continue
            rendered = arg_val.format(**ctx)
            key = (arg_name, rendered)
            if key not in resolved:
                resolved[key] = _resolve_existing_path(
                    rendered,
                    label=(
                        f"{arg_name} rendered from source_grib_root="
                        f"{source_grib_root_original!r}"
                    ),
                )
    return resolved


def _tmp_bundle_path(bundle_out: Path) -> Path:
    return bundle_out.with_name(f".{bundle_out.name}.tmp-{os.getpid()}")


def _verify(
    *,
    bundle_dir: Path,
    filename_tpl: str,
    combos: list[tuple[str, int, int]],
    ctx_base: dict,
    source_grib_root_original: str | None = None,
    source_grib_root_resolved: str | None = None,
) -> dict[str, Any]:
    missing: list[str] = []
    sample_target_vars: list[str] = []

    for date, step, member in combos:
        ctx = {**ctx_base, "date": date, "step": step, "member": member}
        path = bundle_dir / filename_tpl.format(**ctx)
        if not path.exists():
            missing.append(str(path))
            continue
        target_vars = validate_truth_bundle(path)
        if not sample_target_vars:
            sample_target_vars = target_vars[:10]

    if missing:
        raise RuntimeError(f"Missing {len(missing)} bundle(s) after build: {missing[:5]}")

    return {
        "bundle_dir": str(bundle_dir),
        "expected_bundle_count": len(combos),
        "sample_target_vars": sample_target_vars,
        "truth_marker": "yes",
        "source_grib_root_original": source_grib_root_original,
        "source_grib_root_resolved": source_grib_root_resolved,
    }
