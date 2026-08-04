"""Comparison-contract checks for tropical-cyclone distribution plots."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Mapping
import re

from eval._backends.tc.data_types import BoundingBox

_O96_O320_ANALYSIS = "OPER_O320_0001"
_CONTRACT_FIELDS = (
    "geographic_box",
    "support_mode",
    "regrid_resolution_degrees",
    "ensemble_members",
    "lead_times_hours",
    "start_dates",
    "valid_dates",
    "analysis_reference",
)
_PREDICTION_NAME = re.compile(r"predictions_(\d{8})_step(\d{3})\.nc$")
_MEMBER_DIMENSIONS = ("ensemble_member", "member", "realization")


def _parse_init_date(value: object, path: Path) -> datetime:
    if value in (None, ""):
        raise ValueError(f"TC comparison contract: {path} is missing the 'init_date' attribute")
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"TC comparison contract: {path} has invalid init_date={value!r}"
        ) from exc


def _prediction_metadata(path: Path) -> tuple[datetime, int, int]:
    import xarray as xr

    match = _PREDICTION_NAME.match(path.name)
    if match is None:
        raise ValueError(
            f"TC comparison contract: prediction filename must match "
            f"predictions_YYYYMMDD_stepNNN.nc, got {path.name!r}"
        )
    with xr.open_dataset(path) as ds:
        init = _parse_init_date(ds.attrs.get("init_date"), path)
        try:
            lead = int(ds.attrs["lead_step_hours"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"TC comparison contract: {path} is missing a valid 'lead_step_hours' attribute"
            ) from exc
        member_dims = [dim for dim in ds["y_pred"].dims if dim in _MEMBER_DIMENSIONS]
        if len(member_dims) > 1:
            raise ValueError(
                f"TC comparison contract: {path} has ambiguous ensemble dimensions {member_dims}"
            )
        members = int(ds.sizes[member_dims[0]]) if member_dims else 1

    filename_date, filename_step = match.groups()
    if init.strftime("%Y%m%d") != filename_date:
        raise ValueError(
            f"TC comparison contract: {path} filename start date {filename_date} does not match "
            f"init_date {init.strftime('%Y%m%d')}"
        )
    if lead != int(filename_step):
        raise ValueError(
            f"TC comparison contract: {path} filename lead {filename_step} does not match "
            f"lead_step_hours {lead}"
        )
    return init, lead, members


def build_prediction_contract(
    *,
    prediction_files: list[Path],
    bbox: BoundingBox,
    support_mode: str,
    regrid_resolution: float,
    analysis_reference: str,
) -> dict:
    """Build a comparison contract from the metadata of the exact plotted files."""
    if not prediction_files:
        raise ValueError("TC comparison contract: no prediction files were selected")

    starts: set[str] = set()
    leads: set[int] = set()
    valid_dates: set[str] = set()
    member_counts: set[int] = set()
    for path in prediction_files:
        init, lead, members = _prediction_metadata(Path(path))
        starts.add(init.date().isoformat())
        leads.add(lead)
        valid_dates.add((init + timedelta(hours=lead)).date().isoformat())
        member_counts.add(members)

    if len(member_counts) != 1:
        raise ValueError(
            "TC comparison contract: prediction files have different ensemble-member counts: "
            f"{sorted(member_counts)}"
        )

    return {
        "geographic_box": {
            "north": float(bbox.north),
            "south": float(bbox.south),
            "east": float(bbox.east),
            "west": float(bbox.west),
        },
        "support_mode": str(support_mode),
        "regrid_resolution_degrees": float(regrid_resolution),
        "ensemble_members": member_counts.pop(),
        "lead_times_hours": sorted(leads),
        "start_dates": sorted(starts),
        "valid_dates": sorted(valid_dates),
        "analysis_reference": str(analysis_reference),
    }


def validate_curve_support_contract(curves: Mapping[str, object], declared_mode: str) -> None:
    """Reject TC stats when loaded curves do not share one support geometry."""
    if declared_mode not in {"native", "regridded"}:
        raise ValueError(
            "TC curve support contract requires one concrete mode per comparison; "
            f"got {declared_mode!r}"
        )
    if not curves:
        raise ValueError("TC curve support contract cannot validate an empty curve set")

    missing = []
    mode_mismatches = {}
    signatures = {}
    for name, curve in curves.items():
        curve_mode = getattr(curve, "support_mode", "unknown")
        curve_signature = getattr(curve, "support_signature", "unknown")
        if curve_mode == "unknown" or curve_signature == "unknown":
            missing.append(name)
        if curve_mode != declared_mode:
            mode_mismatches[name] = curve_mode
        signatures.setdefault(curve_signature, []).append(name)

    if missing:
        raise ValueError(
            "TC curve support contract is missing loader metadata for: "
            + ", ".join(sorted(missing))
        )
    if mode_mismatches:
        raise ValueError(
            f"TC curve support contract declared {declared_mode!r}, but found "
            f"other modes: {mode_mismatches}"
        )
    if len(signatures) != 1:
        summary = {signature: sorted(names) for signature, names in signatures.items()}
        raise ValueError(
            "TC curve support contract found multiple spatial supports; "
            f"all curves must use one geometry: {summary}"
        )


def require_lane_analysis_reference(lane_name: str, analysis_reference: str | None) -> None:
    """Apply the lane-specific analysis-reference invariant."""
    if lane_name == "o96_o320" and analysis_reference != _O96_O320_ANALYSIS:
        raise ValueError(
            "o96_o320 TC comparisons must use OPER_O320_0001 (OPER O320) as the "
            f"analysis reference; got {analysis_reference!r}"
        )


def validate_comparison_contracts(
    contracts: Mapping[str, Mapping[str, object]],
    *,
    lane_name: str | None = None,
) -> None:
    """Fail with every field mismatch before a comparison plot is produced."""
    if not contracts:
        raise ValueError("TC comparison contract: no experiments were supplied for comparison")

    items = list(contracts.items())
    for name, contract in items:
        missing = [field for field in _CONTRACT_FIELDS if field not in contract]
        if missing:
            raise ValueError(
                f"TC comparison contract for {name!r} is missing required fields: {missing}"
            )
        if lane_name:
            require_lane_analysis_reference(lane_name, str(contract["analysis_reference"]))

    reference_name, reference = items[0]
    differences: list[str] = []
    for name, candidate in items[1:]:
        for field in _CONTRACT_FIELDS:
            if candidate[field] != reference[field]:
                differences.append(
                    f"- {field}: {name}={candidate[field]!r}; "
                    f"{reference_name}={reference[field]!r}"
                )

    if differences:
        raise ValueError(
            "TC comparison contract mismatch; refusing to generate a misleading plot:\n"
            + "\n".join(differences)
        )
