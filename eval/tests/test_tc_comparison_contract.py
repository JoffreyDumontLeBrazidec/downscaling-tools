from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from eval._backends.tc.events import EVENTS
from eval.evaluators.tc.comparison_contract import (
    build_prediction_contract,
    validate_comparison_contracts,
)


def _write_prediction(path: Path, *, init_date: str, step: int, members: int = 10) -> None:
    ds = xr.Dataset(
        data_vars={
            "y_pred": (
                ("sample", "ensemble_member", "grid_point_hres", "weather_state"),
                np.zeros((1, members, 1, 3), dtype=np.float32),
            ),
        },
        coords={
            "sample": [0],
            "ensemble_member": list(range(1, members + 1)),
            "grid_point_hres": [0],
            "weather_state": ["msl", "10u", "10v"],
        },
        attrs={"init_date": init_date, "lead_step_hours": step},
    )
    ds.to_netcdf(path)


def test_build_prediction_contract_uses_file_metadata(tmp_path: Path):
    first = tmp_path / "predictions_20230826_step024.nc"
    second = tmp_path / "predictions_20230827_step048.nc"
    _write_prediction(first, init_date="2023-08-26T00:00:00", step=24)
    _write_prediction(second, init_date="2023-08-27T00:00:00", step=48)

    contract = build_prediction_contract(
        prediction_files=[first, second],
        bbox=EVENTS["idalia"].bbox,
        support_mode="regridded",
        regrid_resolution=0.25,
        analysis_reference="OPER_O320_0001",
    )

    assert contract == {
        "geographic_box": {"north": 40.0, "south": 10.0, "east": -80.0, "west": -100.0},
        "support_mode": "regridded",
        "regrid_resolution_degrees": 0.25,
        "ensemble_members": 10,
        "lead_times_hours": [24, 48],
        "start_dates": ["2023-08-26", "2023-08-27"],
        "valid_dates": ["2023-08-27", "2023-08-29"],
        "analysis_reference": "OPER_O320_0001",
    }


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("geographic_box", {"north": 41.0, "south": 10.0, "east": -80.0, "west": -100.0}),
        ("ensemble_members", 5),
        ("lead_times_hours", [24, 72]),
        ("start_dates", ["2023-08-28"]),
        ("valid_dates", ["2023-08-29"]),
        ("analysis_reference", "OPER_O96_0001"),
    ],
)
def test_validate_comparison_contracts_reports_each_mismatch(
    tmp_path: Path, field: str, replacement: object,
):
    prediction = tmp_path / "predictions_20230826_step024.nc"
    _write_prediction(prediction, init_date="2023-08-26T00:00:00", step=24)
    baseline = build_prediction_contract(
        prediction_files=[prediction],
        bbox=EVENTS["idalia"].bbox,
        support_mode="regridded",
        regrid_resolution=0.25,
        analysis_reference="OPER_O320_0001",
    )
    mismatched = deepcopy(baseline)
    mismatched[field] = replacement

    with pytest.raises(ValueError, match=field):
        validate_comparison_contracts({"reference": baseline, "candidate": mismatched})


def test_validate_comparison_contracts_requires_oper_o320_for_o96_o320(tmp_path: Path):
    prediction = tmp_path / "predictions_20230826_step024.nc"
    _write_prediction(prediction, init_date="2023-08-26T00:00:00", step=24)
    contract = build_prediction_contract(
        prediction_files=[prediction],
        bbox=EVENTS["idalia"].bbox,
        support_mode="regridded",
        regrid_resolution=0.25,
        analysis_reference="OPER_O96_0001",
    )

    with pytest.raises(ValueError, match="OPER_O320_0001"):
        validate_comparison_contracts({"candidate": contract}, lane_name="o96_o320")
