from __future__ import annotations

import pytest

from eval.evaluators.region_plot.core.plotting.metadata import PlotMetadata


@pytest.fixture
def full_metadata() -> PlotMetadata:
    return PlotMetadata(
        region="alps_innsbruck",
        date="2023-08-26 00:00",
        init_date="2023-08-25 00:00",
        lead_hours=24,
        sample_position=3,
        ensemble_member=1,
    )


def test_plot_metadata_creation_uses_defaults() -> None:
    metadata = PlotMetadata(region="amazon_forest")

    assert metadata.region == "amazon_forest"
    assert metadata.date == ""
    assert metadata.init_date == ""
    assert metadata.lead_hours is None
    assert metadata.sample_position == 0
    assert metadata.ensemble_member == 0


def test_plot_metadata_to_title_with_defaults() -> None:
    metadata = PlotMetadata(region="amazon_forest")

    # Readable figure title (region name in words); raw keys such as sample_pos stay off figures.
    assert metadata.to_title() == "Amazon Forest"


def test_plot_metadata_to_title_with_full_metadata(full_metadata: PlotMetadata) -> None:
    title = full_metadata.to_title()

    assert title == "Alps Innsbruck — Init: 2023-08-25 00:00 — Valid: 2023-08-26 00:00 — T+24h"
    assert full_metadata.ensemble_member == 1
