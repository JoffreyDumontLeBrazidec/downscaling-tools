"""Tests for spectra curve naming and for the pairing it unblocks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from eval._backends.scoreboard.spectra import load_spectra_metrics
from eval._backends.spectra import naming


@pytest.mark.parametrize(
    ("field_dir", "weather_state"),
    [
        ("msl_sfc", "msl"),
        ("2t_sfc", "2t"),
        ("10u_sfc", "10u"),
        ("t_850", "t_850"),
        ("z_500", "z_500"),
    ],
)
def test_both_conventions_round_trip(field_dir: str, weather_state: str) -> None:
    """t_850 and z_500 contain an underscore, so the two forms must stay distinct."""
    canonical = naming.canonical_name(
        "ampl", date=20230826, step=120, field_dir=field_dir, token="1", member=1
    )
    short = naming.short_name(
        "ampl", date=20230826, step=120, weather_state=weather_state, member=1
    )

    parsed_canonical = naming.parse(canonical)
    parsed_short = naming.parse(short)

    assert parsed_canonical is not None and parsed_short is not None
    assert parsed_canonical["form"] == "canonical"
    assert parsed_canonical["token"] == "1"
    assert parsed_short["form"] == "short"
    assert parsed_short["token"] is None
    for parsed in (parsed_canonical, parsed_short):
        assert parsed["field_dir"] == field_dir
        assert parsed["weather_state"] == weather_state
        assert parsed["date"] == 20230826
        assert parsed["step"] == 120
        assert parsed["member"] == 1


@pytest.mark.parametrize(
    "name",
    [
        "ampl_20230826_120_bogus_n1.npy",
        "ampl_2023_120_msl_n1.npy",
        "notes.txt",
        "ampl_20230826_120_msl_sfc_a_b_n1.npy",
    ],
)
def test_parse_rejects_names_it_does_not_understand(name: str) -> None:
    assert naming.parse(name) is None


def test_find_prefers_the_canonical_name(tmp_path: Path) -> None:
    canonical = naming.canonical_name(
        "ampl", date=20230826, step=120, field_dir="msl_sfc", token="1", member=1
    )
    short = naming.short_name(
        "ampl", date=20230826, step=120, weather_state="msl", member=1
    )
    (tmp_path / canonical).touch()
    (tmp_path / short).touch()

    found = naming.find(
        tmp_path, "ampl", date=20230826, step=120, field_dir="msl_sfc", member=1
    )

    assert found is not None and found.name == canonical


def test_find_falls_back_to_the_short_name(tmp_path: Path) -> None:
    """The caches already on /perm use the short form and must keep working."""
    short = naming.short_name(
        "ampl", date=20230826, step=120, weather_state="msl", member=1
    )
    (tmp_path / short).touch()

    found = naming.find(
        tmp_path, "ampl", date=20230826, step=120, field_dir="msl_sfc", member=1
    )

    assert found is not None and found.name == short


def test_find_returns_none_when_neither_exists(tmp_path: Path) -> None:
    assert (
        naming.find(tmp_path, "ampl", date=20230826, step=120, field_dir="msl_sfc", member=1)
        is None
    )


# --- the regression this whole step exists for ------------------------------

FIELDS = [("10u", "10u_sfc"), ("10v", "10v_sfc"), ("2t", "2t_sfc"),
          ("msl", "msl_sfc"), ("t_850", "t_850"), ("z_500", "z_500")]


def _write_curves(root: Path, *, canonical: bool, scale: float) -> None:
    wvn = np.arange(1, 401, dtype=np.float64)
    for _field, field_dir in FIELDS:
        d = root / field_dir
        d.mkdir(parents=True, exist_ok=True)
        key = dict(date=20230826, step=120, field_dir=field_dir, token="1", member=1)
        if canonical:
            amp_name = naming.canonical_name("ampl", **key)
            wvn_name = naming.canonical_name("wvn", **key)
        else:
            amp_name = naming.short_name(
                "ampl", date=20230826, step=120,
                weather_state=naming.weather_state_for(field_dir), member=1,
            )
            wvn_name = naming.short_name(
                "wvn", date=20230826, step=120,
                weather_state=naming.weather_state_for(field_dir), member=1,
            )
        np.save(d / amp_name, np.full(400, scale, dtype=np.float64))
        np.save(d / wvn_name, wvn)


def test_short_named_candidate_pairs_with_canonical_reference(tmp_path: Path) -> None:
    """Before the shared helper, every lookup missed and the mean was None.

    The candidate uses the short convention (what _amplitude_computer.py wrote)
    and the reference uses the canonical one, which is the mixture that exists on
    disk today.
    """
    candidate = tmp_path / "run" / "spectra"
    reference = tmp_path / "ref" / "spectra"
    _write_curves(candidate, canonical=False, scale=2.0)
    _write_curves(reference, canonical=True, scale=1.0)
    (tmp_path / "run" / "staging_summary.json").write_text(
        json.dumps({"dates": [20230826], "steps_hours": [120], "ensemble_members": [1]}),
        encoding="utf-8",
    )

    metrics = load_spectra_metrics(tmp_path / "run", reference_root=reference)

    assert metrics["mean"] is not None
    assert metrics["coverage"] != "missing"
    assert all(metrics["missing_reference_pairs"][f] == 0 for f, _ in FIELDS)
    # candidate is uniformly twice the reference, so the relative L2 is 1.0
    for field, _ in FIELDS:
        assert metrics[field] == pytest.approx(1.0)
