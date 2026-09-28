"""Small pure helpers used by the diagnostic figures (storm_maps, spectra_coherence, displacement)."""
from eval._backends.storm_maps.render import _column_titles, _ogrid
from eval.evaluators.displacement.plotter import _box_text, _run_text
from eval.evaluators.spectra_coherence.plot_stratified import _band_label


def test_ogrid_names_octahedral_grids():
    assert _ogrid(421120) == "O320"
    assert _ogrid(6599680) == "O1280"
    assert _ogrid(12345) is None


def test_column_titles_name_the_grids():
    t = _column_titles({"hres": "O1280", "lres": "O320"})
    assert t["truth"] == "Truth (O1280)"
    assert t["model"] == "Model (O1280)"
    assert t["input"] == "Input (O320 interpolated to O1280)"
    assert _column_titles(None)["input"] == "Input (interpolated to the target grid)"


def test_band_label_uses_the_band_table():
    bands = [{"name": "fine", "lo": 300, "hi": 500}, {"name": "near_grid", "lo": 700, "hi": 100000}]
    assert _band_label("fine", bands) == "fine (ℓ 300–500)"
    assert _band_label("near_grid", bands) == "near grid (ℓ ≥ 700)"
    assert _band_label("meso", bands) == "meso"


def test_run_and_box_text_are_readable():
    assert _run_text("predictions") == ""
    assert _run_text("sept_ft400k_ctrl_ja6y") == "sept_ft400k_ctrl_ja6y"
    assert _box_text("humberto_atlantic", [22.0, 42.0, -72.0, -48.0]) == \
        "humberto atlantic (22 to 42°N, -72 to -48°E)"
