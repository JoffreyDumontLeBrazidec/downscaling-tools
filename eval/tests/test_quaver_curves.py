"""The quaver plot phase draws the shared probabilistic figure from quaver's stored curves."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

from eval.evaluators.quaver import plotter  # noqa: E402
from eval.evaluators.quaver.curves import dump_to_curves  # noqa: E402

PARAMS = {"expver": "j9f3", "class_": "rd", "first_reference_date": 20250901,
          "last_reference_date": 20250930}
INPUT = {"stream": "eefo", "grid": "O320"}
REF = {"label": "enfo O1280"}


def _row(legend, expver, cls, score, param, scores, level=None, levtype="sfc", stream="enfo"):
    return {
        "retriever": {"expver": [expver], "class": [cls], "score": score, "parameter": param,
                      "level": level, "levtype": levtype, "domain_name": "n.hem", "stream": [stream]},
        "legend": legend, "scores": scores,
    }


def _dump(tmp_path):
    table_sfc = {"labels": [1.0, 2.0], "data": [
        _row("ML j9f3", "j9f3", "rd", "fcrps", "2t", [1.0, 2.0]),
        _row("eefo O320 input ", "0001", "od", "fcrps", "2t", [1.5, 2.5], stream="eefo"),
        _row("enfo O1280 ", "0001", "od", "fcrps", "2t", [0.9, 1.9]),
        _row("ML j9f3 mem 1", "j9f3", "rd", "sdaf", "2t", [3.0, 3.0]),
    ]}
    table_pl = {"labels": [1.0, 2.0], "data": [
        _row("ML j9f3", "j9f3", "rd", "rmsef", "z", [3.0, 6.0], level=500, levtype="pl"),
    ]}
    path = tmp_path / "dump.json"
    path.write_text(json.dumps([table_sfc, table_pl]))
    return path


def test_dump_to_curves_classifies_and_converts_lead_days(tmp_path):
    rows = dump_to_curves([_dump(tmp_path)], PARAMS, INPUT, REF)
    roles = {(r["variable"], r["metric"], r["series_role"]) for r in rows}
    assert ("2t", "fcrps", "model") in roles
    assert ("2t", "fcrps", "input") in roles
    assert ("2t", "fcrps", "reference") in roles
    assert ("z_500", "rmse_ens_mean", "model") in roles
    assert not any(r["metric"] == "sdaf" for r in rows)
    assert sorted({r["lead_h"] for r in rows}) == [24.0, 48.0]
    z = [r for r in rows if r["variable"] == "z_500"]
    assert z and all(r["native_unit"] == "gpm" for r in z)
    labels = {r["series_role"]: r["series_label"] for r in rows if r["variable"] == "2t"}
    assert labels == {"model": "model j9f3", "input": "input (EEFO O320)",
                      "reference": "reference (ENFO O1280)"}


def test_draw_writes_figure_and_curve_table(tmp_path):
    out = plotter._draw(tmp_path, PARAMS, INPUT, REF, [_dump(tmp_path)])
    assert out == tmp_path
    assert (tmp_path / "quaver_j9f3_probabilistic_scores.pdf").exists()
    assert (tmp_path / "quaver_j9f3_curves.csv").exists()
    assert list((tmp_path / "quaver_j9f3_probabilistic_scores_pages").glob("*.png"))


def test_with_storage_injects_one_storage_key():
    src = "document(\n        plots=x,\n        data=documentdata(),\n        orientation='landscape',\n    )\n"
    patched = plotter._with_storage(src, __import__("pathlib").Path("/tmp/d.json"))
    assert patched.count("storage=") == 1 and "filestorer:file=/tmp/d.json,format=json" in patched
