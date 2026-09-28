"""Turn the score curves that quaver plots into the tidy table of the shared figure.

The plot phase (``plotter.py``) still runs the patched quaver plot scripts, so the numbers
come from exactly the same FDB queries, fair-mean handling and ensemble-size scalings as
before. Quaver is asked to store the data of every panel in a JSON file (its ``storage``
option of ``document``); this module reads that file. Only the drawing changes: the figure is
made by ``eval.plotting.probabilistic.plot_probabilistic_scores``.

Each quaver panel carries one curve per legend entry. A curve is classified by what quaver
recorded about it:

* ``model``      the experiment (its expver equals ``params["expver"]`` in class ``rd``);
* ``input``      the coarse driving ensemble (its legend contains "input", set by the plotter);
* ``reference``  everything else (the self-computed operational ENFO on the output grid, or an
                 operational curve from the template).

Quaver reports geopotential in metres (its CRPS of z500 is a few metres), so ``z`` values are
tagged ``native_unit="gpm"`` and shown in dam.
"""
from __future__ import annotations

import json
from pathlib import Path

_SCORE_TO_METRIC = {"fcrps": "fcrps", "crps": "crps", "spread": "spread", "rmsef": "rmse_ens_mean"}
# Units in which quaver stores each parameter's scores, where they differ from the framework's
# native units. Quaver verification of mean sea level pressure is reported in hPa.
_QUAVER_NATIVE_UNITS = {"z": "gpm", "msl": "hPa"}


def _first(value):
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _variable(retriever: dict) -> str:
    param = str(_first(retriever.get("parameter")))
    level = _first(retriever.get("level"))
    if retriever.get("levtype") == "pl" and level not in (None, "", "None"):
        return f"{param}_{int(float(level))}"
    return param


def _native_unit(retriever: dict) -> str | None:
    return _QUAVER_NATIVE_UNITS.get(str(_first(retriever.get("parameter"))))


def _labels(params: dict, input_params: dict | None, ref_params: dict | None):
    model = f"model {params['expver']}"
    inp = None
    if input_params:
        inp = f"input ({str(input_params.get('stream', 'enfo')).upper()} {input_params['grid']})"
    ref = None
    if ref_params and ref_params.get("label"):
        ref = f"reference ({str(ref_params['label']).replace('enfo', 'ENFO').replace('eefo', 'EEFO')})"
    return model, inp, ref


def _role(row: dict, params: dict) -> str:
    rt = row["retriever"]
    if str(_first(rt.get("expver"))) == str(params["expver"]) and \
            str(_first(rt.get("class", rt.get("class_")))) == str(params.get("class_", "rd")):
        return "model"
    if "input" in str(row.get("legend", "")).lower():
        return "input"
    return "reference"


def dump_to_curves(dump_paths, params: dict, input_params: dict | None = None,
                   ref_params: dict | None = None) -> list[dict]:
    """Tidy rows (see ``eval.plotting.probabilistic``) from quaver storage dumps."""
    model_label, input_label, ref_label = _labels(params, input_params, ref_params)
    rows: list[dict] = []
    seen: set[tuple] = set()
    for path in dump_paths:
        for table in json.loads(Path(path).read_text()):
            leads = [float(x) * 24.0 for x in table["labels"]]  # quaver labels are lead days
            for row in table["data"]:
                rt = row["retriever"]
                metric = _SCORE_TO_METRIC.get(str(rt.get("score")))
                if metric is None or row.get("scores") is None:
                    continue  # per-member standard-deviation panels are not ensemble scores
                role = _role(row, params)
                label = {"model": model_label, "input": input_label or "input",
                         "reference": ref_label or str(row.get("legend", "reference")).strip()}[role]
                variable = _variable(rt)
                domain = str(rt.get("domain_name"))
                key = (metric, variable, domain, role, label)
                if key in seen:
                    continue
                seen.add(key)
                unit = _native_unit(rt)
                for lead, value in zip(leads, row["scores"]):
                    if value is None:
                        continue
                    rows.append({
                        "metric": metric, "variable": variable, "domain": domain,
                        "lead_h": lead, "series_role": role, "series_label": label,
                        "value": float(value), "native_unit": unit,
                    })
    return rows
