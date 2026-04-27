#!/usr/bin/env python3
"""Validate an overfit test: predictions sanity + spectra quality.

Usage (single variable):
    python validate_overfit.py \
        --predictions-nc /path/to/predictions.nc \
        --spectra-json /path/to/spectra_summary.json \
        --variable 2t \
        [--full-field-threshold 0.05] \
        [--residual-threshold 0.10] \
        [--rmse-threshold 2.0] \
        [--temp-min 220] [--temp-max 330]

Usage (multi-variable):
    python validate_overfit.py \
        --predictions-nc /path/to/predictions.nc \
        --spectra-json /path/to/spectra_summary.json \
        --variables 2t,tp \
        [--full-field-threshold 0.05] \
        [--residual-threshold 0.10] \
        [--rmse-threshold 2.0] \
        [--temp-min 180] [--temp-max 340]

Exits 0 on PASS, 1 on FAIL.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def check_predictions(nc_path: str, variable: str, temp_min: float, temp_max: float, rmse_threshold: float) -> dict:
    """Basic sanity checks on predictions.nc."""
    import xarray as xr

    ds = xr.open_dataset(nc_path)
    ws_list = list(ds.coords.get("weather_state", ds.get("weather_state", [])).values)

    if variable not in ws_list:
        return {"pass": False, "reason": f"Variable '{variable}' not found in weather_states: {ws_list}"}

    var_idx = ws_list.index(variable)

    y_pred = ds["y_pred"].values  # [sample, member, grid, vars]
    y_truth = ds["y"].values

    pred_var = y_pred[..., var_idx]
    truth_var = y_truth[..., var_idx]

    nan_frac = np.isnan(pred_var).sum() / pred_var.size
    inf_frac = np.isinf(pred_var).sum() / pred_var.size
    vmin, vmax = float(np.nanmin(pred_var)), float(np.nanmax(pred_var))
    vmean = float(np.nanmean(pred_var))

    rmse = float(np.sqrt(np.nanmean((pred_var - truth_var) ** 2)))
    corr = float(np.corrcoef(pred_var.ravel(), truth_var.ravel())[0, 1])

    results = {
        "variable": variable,
        "nan_fraction": nan_frac,
        "inf_fraction": inf_frac,
        "pred_min": vmin,
        "pred_max": vmax,
        "pred_mean": vmean,
        "rmse": rmse,
        "spatial_correlation": corr,
        "checks": {},
    }

    results["checks"]["no_nan"] = nan_frac == 0.0
    results["checks"]["no_inf"] = inf_frac == 0.0
    results["checks"]["range_ok"] = vmin >= temp_min and vmax <= temp_max
    results["checks"]["rmse_ok"] = rmse < rmse_threshold
    results["checks"]["corr_ok"] = corr > 0.95

    results["pass"] = all(results["checks"].values())
    return results


def check_spectra(spectra_json_path: str, variable: str, ff_threshold: float, res_threshold: float) -> dict:
    """Check spectra summary for a variable."""
    with open(spectra_json_path) as f:
        summary = json.load(f)

    ws_data = summary.get("weather_states", {})
    if variable not in ws_data:
        return {"pass": False, "reason": f"Variable '{variable}' not in spectra summary. Available: {list(ws_data.keys())}"}

    var_data = ws_data[variable]
    scopes = var_data.get("scopes", {})

    ff_score = None
    res_score = None

    if "full_field" in scopes:
        ff_score = scopes["full_field"].get("relative_l2_mean_curve")
    if "residual" in scopes:
        res_score = scopes["residual"].get("relative_l2_mean_curve")

    results = {
        "variable": variable,
        "full_field_relative_l2": ff_score,
        "residual_relative_l2": res_score,
        "checks": {},
    }

    if ff_score is not None and not (isinstance(ff_score, float) and np.isnan(ff_score)):
        results["checks"]["full_field_ok"] = ff_score < ff_threshold
    else:
        results["checks"]["full_field_ok"] = False
        results["full_field_note"] = "NaN or missing — likely lmax below score_wavenumber_min_exclusive"

    if res_score is not None and not (isinstance(res_score, float) and np.isnan(res_score)):
        results["checks"]["residual_ok"] = res_score < res_threshold
    else:
        results["checks"]["residual_ok"] = False
        results["residual_note"] = "NaN or missing"

    results["pass"] = all(results["checks"].values())
    return results


def main():
    p = argparse.ArgumentParser(description="Validate overfit test predictions + spectra.")
    p.add_argument("--predictions-nc", required=True, help="Path to predictions.nc")
    p.add_argument("--spectra-json", required=True, help="Path to spectra_summary.json")
    # Accept either --variable (single) or --variables (comma-separated list)
    p.add_argument("--variable", default=None, help="Single variable to validate (legacy)")
    p.add_argument("--variables", default=None, help="Comma-separated list of variables (e.g. 2t,tp)")
    p.add_argument("--full-field-threshold", type=float, default=0.05)
    p.add_argument("--residual-threshold", type=float, default=0.10)
    p.add_argument("--rmse-threshold", type=float, default=2.0)
    p.add_argument("--temp-min", type=float, default=220.0)
    p.add_argument("--temp-max", type=float, default=330.0)
    args = p.parse_args()

    # Resolve variable list: --variables takes precedence, fall back to --variable
    if args.variables:
        var_list = [v.strip() for v in args.variables.split(",")]
    elif args.variable:
        var_list = [args.variable]
    else:
        var_list = ["2t"]

    print("=" * 60)
    print("OVERFIT VALIDATION TEST")
    print(f"Variables: {', '.join(var_list)}")
    print("=" * 60)

    all_pass = True

    for variable in var_list:
        # --- Predictions sanity ---
        print(f"\n--- Predictions sanity ({variable}) ---")
        pred_result = check_predictions(
            args.predictions_nc, variable,
            args.temp_min, args.temp_max, args.rmse_threshold,
        )
        if "reason" in pred_result:
            print(f"  FAIL: {pred_result['reason']}")
        else:
            print(f"  NaN fraction:  {pred_result['nan_fraction']:.6f}  {'✓' if pred_result['checks']['no_nan'] else '✗'}")
            print(f"  Inf fraction:  {pred_result['inf_fraction']:.6f}  {'✓' if pred_result['checks']['no_inf'] else '✗'}")
            print(f"  Range:         [{pred_result['pred_min']:.2f}, {pred_result['pred_max']:.2f}]  {'✓' if pred_result['checks']['range_ok'] else '✗'} (expected [{args.temp_min}, {args.temp_max}])")
            print(f"  RMSE vs truth: {pred_result['rmse']:.4f} K  {'✓' if pred_result['checks']['rmse_ok'] else '✗'} (threshold {args.rmse_threshold})")
            print(f"  Spatial corr:  {pred_result['spatial_correlation']:.6f}  {'✓' if pred_result['checks']['corr_ok'] else '✗'} (threshold 0.95)")
        print(f"  Predictions: {'PASS' if pred_result['pass'] else 'FAIL'}")

        # --- Spectra quality ---
        print(f"\n--- Spectra quality ({variable}) ---")
        spec_result = check_spectra(
            args.spectra_json, variable,
            args.full_field_threshold, args.residual_threshold,
        )
        if "reason" in spec_result:
            print(f"  FAIL: {spec_result['reason']}")
        else:
            ff = spec_result["full_field_relative_l2"]
            res = spec_result["residual_relative_l2"]
            ff_str = f"{ff:.6f}" if ff is not None else "N/A"
            res_str = f"{res:.6f}" if res is not None else "N/A"
            print(f"  Full-field relative_l2: {ff_str}  {'✓' if spec_result['checks']['full_field_ok'] else '✗'} (threshold {args.full_field_threshold})")
            if "full_field_note" in spec_result:
                print(f"    Note: {spec_result['full_field_note']}")
            print(f"  Residual relative_l2:  {res_str}  {'✓' if spec_result['checks']['residual_ok'] else '✗'} (threshold {args.residual_threshold})")
            if "residual_note" in spec_result:
                print(f"    Note: {spec_result['residual_note']}")
        print(f"  Spectra: {'PASS' if spec_result['pass'] else 'FAIL'}")

        if not (pred_result["pass"] and spec_result["pass"]):
            all_pass = False

    # --- Overall verdict ---
    print("\n" + "=" * 60)
    print(f"VERDICT: {'PASS' if all_pass else 'FAIL'} ({len(var_list)} variable(s): {', '.join(var_list)})")
    print("=" * 60)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
