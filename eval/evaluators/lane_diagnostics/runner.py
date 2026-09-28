"""Build the o1280->o2560 lane's diagnostic figure bundle.

``run`` performs the measurements that need to touch prediction fields and
stores them as small JSON files.  ``plot`` turns those, together with the
artefacts named in the lane configuration, into one labelled bundle: every
figure as its own PDF, all of them concatenated, and a captions file.

This evaluator produces NO scoreboard row.  It is diagnostic: it explains an
already-scored result rather than predicting a new one.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np

from . import compute, figures as F, prepare as P
from .standing_set import build_standing_set

LOG = logging.getLogger(__name__)

MEASUREMENTS = "measurements"


def _cfg(eval_config: dict, key: str, default=None):
    value = eval_config.get(key, default)
    if value is None:
        raise KeyError(f"lane_diagnostics config is missing '{key}'")
    return value


# ---------------------------------------------------------------------------
# run: the reductions that have to open prediction files
# ---------------------------------------------------------------------------

def run(
    predictions_dir,
    lane_config: dict,
    eval_config: dict,
    *,
    output_dir=None,
    overwrite: bool = False,
    checkpoint: str | None = None,
    **kwargs,
) -> Path:
    predictions_dir = Path(predictions_dir).expanduser().resolve()
    output_dir = Path(output_dir) if output_dir else predictions_dir / "evaluators" / "lane_diagnostics"
    meas = output_dir / MEASUREMENTS
    meas.mkdir(parents=True, exist_ok=True)

    box = dict(eval_config.get("box", F.BOX))
    stages = set(eval_config.get("stages", ["sampler_peaks", "loss_budget", "pair_coherence", "box_wind"]))

    if "box_wind" in stages:
        path = meas / "box_wind.json"
        if overwrite or not path.exists():
            rows = compute.box_wind(predictions_dir, box)
            path.write_text(json.dumps(rows) + "\n")
            LOG.info("wrote %s (%d rows)", path, len(rows))

    if "loss_budget" in stages:
        path = meas / "loss_budget.json"
        if overwrite or not path.exists():
            lb = dict(eval_config.get("loss_budget", {}))
            out = compute.loss_budget(
                predictions_dir,
                lane_config.get("precip", {}),
                lb.get("thresholds_mm", [0.1, 1, 2, 5, 10, 20, 30, 50, 100, 200]),
                stride=int(lb.get("stride", 5)),
                member=int(lb.get("member", 0)),
            )
            path.write_text(json.dumps(out, indent=2) + "\n")
            LOG.info("wrote %s", path)

    if "pair_coherence" in stages:
        path = meas / "pair_coherence.json"
        if overwrite or not path.exists():
            # The two lanes must be read at the SAME lead times, or the shape of
            # the comparison is set by which leads each campaign happened to run.
            pair_steps = eval_config.get("pair_steps", [24, 48, 72, 96, 120])
            lanes = {}
            for name, pdir in _cfg(eval_config, "pair_lanes").items():
                lanes[name] = compute.pair_coherence(pdir, channel="msl", member=0,
                                                     steps=pair_steps)
            path.write_text(json.dumps(lanes) + "\n")
            LOG.info("wrote %s", path)

    if "sampler_peaks" in stages:
        path = meas / "sampler_peaks.json"
        if overwrite or not path.exists():
            out = compute.sampler_peaks(dict(_cfg(eval_config, "sampler_arms")))
            path.write_text(json.dumps(out) + "\n")
            LOG.info("wrote %s", path)

    return output_dir


# ---------------------------------------------------------------------------
# score: the handful of numbers the figures rest on
# ---------------------------------------------------------------------------

def score(results_dir, lane_config: dict, eval_config: dict, **kwargs) -> dict:
    results_dir = Path(results_dir)
    meas = results_dir / MEASUREMENTS
    out: dict = {"scoreboard": False,
                 "note": "diagnostic evaluator; deliberately produces no scoreboard row"}

    cap = F.load_capacity(_cfg(eval_config, "capacity")["control"])
    mask = F.colocated_short_lead(cap)
    out["n_cases_total"] = cap["n"]
    out["n_cases_colocated_short_lead"] = int(mask.sum())
    out["mean_gap_closed_hpa"] = float(cap["closed"][mask].mean())
    out["mean_driver_gap_hpa"] = float(cap["gap"][mask].mean())

    scores = P.load_precip_rows(_cfg(eval_config, "precip_scores_json"))
    ranks = P.precip_ranks(scores)
    out["truth_peak_above_all_members_frac"] = (
        ranks["max_mm"]["histogram"][-1] / ranks["max_mm"]["n"])
    out["truth_peak_mean_rank"] = ranks["max_mm"]["mean_rank"]
    spread = P.precip_spread(scores)
    out["member_peak_spread_model_mm"] = float(np.mean(spread["model_sd"]))
    out["member_peak_spread_driver_mm"] = float(np.mean(spread["driver_sd"]))
    out["spread_narrowing_factor"] = (
        out["member_peak_spread_driver_mm"] / out["member_peak_spread_model_mm"])

    lb_path = meas / "loss_budget.json"
    if lb_path.exists():
        lb = json.loads(lb_path.read_text())
        thr = lb["thresholds_mm"]
        if 30.0 in thr:
            k = thr.index(30.0)
            out["sse_share_above_30mm"] = lb["sse_share_above"][k]
            out["point_share_above_30mm"] = lb["point_share_above"][k]
    return out


# ---------------------------------------------------------------------------
# plot: the bundle
# ---------------------------------------------------------------------------

def plot(results_dir, lane_config: dict, eval_config: dict, *, output_dir=None, **kwargs) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else results_dir
    meas = results_dir / MEASUREMENTS
    bundle = output_dir / "plots"
    bundle.mkdir(parents=True, exist_ok=True)

    cap_paths = _cfg(eval_config, "capacity")
    cap_ctrl = F.load_capacity(cap_paths["control"])
    cap_w13 = F.load_capacity(cap_paths["guided"]) if cap_paths.get("guided") else None
    scores = P.load_precip_rows(_cfg(eval_config, "precip_scores_json"))
    scan = P.load_scan(_cfg(eval_config, "scan_jsonl"))

    figs: list[tuple[str, str, object, str]] = []   # number, slug, figure, caption

    # 1 -------------------------------------------------------------------
    fig, cap = F.fig01_capacity_curve(cap_ctrl, cap_w13)
    figs.append(("01", "deepening_capacity_curve", fig, cap))

    # 2 -------------------------------------------------------------------
    entries = []
    dp_delivered, n_dp = P.cyclone_delivered(cap_ctrl, 990.0)
    entries.append({
        "label": "cyclone central pressure\n(driver low below 990 hPa)",
        "required": P.scan_increment(scan, "msl_min_hpa", 990.0),
        "delivered": dp_delivered, "unit": "hPa",
        "support_note": f"{n_dp} co-located cases at or below 72 h against "
                        f"{P.scan_n(scan, 990.0)} paired times"})
    wind_path = meas / "box_wind.json"
    if wind_path.exists():
        wind_rows = json.loads(wind_path.read_text())
        wd, n_w = P.wind_delivered(wind_rows, cap_ctrl, 990.0)
        entries.append({
            "label": "cyclone maximum 10 m wind\n(driver low below 990 hPa)",
            "required": P.scan_increment(scan, "wind_max_ms", 990.0),
            "delivered": wd, "unit": "m/s",
            "support_note": f"{n_w} co-located cases at or below 72 h"})
    for label, stat, key, unit in (
            ("precipitation wet fraction", "wet_frac", "tp_wet_frac", "of the grid"),
            ("precipitation 99th percentile", "p99_mm", "tp_p99_mm", "mm"),
            ("precipitation 99.9th percentile", "p999_mm", "tp_p999_mm", "mm"),
            ("precipitation peak", "max_mm", "tp_max_mm", "mm")):
        entries.append({
            "label": label,
            "required": P.scan_increment(scan, key),
            "delivered": P.precip_delivered_increment(scores, stat),
            "unit": unit,
            "support_note": f"{len(scores['rows'])} slices x 10 members against "
                            f"{len(scan)} paired times"})
    fig, cap = F.fig02_required_vs_delivered(entries)
    figs.append(("02", "required_vs_delivered", fig, cap))

    # 3 -------------------------------------------------------------------
    fig, cap = F.fig03_systematic_by_intensity(P.scan_bins(scan))
    figs.append(("03", "systematic_correction_by_intensity", fig, cap))

    # 4, 5 ----------------------------------------------------------------
    fig, cap = F.fig04_case_trajectories(cap_ctrl)
    figs.append(("04", "per_case_cyclone_trajectory", fig, cap))
    fig, cap = F.fig05_track_divergence(cap_ctrl)
    figs.append(("05", "track_divergence_by_lead", fig, cap))

    # 6, 7, 8 -------------------------------------------------------------
    fig, cap = F.fig06_precip_ceiling(P.precip_peaks(scores))
    figs.append(("06", "precipitation_ceiling", fig, cap))

    systematic = {
        "labels": ["99th percentile", "99.9th percentile", "per-slice peak"],
        "required": [P.scan_increment(scan, k) for k in ("tp_p99_mm", "tp_p999_mm", "tp_max_mm")],
        "delivered": [P.precip_delivered_increment(scores, k)
                      for k in ("p99_mm", "p999_mm", "max_mm")],
        "n_scan": len(scan),
    }
    fig, cap = F.fig07_precip_quantiles(P.precip_campaign_quantiles(scores), systematic)
    figs.append(("07", "precipitation_distribution_and_p99_reversal", fig, cap))

    fig, cap = F.fig08_precip_skill(P.precip_per_step(scores))
    figs.append(("08", "precipitation_skill_by_lead", fig, cap))

    # 9 -------------------------------------------------------------------
    lb_path = meas / "loss_budget.json"
    if lb_path.exists():
        fig, cap = F.fig09_loss_budget(json.loads(lb_path.read_text()))
        figs.append(("09", "precipitation_loss_budget", fig, cap))
    else:
        LOG.error("figure 9 skipped: %s missing", lb_path)

    # 10 ------------------------------------------------------------------
    sp_path = meas / "sampler_peaks.json"
    if sp_path.exists():
        arms = json.loads(sp_path.read_text())
        fig, cap = F.fig10_sampler_arms(arms, list(_cfg(eval_config, "sampler_arms").keys()))
        figs.append(("10", "sampler_arms_peak_distribution", fig, cap))
    else:
        LOG.error("figure 10 skipped: %s missing", sp_path)

    # 11 ------------------------------------------------------------------
    pc_path = meas / "pair_coherence.json"
    if pc_path.exists():
        fig, cap = F.fig11_pair_coherence(json.loads(pc_path.read_text()))
        figs.append(("11", "cross_lane_pair_coherence", fig, cap))
    else:
        LOG.error("figure 11 skipped: %s missing", pc_path)

    # 13, 14 --------------------------------------------------------------
    fig, cap = F.fig13_rank_histograms(P.precip_ranks(scores))
    figs.append(("13", "rank_histogram_precipitation_peak", fig, cap))
    fig, cap = F.fig14_spread_collapse(P.precip_spread(scores))
    figs.append(("14", "ensemble_spread_collapse", fig, cap))

    # write every figure on its own, then the concatenation ----------------
    captions: dict[str, dict] = {}
    for number, slug, fig, cap in figs:
        path = bundle / f"{number}_{slug}.pdf"
        fig.savefig(path, dpi=200)
        captions[number] = {"slug": slug, "file": path.name, "caption": cap}

    # 12: the standing cyclone set and the spectra panel -------------------
    standing = build_standing_set(eval_config, bundle)
    captions.update(standing["captions"])

    combined = bundle / "o1280_o2560_diagnostic_bundle.pdf"
    with PdfPages(combined) as pdf:
        for number, slug, fig, _cap in figs:
            pdf.savefig(fig)
        for fig in standing["figures"]:
            pdf.savefig(fig)
    for _n, _s, fig, _c in figs:
        plt.close(fig)
    for fig in standing["figures"]:
        plt.close(fig)

    lines = ["# o1280 -> o2560 (9 km -> 4.4 km) diagnostic figure bundle", "",
             "Every caption states the support the numbers sit on, the sample size, and the",
             "arm that produced them. This bundle is diagnostic and carries no scoreboard row.",
             ""]
    for number in sorted(captions):
        entry = captions[number]
        lines += [f"## Figure {number.lstrip('0') or '0'} - {entry['slug'].replace('_', ' ')}",
                  f"File: `{entry['file']}`", "", entry["caption"], ""]
    (bundle / "CAPTIONS.md").write_text("\n".join(lines) + "\n")
    (bundle / "manifest.json").write_text(json.dumps(captions, indent=2) + "\n")
    LOG.info("bundle written to %s", bundle)
    return bundle
