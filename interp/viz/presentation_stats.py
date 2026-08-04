#!/usr/bin/env python3
"""Statistically explicit renderer for the o320->o1280 TC presentation plots."""
from __future__ import annotations

import argparse
import json
import os
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/home/ecm5702/perm/interp"
OUT = "/home/ecm5702/perm/interp/presentation"
MEMBER_RUNS = sorted(glob(os.path.join(ROOT, "eec_peak072_m*", "tc_traj", "trajectory", "trajectory.json")))
CFEC_LABELS = ["cfec_010k", "cfec_020k", "cfec_030k", "cfec_050k", "cfec_100k", "cfec_150k", "cfec_200k"]

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 12, "axes.grid": True,
    "grid.alpha": 0.25, "legend.fontsize": 9, "axes.axisbelow": True,
})
C = {"ceiling": "#1b9e9e", "realized": "#d9531e", "truth": "#2ca02c", "interp": "#7f7f7f", "lane2": "#7a4fb5"}


def read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def member_data():
    if len(MEMBER_RUNS) != 10:
        raise RuntimeError(f"expected 10 eecdb member runs, found {len(MEMBER_RUNS)}")
    data = [read(path) for path in MEMBER_RUNS]
    ckpts = {item["checkpoint"] for item in data}
    if len(ckpts) != 1:
        raise RuntimeError(f"member runs use different checkpoints: {ckpts}")
    return data


def ceiling_matrix(data):
    sigmas = sorted(float(item["sigma"]) for item in data[0]["ceiling"])
    matrix = []
    targets = []
    interps = []
    for item in data:
        by_sigma = {float(row["sigma"]): row["metrics"]["msl"] for row in item["ceiling"]}
        if sorted(by_sigma) != sigmas:
            raise RuntimeError("member ceiling sigma grids differ")
        matrix.append([by_sigma[sigma] for sigma in sigmas])
        targets.append(item["summary"]["msl"]["target"])
        interps.append(item["summary"]["msl"]["x_interp"])
    return np.asarray(sigmas), np.asarray(matrix), np.asarray(targets), np.asarray(interps)


def qband(values):
    return np.percentile(values, [10, 50, 90], axis=0)


def fig_ceiling_curves(out):
    data = member_data()
    sigmas, values, targets, interps = ceiling_matrix(data)
    q10, q50, q90 = qband(values)
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.fill_between(sigmas, q10, q90, color=C["ceiling"], alpha=0.20, label="member spread (10–90%)")
    ax.plot(sigmas, q50, "o-", color=C["ceiling"], lw=2.2, ms=7, label="ceiling median (n=10 members)")
    ax.axhline(np.median(targets), ls="--", color=C["truth"], lw=1.6,
               label=f"truth median  {np.median(targets):.1f} hPa")
    ax.axhspan(np.percentile(targets, 10), np.percentile(targets, 90), color=C["truth"], alpha=0.08)
    ax.axhline(np.median(interps), ls=":", color=C["interp"], lw=1.6,
               label=f"x_interp median  {np.median(interps):.1f} hPa")
    low = sigmas == 5.0
    ax.scatter(sigmas[low], q50[low], s=100, facecolors="white", edgecolors="#555", zorder=5,
               label="σ=5 diagnostic only (low-σ box-min artifact)")
    ax.annotate("excluded from the knowledge claim:\nlow-σ raw box-min artifact",
                xy=(5, q50[low][0]), xytext=(7.5, q50[low][0] + 7), fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color="#555"), color="#555")
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.set_xlabel("noised_target-forced σ (high → low ≈ coarse → fine scale)")
    ax.set_ylabel("storm-core msl min (hPa, deeper = better)")
    ax.set_title("eecdb127 o320→o1280 — ceiling across 10 input/target members")
    ax.text(0.02, 0.03, "One ceiling noise draw per member; band is member-to-member spread, not independent-cell CI.",
            transform=ax.transAxes, fontsize=8.2, color="#444", style="italic")
    ax.legend(loc="upper left", fontsize=8.2)
    fig.tight_layout(); fig.savefig(os.path.join(out, "ceiling_curves.png")); plt.close(fig)


def step_of(data):
    text = json.dumps(data.get("checkpoint", ""))
    digits = "".join(ch if ch.isdigit() else " " for ch in text).split()
    return int(digits[-1]) if digits else 0


def load_nested(label):
    path = os.path.join(ROOT, label, "tc_traj", "trajectory", "trajectory.json")
    if not os.path.exists(path):
        path = os.path.join(ROOT, label, "trajectory", "trajectory.json")
    return read(path)


def fig_ceiling_vs_steps(out):
    points = []
    for label in CFEC_LABELS:
        item = load_nested(label)
        points.append((step_of(item) / 1e3, item["summary"]["msl"]["ceiling_extreme"], "cfec83a3 o96→o320", 1))
    eec = read(os.path.join(ROOT, "eecdb127_o1280_195k", "tc_traj", "trajectory", "trajectory.json"))
    points.append((step_of(eec) / 1e3, eec["summary"]["msl"]["ceiling_extreme"], "eecdb127 o320→o1280", 1))
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    cf = points[:-1]
    ax.plot([p[0] for p in cf], [p[1] for p in cf], "o-", color=C["ceiling"], lw=2.2,
            label="cfec83a3 o96→o320 (one pair/checkpoint)")
    e = points[-1]
    ax.scatter([e[0]], [e[1]], marker="D", s=100, color=C["lane2"],
               label="eecdb127 o320→o1280 (one pair; final checkpoint only)")
    ax.set_xlabel("training steps (×10³)"); ax.set_ylabel("deepest ceiling storm-core msl min (hPa)")
    ax.set_title("Knowledge ceiling vs training steps — exploratory, not statistically resolved")
    ax.text(0.02, 0.04, "NOT sufficient for a training-length claim: no replicate input/target pairs at any checkpoint.",
            transform=ax.transAxes, fontsize=9, color="#9b2226", weight="bold")
    ax.text(0.67, 0.13, "Every point: n=1 pair", transform=ax.transAxes, fontsize=8.5, color="#555")
    ax.legend(loc="upper right", fontsize=8.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "ceiling_vs_steps.png")); plt.close(fig)


def fig_compare_traj(out):
    data = member_data()
    sigmas, ceilings, targets, interps = ceiling_matrix(data)
    c10, c50, c90 = qband(ceilings)
    # All ten member runs use the same sampler schedule: aggregate by denoiser-call index.
    paths = []
    for item in data:
        paths.extend(item["trajectories"])
    arrays = []
    sigma_arrays = []
    finals = []
    for tr in paths:
        steps = sorted(tr["steps"], key=lambda row: -row["sigma"])
        arrays.append([row["metrics"]["msl"] for row in steps])
        sigma_arrays.append([row["sigma"] for row in steps])
        finals.append(tr["final"]["msl"])
    lengths = {len(row) for row in arrays}
    if len(lengths) != 1:
        raise RuntimeError(f"sampler trajectory lengths differ: {lengths}")
    realized = np.asarray(arrays)
    sigma_path = np.median(np.asarray(sigma_arrays), axis=0)
    r10, r50, r90 = qband(realized)
    final10, final50, final90 = np.percentile(finals, [10, 50, 90])
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    ax.fill_between(sigma_path, r10, r90, color=C["realized"], alpha=0.18, label="realized sampler 10–90%")
    ax.plot(sigma_path, r50, color=C["realized"], lw=2.3, label=f"realized median final {final50:.1f} hPa")
    ax.fill_between(sigmas, c10, c90, color=C["ceiling"], alpha=0.20, label="ceiling member spread 10–90%")
    ax.plot(sigmas, c50, "o--", color=C["ceiling"], lw=2.2, ms=6, label="ceiling median (n=10 members)")
    ax.axhline(np.median(targets), ls="--", color=C["truth"], lw=1.6,
               label=f"truth median {np.median(targets):.1f} hPa")
    ax.axhline(np.median(interps), ls=":", color=C["interp"], lw=1.6,
               label=f"x_interp median {np.median(interps):.1f} hPa")
    ax.scatter([5], [c50[list(sigmas).index(5.0)]], s=95, facecolors="white", edgecolors="#555", zorder=5,
               label="σ=5 diagnostic artifact")
    usable_sigma = min(sigmas[sigmas >= 10], key=lambda value: abs(value - 20))
    ci = list(sigmas).index(usable_sigma)
    gap = final50 - c50[ci]
    ax.annotate(f"median gap at σ={usable_sigma:g}: {gap:+.1f} hPa\n(positive = sampler shallower)",
                xy=(usable_sigma, c50[ci]), xytext=(usable_sigma * 1.8, c50[ci] - 8), fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color="#333"), color="#333")
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.set_xlabel("noise level σ (high → low ≈ coarse → fine scale)")
    ax.set_ylabel("storm-core msl min (hPa, deeper = better)")
    ax.set_title("eecdb127 o320→o1280 — paired ceiling vs realized sampler")
    ax.text(0.02, 0.03, "n=10 input/target members × 8 seeds = 80 paired sampler trajectories; σ=5 excluded from interpretation.",
            transform=ax.transAxes, fontsize=8.2, color="#444", style="italic")
    ax.legend(loc="upper left", fontsize=8.1)
    fig.tight_layout(); fig.savefig(os.path.join(out, "compare_traj.png")); plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig_ceiling_curves(args.out)
    fig_ceiling_vs_steps(args.out)
    fig_compare_traj(args.out)
