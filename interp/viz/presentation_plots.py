#!/usr/bin/env python3
"""Presentation figures for o320->o1280 TC performance.

Pure renderer: reads existing interp `trajectory.json` files (no GPU, no model
load) and emits a consistent set of presentation PNGs. The trajectory probes
themselves are produced by `python -m interp trajectory`; this script only
plots their JSON so the figures share one style.

Figures
-------
1. ceiling_curves.png   - eecdb127 (o320->o1280): noised-target-forced ceiling
                          (storm-core msl min) vs noise level sigma.
2. ceiling_vs_steps.png - cross-lane: cfec83a3 (o96->o320) ceiling-vs-steps curve
                          + eecdb127 (o320->o1280) single 194k point. Per-lane
                          truth reference lines (no intermediate eecdb ckpts on
                          disk, so only its final point is available).
3. compare_traj.png     - eecdb127: ceiling ("knows") vs realized free sampler
                          ("commits"), with truth + x_interp baselines.

Usage:  python presentation_plots.py [--out DIR]
"""
from __future__ import annotations
import argparse, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/ecm5702/perm/interp"

# --- consistent style -------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 12, "axes.grid": True,
    "grid.alpha": 0.25, "legend.fontsize": 9, "axes.axisbelow": True,
})
C = {
    "ceiling": "#1b9e9e",     # teal  - what the model KNOWS
    "realized": "#d9531e",    # orange/red - what it COMMITS to
    "truth": "#2ca02c",       # green - target truth
    "interp": "#7f7f7f",      # grey  - linear x_interp baseline
    "lane2": "#7a4fb5",       # purple - o320->o1280 on the cross-lane plot
}

# label -> (run dir under ROOT, pretty name)
def load(label: str) -> dict:
    """Find and load a run's trajectory.json (handles the tc_traj/ nesting)."""
    for sub in ("tc_traj", "."):
        p = os.path.join(ROOT, label, sub, "trajectory", "trajectory.json")
        if os.path.exists(p):
            return json.load(open(p))
    raise FileNotFoundError(f"no trajectory.json for {label}")

def step_of(d: dict) -> int:
    m = re.search(r"step_(\d+)", json.dumps(d.get("checkpoint", "")) + json.dumps(d))
    return int(m.group(1)) if m else 0

def ceiling_xy(d: dict, field="msl"):
    cs = sorted(d["ceiling"], key=lambda c: c["sigma"])
    return (np.array([c["sigma"] for c in cs]),
            np.array([c["metrics"][field] for c in cs]))

# ---------------------------------------------------------------------------
def fig_ceiling_curves(out):
    d = load("eecdb127_o1280_195k")
    sig, msl = ceiling_xy(d)
    tgt = d["summary"]["msl"]["target"]
    xin = d["summary"]["msl"]["x_interp"]
    deep = d["summary"]["msl"]["ceiling_extreme"]
    deep_s = d["summary"]["msl"]["ceiling_extreme_sigma"]

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.plot(sig, msl, "o-", color=C["ceiling"], lw=2.2, ms=7,
            label="noised_target-forced ceiling (msl min)")
    ax.axhline(tgt, ls="--", color=C["truth"], lw=1.6, label=f"truth (target)  {tgt:.1f} hPa")
    ax.axhline(xin, ls=":", color=C["interp"], lw=1.6, label=f"x_interp baseline  {xin:.1f} hPa")
    ax.annotate(f"deepest ceiling\n{deep:.1f} hPa @ σ={deep_s:.0f}",
                xy=(deep_s, deep), xytext=(deep_s*2.2, deep+6),
                fontsize=9, color=C["ceiling"],
                arrowprops=dict(arrowstyle="->", color=C["ceiling"]))
    ax.set_xscale("log")
    ax.invert_xaxis()  # high sigma (coarse) -> low sigma (fine)
    ax.set_xlabel("noised_target-forced  σ   (high → low  ≈  coarse → fine scale)")
    ax.set_ylabel("storm-core msl min  (hPa,  deeper = better)")
    ax.set_title("eecdb127  o320→o1280  —  what the denoiser knows (Franklin)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    p = os.path.join(out, "ceiling_curves.png"); fig.savefig(p); plt.close(fig)
    print("wrote", p)

def fig_ceiling_vs_steps(out):
    cfec_labels = ["cfec_010k", "cfec_020k", "cfec_030k", "cfec_050k",
                   "cfec_100k", "cfec_150k", "cfec_200k"]
    steps, ceil = [], []
    tgt_cfec = None
    for lab in cfec_labels:
        d = load(lab); s = d["summary"]["msl"]
        steps.append(step_of(d)); ceil.append(s["ceiling_extreme"]); tgt_cfec = s["target"]
    steps = np.array(steps); ceil = np.array(ceil)

    e = load("eecdb127_o1280_195k"); es = e["summary"]["msl"]
    e_step = step_of(e); e_ceil = es["ceiling_extreme"]; tgt_eec = es["target"]

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.plot(steps/1e3, ceil, "o-", color=C["ceiling"], lw=2.2, ms=6,
            label="cfec83a3  o96→o320  (deepest ceiling)")
    ax.axhline(tgt_cfec, ls="--", color=C["ceiling"], lw=1.2, alpha=0.6,
               label=f"o96→o320 truth  {tgt_cfec:.1f} hPa")
    ax.plot([e_step/1e3], [e_ceil], "D", color=C["lane2"], ms=11,
            label=f"eecdb127  o320→o1280  @ {e_step/1e3:.0f}k  (only ckpt on disk)")
    ax.axhline(tgt_eec, ls="--", color=C["lane2"], lw=1.2, alpha=0.6,
               label=f"o320→o1280 truth  {tgt_eec:.1f} hPa")
    ax.set_xlabel("training steps  (×10³)")
    ax.set_ylabel("deepest ceiling  storm-core msl min  (hPa)")
    ax.set_title("Knowledge ceiling vs training steps  (Franklin)")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.text(0.02, 0.04,
            "ceiling is set early & ~flat: capacity, not training length, is the ceiling",
            transform=ax.transAxes, fontsize=8.5, style="italic", color="#444")
    fig.tight_layout()
    p = os.path.join(out, "ceiling_vs_steps.png"); fig.savefig(p); plt.close(fig)
    print("wrote", p)

def fig_compare_traj(out):
    d = load("eecdb127_o1280_195k")
    sig_c, msl_c = ceiling_xy(d)
    s = d["summary"]["msl"]
    tgt, xin = s["target"], s["x_interp"]

    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    # realized free sampler: one faint line per seed (msl vs sigma along sampling)
    seed_finals = []
    for i, tr in enumerate(d["trajectories"]):
        stp = sorted(tr["steps"], key=lambda x: -x["sigma"])
        sg = np.array([x["sigma"] for x in stp])
        ms = np.array([x["metrics"]["msl"] for x in stp])
        ax.plot(sg, ms, "-", color=C["realized"], lw=0.8, alpha=0.30,
                label="realized free sampler (per seed)" if i == 0 else None)
        seed_finals.append(tr["final"]["msl"])
    # mean realized final
    rfin = float(np.mean(seed_finals))
    ax.axhline(rfin, ls="-", color=C["realized"], lw=2.0,
               label=f"realized mean final  {rfin:.1f} hPa")
    # ceiling
    ax.plot(sig_c, msl_c, "o--", color=C["ceiling"], lw=2.2, ms=6,
            label="noised_target-forced ceiling")
    # references
    ax.axhline(tgt, ls="--", color=C["truth"], lw=1.6, label=f"truth  {tgt:.1f} hPa")
    ax.axhline(xin, ls=":", color=C["interp"], lw=1.6, label=f"x_interp  {xin:.1f} hPa")

    # annotate the "knows but doesn't commit" gap
    deep = s["ceiling_extreme"]; deep_s = s["ceiling_extreme_sigma"]
    ax.annotate("", xy=(deep_s, deep), xytext=(deep_s, rfin),
                arrowprops=dict(arrowstyle="<->", color="#333", lw=1.4))
    ax.text(deep_s*1.25, (deep+rfin)/2,
            f"knows but doesn't\ncommit  ≈ {rfin-deep:.0f} hPa",
            fontsize=9, color="#333", va="center")
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.set_xlabel("noise level  σ   (high → low  ≈  coarse → fine scale)")
    ax.set_ylabel("storm-core msl min  (hPa,  deeper = better)")
    ax.set_title("eecdb127  o320→o1280  —  ceiling vs realized (Franklin)")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    p = os.path.join(out, "compare_traj.png"); fig.savefig(p); plt.close(fig)
    print("wrote", p)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "presentation"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_ceiling_curves(a.out)
    fig_ceiling_vs_steps(a.out)
    fig_compare_traj(a.out)
    print("done ->", a.out)
