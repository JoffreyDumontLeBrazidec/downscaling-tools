"""The o1280->o2560 (9 km -> 4.4 km) lane diagnostic figure set.

Each function here builds one figure and returns it together with the caption
that belongs under it.  Every caption names the support the numbers were
measured on, how many cases went into them, and which arm produced them,
because on this lane the same quantity measured on two different supports gives
two different answers and the difference has already caused one wrong verdict.

Three rules are enforced by construction rather than by convention:

* Cyclone depth is never read off a pooled extreme.  Every pressure and wind
  readout below is computed per (date, lead step, ensemble member).
* Every cyclone readout is split by DRIVER QUALITY, meaning how far the driver's
  own box minimum sits from the target's, and what is plotted is the fraction of
  that driver-to-target gap the model closes, not the raw depth against truth.
* Beyond about 72 hours the target's storm is somewhere materially different
  from the driver's on a majority of members, so those leads are excluded from
  intensity statistics or shaded where they are shown.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- fixed properties of this lane's diagnostic supports --------------------

BOX = {"south": 15.0, "north": 40.0, "west": -80.0, "east": -35.0}
BOX_TEXT = "North Atlantic box 15-40N, 80-35W"
TRAINED_LEAD_MAX = 72       # training covered lead times 6-72 h
COLOCATION_DEG = 2.0        # a case counts as the same storm within this distance

SUPPORT_BOX = (f"Support: {BOX_TEXT}, one box minimum per (date, lead step, member); "
               "never a pooled extreme.")
SUPPORT_GLOBAL = ("Support: the complete global O2560 grid, 26,306,560 points, "
                  "one value per (date, lead step) slice.")
SUPPORT_TROPICS = ("Support: global tropical belt, latitude within 30 degrees of the "
                   "equator, on the paired training datasets.")
SUPPORT_BOXLANE = ("Support: the Humberto box lane, one value per (date, lead step, "
                   "member) field over the box.")

ARM_CTRL = ("Arm: checkpoint fccc23df, epoch 456, step 300,000, unguided, sampler "
            "piecewise-30 with a noise ceiling of 1e3 (this lane's default before "
            "2026-08-27).")
ARM_W13 = ("Guided arm: the same checkpoint with mass-only autoguidance at weight 1.3, "
           "same sampler.")
CAMPAIGN = ("Campaign: Humberto, 5 initialisations 2025-09-26 to 2025-09-30, 10 ensemble "
            "members, 20 six-hourly lead times, the clean post-deaccumulation-fix run.")

C_TRUTH = "#111111"
C_DRIVER = "#1f77b4"
C_MODEL = "#d62728"
C_GUIDED = "#ff7f0e"

DRIVER_BINS = [
    ("driver deeper\nthan target", -1e9, 0.0),
    ("0-5", 0.0, 5.0),
    ("5-10", 5.0, 10.0),
    ("10-20", 10.0, 20.0),
    ("20-30", 20.0, 30.0),
    ("30+", 30.0, 1e9),
]


# ---------------------------------------------------------------------------
# small shared helpers
# ---------------------------------------------------------------------------

def finish(fig, title: str, caption: str, *, left_inches: float = 1.05):
    """Stamp a title on a figure and lay the caption out underneath it.

    The page grows downwards to make room for the caption rather than the axes
    shrinking upwards. Shrinking the axes was the wrong way round: it left the
    plotting area shorter than its own rotated y-axis label, which matplotlib
    then clipped at the edge of the page.
    """
    wrapped = "\n".join(textwrap.fill(part, 132) for part in caption.split("\n"))
    n_lines = wrapped.count("\n") + 1

    width, height = fig.get_size_inches()
    caption_inches = 0.135 * n_lines + 0.20
    title_inches = 0.45
    total = height + caption_inches + title_inches
    fig.set_size_inches(width, total)
    fig.subplots_adjust(
        bottom=(caption_inches + 0.62) / total,
        top=1.0 - title_inches / total,
        left=left_inches / width,
        right=1.0 - 0.25 / width,
    )
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.0 - 0.16 / total)
    fig.text(0.012, 0.10 / total, wrapped, ha="left", va="bottom", fontsize=8.2,
             family="DejaVu Sans", color="#333333")
    return fig, caption


def _sem(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0


def _bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """Percentile bootstrap interval for a mean, so intervals are honest on small bins."""
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _bootstrap_ratio_ci(num: np.ndarray, den: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """Percentile bootstrap interval for a ratio of two sums over the same cases."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    if num.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, num.size, size=(n_boot, num.size))
    ratios = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    return float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


def _bin_value(bins: list[dict], label: str, key: str) -> float:
    """Look one measured bin up by its label, so captions quote the plotted number."""
    for b in bins:
        if b["label"] == label:
            return float(b[key])
    return float("nan")


def _great_circle_deg(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Angular separation in degrees between two sets of points."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    cosd = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl)
    return np.degrees(np.arccos(np.clip(cosd, -1.0, 1.0)))


def load_capacity(path: str | Path) -> dict:
    """Load the per-case box-minimum table and derive the columns the figures need."""
    rows = json.loads(Path(path).read_text())
    out = {k: np.asarray([r[k] for r in rows]) for k in
           ("step", "member", "truth", "interp", "model",
            "truth_lat", "truth_lon", "interp_lat", "interp_lon",
            "model_lat", "model_lon")}
    out["date"] = np.asarray([r["date"] for r in rows])
    out["gap"] = out["interp"] - out["truth"]          # >0: driver too shallow
    out["closed"] = out["interp"] - out["model"]       # >0: model deepened
    out["sep_model_truth"] = _great_circle_deg(
        out["model_lat"], out["model_lon"], out["truth_lat"], out["truth_lon"])
    out["n"] = len(rows)
    return out


def colocated_short_lead(cap: dict) -> np.ndarray:
    """Cases that are the same storm and inside the trained lead range."""
    return (cap["sep_model_truth"] <= COLOCATION_DEG) & (cap["step"] <= TRAINED_LEAD_MAX)


def bin_by_driver_quality(cap: dict, mask: np.ndarray):
    """Group the selected cases into the standing driver-quality bins."""
    out = []
    for label, lo, hi in DRIVER_BINS:
        sel = mask & (cap["gap"] > lo) & (cap["gap"] <= hi)
        out.append((label, sel))
    return out


# ---------------------------------------------------------------------------
# Figure 1 - the deepening capacity curve
# ---------------------------------------------------------------------------

def fig01_capacity_curve(cap_ctrl: dict, cap_w13: dict | None):
    """Gap closed against driver quality, in hectopascals and as a fraction.

    The fraction is the total closed divided by the total gap inside a bin, not
    the average of the per-case ratios. Averaging ratios would be dominated by
    the cases where the driver was already almost right, because a gap close to
    zero in the denominator makes the ratio explode.
    """
    fig, axs = plt.subplots(1, 2, figsize=(14.5, 5.6))
    arms = [("control", cap_ctrl, C_MODEL)]
    if cap_w13 is not None:
        arms.append(("mass-only autoguidance, weight 1.3", cap_w13, C_GUIDED))

    width = 0.38
    xs = np.arange(len(DRIVER_BINS))
    counts_text, story = [], []
    for k, (name, cap, colour) in enumerate(arms):
        mask = colocated_short_lead(cap)
        means, errs, fracs, flo, fhi, ns = [], [], [], [], [], []
        for label, sel in bin_by_driver_quality(cap, mask):
            closed, gap = cap["closed"][sel], cap["gap"][sel]
            ns.append(int(sel.sum()))
            means.append(float(closed.mean()) if sel.sum() else np.nan)
            errs.append(_sem(closed) if sel.sum() else 0.0)
            if sel.sum() and label != DRIVER_BINS[0][0]:
                fracs.append(float(closed.sum() / gap.sum()))
                lo, hi = _bootstrap_ratio_ci(closed, gap)
                flo.append(fracs[-1] - lo)
                fhi.append(hi - fracs[-1])
            else:
                fracs.append(np.nan)
                flo.append(0.0)
                fhi.append(0.0)
        off = (k - (len(arms) - 1) / 2) * width
        axs[0].bar(xs + off, means, width, yerr=errs, capsize=3,
                   color=colour, alpha=0.85, label=name)
        axs[1].bar(xs + off, fracs, width, yerr=[flo, fhi], capsize=3,
                   color=colour, alpha=0.85, label=name)
        counts_text.append(f"{name}: n = {sum(ns)} cases ({', '.join(str(v) for v in ns)} by bin)")
        story.append(
            f"{name} adds {means[1]:.1f} hPa where the driver is within 5 hPa and "
            f"{means[-1]:.1f} hPa where it is more than 30 hPa away, closing "
            f"{100*fracs[1]:.0f} per cent of the gap in the first case and "
            f"{100*fracs[-1]:.0f} per cent in the last")

    axs[0].set_ylabel("mean pressure the model adds (hPa)")
    axs[0].set_title("How much deepening the model adds")
    axs[1].axhline(1.0, color=C_TRUTH, lw=1.2, ls="--")
    axs[1].text(len(DRIVER_BINS) - 0.55, 1.03, "the whole gap closed", ha="right",
                fontsize=8.5, color="#555555")
    axs[1].set_ylabel("fraction of the driver-to-target gap closed")
    axs[1].set_title("The same result as a fraction of what was needed")
    axs[1].set_ylim(0, 1.45)
    for ax in axs:
        ax.set_xticks(xs)
        ax.set_xticklabels([b[0] for b in DRIVER_BINS], fontsize=9)
        ax.set_xlabel("driver quality: driver box minimum minus target box minimum (hPa)")
        ax.axhline(0.0, color="#000000", lw=0.8)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(axis="y", alpha=0.25)

    caption = (
        "Left: the mean pressure the downscaler subtracts from its driver's box minimum, "
        "grouped by how far that driver already sits from the target. Right: the same "
        "amount as the fraction of the driver-to-target gap that was closed, which is the "
        "pre-registered readout on this lane; it is the total closed divided by the total "
        "gap within a bin, with a 95 per cent bootstrap interval. The leftmost group has "
        "no fraction because there the driver is already DEEPER than the target, so there "
        "is no gap to close and the model's extra deepening only makes it worse. "
        + "; ".join(story) + ". The absolute amount added grows with the size of the gap "
        "but nowhere near fast enough to keep up with it, so the fraction closed falls "
        "steadily. Guidance lifts every group, including the groups where no deepening "
        "was needed, which is why it scores as no improvement overall.\n"
        + SUPPORT_BOX + " " + CAMPAIGN + "\n" + ARM_CTRL + " " + ARM_W13 + "\n"
        "Cases kept: model and target minima within "
        f"{COLOCATION_DEG:.0f} degrees of each other and lead time at most "
        f"{TRAINED_LEAD_MAX} h, the trained range. " + "; ".join(counts_text) + "."
    )
    return finish(fig, "1. What the model can add to its driver, and what was needed", caption)


# ---------------------------------------------------------------------------
# Figure 2 - required against delivered
# ---------------------------------------------------------------------------

def fig02_required_vs_delivered(entries: list[dict]):
    """entries: label, required, delivered, unit, support_note."""
    fig, ax = plt.subplots(figsize=(14.5, 6.2))
    labels = [e["label"] for e in entries]
    fracs = [e["delivered"] / e["required"] if e["required"] else np.nan for e in entries]
    ys = np.arange(len(entries))[::-1]
    colours = [C_MODEL if f < 0.5 else "#2ca02c" for f in fracs]
    ax.barh(ys, fracs, color=colours, alpha=0.85, height=0.6)
    ax.axvline(1.0, color=C_TRUTH, ls="--", lw=1.2)
    ax.text(1.01, ys.min() - 0.55, "the whole systematic correction", fontsize=9,
            color="#555555", va="bottom")
    for y, e, f in zip(ys, entries, fracs):
        ax.text(min(f, 1.05) + 0.02, y,
                f"{f*100:.0f}%   (needed {e['required']:+.3g} {e['unit']}, "
                f"model adds {e['delivered']:+.3g} {e['unit']})",
                va="center", fontsize=9)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, 2.05)
    ax.set_xlabel("fraction of the systematic 9 km to 4.4 km correction that the model delivers")
    ax.grid(axis="x", alpha=0.25)
    fig.set_size_inches(15.0, 6.4)

    caption = (
        "The bar is the model's own increment over its interpolated driver divided by the "
        "systematic increment measured between the two training datasets. The model "
        "reproduces most of the correction for cyclone pressure and wind and for the "
        "precipitation distribution through the 99.9th percentile, including the reversal "
        "at the 99th percentile where the target is genuinely LIGHTER than its driver. It "
        "delivers about a twentieth of the correction at the rare intense peak.\n"
        "TWO SUPPORTS ARE CROSSED HERE AND THE COMPARISON IS ONLY AS GOOD AS THAT. The "
        "required column is measured on the paired training datasets over the global "
        "tropical belt; the delivered column is measured on the Humberto campaign, "
        "cyclone quantities over the " + BOX_TEXT + " and precipitation quantities on the "
        "global grid. The two also use different driving products: the systematic scan "
        "pairs the deterministic high-resolution forecast the model trained against, "
        "while the campaign uses interpolated ensemble members.\n"
        + ARM_CTRL + " " + CAMPAIGN + "\n"
        + "; ".join(f"{e['label'].replace(chr(10), ' ')}: {e['support_note']}"
                    for e in entries) + "."
    )
    return finish(fig, "2. Everything except the rare intense convective peak",
                  caption, left_inches=3.1)


# ---------------------------------------------------------------------------
# Figure 3 - the systematic correction against storm intensity
# ---------------------------------------------------------------------------

def fig03_systematic_by_intensity(scan_bins: list[dict]):
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.4))
    xs = np.arange(len(scan_bins))
    labels = [b["label"] for b in scan_bins]

    dp = np.array([b["deepening"] for b in scan_bins])
    dlo = np.array([b["deepening_lo"] for b in scan_bins])
    dhi = np.array([b["deepening_hi"] for b in scan_bins])
    axs[0].bar(xs, dp, color=C_MODEL, alpha=0.85,
               yerr=[dp - dlo, dhi - dp], capsize=4)
    axs[0].axhline(-10.0, color="#555555", ls=":", lw=1.2)
    axs[0].text(0.02, 0.06, "saturation near 10 hPa", transform=axs[0].transAxes,
                fontsize=9, color="#555555")
    axs[0].set_ylabel("systematic deepening, target minus driver (hPa)")
    axs[0].set_title("Cyclone central pressure")
    axs[0].invert_yaxis()

    wg = np.array([b["wind_gain"] for b in scan_bins])
    wlo = np.array([b["wind_gain_lo"] for b in scan_bins])
    whi = np.array([b["wind_gain_hi"] for b in scan_bins])
    axs[1].bar(xs, wg, color=C_DRIVER, alpha=0.85,
               yerr=[wg - wlo, whi - wg], capsize=4)
    axs[1].set_ylabel("systematic wind gain, target minus driver (m/s)")
    axs[1].set_title("Maximum 10 m wind speed")

    for ax in axs:
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_xlabel("selection: the driver's deepest tropical low is below this pressure")
        ax.grid(axis="y", alpha=0.25)
        ax.axhline(0.0, color="#000000", lw=0.8)
    for x, b in zip(xs, scan_bins):
        axs[0].annotate(f"n={b['n']}", (x, 0), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=8, color="#555555")

    caption = (
        "The two training datasets hold the SAME period at 9 km and at 4.4 km, so the "
        "average difference between them is the systematic effect of resolution and "
        "physics that a downscaler could learn: the chaotic disagreement between two "
        "integrations averages away, a systematic offset does not. For genuine tropical "
        "cyclones, meaning a driver low below 990 hPa, that effect is "
        f"{abs(_bin_value(scan_bins, '< 990 hPa', 'deepening')):.2f} hPa of deepening, and "
        "it saturates near 10 hPa rather than growing without limit: the deepest selection "
        "does not get a proportionally larger correction. Error bars are 95 per cent "
        "bootstrap intervals on the mean.\n"
        + SUPPORT_TROPICS + f" Sample: {scan_bins[0]['n']} paired forecast times in the "
        "widest selection shown, from a scan of the two training datasets every third "
        "time. No model is involved in this figure; it is a property of the data the model "
        "was trained on.\n"
        "The earlier reading of this scan, made on 2026-08-28 when 1,333 samples had been "
        "written, gave 6.23 hPa for the same selection. The scan has since finished more "
        "samples and the number has moved slightly; the verdict is unchanged.\n"
        "Note the earlier reading that the systematic difference is about 30 hPa is "
        "RETRACTED: that came from a single global extreme over 15 months, which is not a "
        "systematic effect."
    )
    return finish(fig, "3. How large the true 9 km to 4.4 km correction is, by storm strength", caption)


# ---------------------------------------------------------------------------
# Figure 4 - per-case cyclone trajectory
# ---------------------------------------------------------------------------

def fig04_case_trajectories(cap: dict):
    dates = sorted(set(cap["date"].tolist()))
    fig, axs = plt.subplots(1, len(dates), figsize=(4.0 * len(dates), 5.0), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, date in zip(axs, dates):
        sel = cap["date"] == date
        steps = np.unique(cap["step"][sel])
        for key, colour, name in (("truth", C_TRUTH, "IEKM 4.4 km target"),
                                  ("interp", C_DRIVER, "interpolated 9 km driver"),
                                  ("model", C_MODEL, "downscaler")):
            med = np.array([np.median(cap[key][sel & (cap["step"] == s)]) for s in steps])
            lo = np.array([np.percentile(cap[key][sel & (cap["step"] == s)], 10) for s in steps])
            hi = np.array([np.percentile(cap[key][sel & (cap["step"] == s)], 90) for s in steps])
            ax.plot(steps, med, color=colour, lw=1.8, label=name)
            ax.fill_between(steps, lo, hi, color=colour, alpha=0.16, linewidth=0)
        ax.axvspan(TRAINED_LEAD_MAX, steps.max(), color="#bbbbbb", alpha=0.28, zorder=0)
        ax.axvline(TRAINED_LEAD_MAX, color="#555555", ls="--", lw=1.0)
        ax.set_title(f"initialised {date[:4]}-{date[4:6]}-{date[6:]}", fontsize=10)
        ax.set_xlabel("lead time (h)")
        ax.grid(alpha=0.25)
    axs[0].set_ylabel("box minimum sea-level pressure (hPa)")
    axs[0].invert_yaxis()
    axs[0].legend(fontsize=8.5, loc="lower left")
    axs[-1].text(0.98, 0.02, "shaded: beyond the trained\nlead range and off-track",
                 transform=axs[-1].transAxes, ha="right", va="bottom", fontsize=8,
                 color="#555555")

    caption = (
        "One panel per initialisation. The line is the median over the ten ensemble "
        "members and the band spans the 10th to the 90th member percentile. On 26 and 27 "
        "September the target rapid-intensifies, the driver never does, and the model "
        "adds only a scattered few hectopascals on top of a driver that has missed the "
        "event: that is a forecast error the downscaler is not able to repair, not a "
        "failure to make deep cyclones. On 29 and 30 September the driver is close and "
        "the model tracks the target.\n"
        + SUPPORT_BOX + " " + CAMPAIGN + " " + ARM_CTRL + "\n"
        f"Sample: {cap['n']} cases, 5 dates x 20 lead times x 10 members. The shaded "
        f"region beyond {TRAINED_LEAD_MAX} h is outside the trained lead range AND is "
        "where the target's storm has usually moved away from the driver's, so any "
        "intensity comparison there is mixing an intensity error with a track error."
    )
    return finish(fig, "4. The same storm, case by case", caption)


# ---------------------------------------------------------------------------
# Figure 5 - track divergence against lead
# ---------------------------------------------------------------------------

def fig05_track_divergence(cap: dict):
    steps = np.unique(cap["step"])
    frac = np.array([float((cap["sep_model_truth"][cap["step"] == s] > COLOCATION_DEG).mean())
                     for s in steps])
    n_per = np.array([int((cap["step"] == s).sum()) for s in steps])

    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    ax.plot(steps, 100 * frac, "-o", color=C_MODEL, lw=2.0, ms=5)
    ax.axvline(TRAINED_LEAD_MAX, color="#555555", ls="--", lw=1.4)
    ax.axvspan(TRAINED_LEAD_MAX, steps.max(), color="#bbbbbb", alpha=0.25, zorder=0)
    ax.axhline(50, color=C_TRUTH, ls=":", lw=1.2)
    ax.text(TRAINED_LEAD_MAX + 1.5, 6, "proposed cut-off for cyclone verification\n"
                                       "on this lane, 72 h", fontsize=9, color="#333333")
    ax.text(steps.min() + 1, 52, "half the members looking at a different low",
            fontsize=9, color="#555555")
    for s, f in zip(steps, frac):
        if s in (6, 24, 48, 72, 96, 120):
            ax.annotate(f"{100*f:.0f}%", (s, 100 * f), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=9)
    ax.set_xlabel("lead time (h)")
    ax.set_ylabel("members whose box minimum is more than 2 degrees from the target's (%)")
    ax.grid(alpha=0.25)
    ax.set_ylim(0, 100)

    caption = (
        "How often the model and the target are no longer describing the same low. "
        f"{100*frac[0]:.0f} per cent of members at {steps[0]:.0f} hours, "
        f"{100*frac[-1]:.0f} per cent at {steps[-1]:.0f} hours, and it passes half the "
        f"members between {steps[np.argmax(frac > 0.5)] if (frac > 0.5).any() else float('nan'):.0f} "
        "hours and the end of the forecast. This is why every "
        "intensity number in this bundle is restricted to at most 72 hours or is shown "
        "with the longer leads shaded: past that point a track error would otherwise be "
        "read as an intensity error. The cut-off coincides with the trained lead range, "
        "which covered 6 to 72 hours.\n"
        + SUPPORT_BOX + " " + CAMPAIGN + " " + ARM_CTRL + "\n"
        f"Sample: {cap['n']} cases in total, {n_per[0]} per lead time (5 dates x 10 members)."
    )
    return finish(fig, "5. Beyond 72 hours the storms are no longer the same storm", caption)


# ---------------------------------------------------------------------------
# Figure 6 - the precipitation ceiling
# ---------------------------------------------------------------------------

def fig06_precip_ceiling(peaks: dict):
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.4),
                            gridspec_kw={"width_ratios": [1.35, 1.0]})
    series = [("downscaler\n(10 members x 100 slices)", peaks["model"], C_MODEL),
              ("interpolated 9 km driver\n(10 members x 100 slices)", peaks["driver"], C_DRIVER),
              ("IEKM 4.4 km target\n(100 slices)", peaks["truth"], C_TRUTH)]
    bins = np.linspace(0, max(np.max(s[1]) for s in series) * 1.02, 60)
    for name, values, colour in series:
        axs[0].hist(values, bins=bins, density=True, histtype="step", lw=2.0,
                    color=colour, label=name)
    axs[0].set_xlabel("largest six-hour precipitation total anywhere on the globe (mm)")
    axs[0].set_ylabel("probability density")
    axs[0].legend(fontsize=8.5)
    axs[0].grid(alpha=0.25)
    axs[0].set_title("Distribution of the per-slice peak")

    cvs, meds = [], []
    for name, values, colour in series:
        v = np.asarray(values, dtype=float)
        cvs.append(v.std() / v.mean())
        meds.append(np.median(v))
    axs[1].bar(np.arange(3), cvs, color=[C_MODEL, C_DRIVER, C_TRUTH], alpha=0.85)
    for i, (c, m) in enumerate(zip(cvs, meds)):
        axs[1].text(i, c + 0.008, f"{c:.3f}\nmedian {m:.0f} mm", ha="center", fontsize=9)
    axs[1].set_xticks(np.arange(3))
    axs[1].set_xticklabels(["downscaler", "driver", "target"], fontsize=9)
    axs[1].set_ylabel("coefficient of variation of the peak")
    axs[1].set_ylim(0, max(cvs) * 1.35)
    axs[1].grid(axis="y", alpha=0.25)
    axs[1].set_title("How much the peak varies case to case")

    caption = (
        "The coefficient of variation is the standard deviation divided by the mean, so it "
        "says how much a quantity moves from case to case relative to its own size. The "
        "target's largest six-hour rainfall varies by a factor of about seven across "
        "cases; the model's varies by less than a factor of two, and it does not even "
        f"follow the variation of the driver it can see ({cvs[0]:.3f} against "
        f"{cvs[1]:.3f} for the driver and {cvs[2]:.3f} for the target). A model drawing "
        "correctly from the conditional distribution would vary roughly like the target "
        "does. The near-constant peak is therefore not a placement problem, which would "
        "leave the distribution of peak values intact and only move them around the map.\n"
        + SUPPORT_GLOBAL + " " + CAMPAIGN + " " + ARM_CTRL + "\n"
        f"Sample: {len(peaks['model'])} member-slices for the model and for the driver, "
        f"{len(peaks['truth'])} slices for the target."
    )
    return finish(fig, "6. The model's rainfall peak barely moves", caption)


# ---------------------------------------------------------------------------
# Figure 7 - the precipitation distribution and the sign reversal at the 99th
# ---------------------------------------------------------------------------

def fig07_precip_quantiles(campaign: dict, systematic: dict):
    """Where the three precipitation distributions sit, and what the step changes.

    The left panel is drawn as a RATIO to the target rather than as three raw
    curves. Raw curves span three decades, so a model sitting ten per cent away
    from the target is indistinguishable from one sitting on top of it, and the
    whole point of the panel is exactly that difference.
    """
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.6))

    xs = np.arange(len(campaign["labels"]))
    truth = np.asarray(campaign["truth"], dtype=float)
    for name, key, colour, marker in (("downscaler", "model", C_MODEL, "o"),
                                      ("interpolated 9 km driver", "driver", C_DRIVER, "s")):
        ratio = np.asarray(campaign[key], dtype=float) / truth
        axs[0].plot(xs, ratio, "-" + marker, color=colour, lw=2.0, ms=7, label=name)
        for x, v in zip(xs, ratio):
            axs[0].annotate(f"{v:.2f}", (x, v), textcoords="offset points",
                            xytext=(0, 8 if colour == C_MODEL else -14),
                            ha="center", fontsize=8.5, color=colour)
    axs[0].axhline(1.0, color=C_TRUTH, lw=2.0, ls="--", label="IEKM 4.4 km target")
    axs[0].set_xticks(xs)
    axs[0].set_xticklabels(campaign["labels"], fontsize=9)
    axs[0].set_ylabel("value divided by the target's value")
    axs[0].set_xlabel("position in the distribution")
    axs[0].legend(fontsize=9)
    axs[0].grid(alpha=0.25)
    axs[0].set_title("Each field relative to the target it should match")

    xs2 = np.arange(len(systematic["labels"]))
    width = 0.38
    axs[1].bar(xs2 - width / 2, systematic["required"], width, color="#555555",
               alpha=0.9, label="systematic target minus driver")
    axs[1].bar(xs2 + width / 2, systematic["delivered"], width, color=C_MODEL,
               alpha=0.9, label="model minus its own driver")
    axs[1].axhline(0.0, color="#000000", lw=1.0)
    axs[1].set_yscale("symlog", linthresh=1.0)
    neg = [i for i, v in enumerate(systematic["required"]) if v < 0]
    for i in neg:
        axs[1].axvspan(i - 0.5, i + 0.5, color="#9ecae1", alpha=0.30, zorder=0)
    for x, v in zip(xs2, systematic["required"]):
        axs[1].annotate(f"{v:+.2f}", (x - width / 2, v), textcoords="offset points",
                        xytext=(0, 5 if v >= 0 else -12), ha="center", fontsize=8.5)
    for x, v in zip(xs2, systematic["delivered"]):
        axs[1].annotate(f"{v:+.2f}", (x + width / 2, v), textcoords="offset points",
                        xytext=(0, 5 if v >= 0 else -12), ha="center", fontsize=8.5,
                        color=C_MODEL)
    if neg:
        axs[1].set_title("Shaded: the target is LIGHTER than its driver here")
    else:
        axs[1].set_title("What the step from 9 km to 4.4 km changes")
    axs[1].set_xticks(xs2)
    axs[1].set_xticklabels(systematic["labels"], fontsize=9)
    axs[1].set_ylabel("increment (mm), symmetric-log scale")
    axs[1].set_xlabel("position in the distribution")
    axs[1].legend(fontsize=9, loc="lower left")
    axs[1].grid(axis="y", alpha=0.25)
    top = max(max(systematic["required"]), max(systematic["delivered"]))
    axs[1].set_ylim(top=top * 4.0)

    caption = (
        "Left: the model and its interpolated driver, each divided by the target at the "
        "same place in the precipitation distribution, so the flat line at one is the "
        "target. The model tracks its driver closely everywhere and the two fall short "
        "together at the peak. Right: the systematic increment between the two training "
        "datasets beside the increment the model actually applies, on a scale that is "
        "linear near zero and logarithmic beyond, so a small negative bar and a large "
        "positive one are both visible. Going from 9 km to 4.4 km makes the moderate-heavy "
        "rain LIGHTER, the shaded 99th-percentile column, while making the rare peak far "
        "heavier. The model reproduces the light direction, including its sign, and almost "
        "none of the heavy one.\n"
        "TWO SUPPORTS, and the comparison is only as good as that. The left panel and the "
        "delivered bars are the Humberto campaign on the complete global O2560 grid. The "
        "required bars are the mean difference between the paired training datasets over "
        "the global tropical belt. Both sides are means over their own samples, so the two "
        "kinds of average match.\n"
        + CAMPAIGN + " " + ARM_CTRL + f" Sample: {campaign['n_member_slices']} member-slices "
        f"for the model and driver, {campaign['n_slices']} slices for the target; "
        f"{systematic['n_scan']} paired times for the systematic bars."
    )
    return finish(fig, "7. The precipitation distribution, including the reversal at the 99th percentile", caption)


# ---------------------------------------------------------------------------
# Figure 8 - precipitation skill against lead time
# ---------------------------------------------------------------------------

def fig08_precip_skill(per_step: dict):
    steps = np.asarray(per_step["steps"], dtype=float)
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.2))
    axs[0].plot(steps, per_step["model_rmse"], "-o", color=C_MODEL, lw=2.0, ms=4,
                label="downscaler")
    axs[0].plot(steps, per_step["baseline_rmse"], "-s", color=C_DRIVER, lw=2.0, ms=4,
                label="interpolated 9 km driver")
    axs[0].set_ylabel("root-mean-square error against the target (mm per 6 h)")
    axs[0].set_title("Pointwise error, lower is better")
    axs[1].plot(steps, per_step["model_corr"], "-o", color=C_MODEL, lw=2.0, ms=4,
                label="downscaler")
    axs[1].plot(steps, per_step["baseline_corr"], "-s", color=C_DRIVER, lw=2.0, ms=4,
                label="interpolated 9 km driver")
    axs[1].set_ylabel("correlation with the target")
    axs[1].set_title("Pattern agreement, higher is better")
    for ax in axs:
        ax.axvline(TRAINED_LEAD_MAX, color="#555555", ls="--", lw=1.0)
        ax.axvspan(TRAINED_LEAD_MAX, steps.max(), color="#bbbbbb", alpha=0.22, zorder=0)
        ax.set_xlabel("lead time (h)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)

    caption = (
        "The interpolated driver is the baseline any downscaler has to beat, because it is "
        "what you get for free by putting the 9 km field on the 4.4 km grid. The model "
        "never beats it at any lead time, on either measure, and the two converge as the "
        "forecast ages. A sharp feature placed slightly wrongly is penalised twice by a "
        "pointwise error, once for being absent where the target has it and once for being "
        "present where the target does not, so this figure alone does not separate a model "
        "that adds nothing from one that adds detail in the wrong place. It is figures 6, "
        "13 and 14 that make that separation.\n"
        + SUPPORT_GLOBAL + " " + CAMPAIGN + " " + ARM_CTRL + "\n"
        f"Sample: {per_step['n_slices']} slices, each averaged over 10 members; "
        f"{len(steps)} lead times. Leads beyond {TRAINED_LEAD_MAX} h are shaded because "
        "they are outside the trained range."
    )
    return finish(fig, "8. Precipitation skill against lead time, model and baseline", caption)


# ---------------------------------------------------------------------------
# Figure 9 - the loss budget
# ---------------------------------------------------------------------------

def fig09_loss_budget(budget: dict):
    thr = np.asarray(budget["thresholds_mm"], dtype=float)
    sse = 100 * np.asarray(budget["sse_share_above"], dtype=float)
    pts = 100 * np.asarray(budget["point_share_above"], dtype=float)

    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    ax.plot(thr, sse, "-o", color=C_MODEL, lw=2.2, ms=6,
            label="share of the squared-error budget")
    ax.plot(thr, pts, "-s", color=C_DRIVER, lw=2.2, ms=6,
            label="share of the grid points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("threshold on the target's six-hour precipitation (mm)")
    ax.set_ylabel("share of the total (%)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=9.5)
    ax.set_xlim(right=float(thr.max()) * 6.0)
    for t in (30.0, 100.0):
        if t in thr:
            k = int(np.where(thr == t)[0][0])
            ax.annotate(f"above {t:.0f} mm: {pts[k]:.3f}% of the grid\n"
                        f"carries {sse[k]:.1f}% of the squared error",
                        (t, sse[k]), textcoords="offset points", xytext=(12, 10),
                        fontsize=9, color="#333333",
                        arrowprops=dict(arrowstyle="-", color="#888888", lw=0.8))

    caption = (
        "This figure removes one candidate explanation. If the training objective simply "
        "could not see the heavy tail, there would be no gradient pressure to reproduce it "
        "and the flat ceiling would follow. It can see it: a tiny fraction of the grid "
        "carries a large share of the total squared error for precipitation, so the tail "
        "dominates rather than disappears in the objective. Both axes are logarithmic, and "
        "the two curves are shares of the same totals, so their vertical distance is how "
        "much more heavily the tail counts than its area would suggest.\n"
        + SUPPORT_GLOBAL + " " + CAMPAIGN + " " + ARM_CTRL + "\n"
        f"Sample: {budget['n_slices']} slices, member index {budget['member_index']}, "
        f"{budget['n_points_total']:.3g} grid points in total. The threshold is applied to "
        "the TARGET's value at each point, and the squared error is between the model and "
        "the target at that same point."
    )
    return finish(fig, "9. The heavy tail is a large part of the precipitation loss", caption)


# ---------------------------------------------------------------------------
# Figure 10 - sampler arms
# ---------------------------------------------------------------------------

def fig10_sampler_arms(arms: dict, order: list[str]):
    present = [k for k in order if arms.get(k, {}).get("n_member_slices", 0) > 0]
    missing = [k for k in order if k not in present]
    fig, ax = plt.subplots(figsize=(12.5, 6.0))
    data = [np.asarray(arms[k]["peaks_mm"], dtype=float) for k in present]
    parts = ax.violinplot(data, positions=np.arange(len(present)), widths=0.8,
                          showmedians=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(C_MODEL)
        body.set_alpha(0.35)
    parts["cmedians"].set_color(C_TRUTH)

    truth = None
    for k in present:
        if arms[k].get("truth_peaks_mm"):
            truth = np.asarray(arms[k]["truth_peaks_mm"], dtype=float)
            break
    if truth is not None:
        ax.axhline(float(np.median(truth)), color=C_TRUTH, ls="--", lw=1.4)
        ax.text(len(present) - 0.5, float(np.median(truth)) * 1.02,
                f"median target peak in these boxes, {np.median(truth):.0f} mm",
                ha="right", fontsize=9, color="#333333")

    labels, medians, cvs = [], [], []
    for i, k in enumerate(present):
        v = np.asarray(arms[k]["peaks_mm"], dtype=float)
        medians.append(float(np.median(v)))
        cvs.append(float(v.std() / v.mean()))
        ax.text(i, v.max() * 1.02, f"cv {cvs[-1]:.3f}\nmedian {medians[-1]:.0f} mm\nn={v.size}",
                ha="center", fontsize=8.5)
        sm = arms[k]["sampler"]
        labels.append(f"{k}\nsteps {sm.get('num_steps')}, ceiling {sm.get('sigma_max'):g}\n"
                      f"churn {sm.get('S_churn')}")
    ax.set_xticks(np.arange(len(present)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("box peak six-hour precipitation (mm)")
    ax.set_ylim(top=max(np.max(d) for d in data) * 1.22)
    ax.grid(axis="y", alpha=0.25)

    # the two arms that share a configuration measure the run-to-run scatter
    repeats = [i for i, k in enumerate(present) if k.startswith("ceiling 1e6")]
    scatter_line = ""
    if len(repeats) == 2:
        a, b = medians[repeats[0]], medians[repeats[1]]
        scatter_line = (
            f"Two arms carry IDENTICAL settings and differ only in being separate runs; "
            f"their medians are {a:.0f} and {b:.0f} mm, so about {100*abs(a-b)/((a+b)/2):.0f} "
            "per cent is run-to-run scatter and any smaller difference means nothing. ")
    base = medians[0]
    effects = "; ".join(f"{k}: median {m:.0f} mm ({100*(m-base)/base:+.0f} per cent), "
                        f"coefficient of variation {c:.3f}"
                        for k, m, c in zip(present, medians, cvs))

    caption = (
        "All arms are the SAME checkpoint on the SAME cases, so the sampler is the only "
        "thing that differs and any change here is the sampler's doing. Widening the noise "
        "ceiling by a factor of a thousand barely moves the peak. Raising the churn, which "
        "is how much extra noise is injected and then removed again at each sampling step, "
        "does lift the field, but it lifts the WHOLE distribution while REDUCING the "
        "coefficient of variation, which is the signature of a raised roughness floor "
        "rather than of rare intense cells appearing. That places the limit in the trained "
        "weights rather than in how they are sampled. " + effects + ". " + scatter_line + "\n"
        + SUPPORT_BOXLANE + " Checkpoint fccc23df step 300,000 throughout. These arms "
        "belong to a sampler screen run by another session and were read without "
        "modification.\n"
        "Sample sizes are annotated on each arm and are NOT pooled, because they differ. "
        + (f"NOT SHOWN because no prediction files existed when this bundle was built: "
           f"{', '.join(missing)}. " if missing else "")
        + ("The boxed predictions carry no target precipitation field, so there is no "
           "target line on this figure; for how far the model's peak sits below the "
           "target's, see figures 6 and 14 on the global support."
           if truth is None else "")
    )
    return finish(fig, "10. No sampler setting moves the ceiling", caption,
                  left_inches=1.35)


# ---------------------------------------------------------------------------
# Figure 11 - cross-lane pair coherence
# ---------------------------------------------------------------------------

def fig11_pair_coherence(lanes: dict):
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.4))
    colours = {"o1280_o2560": C_MODEL, "o320_o1280": C_DRIVER}
    for name, rows in lanes.items():
        steps = sorted({r["step"] for r in rows})
        ri = [float(np.mean([r["rmse_interp"] for r in rows if r["step"] == s])) for s in steps]
        rm = [float(np.mean([r["rmse_model"] for r in rows if r["step"] == s])) for s in steps]
        colour = colours.get(name, "#7f7f7f")
        axs[0].plot(steps, ri, "-o", color=colour, lw=2.0, ms=5,
                    label=f"{name}: interpolated input against target")
        axs[0].plot(steps, rm, "--s", color=colour, lw=1.6, ms=4, alpha=0.7,
                    label=f"{name}: model against target")
        gain = [100 * (a - b) / a for a, b in zip(ri, rm)]
        axs[1].plot(steps, gain, "-o", color=colour, lw=2.0, ms=5, label=name)
    axs[0].set_ylabel("sea-level pressure error (Pa)")
    axs[0].set_title("How far the interpolated input already is from the target")
    axs[1].axhline(0.0, color="#000000", lw=0.8)
    axs[1].set_ylabel("error removed (% of the input's)")
    axs[1].set_title("What the downscaler adds on top")
    for ax in axs:
        ax.set_xlabel("lead time (h)")
        ax.legend(fontsize=8.5)
        ax.grid(alpha=0.25)

    caption = (
        "The obvious explanation for this lane's small gains was that it pairs two "
        "different integrations, an operational 9 km input against a separate 4.4 km "
        "target, while the coarse lane pairs two members of one system, so the 4.4 km pair "
        "would be looser and less learnable. Measured, that is false: the 9 km input is "
        "ALREADY CLOSER to its 4.4 km target than the 31 km input is to its 9 km target, at "
        "every lead time. What separates the lanes is the size of the prize. At 31 km a "
        "cyclone core is unresolved and 9 km is where an eyewall first becomes "
        "representable, so the coarse lane has a large systematic gap to close. At 9 km "
        "that threshold is already behind us.\n"
        "ONE SUPPORT PER CURVE AND THEY ARE NOT THE SAME SUPPORT: each lane is scored on "
        "its OWN target grid, against its own target, so the two lanes' absolute values "
        "are not comparable point for point; what is comparable is the shape and the "
        "ordering within each lane, and the right-hand panel, which is each lane's gain "
        "relative to its own baseline.\n"
        "Global field, sea-level pressure, one ensemble member per lane. Sample sizes: "
        + "; ".join(f"{k}: {len(v)} fields" for k, v in lanes.items()) + "."
    )
    return finish(fig, "11. The 9 km to 4.4 km pair is tighter, not looser", caption)


# ---------------------------------------------------------------------------
# Figure 13 - rank histograms
# ---------------------------------------------------------------------------

def fig13_rank_histograms(ranks: dict, n_members: int = 10):
    n_ranks = n_members + 1
    fig = plt.figure(figsize=(13.5, 8.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.5, 1.0], hspace=0.62, wspace=0.32)
    ax = fig.add_subplot(gs[0, :])
    hist = np.asarray(ranks["max_mm"]["histogram"], dtype=float)
    n = hist.sum()
    expected = n / n_ranks
    bars = ax.bar(np.arange(1, n_ranks + 1), hist, color=C_DRIVER, alpha=0.8)
    bars[-1].set_color(C_MODEL)
    bars[0].set_color("#9467bd")
    ax.axhline(expected, color=C_TRUTH, ls="--", lw=1.4)
    ax.text(1.2, expected * 1.12, f"what a calibrated ensemble would give, {expected:.1f} per bin",
            fontsize=9, color="#333333")
    ax.annotate(f"{hist[-1]/n*100:.0f}% of slices: the target's peak is above ALL ten members\n"
                f"(a calibrated ensemble gives {100/n_ranks:.0f}%)",
                (n_ranks, hist[-1]), textcoords="offset points", xytext=(-14, -30),
                ha="right", fontsize=10, color="#8b1a1a", fontweight="bold")
    ax.set_xticks(np.arange(1, n_ranks + 1))
    ax.set_xlabel(f"rank of the target among the {n_members} members and itself "
                  f"(1 = below every member, {n_ranks} = above every member)")
    ax.set_ylabel("number of slices")
    ax.set_title("Where the target's precipitation PEAK falls inside the ensemble", fontsize=11)
    ax.grid(axis="y", alpha=0.25)

    small = [("99.9th percentile", "p999_mm"), ("99th percentile", "p99_mm"),
             ("wet fraction", "wet_frac"), ("grid mean", "mean_mm")]
    for k, (title, key) in enumerate(small):
        axk = fig.add_subplot(gs[1, k])
        h = np.asarray(ranks[key]["histogram"], dtype=float)
        axk.bar(np.arange(1, n_ranks + 1), h, color="#7f7f7f", alpha=0.85)
        axk.axhline(h.sum() / n_ranks, color=C_TRUTH, ls="--", lw=1.1)
        axk.set_title(f"{title}\nmean rank {ranks[key]['mean_rank']:.2f}", fontsize=9.5)
        axk.set_xticks([1, n_ranks])
        axk.set_xticklabels(["below all", "above all"], fontsize=8)
        axk.grid(axis="y", alpha=0.2)
        if k == 0:
            axk.set_ylabel("slices")

    caption = (
        "A diffusion model is trained to DRAW from the distribution of the target, so if "
        "the target is a fair extra member of its own ensemble, its rank among the eleven "
        "values is equally likely to be anywhere, its mean rank is 6.0, and it sits above "
        f"all ten members in {100/n_ranks:.0f} per cent of slices. For the peak it sits "
        f"above all ten in {hist[-1]/n*100:.0f} per cent, and it falls outside the "
        f"ensemble's range altogether in {(hist[0]+hist[-1])/n*100:.0f} per cent of slices "
        f"against an expected {200/n_ranks:.0f}. That is an ensemble too narrow and biased "
        "low at the peak. The small panels show the same test for the bulk of the "
        "distribution: the model is biased HIGH there, the target ranking near the bottom, "
        "while the grid mean is close to calibrated. The model lays down roughly the right "
        "total water and distributes it wrongly.\n"
        "This figure is the reason the earlier explanation, that the placement of an "
        "intense cell is unpredictable so the model learns not to commit, was withdrawn. "
        "Unpredictability would show as DIVERSITY ACROSS MEMBERS, each member carrying a "
        "realistic peak somewhere plausible. A suppressed tail in every member is a defect "
        "of the learned distribution, not a limit set by information.\n"
        + SUPPORT_GLOBAL + " " + CAMPAIGN + " " + ARM_CTRL +
        f" Sample: {int(n)} slices, each compared against its own 10 members."
    )
    return finish(fig, "13. The target's rainfall peak falls outside the ensemble", caption)


# ---------------------------------------------------------------------------
# Figure 14 - ensemble spread collapse
# ---------------------------------------------------------------------------

def fig14_spread_collapse(spread: dict):
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.8),
                            gridspec_kw={"width_ratios": [1.6, 1.0]})
    order = np.argsort(spread["truth"])
    x = np.arange(len(order))
    for key, colour, name in (("driver", C_DRIVER, "interpolated 9 km driver"),
                              ("model", C_MODEL, "downscaler")):
        lo = np.asarray(spread[f"{key}_min"])[order]
        hi = np.asarray(spread[f"{key}_max"])[order]
        med = np.asarray(spread[f"{key}_med"])[order]
        axs[0].fill_between(x, lo, hi, color=colour, alpha=0.30, linewidth=0, label=f"{name}, member range")
        axs[0].plot(x, med, color=colour, lw=1.4)
    axs[0].plot(x, np.asarray(spread["truth"])[order], color=C_TRUTH, lw=2.0,
                label="IEKM 4.4 km target")
    axs[0].set_xlabel("the 100 slices, ordered by the target's own peak")
    axs[0].set_ylabel("global peak six-hour precipitation (mm)")
    axs[0].legend(fontsize=9)
    axs[0].grid(alpha=0.25)
    axs[0].set_title("Ten members in, and how wide they come out")

    sd_model = np.asarray(spread["model_sd"], dtype=float)
    sd_driver = np.asarray(spread["driver_sd"], dtype=float)
    axs[1].bar([0, 1], [sd_driver.mean(), sd_model.mean()],
               color=[C_DRIVER, C_MODEL], alpha=0.85)
    for i, v in enumerate([sd_driver.mean(), sd_model.mean()]):
        axs[1].text(i, v + 1.2, f"{v:.1f} mm", ha="center", fontsize=11, fontweight="bold")
    ratio = sd_driver.mean() / sd_model.mean()
    axs[1].set_xticks([0, 1])
    axs[1].set_xticklabels(["driver ensemble\n(what went in)",
                            "downscaler ensemble\n(what came out)"], fontsize=9.5)
    axs[1].set_ylabel("spread of the ten peaks within a slice (mm)")
    axs[1].set_title(f"The ensemble comes out {ratio:.1f} times narrower")
    axs[1].grid(axis="y", alpha=0.25)

    caption = (
        "For each slice the ten driving members are ten genuinely different weather states, "
        "and the model sees them one at a time. The band is the range of the ten peak "
        "values; the spread on the right is the mean over slices of the standard deviation "
        "of those ten values within a slice. The driver's peaks scatter widely from member "
        f"to member, {sd_driver.mean():.1f} mm on average. The model's come out at "
        f"{sd_model.mean():.1f} mm, so the ensemble is {ratio:.1f} times NARROWER at the "
        "extreme peak than the one it was conditioned on, while the target sits above it. "
        "The model is not passing through the diversity it was given, let alone adding the "
        "extra diversity that going to 4.4 km should produce.\n"
        "Keep the two questions apart. Whether the model can put an intense cell in the "
        "right place is bounded by the information in a 9 km driver and is not what this "
        "figure measures. Whether its ensemble has the right distribution of peak values is "
        "not bounded by that information, and that is what is failing here.\n"
        + SUPPORT_GLOBAL + " " + CAMPAIGN + " " + ARM_CTRL +
        f" Sample: {len(spread['truth'])} slices x 10 members for both ensembles."
    )
    return finish(fig, "14. The ensemble collapses the spread it was given", caption)
