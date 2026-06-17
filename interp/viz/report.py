"""Per-checkpoint interp report — ONE PDF from the run dir's JSONs.

Layout: ONE folder per checkpoint. Tool JSONs live either directly under the
run dir (<run_dir>/<tool_subdir>/<tool>.json — scene-agnostic diagnostics) or
inside CASE-STUDY subdirs (<run_dir>/<case>/<tool_subdir>/<tool>.json, e.g.
humberto/, amazon/, png_precip/). Feature-permutation puts the case studies
side by side (heatmap columns); integrated gradients renders ONE self-contained
page per case (driver maps + self-pinned top-driver bars + that case's
locality-vs-sigma panel); the remaining diagnostics (ablation, activation
norms, CKA) are global and rendered once.

Writes <run_dir>/plots/report.pdf plus <run_dir>/plots/methods.md (the method
descriptions live in the markdown next to the PDF, NOT as pages inside it).
--per-tool also writes one PDF per tool under the legacy names.

Usage
-----
    python -m interp.viz --run-dir ~/perm/interp/85884ee7_189k
    python -m interp.viz --run-dir ... --tools cka,activation_patching --per-tool
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm, PowerNorm, TwoSlopeNorm

from interp.viz.panels import (PRECIP_CMAP, SEQ_CMAP, SEQ_CMAP2, fig_grid,
                               geo_panel, heatmap, loglog, ranked_barh)

LOGGER = logging.getLogger("interp.viz")

SURFACE_TARGETS = ["10u", "10v", "2t", "msl", "tp"]

LRES_C, HRES_C, BOTH_C = "#1f77b4", "#ff7f0e", "#2ca02c"


def _title(ckpt_id, *parts):
    return f"{ckpt_id} · " + " · ".join(str(p) for p in parts)


def _targets_of(d: dict) -> list[str]:
    return [t for t in SURFACE_TARGETS if t in d]


def _pretty_case(name) -> str:
    """Display name for a case-study dir (underscores -> spaces)."""
    return str(name).replace("_", " ") if name else ""


def _pretty_region(spec: str) -> str:
    """Human-readable label for a region spec used in titles.

    'bbox:15,32,-70,-50' -> 'box [15-32N, 70-50W]'; named regions title-cased.
    """
    s = str(spec)
    if s.startswith("bbox:"):
        try:
            la0, la1, lo0, lo1 = (float(x) for x in s[len("bbox:"):].split(","))
        except Exception:
            return s
        def _lat(v):
            return f"{abs(v):g}{'N' if v >= 0 else 'S'}"
        def _lon(v):
            return f"{abs(v):g}{'E' if v >= 0 else 'W'}"
        box = f"[{_lat(la0)}-{_lat(la1)}, {_lon(lo0)}-{_lon(lo1)}]"
        # Name the known case boxes so titles read in plain English.
        if la0 >= 10 and la1 <= 35 and lo0 >= -75 and lo1 <= -45:
            return f"N. Atlantic TC box {box}"
        return f"box {box}"
    return s.replace("_", " ")


# ---------------------------------------------------------------------------
# feature permutation (per-sigma): one heatmap grid, targets x region columns
# ---------------------------------------------------------------------------

def _perm_sources(first, which):
    """(names, [(row_offset, sigma_result_key), ...], group_label) for a
    permutation view, or None if that view's data is absent. 'input' = the
    in_lres + in_hres pathways; 'noisy' = the noised-target (noisy_hres) one."""
    if which == "noisy":
        ni = first.get("noisy_importance")
        if not ni:
            return None
        names = [i["name"] for _, i in sorted(ni.items(), key=lambda kv: int(kv[0]))]
        return names, [(0, "noisy_importance")], "noisy_hres (noised-target channels)"
    ln = [i["name"] for _, i in sorted(first["lres_importance"].items(),
                                       key=lambda kv: int(kv[0]))]
    hn = [i["name"] for _, i in sorted(first["hres_importance"].items(),
                                       key=lambda kv: int(kv[0]))]
    return ln + hn, [(0, "lres_importance"), (len(ln), "hres_importance")], "in_lres + in_hres"


def _perm_order(score, all_names, target, which, top_k):
    """Row order for a permutation heatmap: top-`top_k` by score, with forcing
    channels dropped from the noisy view and the target's OWN variable always
    pinned in (item 8). Returns (order, self_idx)."""
    drop = {i for i, n in enumerate(all_names) if which == "noisy" and _is_forcing(n)}
    cand = [int(i) for i in np.argsort(-score) if i not in drop]
    order = cand[:top_k]
    self_idx = next((i for i in cand if all_names[i] == target), None)
    if self_idx is not None and self_idx not in order:
        order = order[:max(0, top_k - 1)] + [self_idx]
    return order, self_idx


def _render_permutation_one(data, ckpt_id, which):
    sigmas = [r["sigma"] for r in data["sigma_results"]]
    first = data["sigma_results"][0]
    src = _perm_sources(first, which)
    if src is None:
        return None
    all_names, src_specs, group_label = src
    targets = first.get("surface_targets") or _targets_of(first["baseline_mse_per_target"])
    probe = first[src_specs[0][1]]
    any_entry = probe.get("0") or next(iter(probe.values()))
    regions = [r for r in (data.get("regions") or [])
               if any_entry.get("region_paired_mse_per_target", {}).get(r)]
    has_extreme = bool(any_entry.get("extreme_paired_mse_per_target"))

    def matrix(target, key, region):
        M = np.zeros((len(all_names), len(sigmas)))
        for j, res in enumerate(data["sigma_results"]):
            for off, src_key in src_specs:
                for idx_str, info in res.get(src_key, {}).items():
                    row = off + int(idx_str)
                    if key == "global":
                        M[row, j] = info["paired_mse_per_target"].get(target, 0.0)
                    else:
                        M[row, j] = info.get(key, {}).get(region, {}).get(target, 0.0)
        return M

    columns = []  # (label, key, region)
    for r in regions[:2]:
        columns.append((r, "region_paired_mse_per_target", r))
        if has_extreme:
            columns.append((f"{r} tail p{data.get('extreme_percentile'):g}",
                            "extreme_paired_mse_per_target", r))
    if not columns:
        columns = [("global", "global", None)]
    columns = columns[:4]

    top_k = 15
    n_t = len(targets)
    fig, axes = plt.subplots(n_t, len(columns),
                             figsize=(5.5 * len(columns), 3.8 * n_t), squeeze=False)
    for row, target in enumerate(targets):
        mats = {lbl: matrix(target, key, region) for lbl, key, region in columns}
        score = np.zeros(len(all_names))
        for M in mats.values():
            score = np.maximum(score, M.max(axis=1))
        order, self_idx = _perm_order(score, all_names, target, which, top_k)
        names_top = [all_names[i] + (" (self)" if i == self_idx else "") for i in order]
        for c, (lbl, _, _) in enumerate(columns):
            M = mats[lbl][order]
            pos = M[M > 0]
            vmin = max(pos.min() if pos.size else 1e-10, 1e-10)
            vmax = max(float(M.max()), vmin * 10)
            heatmap(axes[row][c], M, [f"{s:g}" for s in sigmas], names_top,
                    title=f"output {target} · scored over {_pretty_region(lbl)}",
                    cmap=SEQ_CMAP, norm=LogNorm(vmin=vmin, vmax=vmax),
                    cbar_label="paired MSE (log)", fig=fig)
            axes[row][c].set_xlabel("sigma")
    fig.suptitle(_title(ckpt_id, f"feature permutation · {group_label} · paired MSE, top-15"),
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return fig


def render_permutation(data: dict, ckpt_id: str) -> list:
    """Per-sigma permutation heatmaps: the in_lres+in_hres view, then (when the
    noisy-target channels were permuted) a separate noisy_hres view (item 8)."""
    return [f for f in (_render_permutation_one(data, ckpt_id, w)
                        for w in ("input", "noisy")) if f is not None]


# ---------------------------------------------------------------------------
# feature permutation (full sampling): top-K bars per target
# ---------------------------------------------------------------------------

def render_full_sampling(data: dict, ckpt_id: str) -> list:
    result = data["result"]
    targets = result.get("surface_targets") or _targets_of(result["baseline_mse_per_target"])
    items = []
    for group in ("lres", "hres"):
        for info in result[f"{group}_importance"].values():
            items.append((info["name"], group, info["paired_mse_per_target"],
                          info.get("paired_mse_per_target_std", {})))

    fig, axes = fig_grid(len(targets), cols=2, panel_w=8, panel_h=6)
    for ax, target in zip(axes, targets):
        ranked = sorted(items, key=lambda x: x[2].get(target, 0.0), reverse=True)
        top = ranked[:15]
        floor = float(np.median([t[2][target] for t in ranked[len(ranked) // 2:]]))
        ranked_barh(ax, [t[0] for t in top], [t[2][target] for t in top],
                    colors=[LRES_C if t[1] == "lres" else HRES_C for t in top],
                    stds=[t[3].get(target, 0.0) for t in top],
                    xlabel="paired MSE (full sampling)", title=f"target = {target}",
                    floor=floor)
    fig.suptitle(_title(ckpt_id, "full-sampling permutation",
                        f"num_steps={result['num_steps']} reps={result['n_repeats']}"),
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return [fig]


# ---------------------------------------------------------------------------
# conditioning ablation: per-target curves + correlation summary
# ---------------------------------------------------------------------------

def render_ablation(data: dict, ckpt_id: str) -> list:
    """Grid: rows = targets, columns = REGIONS (one panel per region, 3 curves
    each — readable, no 6-line overlays), plus a global correlation summary."""
    sigmas = [r["sigma"] for r in data["sigma_results"]]
    first = data["sigma_results"][0]
    regions = data.get("regions")
    targets = data.get("surface_targets")
    has_region = bool(regions and targets
                      and first["ablate_lres"].get("region_per_target_mse"))
    if not targets:
        name_to_idx = {"10u": 0, "10v": 1, "2t": 3, "msl": 4}
        per_var_len = len(first["ablate_lres"]["per_var_mse"])
        targets = [t for t, i in name_to_idx.items() if i < per_var_len]
    third = "ablate_noisy_hres" if "ablate_noisy_hres" in first else "ablate_both"
    third_lbl = "zero noisy_hres" if third == "ablate_noisy_hres" else "zero both"
    plot_regions = list(regions)[:3] if has_region else ["global"]

    cols = len(plot_regions)
    rows = len(targets)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.3 * rows),
                             squeeze=False)
    pathways = (("zero in_lres", "ablate_lres", LRES_C, "o"),
                ("zero in_hres", "ablate_hres", HRES_C, "s"),
                (third_lbl, third, BOTH_C, "^"))
    for row, target in enumerate(targets):
        for col, rname in enumerate(plot_regions):
            ax = axes[row][col]
            series, styles = {}, {}
            for label, key, color, marker in pathways:
                if key not in first:
                    continue
                if has_region:
                    series[label] = [r[key]["region_per_target_mse"][rname][target]
                                     for r in data["sigma_results"]]
                else:
                    name_to_idx = {"10u": 0, "10v": 1, "2t": 3, "msl": 4}
                    series[label] = [r[key]["per_var_mse"][name_to_idx[target]]
                                     for r in data["sigma_results"]]
                styles[label] = dict(color=color, marker=marker)
            loglog(ax, sigmas, series, styles=styles,
                   ylabel="MSE(ablated vs full output)",
                   title=f"target {target} · region {_pretty_region(rname)}")

    fig.suptitle(_title(ckpt_id, "conditioning ablation — MSE of zeroing each input "
                                 "pathway, per target × region"), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return [fig]


# ---------------------------------------------------------------------------
# activations: 3 whole-module norm panels (+ per-block l2 heatmap when present)
# ---------------------------------------------------------------------------

def render_activation_norms(data: dict, ckpt_id: str) -> list:
    sigmas_raw = sorted(data["profiles"].keys(), key=float)
    sigmas = [float(s) for s in sigmas_raw]
    modules = ["encoder", "processor", "decoder"]

    block_names = sorted({k for s in sigmas_raw for k in data["profiles"][s]
                          if k.startswith("processor.layer_")})
    n_panels = 3 + (1 if block_names else 0)
    fig, axes = fig_grid(n_panels, cols=2 if block_names else 3,
                         panel_w=7, panel_h=4.5)
    for ax, module in zip(axes[:3], modules):
        series = {stat: [data["profiles"][s].get(module, {}).get(key, np.nan)
                         for s in sigmas_raw]
                  for stat, key in (("L2 norm", "l2_norm"), ("std", "std"),
                                    ("max |·|", "max_abs"))}
        loglog(ax, sigmas, series, ylabel="activation magnitude", title=module)
    if block_names:
        M = np.array([[data["profiles"][s][b]["l2_norm"] for s in sigmas_raw]
                      for b in block_names])
        heatmap(axes[3], M, [f"{s:g}" for s in sigmas],
                [b.replace("processor.", "") for b in block_names],
                title="per-block L2 norm", cmap=SEQ_CMAP2, cbar_label="L2", fig=fig)
        axes[3].set_xlabel("sigma")
    fig.suptitle(_title(ckpt_id, "activation norms (encoder · processor · decoder) — "
                                 "GLOBAL field, not case-restricted"), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return [fig]


def render_cka(data: dict, ckpt_id: str) -> list:
    sigma_keys = [s for s in data["sigmas"] if str(s) in data["cka_matrices"]]
    fig, axes = fig_grid(len(sigma_keys) + 1, cols=3, panel_w=5.5, panel_h=5)
    for ax, sig in zip(axes[:-1], sigma_keys):
        entry = data["cka_matrices"][str(sig)]
        names = entry["layer_names"]
        heatmap(ax, entry["matrix"], names, names, title=f"σ = {sig:g}",
                cmap="RdBu_r", vmin=0.0, vmax=1.0, cbar_label="CKA",
                hlines=([7.5] if len(names) == 16 else []), fig=fig,
                ytick_fontsize=6)
        ax.set_xticklabels(names, rotation=90, fontsize=6)
        if len(names) == 16:
            ax.axvline(7.5, color="black", linewidth=0.6)

    wa, wb, cr = [], [], []
    for sig in sigma_keys:
        m = np.array(data["cka_matrices"][str(sig)]["matrix"])
        if m.shape[0] < 16:
            continue
        ca, cb, cross = m[:8, :8], m[8:, 8:], m[:8, 8:]
        off = ~np.eye(8, dtype=bool)
        wa.append(ca[off].mean())
        wb.append(cb[off].mean())
        cr.append(cross.mean())
    loglog(axes[-1], sigma_keys[:len(wa)],
           {"within chunk-A (L0-7)": wa, "within chunk-B (L8-15)": wb,
            "cross-chunk": cr},
           ylabel="mean CKA", title="chunk summary", logy=False, ylim=(0.5, 1.0))
    fig.suptitle(_title(ckpt_id, "CKA layer similarity (processor blocks) — "
                                 "GLOBAL field, not case-restricted"), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return [fig]


# ---------------------------------------------------------------------------
# activation patching: per-target rows x (residual | grid_region | stage)
# ---------------------------------------------------------------------------

def render_patching(data: dict, ckpt_id: str) -> list:
    targets = data["surface_targets"]
    sigmas = [r["sigma"] for r in data["sigma_results"]]
    block_names = data["block_names"]
    chunk_boundary = data.get("chunk_boundary", len(block_names) // 2)
    stage_names = data.get("stage_names", [])
    storm = data.get("storm", {})
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-0.2, vmax=1.0)
    block_rows = block_names + ["chunk-A", "chunk-B"]

    modes = [m for m in ("residual", "grid_region", "stage")
             if any(m in r for r in data["sigma_results"])]
    fig, axes = plt.subplots(len(targets), len(modes),
                             figsize=(6.4 * len(modes), 3.3 * len(targets)),
                             squeeze=False)

    def block_matrix(mode, t):
        M = np.full((len(block_rows), len(sigmas)), np.nan)
        for j, res in enumerate(data["sigma_results"]):
            d = res.get(mode)
            if not d:
                continue
            for i, name in enumerate(block_names):
                M[i, j] = d["recovery_per_block"][name].get(t, np.nan)
            M[len(block_names), j] = d["recovery_per_chunk"]["A"].get(t, np.nan)
            M[len(block_names) + 1, j] = d["recovery_per_chunk"]["B"].get(t, np.nan)
        return M

    for r, t in enumerate(targets):
        for c, mode in enumerate(modes):
            ax = axes[r][c]
            if mode == "stage":
                M = np.full((len(stage_names), len(sigmas)), np.nan)
                for j, res in enumerate(data["sigma_results"]):
                    sg = res.get("stage")
                    if sg:
                        for i, sn in enumerate(stage_names):
                            M[i, j] = sg["recovery_per_stage"][sn].get(t, np.nan)
                heatmap(ax, M, [f"{s:g}" for s in sigmas], stage_names,
                        title=f"stage · {t}", cmap="RdBu_r", norm=norm,
                        cbar_label="recovery", fig=fig, ytick_fontsize=8)
            else:
                M = block_matrix(mode, t)
                heatmap(ax, M, [f"{s:g}" for s in sigmas], block_rows,
                        title=f"{mode} · {t}", cmap="RdBu_r", norm=norm,
                        cbar_label="recovery", fig=fig, ytick_fontsize=6,
                        hlines=[chunk_boundary - 0.5])
                ax.axhline(len(block_names) - 0.5, color="grey", linewidth=1.2,
                           linestyle="--")
            ax.set_xlabel("sigma")

    storm_txt = "%s @ (%.1f,%.1f) r=%sdeg" % (
        storm.get("label", "?"), storm.get("center_lat", 0.0),
        storm.get("center_lon", 0.0), storm.get("radius_deg", "?"))
    fig.suptitle(_title(ckpt_id, f"activation patching ({data.get('corruption', '?')}) "
                                 f"— {storm_txt}"), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return [fig]


# ---------------------------------------------------------------------------
# integrated gradients: ONE consolidated page per case study (driver maps +
# self-pinned top-driver bars + a full-width locality-vs-sigma panel). The page
# header states the geometry: the scalar is the target averaged over the probe
# DISK (the "circle"), while the attribution itself spans ALL global input
# cells (the maps are only a +/-zoom-deg crop for legibility).
# ---------------------------------------------------------------------------

IG_N_MAPS = 3  # signed driver maps shown per target row (all vars stay in JSON)


def _ig_index(data: dict):
    """(targets, functionals, sigmas, disp_sigma, idx, spatial) for one IG JSON."""
    targets = data["surface_targets"]
    functionals = data["functionals"]
    sigmas = sorted({e["sigma"] for e in data["results"]})
    map_sigma = data.get("map_sigma")
    disp_sigma = map_sigma if (map_sigma in sigmas) else sigmas[-1]
    idx = {(e["functional"], e["target"], e["sigma"]): e for e in data["results"]}
    spatial = [f for f in ("eye", *[f for f in functionals if f.startswith("box:")])
               if f in functionals]
    return targets, functionals, sigmas, disp_sigma, idx, spatial


# Edge colour + name tag per pathway, so the three conditioning groups are
# distinguishable even when they share a variable name (t_1000 in_lres vs the
# noised-target t_1000). 'ntgt' is the JSON key for the noisy_hres pathway.
_GROUP_EDGE = {"lres": "none", "hres": "black", "ntgt": "#2ca02c"}
_GROUP_TAG = {"lres": " [in_lres]", "hres": " [in_hres]", "ntgt": " [noisy_hres]"}
_GROUP_SELF = {"lres": "self in_lres", "ntgt": "self noisy_hres"}


def _is_forcing(name: str) -> bool:
    """Static/time FORCING channels (cos_*, sin_*, lsm, surface z). They sit in
    the output schema as passthrough, so the model does not actually denoise
    them — their 'noisy_hres' attribution is a spurious artifact and is excluded
    from the noisy-target pathway (they remain valid in_lres/in_hres inputs)."""
    return name.startswith("cos_") or name.startswith("sin_") or name in ("lsm", "z")


def _ig_bars_ax(ax, e, target=None, topk=12):
    """One top-drivers bar panel for one IG result entry.

    Merges the three conditioning pathways — in_lres (no edge), in_hres (black
    edge), noisy_hres (green edge) — tagged so a variable appearing in more than
    one pathway (t_1000 [in_lres] vs t_1000 [noisy_hres]) is unambiguous. Fill =
    sign (red +, blue −). Forcing channels are dropped from noisy_hres (they are
    passthrough, not denoised). The target's OWN in_lres input AND its own
    noisy_hres channel are both pinned in, each annotated with its global rank.
    """
    items = []
    for grp in ("lres", "hres", "ntgt"):
        for v in e.get(grp, {}).values():
            if grp == "ntgt" and _is_forcing(v["name"]):
                continue
            items.append((v["name"], grp, v["mean_abs"], v["signed_mean"]))
    if not items:
        ax.axis("off")
        return
    items.sort(key=lambda x: x[2], reverse=True)
    # rank of the target's own in_lres input and own noisy_hres channel
    pins = {}
    for grp in ("lres", "ntgt"):
        r = next((i for i, x in enumerate(items)
                  if target is not None and x[0] == target and x[1] == grp), None)
        if r is not None:
            pins[grp] = r
    top = items[:topk]
    extra = [items[r] for r in pins.values() if r >= topk]
    if extra:
        top = top[:max(0, topk - len(extra))] + extra
    labels = []
    for x in top:
        tag = _GROUP_TAG[x[1]]
        if target is not None and x[0] == target and x[1] in _GROUP_SELF:
            tag = f" ({_GROUP_SELF[x[1]]} #{pins[x[1]] + 1})"
        labels.append(f"{x[0]}{tag}")
    ranked_barh(ax, labels, [x[2] for x in top],
                colors=["#d62728" if x[3] >= 0 else "#1f77b4" for x in top],
                edgecolors=[_GROUP_EDGE[x[1]] for x in top],
                xlabel="mean |attr|")


def _locality_str(e, probe_r):
    """Probe-relative locality (item 3): '% inside the output disk, % inside
    disk + buffer'. Falls back to the fixed 500 km figure for older JSONs."""
    c = e.get("coherence") or {}
    fp, fpb, buf = (c.get("frac_within_probe"),
                    c.get("frac_within_probe_plus_buffer"), c.get("buffer_km"))
    if fp is not None and fpb is not None and probe_r:
        return f"{fp * 100:.0f}% in disk, {fpb * 100:.0f}% in +{buf:g}km"
    f5 = c.get("frac_within_500km")
    return f"{f5 * 100:.0f}% within 500 km" if f5 is not None else "—"


def _case_geom(d: dict):
    """Resolve the spatial-functional geometry shared by every IG page for one
    case: dict(fkey, probe_r, zoom_deg, tgts, idx, sigmas, disp_sigma) or None."""
    targets, _, sigmas, disp_sigma, idx, spatial = _ig_index(d)
    if not spatial:
        return None
    fkey = spatial[0]
    probe_key = fkey.split(":", 1)[1] if fkey.startswith("box:") else fkey
    probe_r = (d.get("boxes", {}).get(probe_key, {}) or {}).get("radius_km")
    tgts = [t for t in targets if (idx.get((fkey, t, disp_sigma)) or {}).get("zoom")]
    return dict(fkey=fkey, probe_r=probe_r, zoom_deg=d.get("zoom_deg", 12.0),
                tgts=tgts, idx=idx, sigmas=sigmas, disp_sigma=disp_sigma)


def _ig_context_page(d: dict, case_label, ckpt_id: str):
    """Overview of the OBSERVED field(s) this case probes (items 12 & 13): you
    SEE the storm / precip we study, with its peak value annotated, so the
    event's significance is obvious before reading any attribution."""
    g = _case_geom(d)
    if not g or not g["tgts"]:
        return None
    fkey, probe_r, idx, ds, tgts = (g["fkey"], g["probe_r"], g["idx"],
                                    g["disp_sigma"], g["tgts"])

    def zoom_of(t):
        return (idx.get((fkey, t, ds)) or {}).get("zoom", {})

    # Candidate context panels in priority order; keep up to 3 available ones.
    # Each: (zoom, vals, title, cmap, cbar, peak_mode, vlo, vhi, gamma). The
    # colour limits (vlo, vhi) ALWAYS span the field's FULL range, and the title
    # quotes those same numbers — so the colourbar and the annotated extreme are
    # consistent by construction and the feature we advertise (the storm low,
    # the precip peak) is always on-scale. gamma<1 applies a PowerNorm (sqrt) so
    # the heavily skewed precip field shows its pattern without clipping its peak.
    candidates = []
    if "tp" in tgts:
        z = zoom_of("tp"); obs = np.asarray(z.get("obs", []), float) * 1000.0  # m->mm
        if obs.size:
            vlo, vhi = 0.0, float(np.nanmax(obs))
            candidates.append((z, obs, f"observed tp — peak {vhi:.0f} mm",
                               PRECIP_CMAP, "tp (mm)", "max", vlo, vhi, 0.5))
    if "msl" in tgts:
        z = zoom_of("msl"); obs = np.asarray(z.get("obs", []), float) / 100.0  # Pa->hPa
        if obs.size:
            vlo, vhi = float(np.nanmin(obs)), float(np.nanmax(obs))
            candidates.append((z, obs, f"observed msl — min {vlo:.0f} hPa",
                               "cmc.lipari", "msl (hPa)", "min", vlo, vhi, 1.0))
    if "10u" in tgts and "10v" in tgts:
        zu, zv = zoom_of("10u"), zoom_of("10v")
        ou, ov = np.asarray(zu.get("obs", []), float), np.asarray(zv.get("obs", []), float)
        if ou.size and ou.shape == ov.shape:
            spd = np.sqrt(ou ** 2 + ov ** 2)
            vhi = float(np.nanmax(spd))
            candidates.append((zu, spd, f"observed 10 m wind — max {vhi:.0f} m/s",
                               "cmc.batlow", "wind (m/s)", "max", 0.0, vhi, 1.0))
    if "2t" in tgts:
        z = zoom_of("2t"); obs = np.asarray(z.get("obs", []), float) - 273.15  # K->°C
        if obs.size:
            vlo, vhi = float(np.nanmin(obs)), float(np.nanmax(obs))
            candidates.append((z, obs, f"observed 2t — {vlo:.0f}..{vhi:.0f} °C",
                               "cmc.lajolla", "2t (°C)", None, vlo, vhi, 1.0))
    panels = candidates[:3]
    if not panels:
        return None

    fig = plt.figure(figsize=(max(8.0, 5.6 * len(panels)), 5.2))
    gs = fig.add_gridspec(1, len(panels))
    for j, (z, vals, title, cmap, cbl, peak_mode, vlo, vhi, gamma) in enumerate(panels):
        plat, plon = np.asarray(z["lat"], float), np.asarray(z["lon"], float)
        peak = None
        if peak_mode is not None and vals.size:
            pidx = int(np.nanargmax(vals) if peak_mode == "max" else np.nanargmin(vals))
            peak = (float(plat[pidx]), float(plon[pidx]))
        if vhi <= vlo:
            vhi = vlo + 1e-9
        norm = PowerNorm(gamma, vmin=vlo, vmax=vhi) if gamma != 1.0 else None
        geo_panel(fig, gs[0, j], plat, plon, vals, title=title, diverging=False,
                  cmap=cmap, cbar_label=cbl, vmin=vlo, vmax=vhi, norm=norm,
                  center=(z.get("center_lat"), z.get("center_lon")),
                  probe_radius_km=probe_r, peak=peak)
    head = " · ".join(str(p) for p in (_pretty_case(case_label), fkey) if p)
    fig.suptitle(_title(ckpt_id, f"case overview · {head}  —  observed field(s) probed")
                 + "\n★ probe centre · ring = output disk · ✕ = field peak", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def _ig_maps_page(d: dict, case_label, ckpt_id: str, sigma):
    """Driver maps + self-pinned bars for ONE sigma (rows = surface targets)."""
    g = _case_geom(d)
    if not g:
        return None
    fkey, probe_r, zoom_deg, idx = g["fkey"], g["probe_r"], g["zoom_deg"], g["idx"]
    tgts = [t for t in g["tgts"] if (idx.get((fkey, t, sigma)) or {}).get("zoom")]
    if not tgts:
        return None
    nrows, ncols = len(tgts), IG_N_MAPS + 1
    fig = plt.figure(figsize=(4.7 * ncols, 4.0 * nrows))
    gs = fig.add_gridspec(nrows, ncols)
    for r, t in enumerate(tgts):
        e = idx[(fkey, t, sigma)]
        z = e["zoom"]
        center = (z.get("center_lat"), z.get("center_lon"))
        lat, lon = np.asarray(z.get("lat", [])), np.asarray(z.get("lon", []))
        varmaps = list(z.get("vars", {}).items())[:IG_N_MAPS]
        for j in range(IG_N_MAPS):
            if j < len(varmaps) and lat.size:
                vname, vals = varmaps[j]
                geo_panel(fig, gs[r, j], lat, lon, np.asarray(vals),
                          title=f"{vname} → {t}  (signed)", diverging=True,
                          center=center, probe_radius_km=probe_r, obs=z.get("obs"))
            else:
                fig.add_subplot(gs[r, j]).axis("off")
        axb = fig.add_subplot(gs[r, IG_N_MAPS])
        _ig_bars_ax(axb, e, target=t)
        axb.set_title(f"{t} drivers · {_locality_str(e, probe_r)}", fontsize=8)
    head = " · ".join(str(p) for p in (_pretty_case(case_label), fkey) if p)
    rtxt = f" R={probe_r:g} km" if probe_r else ""
    fig.suptitle(
        _title(ckpt_id, f"integrated gradients · {head} · σ={sigma:g}") + "\n"
        + f"output 'circle' = mean of target over the {fkey} output disk{rtxt}; "
          f"attribution over ALL global input cells (maps cropped ±{zoom_deg:g}°, "
          f"contours = observed target). bars: in_lres / in_hres [black] / noisy_hres [green], "
          f"red + blue −, self pinned.",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


# Which (pathway, variable) radial profiles the locality page shows. Each is a
# panel; each panel draws one curve per sigma. Aggregated over surface targets.
IG_LOCALITY_VARS = [("lres", "2t"), ("ntgt", "2t"), ("hres", "lsm")]


def _ig_locality_page(d: dict, case_label, ckpt_id: str):
    """Radial-locality page (item 5): for each of a few input variables, the
    cumulative fraction of its (target-aggregated) |attribution| within X km of
    the probe centre, as a function of X — one curve per sigma. Reveals how
    LOCAL the model's use of that input is, and how it sharpens/spreads with
    noise. Storm-relative + averaged over members, so the moving TC centre does
    not blur it."""
    g = _case_geom(d)
    if not g:
        return None
    fkey, probe_r = g["fkey"], g["probe_r"]
    radial = [r for r in d.get("radial_locality", []) if r["functional"] == fkey]
    if not radial:
        return None
    by_sigma = {r["sigma"]: r for r in radial}
    sigmas = sorted(by_sigma)
    edges = np.asarray(radial[0]["edges_km"], float)

    panels = [(grp, var) for grp, var in IG_LOCALITY_VARS
              if any(var in by_sigma[s].get(grp, {}) for s in sigmas)]
    if not panels:
        return None

    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 5.0),
                             squeeze=False)
    for ax, (grp, var) in zip(axes[0], panels):
        for s in sigmas:
            ys = by_sigma[s].get(grp, {}).get(var)
            if ys is not None:
                ax.plot(edges, ys, marker="o", ms=3, label=f"σ={s:g}")
        if probe_r:
            ax.axvline(probe_r, color="grey", ls="--", lw=0.8)
            ax.text(probe_r, 0.02, f" output disk {probe_r:g} km", fontsize=6,
                    rotation=90, va="bottom", ha="left", color="grey")
        ax.set_ylim(0.0, 1.02)
        ax.set_xlim(0, float(edges.max()))
        ax.set_xlabel("radius X around probe centre (km)")
        ax.set_ylabel("cumulative fraction of |attribution| within X")
        tag = {"lres": "in_lres", "hres": "in_hres", "ntgt": "noisy_hres"}[grp]
        ax.set_title(f"{tag} {var}", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    head = " · ".join(str(p) for p in (_pretty_case(case_label), fkey) if p)
    fig.suptitle(_title(ckpt_id, f"IG · {head} · radial locality")
                 + "\nhow local is each input's influence on the probe — "
                   "cumulative |attribution| vs radius, per sigma (aggregated over "
                   "surface targets)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def _ig_case_figs(d: dict, case_label, ckpt_id: str) -> list:
    """All pages for one IG case: overview → one map page per sigma → locality."""
    g = _case_geom(d)
    if not g or not g["tgts"]:
        return []
    figs = [_ig_context_page(d, case_label, ckpt_id)]
    figs += [_ig_maps_page(d, case_label, ckpt_id, s) for s in g["sigmas"]]
    figs.append(_ig_locality_page(d, case_label, ckpt_id))
    return [f for f in figs if f is not None]


def render_ig(data: dict, ckpt_id: str) -> list:
    """Single-case IG (base-level JSON) — overview + per-sigma maps + locality."""
    return _ig_case_figs(data, None, ckpt_id)


# ---------------------------------------------------------------------------
# case-study side-by-side renderers (one column per case within each method)
# ---------------------------------------------------------------------------

def _render_permutation_cases_one(datas, ckpt_id, which):
    """One heatmap grid: rows = union of surface targets, columns = case studies
    (each scored over its own region), for one pathway view ('input' or
    'noisy'). Returns the figure, or None if that view's data is absent."""
    cases = list(datas)
    all_targets = [t for t in SURFACE_TARGETS
                   if any(t in d["sigma_results"][0].get("surface_targets", [])
                          for d in datas.values())]
    top_k = 15
    fig, axes = plt.subplots(len(all_targets), len(cases),
                             figsize=(5.5 * len(cases), 3.8 * len(all_targets)),
                             squeeze=False)
    drew = False
    for col, case in enumerate(cases):
        d = datas[case]
        first = d["sigma_results"][0]
        src = _perm_sources(first, which)
        if src is None:
            for row in range(len(all_targets)):
                axes[row][col].axis("off")
            continue
        all_names, src_specs, _ = src
        sigmas = [r["sigma"] for r in d["sigma_results"]]
        region = (d.get("regions") or ["global"])[0]
        for row, target in enumerate(all_targets):
            ax = axes[row][col]
            if target not in first.get("surface_targets", []):
                ax.axis("off")
                continue
            M = np.zeros((len(all_names), len(sigmas)))
            for j, res in enumerate(d["sigma_results"]):
                for off, src_key in src_specs:
                    for idx_str, info in res.get(src_key, {}).items():
                        M[off + int(idx_str), j] = (
                            info.get("region_paired_mse_per_target", {})
                                .get(region, {}).get(target)
                            or info["paired_mse_per_target"].get(target, 0.0))
            order, self_idx = _perm_order(M.max(axis=1), all_names, target, which, top_k)
            Mt = M[order]
            pos = Mt[Mt > 0]
            vmin = max(pos.min() if pos.size else 1e-10, 1e-10)
            vmax = max(float(Mt.max()), vmin * 10)
            heatmap(ax, Mt, [f"{s:g}" for s in sigmas],
                    [all_names[i] + (" (self)" if i == self_idx else "") for i in order],
                    title=f"{_pretty_case(case)}: output {target} scored over {_pretty_region(region)}",
                    cmap=SEQ_CMAP, norm=LogNorm(vmin=vmin, vmax=vmax),
                    cbar_label="paired MSE (log)", fig=fig)
            ax.set_xlabel("sigma")
            drew = True
    if not drew:
        plt.close(fig)
        return None
    grp = "in_lres + in_hres" if which == "input" else "noisy_hres (noised target)"
    fig.suptitle(_title(ckpt_id, f"feature permutation · {grp} — output disturbance "
                                 "per case region (paired MSE, top-15)"), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return fig


def render_permutation_cases(datas: dict, ckpt_id: str) -> list:
    """Per-pathway permutation grids across cases: in_lres+in_hres, then the
    noisy_hres view when the noised-target channels were permuted (item 8)."""
    return [f for f in (_render_permutation_cases_one(datas, ckpt_id, w)
                        for w in ("input", "noisy")) if f is not None]


def render_ig_cases(datas: dict, ckpt_id: str) -> list:
    """Per case: overview page → one driver-map page per sigma → locality page.
    Delegates to _ig_case_figs so the single-case and multi-case paths match."""
    figs = []
    for case, d in datas.items():
        figs += _ig_case_figs(d, case, ckpt_id)
    return figs


def _cases_addendum(tool: str, datas: dict) -> str:
    lines = ["", "Cases (side by side):"]
    for c, d in datas.items():
        if tool == "feature_permutation":
            lines.append(f"  {c}: regions={d.get('regions')}, "
                         f"batch={d.get('batch_size')}, "
                         f"bundles={[Path(p).name for p in d.get('bundle_paths', [])][:1]}")
        elif tool == "integrated_gradients":
            bx = {n: (round(b.get("lat", 0), 1), round(b.get("lon", 0), 1),
                      b.get("radius_km")) for n, b in d.get("boxes", {}).items()}
            lines.append(f"  {c}: functionals={d.get('functionals')}, probes={bx}")
        else:
            lines.append(f"  {c}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# method descriptions: rendered to plots/methods.md NEXT TO the report (not as
# pages inside the PDF). One describe_*() per method returns plain text.
# ---------------------------------------------------------------------------

def _fmt_sigmas(data):
    sigs = data.get("sigmas") or [r["sigma"] for r in data.get("sigma_results", [])]
    return ", ".join(f"{s:g}" for s in sigs)


def _meta_args(meta):
    return (meta or {}).get("args", {})


def describe_ig(data, meta):
    a = _meta_args(meta)
    lines = [
        "Integrated Gradients attributes a SCALAR functional of the model output back to every input cell and "
        "variable (lres conditioning + hres forcings), by integrating input-gradients along the path from a "
        f"baseline ({data.get('baseline', 'zeros')}) to the true input ({data.get('ig_steps')} Riemann steps, "
        "diffusion noise held fixed). Positive attribution = the input pushes the functional up. Computed at "
        f"sigmas {_fmt_sigmas(data)} — rendered as ONE driver-map page per sigma. Each case also gets a "
        f"'case overview' page (the observed field we probe) and a radial-locality page. Per-case functionals, "
        "probe centres/radii and event bundles are listed under 'Cases' below.",
        "",
        f"SAMPLES: each case is averaged over n_samples={data.get('n_samples', 1)} ensemble members. Because each "
        "member's storm sits in a slightly different place, only ALIGNMENT-SAFE quantities are averaged — the "
        "driver bars (global per-variable means) and everything measured relative to each member's OWN probe "
        f"centre (coherence, radial locality). The MAPS are shown for member {data.get('maps_from_member', 0)} "
        "only, so the storm structure stays sharp instead of being blurred across members.",
        "",
        "OUTPUT side ('the circle'): the scalar is the area-weighted MEAN of ONE surface target over a single "
        "disk of output cells — NOT the whole globe. eye = a tight disk centred on the auto-detected storm core "
        "(msl minimum for cyclones, tp maximum for precip); box = a named disk of a stated radius (e.g. amazon "
        "R=1200 km). That disk is the only output region that defines the scalar.",
        "INPUT side: the gradient of that one scalar is taken w.r.t. EVERY input cell of the WHOLE GLOBE, across "
        "THREE conditioning pathways — in_lres (coarse atmospheric state), in_hres (static/time forcings) and "
        "noisy_hres (the noised hi-res target the denoiser refines). In the driver bars these are distinguished by "
        "edge colour (in_lres none, in_hres black, noisy_hres green) and an [in_lres]/[in_hres]/[noisy_hres] name "
        "tag, so t_1000 [in_lres] and t_1000 [noisy_hres] never collide. Forcing channels (cos_*, sin_*, lsm, "
        "surface z) are EXCLUDED from noisy_hres: they sit in the output schema as passthrough and are not "
        "denoised, so their noisy_hres attribution is a spurious artifact. The maps show a ±zoom-deg crop; the "
        "attribution is global.",
        "LOCALITY: the bar-panel caption reports how much of the GLOBAL input |attribution| sits inside the "
        f"output disk and inside disk + {data.get('probe_buffer_km', 350):g} km (probe-relative). The radial-"
        "locality page plots the full cumulative-fraction-vs-radius curve for in_lres 2t, noisy_hres 2t and "
        "in_hres lsm, one curve per sigma. On the maps the solid ring is the output-disk radius, a dashed ring "
        "marks 500 km context, and the star is the auto-detected centre; the target's own in_lres input AND its "
        "own noisy_hres channel are both pinned into the bars (annotated 'self in_lres' / 'self noisy_hres').",
    ]
    if "tail" in data.get("functionals", []):
        side = a.get("tail_side", "abs")
        lines.append(f"tail = mean of the target field over the cells where the OBSERVED target is in its "
                     f"p{data.get('tail_percentile', 99):g} tail (side={side}; auto = low tail for msl i.e. "
                     f"cyclones, |.| tail otherwise) within region "
                     f"'{a.get('tail_region', 'global')}' — 'which inputs matter exactly where the field is "
                     "extreme'.")
    spectral = [f for f in data.get("functionals", []) if f.startswith("spectral")]
    if spectral:
        lines.append(f"{spectral[0]} = high-wavenumber power (top {100 * (1 - float(a.get('spectral_cutoff', 0.5))):g}% "
                     f"of radial k after regridding the region to a {a.get('spectral_ngrid', 64)}x"
                     f"{a.get('spectral_ngrid', 64)} raster) of the target field over that region — 'which inputs "
                     "create the small scales'. NOTE: with the zeros baseline, attribution scales with the raw "
                     "input value, which inflates large constant fields (cos_latitude etc.); cross-check rankings "
                     "with --baseline mean.")
    lines.append("")
    lines.append(f"Event bundle(s): {', '.join(Path(p).name for p in data.get('bundle_paths', []))}.")
    return "\n".join(lines)


def describe_patching(data, meta):
    st = data.get("storm", {})
    return "\n".join([
        f"Causal activation patching: corrupt the model ({data.get('corruption')}), then splice CLEAN reference "
        "activations back in and measure per-target RECOVERY (1 = output fully restored, 0 = no effect) at sigmas "
        f"{_fmt_sigmas(data)}. Same diffusion noise for reference / corrupted / patched runs.",
        "",
        "residual    — per-block localizer: pseudo-random corruption is injected into EVERY processor block's "
        f"output (scale {data.get('residual_noise_scale')}x norm); patching block L clean shows how much of the "
        "field block L commits. Sanity: NONE=0, ALL=1.",
        "grid_region — patch only the storm-region hidden nodes "
        f"({st.get('n_hidden_region', '?')} nodes within {st.get('radius_deg', '?')} deg of "
        f"({st.get('center_lat', 0):.1f}, {st.get('center_lon', 0):.1f})); recovery measured over the storm region "
        "on the output grid. patch-all < 1 is expected (the corrupted conditioning also feeds the decoder skip "
        "path).",
        "stage       — patch at network cut points (enc_data / enc_hidden / enc_both / proc_out); enc_both must "
        "recover ~1.0 (the corruption only enters through the encoder).",
    ])


def describe_permutation(data, meta):
    a = _meta_args(meta)
    ext = data.get("extreme_percentile")
    txt = [
        f"Feature permutation importance (single denoiser step): each input variable is shuffled across the batch "
        f"(batch = {data.get('batch_size', '?')} event-bundle members), the denoiser is re-run at each sigma with "
        "the SAME noise, and the importance is the paired MSE between permuted and unpermuted outputs (shared "
        f"noise cancels). Repeats: {a.get('n_repeats', '?')}. Log color scale.",
        "",
        "WHAT IS COMPARED PER CASE: the input field shuffled is GLOBAL (the whole low-res conditioning + high-res "
        "forcing), but the output change (paired MSE) is measured ONLY inside that case's region — the N. Atlantic "
        "TC box for the Humberto case, the Amazon rainforest box for the Amazon case. So a column answers: 'which "
        "global inputs most disturb the model's output over THIS region?'. Heatmaps show the top-15 variables "
        f"(rows) vs sigma (columns), one panel row per surface target; regions: {', '.join(data.get('regions', []))}.",
        "",
        "NOTE: the case studies in this checkpoint share ONE event bundle (the Humberto window), so the two "
        "regions are scored on the same global inputs. The two panels look similar because the dominant local "
        "drivers (2t, t_1000, skt) are important everywhere; the magnitudes do differ between regions (e.g. the "
        "Amazon 2t response is ~2x the Atlantic-box one). They are NOT identical data.",
    ]
    if data.get("sigma_results", [{}])[0].get("noisy_importance"):
        txt.append("")
        txt.append("A SEPARATE 'noisy_hres' permutation view shuffles the channels of the noised TARGET the "
                   "denoiser is handed to refine (the same pathway the conditioning ablation zeroes), rather than "
                   "the lres/hres conditioning inputs — so you can compare how much the model leans on the input "
                   "atmosphere vs the (noised) target it is denoising, per noise level.")
    if ext:
        txt.append(f"'tail' columns restrict the paired MSE further to the p{ext:g} extreme cells of the observed "
                   "target within the region — importance specifically for the extremes.")
    if data.get("mode") == "sampling" or "result" in data:
        txt.append("The full-sampling variant permutes inputs around the ENTIRE Heun sampling trajectory "
                   "(end-to-end importance for the final output) and is shown as bar charts instead.")
    return "\n".join(txt)


def describe_ablation(data, meta):
    return "\n".join([
        "Conditioning ablation: at each sigma, the denoiser is run with one conditioning pathway zeroed — lres "
        "(coarse atmospheric state), hres (static/time forcings), or the noisy target itself — and compared to the "
        "fully-conditioned output. Curves: MSE(ablated, full) per surface target, area-weighted per region "
        f"({', '.join(data.get('regions') or ['global'])}); summary panel: correlation(ablated, full) over all "
        "channels (corr ~1 = pathway unused, corr ~0 = pathway essential at that noise level). Sigmas: "
        f"{_fmt_sigmas(data)}.",
    ])


def describe_norms(data, meta):
    return ("Activation norms: forward hooks record the magnitude (L2 / std / max) of the encoder, processor and "
            f"decoder outputs while denoising at sigmas {_fmt_sigmas(data)} — which parts of the network are "
            "active at which noise regime. The per-block heatmap (when present) breaks the processor into its "
            "16 blocks. SCOPE: computed on the FULL GLOBAL field of the event batch — activations are not "
            "restricted to any case-study region (hidden-grid nodes mix information globally).")


def describe_cka(data, meta):
    return ("CKA layer similarity: linear Centered Kernel Alignment between the activations of every pair of "
            f"processor blocks, per sigma ({_fmt_sigmas(data)}). CKA ~1 = two blocks compute near-identical "
            "representations. Block clusters reveal processing stages; the chunk-summary panel tracks "
            "within-chunk vs cross-chunk similarity vs sigma (the 16 blocks are chained as 2 chunks of 8). "
            "SCOPE: computed on the FULL GLOBAL field of the event batch, not case-restricted.")


# ---------------------------------------------------------------------------
# trajectory (A1): storm-core x̂₀ vs σ — realized vs teacher-forced ceiling
# ---------------------------------------------------------------------------

_TRAJ_UNIT = {"msl": "hPa", "wind10m": "m/s", "10u": "m/s", "10v": "m/s",
              "2t": "K", "tp": "mm"}


def _traj_panel(ax, data, name):
    """One metric: realized x̂₀ (seed mean + min/max band) vs the teacher-forced
    ceiling, with target and x_interp reference lines, on a reversed-log σ axis.
    For msl the y-axis is inverted so a deeper storm reads upward."""
    refs, ceil, trajs = data["references"], data["ceiling"], data["trajectories"]
    n_calls = min((len(t["steps"]) for t in trajs), default=0)
    xs, mean, lo, hi = [], [], [], []
    for k in range(n_calls):
        vals = [t["steps"][k]["metrics"][name] for t in trajs
                if name in t["steps"][k]["metrics"]]
        if not vals:
            continue
        xs.append(float(np.mean([t["steps"][k]["sigma"] for t in trajs])))
        mean.append(float(np.mean(vals)))
        lo.append(float(np.min(vals)))
        hi.append(float(np.max(vals)))
    if xs:
        o = np.argsort(xs)[::-1]
        xs, mean = np.asarray(xs)[o], np.asarray(mean)[o]
        lo, hi = np.asarray(lo)[o], np.asarray(hi)[o]
        ax.fill_between(xs, lo, hi, color="#D85A30", alpha=0.18, lw=0)
        ax.plot(xs, mean, color="#D85A30", lw=2.0, marker="o", ms=3, label="realized x̂₀")
    cs = [c["sigma"] for c in ceil if name in c["metrics"]]
    cv = [c["metrics"][name] for c in ceil if name in c["metrics"]]
    if cs:
        o = np.argsort(cs)[::-1]
        ax.plot(np.asarray(cs)[o], np.asarray(cv)[o], color="#1D9E75", lw=2.0,
                ls="--", marker="s", ms=3, label="ceiling (teacher-forced)")
    if name in refs.get("target", {}):
        ax.axhline(refs["target"][name], color="#444441", ls=":", lw=1.6, label="target")
    if name in refs.get("x_interp", {}):
        ax.axhline(refs["x_interp"][name], color="#B4B2A9", ls="-", lw=1.6, label="x_interp")
    ax.set_xscale("log")
    ax.invert_xaxis()
    if name == "msl":
        ax.invert_yaxis()
    ax.set_xlabel("noise level σ  (sampling →)")
    ax.set_ylabel(f"{name} storm-core ({_TRAJ_UNIT.get(name, '')})")
    ax.set_title(name, fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6, loc="best")


def render_trajectory(data: dict, ckpt_id: str) -> list:
    """One page: a σ-trajectory panel per reported metric (msl + wind10m lead)."""
    metrics = data.get("metrics_reported") or list(data["references"]["target"].keys())
    lead = [m for m in ("msl", "wind10m") if m in metrics]
    ordered = lead + [m for m in metrics if m not in lead]
    if not ordered:
        return []
    fig, axes = fig_grid(len(ordered), cols=min(3, len(ordered)), panel_w=5.2, panel_h=4.2)
    for ax, name in zip(axes, ordered):
        _traj_panel(ax, data, name)
    b = data.get("box", {})
    fig.suptitle(
        _title(ckpt_id, "trajectory · x̂₀ birth–commit–erase") + "\n"
        + f"storm box ({b.get('lat', 0):.1f},{b.get('lon', 0):.1f}) R={b.get('radius_km', 0):g} km · "
          f"realized (coral, {len(data.get('seeds', []))} seeds, band = min/max) vs teacher-forced ceiling "
          "(teal) vs target (dotted) vs x_interp (grey). For msl, deeper = up.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return [fig]


def describe_trajectory(data, meta):
    b = data.get("box", {})
    return "\n".join([
        "Diffusion trajectory (A1): the REAL Heun sampler is run and at EVERY denoiser call the model's current "
        "clean-field estimate x̂₀ = D(x_t, σ) (Tweedie) is captured, reconstructed to PHYSICAL units (the model's "
        "own add_interp_to_state) and reduced to a storm-core intensity over a fixed box "
        f"(centre ({b.get('lat', 0):.1f},{b.get('lon', 0):.1f}), R={b.get('radius_km', 0):g} km, "
        f"{b.get('n_cells', '?')} cells; auto-detected from the observed msl minimum). Plotting intensity vs σ "
        "shows WHEN the storm is committed and whether the low-σ steps erase it.",
        "",
        "Four series per surface target, same box: realized = per-seed x̂₀ trajectory (band = min/max over "
        f"{len(data.get('seeds', []))} seeds); ceiling = teacher-forced denoiser probe (feed the TRUE residual + "
        "noise at each σ — 'what the model knows'); target = observed field; x_interp = coarse input. The "
        "ceiling-minus-realized-final gap is the 'knows-but-forgets' diagnostic: a deep ceiling with a shallow "
        "realized endpoint means the denoiser knows the depth but the sampling trajectory does not commit to it.",
        "",
        "Per-target storm-core reduction over the box: msl = MIN (hPa, the deep eye core, only a few cells); "
        "winds/2t/tp = robust p99 (a percentile, not raw max, so the single-cell fine-scale spikes the diffusion "
        "injects at low sigma do not dominate); wind10m = p99 wind speed. Never aggregated across targets.",
        "",
        f"Event bundle(s): {', '.join(Path(p).name for p in data.get('bundle_paths', []))}.",
    ])


def render_seeding(data: dict, ckpt_id: str) -> list:
    """A2 seeding-σ sweep: final storm-core depth vs the σ at which the true storm was planted."""
    runs = sorted(data.get("runs", []), key=lambda r: r["seed_sigma"])
    if not runs:
        return []
    refs = data.get("references", {})
    free = data.get("free") or {}
    metrics = [m for m in ("msl", "wind10m") if any(m in r["final_mean"] for r in runs)]
    fig, axes = fig_grid(len(metrics), cols=len(metrics), panel_w=6.0, panel_h=4.6)
    for ax, name in zip(axes, metrics):
        xs = [r["seed_sigma"] for r in runs]
        ys = [r["final_mean"].get(name) for r in runs]
        ax.plot(xs, ys, color="#D85A30", lw=2.0, marker="o", ms=4, label="seeded → free")
        if name in free:
            ax.axhline(free[name], color="#888780", ls="--", lw=1.6, label="free (no seed)")
        if name in refs.get("target", {}):
            ax.axhline(refs["target"][name], color="#444441", ls=":", lw=1.6, label="target")
        if name in refs.get("x_interp", {}):
            ax.axhline(refs["x_interp"][name], color="#B4B2A9", lw=1.6, label="x_interp")
        cx = data.get("summary", {}).get(name, {}).get("crossover_seed_sigma")
        if cx:
            ax.axvline(cx, color="#1D9E75", ls="-.", lw=1.4)
            ax.text(cx, ax.get_ylim()[1], f" crit σ≈{cx:g}", color="#0F6E56", fontsize=7, va="top")
        ax.set_xscale("log")
        ax.invert_xaxis()
        if name == "msl":
            ax.invert_yaxis()
        ax.set_xlabel("σ_seed — true storm planted above this σ, free sampling below")
        ax.set_ylabel(f"{name} final storm-core ({_TRAJ_UNIT.get(name, '')})")
        ax.set_title(name, fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(
        _title(ckpt_id, "seeding-σ sweep (A2) — when is storm depth decided?") + "\n"
        "final storm-core depth vs σ_seed. Right (small σ_seed) = teacher-forced almost all the way → "
        "committed/deep; left (large σ_seed) = planted early then long free fall → free/shallow. A sharp step "
        "(green) = critical window (fix the noise schedule); a smooth ramp = low-σ refinement (fix with guidance).",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return [fig]


def describe_seeding(data, meta):
    s = data.get("summary", {}).get("msl", {})
    return "\n".join([
        "Seeding-σ sweep (A2): for each σ_seed, the TRUE storm is planted into the sampler at noise level "
        "σ_seed (y = true_residual + σ_seed·ε) and the REAL production Heun sampler then runs FREELY for every "
        "σ below σ_seed. The plot is final storm-core depth vs σ_seed — 'how late can we hand the model the "
        "answer and still have it commit?'.",
        "",
        "Small σ_seed = the storm is teacher-forced down to a low noise level (short free tail) → committed/deep "
        "(≈ the ceiling). Large σ_seed = planted early then the sampler free-falls → ≈ the free run (shallow). A "
        "SHARP step means storm depth is decided in a narrow critical window of σ (→ fix the training noise "
        "schedule / high-σ coverage); a SMOOTH ramp means depth is set by low-σ refinement (→ guidance / churn).",
        "",
        f"This checkpoint: msl committed≈{s.get('committed')}, free≈{s.get('free')}, "
        f"crossover σ_seed≈{s.get('crossover_seed_sigma')}.",
        "",
        f"Event bundle(s): {', '.join(Path(p).name for p in data.get('bundle_paths', []))}.",
    ])


DESCRIBERS = {
    "integrated_gradients": ("Integrated gradients — what drives each target", describe_ig),
    "activation_patching": ("Activation patching — which stage commits each field", describe_patching),
    "feature_permutation": ("Feature permutation — which inputs matter", describe_permutation),
    "feature_permutation_full_sampling": (
        "Feature permutation (full sampling) — end-to-end input importance", describe_permutation),
    "conditioning_ablation": ("Conditioning ablation — what each pathway carries", describe_ablation),
    "activation_profiling": ("Activation norms — where the network is active", describe_norms),
    "cka": ("CKA — how processor representations are organized", describe_cka),
    "trajectory": ("Trajectory — when the storm is born, committed and erased", describe_trajectory),
    "trajectory_seeding": ("Seeding-σ sweep — when is storm depth decided", describe_seeding),
}


# ---------------------------------------------------------------------------
# registry + dispatcher
# ---------------------------------------------------------------------------

# tool -> (candidate subdirs searched in order, json filename, renderer)
TOOL_REGISTRY = {
    "integrated_gradients": (["integrated_gradients"],
                             "integrated_gradients.json", render_ig),
    "activation_patching": (["activation_patching"],
                            "activation_patching.json", render_patching),
    "feature_permutation": (["feature_permutation", "feature_permutation_per_target"],
                            "permutation_importance.json", render_permutation),
    "feature_permutation_full_sampling": (
        ["feature_permutation_full_sampling", "feature_permutation_full_sampling_per_target"],
        "permutation_importance_full_sampling.json", render_full_sampling),
    "conditioning_ablation": (["conditioning_ablation"],
                              "conditioning_ablation.json", render_ablation),
    "activation_profiling": (["activations", "activation_profiling"],
                             "activation_profiling.json", render_activation_norms),
    "cka": (["activations", "cka", "cka_analysis", "cka_analysis_fixed"],
            "cka_analysis.json", render_cka),
    "trajectory": (["trajectory"], "trajectory.json", render_trajectory),
    "trajectory_seeding": (["trajectory"], "seeding.json", render_seeding),
}


def find_json(run_dir: Path, tool: str) -> Path | None:
    subdirs, json_name, _ = TOOL_REGISTRY[tool]
    for sd in subdirs:
        p = run_dir / sd / json_name
        if p.exists():
            return p
    return None


TOOL_SUBDIR_NAMES = {sd for sds, _, _ in TOOL_REGISTRY.values() for sd in sds} | {"plots"}

# Tools whose JSON is kept on disk but NOT rendered into the report (item 8:
# activation patching removed until the method is settled). Its subdir name
# stays in TOOL_SUBDIR_NAMES above so discover_cases still treats it as a tool
# dir, not a case.
DISABLED_TOOLS = {"activation_patching"}


def discover_cases(run_dir: Path) -> dict:
    """Case-study subdirs: <run_dir>/<case>/<tool_subdir>/<tool>.json.

    A case dir is any non-tool subdir that contains at least one tool JSON.
    Dirs starting with '_' (e.g. _archive) are ignored.
    """
    cases = {}
    for sub in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        if sub.name in TOOL_SUBDIR_NAMES or sub.name.startswith("_"):
            continue
        if any(find_json(sub, t) for t in TOOL_REGISTRY):
            cases[sub.name] = sub
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser(description="Interp report renderer")
    parser.add_argument("--run-dir", required=True,
                        help="Per-checkpoint root, e.g. ~/perm/interp/59e4_300k")
    parser.add_argument("--tools", default=None,
                        help="Comma-separated subset of tools; default all detected")
    parser.add_argument("--per-tool", action="store_true",
                        help="Also write one PDF per tool (legacy filenames)")
    parser.add_argument("--no-overwrite", action="store_true",
                        help="Skip per-tool PDFs that already exist")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise SystemExit(f"run-dir does not exist: {run_dir}")
    ckpt_id = run_dir.name
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    only = set(args.tools.split(",")) if args.tools else None

    cases = discover_cases(run_dir)
    if cases:
        LOGGER.info("case-study dirs: %s", list(cases))

    report_figs = []
    methods_sections = []  # (title, body) -> plots/methods.md (item 1)
    for tool, (_, _, renderer) in TOOL_REGISTRY.items():
        if only is not None and tool not in only:
            continue
        if tool in DISABLED_TOOLS:
            LOGGER.info("[%s] disabled — not rendered", tool)
            continue
        base_path = find_json(run_dir, tool)
        case_paths = {c: p for c, p in ((c, find_json(d, tool))
                                        for c, d in cases.items()) if p}
        if base_path is None and not case_paths:
            LOGGER.info("[%s] no JSON found — skip", tool)
            continue
        LOGGER.info("[%s] rendering (base=%s, cases=%s)", tool,
                    bool(base_path), list(case_paths))
        figs = []
        datas = {}
        try:
            if base_path is not None:
                figs += renderer(json.loads(base_path.read_text()), ckpt_id)
            if case_paths:
                datas = {c: json.loads(p.read_text()) for c, p in case_paths.items()}
                if tool == "feature_permutation" and len(datas) > 1:
                    figs += render_permutation_cases(datas, ckpt_id)
                elif tool == "integrated_gradients" and len(datas) > 1:
                    figs += render_ig_cases(datas, ckpt_id)
                else:
                    for c, d in datas.items():
                        figs += renderer(d, f"{ckpt_id} · {_pretty_case(c)}")
        except Exception:
            LOGGER.exception("[%s] renderer failed — skipping", tool)
            continue
        if figs and tool in DESCRIBERS:
            # Method descriptions go to plots/methods.md NEXT TO the PDF, not
            # into the PDF itself (item 1).
            meta_src = base_path or next(iter(case_paths.values()))
            meta_path = meta_src.parent / "run_meta.json"
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            desc_data = (json.loads(base_path.read_text()) if base_path is not None
                         else next(iter(datas.values())))
            title, describe = DESCRIBERS[tool]
            try:
                body = describe(desc_data, meta)
                if len(datas) > 1:
                    body += "\n" + _cases_addendum(tool, datas)
                methods_sections.append((title, body))
            except Exception:
                LOGGER.exception("[%s] description text failed — skipping it", tool)
        if args.per_tool:
            out_pdf = plots_dir / f"{tool}.pdf"
            if not (out_pdf.exists() and args.no_overwrite):
                with PdfPages(out_pdf) as pdf:
                    for f in figs:
                        pdf.savefig(f)
                LOGGER.info("  wrote %s", out_pdf.name)
        report_figs.extend(figs)

    if report_figs:
        report_pdf = plots_dir / "report.pdf"
        with PdfPages(report_pdf) as pdf:
            for f in report_figs:
                pdf.savefig(f)
                plt.close(f)
        LOGGER.info("Done. %d pages -> %s", len(report_figs), report_pdf)
    else:
        LOGGER.warning("Nothing rendered (no JSONs found in %s)", run_dir)

    # Method descriptions as markdown NEXT TO the report (item 1).
    if methods_sections:
        md = [f"# Interp methods — {ckpt_id}", "",
              "Method descriptions for the figures in `report.pdf` (one section per "
              "method present in this checkpoint's report). Generated by "
              "`python -m interp.viz`; not embedded in the PDF.", ""]
        for title, body in methods_sections:
            md.append(f"## {title}")
            md.append("")
            md.append(body.strip())
            md.append("")
        methods_md = plots_dir / "methods.md"
        methods_md.write_text("\n".join(md))
        LOGGER.info("Methods -> %s", methods_md)


if __name__ == "__main__":
    main()
