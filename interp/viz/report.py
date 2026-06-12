"""Per-checkpoint interp report — ONE PDF from the run dir's JSONs.

Reads  <run_dir>/<tool_subdir>/<tool>.json  (old and new schemas)
Writes <run_dir>/plots/report.pdf          (default; --per-tool for one PDF
                                            per tool under the legacy names)

Page order follows the headline-first convention: storm-centered attribution
maps (what drives the extreme) lead; ranking bars and profile curves follow.

Usage
-----
    python -m interp.viz --run-dir ~/perm/interp/59e4_300k
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
from matplotlib.colors import LogNorm, TwoSlopeNorm

from interp.viz.panels import fig_grid, geo_panel, heatmap, loglog, ranked_barh

LOGGER = logging.getLogger("interp.viz")

SURFACE_TARGETS = ["10u", "10v", "2t", "msl", "tp"]

LRES_C, HRES_C, BOTH_C = "#1f77b4", "#ff7f0e", "#2ca02c"


def _title(ckpt_id, *parts):
    return f"{ckpt_id} · " + " · ".join(str(p) for p in parts)


def _targets_of(d: dict) -> list[str]:
    return [t for t in SURFACE_TARGETS if t in d]


# ---------------------------------------------------------------------------
# feature permutation (per-sigma): one heatmap grid, targets x region columns
# ---------------------------------------------------------------------------

def render_permutation(data: dict, ckpt_id: str) -> list:
    sigmas = [r["sigma"] for r in data["sigma_results"]]
    first = data["sigma_results"][0]
    targets = first.get("surface_targets") or _targets_of(first["baseline_mse_per_target"])
    any_entry = first["lres_importance"]["0"]
    regions = [r for r in (data.get("regions") or [])
               if any_entry.get("region_paired_mse_per_target", {}).get(r)]
    has_extreme = bool(any_entry.get("extreme_paired_mse_per_target"))

    lres_names = [i["name"] for _, i in sorted(first["lres_importance"].items(),
                                               key=lambda kv: int(kv[0]))]
    hres_names = [i["name"] for _, i in sorted(first["hres_importance"].items(),
                                               key=lambda kv: int(kv[0]))]
    all_names = lres_names + hres_names
    n_lres = len(lres_names)

    def matrix(target, key, region):
        """(n_vars x n_sigmas) importance matrix for one column spec."""
        M = np.zeros((len(all_names), len(sigmas)))
        for j, res in enumerate(data["sigma_results"]):
            for off, src in [(0, res["lres_importance"]), (n_lres, res["hres_importance"])]:
                for idx_str, info in src.items():
                    row = off + int(idx_str)
                    if key == "global":
                        M[row, j] = info["paired_mse_per_target"].get(target, 0.0)
                    else:
                        M[row, j] = info.get(key, {}).get(region, {}).get(target, 0.0)
        return M

    # Columns: each region, plus its tail twin when present; fallback = global.
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
        order = np.argsort(-score)[:top_k]
        names_top = [all_names[i] for i in order]
        for c, (lbl, _, _) in enumerate(columns):
            M = mats[lbl][order]
            pos = M[M > 0]
            vmin = max(pos.min() if pos.size else 1e-10, 1e-10)
            vmax = max(float(M.max()), vmin * 10)
            heatmap(axes[row][c], M, [f"{s:g}" for s in sigmas], names_top,
                    title=f"{target} · {lbl}", cmap="viridis",
                    norm=LogNorm(vmin=vmin, vmax=vmax),
                    cbar_label="paired MSE (log)", fig=fig)
            axes[row][c].set_xlabel("sigma")
    fig.suptitle(_title(ckpt_id, "feature permutation (single-step, paired MSE, top-15)"),
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return [fig]


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
    region_styles = {"global": "-", "amazon": "--", "amazon_rainforest": "--",
                     "tropics": ":"}

    fig, axes = fig_grid(len(targets) + 1, cols=3, panel_w=6, panel_h=4.5)
    for ax, target in zip(axes[:-1], targets):
        series, styles = {}, {}
        if has_region:
            preferred = ("amazon_rainforest", "amazon", "tc_franklin", "tc_humberto",
                         "tropics", "global")
            plot_regions = [r for r in preferred if r in regions] or list(regions)
            for rname in plot_regions[:3]:
                ls = region_styles.get(rname, "-")
                for label, key, color, marker in (
                        (f"zero in_lres · {rname}", "ablate_lres", LRES_C, "o"),
                        (f"zero in_hres · {rname}", "ablate_hres", HRES_C, "s"),
                        (f"{third_lbl} · {rname}", third, BOTH_C, "^")):
                    if key not in first:
                        continue
                    series[label] = [r[key]["region_per_target_mse"][rname][target]
                                     for r in data["sigma_results"]]
                    styles[label] = dict(color=color, marker=marker, linestyle=ls)
            title = f"target = {target} ({', '.join(plot_regions[:3])})"
        else:
            name_to_idx = {"10u": 0, "10v": 1, "2t": 3, "msl": 4}
            idx = name_to_idx[target]
            for label, key, color, marker in (("zero in_lres", "ablate_lres", LRES_C, "o"),
                                              ("zero in_hres", "ablate_hres", HRES_C, "s"),
                                              (third_lbl, third, BOTH_C, "^")):
                series[label] = [r[key]["per_var_mse"][idx] for r in data["sigma_results"]]
                styles[label] = dict(color=color, marker=marker)
            title = f"target = {target}"
        loglog(ax, sigmas, series, styles=styles,
               ylabel="MSE(ablated, full)", title=title)

    ax = axes[-1]
    series = {}
    for label, key, marker in (("zero in_lres", "ablate_lres", "o"),
                               ("zero in_hres", "ablate_hres", "s"),
                               (third_lbl, third, "^")):
        series[label] = [r[key]["correlation_with_full"] for r in data["sigma_results"]]
    loglog(ax, sigmas, series, ylabel="correlation(ablated, full)",
           title="correlation summary (all channels, global)", logy=False)

    fig.suptitle(_title(ckpt_id, "conditioning ablation (zero lres / hres / noisy vs full)"),
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
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
                title="per-block L2 norm", cmap="viridis", cbar_label="L2", fig=fig)
        axes[3].set_xlabel("sigma")
    fig.suptitle(_title(ckpt_id, "activation norms (encoder · processor · decoder)"),
                 fontsize=12)
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
    fig.suptitle(_title(ckpt_id, "CKA layer similarity (processor blocks)"), fontsize=12)
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
# integrated gradients: extreme-centered driver maps lead, then locality + bars
# ---------------------------------------------------------------------------

def render_ig(data: dict, ckpt_id: str) -> list:
    targets = data["surface_targets"]
    functionals = data["functionals"]
    sigmas = sorted({e["sigma"] for e in data["results"]})
    map_sigma = data.get("map_sigma")
    disp_sigma = map_sigma if (map_sigma in sigmas) else sigmas[-1]
    idx = {(e["functional"], e["target"], e["sigma"]): e for e in data["results"]}
    spatial = [f for f in ("eye", *[f for f in functionals if f.startswith("box:")])
               if f in functionals]
    figs = []

    # 1) per-target signed driver maps at disp_sigma (eye preferred)
    fkey = spatial[0] if spatial else None
    if fkey:
        # Inner dashed ring = the probe's actual radius; outer = 500 km context.
        probe_key = fkey.split(":", 1)[1] if fkey.startswith("box:") else fkey
        probe_r = (data.get("boxes", {}).get(probe_key, {}) or {}).get("radius_km")
        rings = (probe_r, 500.0) if probe_r and probe_r != 500.0 else (200.0, 500.0)
        for t in targets:
            e = idx.get((fkey, t, disp_sigma))
            if e is None or "zoom" not in e:
                continue
            z = e["zoom"]
            center = (z["center_lat"], z["center_lon"])
            lat, lon = np.asarray(z["lat"]), np.asarray(z["lon"])
            varmaps = list(z["vars"].items())[:4]
            fig = plt.figure(figsize=(4.8 * len(varmaps), 4.8))
            for j, (vname, vals) in enumerate(varmaps):
                geo_panel(fig, 1, len(varmaps), 1 + j, lat, lon, np.asarray(vals),
                          title=f"{vname}  (signed)", diverging=True, center=center,
                          rings=rings)
            loc = e.get("coherence", {}).get("frac_within_500km")
            locstr = (f"{loc*100:.0f}% of influence within 500 km" if loc is not None
                      else "")
            fig.suptitle(_title(ckpt_id,
                                f"integrated gradients · {fkey} — what drives the {t} extreme",
                                f"σ={disp_sigma:g}", locstr), fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, 0.92))
            figs.append(fig)

        # 2) locality vs sigma
        fig, ax = plt.subplots(1, 1, figsize=(9, 5))
        series = {t: [(idx.get((fkey, t, s), {}) or {}).get("coherence", {})
                      .get("frac_within_500km", np.nan) for s in sigmas]
                  for t in targets}
        loglog(ax, sigmas, series, logy=False, ylim=(0.0, 1.02),
               ylabel="fraction of input influence within 500 km",
               title=f"Locality of the receptive field on the {fkey}, vs σ")
        fig.suptitle(_title(ckpt_id, f"integrated gradients · {fkey} · locality vs sigma"),
                     fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        figs.append(fig)

    # 3) top-12 driver bars per target, one page per ranking functional
    for rank_f in [f for f in ([fkey] if fkey else []) + ["tail"]
                   + [f for f in functionals if f.startswith("spectral")]
                   if f in functionals]:
        fig, axes = fig_grid(len(targets), cols=2, panel_w=7, panel_h=3.4)
        drew = False
        for ax, t in zip(axes, targets):
            e = idx.get((rank_f, t, disp_sigma))
            if e is None:
                ax.axis("off")
                continue
            drew = True
            items = ([(v["name"], "lres", v["mean_abs"], v["signed_mean"])
                      for v in e["lres"].values()]
                     + [(v["name"], "hres", v["mean_abs"], v["signed_mean"])
                        for v in e["hres"].values()])
            items.sort(key=lambda x: x[2], reverse=True)
            top = items[:12]
            ranked_barh(ax, [x[0] for x in top], [x[2] for x in top],
                        colors=["#d62728" if x[3] >= 0 else "#1f77b4" for x in top],
                        edgecolors=["black" if x[1] == "hres" else "none" for x in top],
                        xlabel="mean |attr|", title=t)
        if not drew:
            plt.close(fig)
            continue
        fig.suptitle(_title(ckpt_id,
                            f"integrated gradients · top drivers · {rank_f} · σ={disp_sigma:g}",
                            "red=+ / blue=− · hres edged"), fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        figs.append(fig)
    return figs


# ---------------------------------------------------------------------------
# method description pages (one short text page before each method's figures)
# ---------------------------------------------------------------------------

import textwrap


def _text_page(title: str, body: str):
    fig = plt.figure(figsize=(11.5, 4.6))
    fig.text(0.04, 0.93, title, fontsize=13, fontweight="bold", va="top")
    wrapped = "\n".join(
        textwrap.fill(par, width=125) if par.strip() else ""
        for par in body.strip().split("\n"))
    fig.text(0.04, 0.82, wrapped, fontsize=9, va="top", linespacing=1.45)
    return fig


def _fmt_sigmas(data):
    sigs = data.get("sigmas") or [r["sigma"] for r in data.get("sigma_results", [])]
    return ", ".join(f"{s:g}" for s in sigs)


def _meta_args(meta):
    return (meta or {}).get("args", {})


def describe_ig(data, meta):
    a = _meta_args(meta)
    boxes = data.get("boxes", {})
    lines = [
        "Integrated Gradients attributes a SCALAR functional of the model output back to every input cell and "
        "variable (lres conditioning + hres forcings), by integrating input-gradients along the path from a "
        f"baseline ({data.get('baseline', 'zeros')}) to the true input ({data.get('ig_steps')} Riemann steps, "
        "diffusion noise held fixed). Positive attribution = the input pushes the functional up. Computed at "
        f"sigmas {_fmt_sigmas(data)}; per-cell maps stored at sigma={data.get('map_sigma')}.",
        "",
        "Functionals in this run: " + ", ".join(data.get("functionals", [])) + ".",
    ]
    if any(f == "eye" or f.startswith("box:") for f in data.get("functionals", [])):
        probes = "; ".join(f"{n}: ({b.get('lat', 0):.1f}, {b.get('lon', 0):.1f}) R={b.get('radius_km', '?')} km, "
                           f"{b.get('n_cells', '?')} cells" for n, b in boxes.items())
        lines.append(f"eye/box = mean of the target field over a storm-centered disk ({probes}). On the maps, the "
                     "inner dashed ring is the probe radius itself, the outer ring is 500 km context; the star is "
                     "the auto-detected msl minimum.")
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
        f"noise cancels). Repeats: {a.get('n_repeats', '?')}. Heatmaps show the top-15 variables (rows, lres + "
        "hres) vs sigma (columns), one panel row per surface target; columns are area-weighted region "
        f"restrictions ({', '.join(data.get('regions', []))}). Log color scale.",
    ]
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
            "16 blocks.")


def describe_cka(data, meta):
    return ("CKA layer similarity: linear Centered Kernel Alignment between the activations of every pair of "
            f"processor blocks, per sigma ({_fmt_sigmas(data)}). CKA ~1 = two blocks compute near-identical "
            "representations. Block clusters reveal processing stages; the chunk-summary panel tracks "
            "within-chunk vs cross-chunk similarity vs sigma (the 16 blocks are chained as 2 chunks of 8).")


DESCRIBERS = {
    "integrated_gradients": ("Integrated gradients — what drives each target", describe_ig),
    "activation_patching": ("Activation patching — which stage commits each field", describe_patching),
    "feature_permutation": ("Feature permutation — which inputs matter", describe_permutation),
    "feature_permutation_full_sampling": (
        "Feature permutation (full sampling) — end-to-end input importance", describe_permutation),
    "conditioning_ablation": ("Conditioning ablation — what each pathway carries", describe_ablation),
    "activation_profiling": ("Activation norms — where the network is active", describe_norms),
    "cka": ("CKA — how processor representations are organized", describe_cka),
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
}


def find_json(run_dir: Path, tool: str) -> Path | None:
    subdirs, json_name, _ = TOOL_REGISTRY[tool]
    for sd in subdirs:
        p = run_dir / sd / json_name
        if p.exists():
            return p
    return None


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

    report_figs = []
    for tool, (_, _, renderer) in TOOL_REGISTRY.items():
        if only is not None and tool not in only:
            continue
        json_path = find_json(run_dir, tool)
        if json_path is None:
            LOGGER.info("[%s] no JSON found — skip", tool)
            continue
        LOGGER.info("[%s] rendering from %s", tool, json_path.relative_to(run_dir))
        try:
            data = json.loads(json_path.read_text())
            figs = renderer(data, ckpt_id)
        except Exception:
            LOGGER.exception("[%s] renderer failed — skipping", tool)
            continue
        if figs and tool in DESCRIBERS:
            meta_path = json_path.parent / "run_meta.json"
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            title, describe = DESCRIBERS[tool]
            try:
                figs.insert(0, _text_page(f"{ckpt_id} · {title}", describe(data, meta)))
            except Exception:
                LOGGER.exception("[%s] description page failed — skipping it", tool)
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


if __name__ == "__main__":
    main()
