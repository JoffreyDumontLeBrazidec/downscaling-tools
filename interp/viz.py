"""Interpretability viz CLI — one PDF per method, all in <run_dir>/plots/.

Conventions
-----------
- Reads from ~/perm/interp/<ckpt_id>/<tool>/<tool_result>.json
- Writes ONE PDF per tool to ~/perm/interp/<ckpt_id>/plots/<tool>.pdf
- Multi-panel layout inside each PDF (one panel per surface target where applicable)
- log-x for sigma; viridis for importance; RdBu for CKA
- matplotlib only (already in .ds-ag venv)

Usage
-----
    python -m interp.viz --run-dir ~/perm/interp/59e4_300k
    python -m interp.viz --run-dir ~/perm/interp/59e4_300k --tools cka
    python -m interp.viz --run-dir ~/perm/interp/59e4_300k --no-overwrite
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.backends.backend_pdf import PdfPages

LOGGER = logging.getLogger("interp.viz")

SURFACE_TARGETS = ["10u", "10v", "2t", "msl", "tp"]

# Tool name -> (subdir, expected JSON filename, renderer key)
TOOL_REGISTRY = {
    "feature_permutation": ("feature_permutation", "permutation_importance.json", "feature_permutation"),
    "feature_permutation_full_sampling": (
        "feature_permutation_full_sampling",
        "permutation_importance_full_sampling.json",
        "full_sampling",
    ),
    "conditioning_ablation": (
        "conditioning_ablation",
        "conditioning_ablation.json",
        "conditioning_ablation",
    ),
    "activation_profiling": (
        "activation_profiling",
        "activation_profiling.json",
        "activation_profiling",
    ),
    "cka": ("cka", "cka_analysis.json", "cka"),
    "activation_patching": (
        "activation_patching",
        "activation_patching.json",
        "activation_patching",
    ),
    "integrated_gradients": (
        "integrated_gradients",
        "integrated_gradients.json",
        "integrated_gradients",
    ),
}


def _title(ckpt_id: str, *parts) -> str:
    return f"{ckpt_id} · " + " · ".join(str(p) for p in parts)


def _existing_targets_from_dict(per_target_dict: dict) -> list[str]:
    return [t for t in SURFACE_TARGETS if t in per_target_dict]


# ---------------------------------------------------------------------------
# feature_permutation: 4 targets × (heatmap | lines), single PDF
# ---------------------------------------------------------------------------

def plot_feature_permutation(json_path: Path, out_pdf: Path, ckpt_id: str):
    data = json.loads(json_path.read_text())
    sigmas = [r["sigma"] for r in data["sigma_results"]]
    targets = data["sigma_results"][0].get("surface_targets") or _existing_targets_from_dict(
        data["sigma_results"][0]["baseline_mse_per_target"]
    )
    regions = data.get("regions") or []
    has_region = bool(regions) and \
        "region_paired_mse_per_target" in data["sigma_results"][0]["lres_importance"]["0"]
    LOGGER.info("feature_permutation: sigmas=%s targets=%s regions=%s",
                sigmas, targets, regions if has_region else None)

    # Stable var-name order across both heatmaps
    lres_names = [info["name"] for _, info in sorted(
        data["sigma_results"][0]["lres_importance"].items(), key=lambda kv: int(kv[0]))]
    hres_names = [info["name"] for _, info in sorted(
        data["sigma_results"][0]["hres_importance"].items(), key=lambda kv: int(kv[0]))]
    all_names = lres_names + hres_names
    n_lres = len(lres_names)

    def _matrix_for(region):
        out = {}
        for target in targets:
            M = np.zeros((len(all_names), len(sigmas)))
            for j, res in enumerate(data["sigma_results"]):
                for n_offset, src_dict in [(0, res["lres_importance"]),
                                            (n_lres, res["hres_importance"])]:
                    for idx_str, info in src_dict.items():
                        row = n_offset + int(idx_str)
                        if region is None:
                            M[row, j] = info["paired_mse_per_target"].get(target, 0.0)
                        else:
                            M[row, j] = (
                                info.get("region_paired_mse_per_target", {})
                                    .get(region, {}).get(target, 0.0)
                            )
            out[target] = M
        return out

    # Compact regional view: amazon | tropics side-by-side, top-15, no lines col.
    # Row ordering: by max(amazon, tropics) across sigma so the rows are the
    # regionally-most-important inputs (identical order in both columns).
    if has_region and "amazon" in regions and "tropics" in regions:
        regs = ["amazon", "tropics"]
        M_by_region = {r: _matrix_for(r) for r in regs}
        layout_regions = regs
    else:
        # Backward-compat fallback: legacy 2-col (global heatmap | top-8 lines).
        # We keep this path for old JSONs that lack regional fields.
        layout_regions = None
        M_global = _matrix_for(None)

    n_targets = len(targets)
    top_k = 15

    if layout_regions is not None:
        cols = len(layout_regions)
        fig, axes = plt.subplots(n_targets, cols,
                                  figsize=(5.5 * cols, 3.8 * n_targets))
        axes = np.atleast_2d(axes).reshape(n_targets, cols)
        for row, target in enumerate(targets):
            score = np.zeros(len(all_names))
            for r in layout_regions:
                score = np.maximum(score, M_by_region[r][target].max(axis=1))
            row_order = np.argsort(-score)[:top_k]
            names_top = [all_names[i] for i in row_order]
            for c, region in enumerate(layout_regions):
                M = M_by_region[region][target][row_order]
                pos = M[M > 0]
                vmin = max(pos.min() if pos.size else 1e-10, 1e-10)
                vmax = max(float(M.max()), vmin * 10)
                ax = axes[row, c]
                im = ax.imshow(M, aspect="auto", cmap="viridis",
                               norm=LogNorm(vmin=vmin, vmax=vmax))
                ax.set_xticks(range(len(sigmas)))
                ax.set_xticklabels([f"{s:g}" for s in sigmas])
                ax.set_yticks(range(top_k))
                ax.set_yticklabels(names_top, fontsize=7)
                ax.set_xlabel("sigma")
                ax.set_title(f"{target} · {region}")
                fig.colorbar(im, ax=ax, label="paired MSE (log)")
        suffix = " · amazon vs tropics (top-15)"
    else:
        # legacy 2-col fallback
        fig, axes = plt.subplots(n_targets, 2, figsize=(14, 4.0 * n_targets))
        axes = np.atleast_2d(axes).reshape(n_targets, 2)
        for row, target in enumerate(targets):
            M = M_global[target]
            row_order = np.argsort(-M.max(axis=1))[:top_k]
            M_top = M[row_order]
            names_top = [all_names[i] for i in row_order]
            ax_h = axes[row, 0]
            pos = M_top[M_top > 0]
            vmin = max(pos.min() if pos.size else 1e-10, 1e-10)
            vmax = max(float(M_top.max()), vmin * 10)
            im = ax_h.imshow(M_top, aspect="auto", cmap="viridis",
                             norm=LogNorm(vmin=vmin, vmax=vmax))
            ax_h.set_xticks(range(len(sigmas)))
            ax_h.set_xticklabels([f"{s:g}" for s in sigmas])
            ax_h.set_yticks(range(top_k))
            ax_h.set_yticklabels(names_top, fontsize=7)
            ax_h.set_xlabel("sigma")
            ax_h.set_title(f"top-{top_k} inputs · target = {target}")
            fig.colorbar(im, ax=ax_h, label="paired MSE (log)")
            ax_l = axes[row, 1]
            for r_i in range(min(8, len(row_order))):
                idx = row_order[r_i]
                ax_l.plot(sigmas, M[idx], marker="o", label=all_names[idx])
            ax_l.set_xscale("log"); ax_l.set_yscale("log")
            ax_l.set_xlabel("sigma"); ax_l.set_ylabel("paired MSE")
            ax_l.set_title(f"top-8 vs σ · target = {target}")
            ax_l.legend(loc="best", fontsize=7, ncol=2)
            ax_l.grid(True, which="both", alpha=0.3)
        suffix = ""

    fig.suptitle(_title(ckpt_id,
                        "feature_permutation (single-step, paired-MSE per surface target)" + suffix),
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_pdf)
    plt.close(fig)
    LOGGER.info("  wrote %s", out_pdf.name)


# ---------------------------------------------------------------------------
# feature_permutation_full_sampling: bar chart per target, single PDF
# ---------------------------------------------------------------------------

def plot_full_sampling(json_path: Path, out_pdf: Path, ckpt_id: str):
    data = json.loads(json_path.read_text())
    result = data["result"]
    targets = result.get("surface_targets") or _existing_targets_from_dict(
        result["baseline_mse_per_target"])
    lres = result["lres_importance"]
    hres = result["hres_importance"]
    all_items = []
    for idx_str, info in lres.items():
        all_items.append((info["name"], "lres", info["paired_mse_per_target"],
                          info.get("paired_mse_per_target_std", {})))
    for idx_str, info in hres.items():
        all_items.append((info["name"], "hres", info["paired_mse_per_target"],
                          info.get("paired_mse_per_target_std", {})))

    n = len(targets)
    cols = 2
    rows = (n + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows))
    axes = np.atleast_2d(axes).reshape(rows, cols)

    for i, target in enumerate(targets):
        ax = axes[i // cols, i % cols]
        ranked = sorted(all_items, key=lambda x: x[2].get(target, 0.0), reverse=True)
        top = ranked[:15]
        names = [t[0] for t in top]
        kinds = [t[1] for t in top]
        vals = np.array([t[2][target] for t in top])
        stds = np.array([t[3].get(target, 0.0) for t in top])
        floor = float(np.median([t[2][target] for t in ranked[len(ranked) // 2:]]))
        y = np.arange(len(names))[::-1]
        colors = ["#1f77b4" if k == "lres" else "#ff7f0e" for k in kinds]
        ax.barh(y, vals, xerr=stds, color=colors, alpha=0.85)
        ax.axvline(floor, color="grey", linestyle=":", label=f"baseline floor = {floor:.3g}")
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("paired MSE (full sampling)")
        ax.set_title(f"target = {target}")
        ax.legend(loc="lower right", fontsize=7)
        ax.grid(True, axis="x", alpha=0.3)

    # Hide any unused panels
    for k in range(n, rows * cols):
        axes[k // cols, k % cols].axis("off")

    fig.suptitle(_title(ckpt_id, "full-sampling permutation",
                         f"num_steps={result['num_steps']} reps={result['n_repeats']}"),
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_pdf)
    plt.close(fig)
    LOGGER.info("  wrote %s", out_pdf.name)


# ---------------------------------------------------------------------------
# conditioning_ablation: 4 per-target panels + 1 correlation panel
# ---------------------------------------------------------------------------

def plot_conditioning_ablation(json_path: Path, out_pdf: Path, ckpt_id: str):
    data = json.loads(json_path.read_text())
    sigmas = [r["sigma"] for r in data["sigma_results"]]
    # Detect region-aware JSON (new) vs old global-only JSON
    regions = data.get("regions")
    targets = data.get("surface_targets")
    has_region = (
        regions is not None and targets is not None
        and "region_per_target_mse" in data["sigma_results"][0]["ablate_lres"]
        and data["sigma_results"][0]["ablate_lres"]["region_per_target_mse"]
    )
    if not targets:
        # fall back to per_var index map
        name_to_idx = {"10u": 0, "10v": 1, "2t": 3, "msl": 4, "tp": None}
        per_var_len = len(data["sigma_results"][0]["ablate_lres"]["per_var_mse"])
        targets = [t for t, idx in name_to_idx.items() if idx is not None and idx < per_var_len]
    LOGGER.info("conditioning_ablation: sigmas=%s targets=%s regions=%s",
                sigmas, targets, regions)

    n_panels = len(targets) + 1  # +1 for correlation summary
    cols = 3
    rows = (n_panels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows))
    axes = np.atleast_2d(axes).reshape(rows, cols)

    # Color: blue=lres, orange=hres, green=both; linestyle per region.
    LRES_C, HRES_C, BOTH_C = "#1f77b4", "#ff7f0e", "#2ca02c"
    region_styles = {"global": "-", "amazon": "--", "tropics": ":"}

    for i, target in enumerate(targets):
        ax = axes[i // cols, i % cols]
        if has_region:
            # Per-region {in_lres, in_hres, noisy_hres} curves.
            # Pick whichever regions are present; default is amazon_rainforest only.
            preferred = ("amazon_rainforest", "amazon", "tc_franklin", "tc_humberto", "tropics")
            plot_regions = [r for r in preferred if r in regions] or list(regions)
            # Track whether the JSON has the new noisy_hres key
            has_noisy = "ablate_noisy_hres" in data["sigma_results"][0]
            for rname in plot_regions:
                ls = region_styles.get(rname, "-")
                lres = [r["ablate_lres"]["region_per_target_mse"][rname][target]
                        for r in data["sigma_results"]]
                hres = [r["ablate_hres"]["region_per_target_mse"][rname][target]
                        for r in data["sigma_results"]]
                ax.plot(sigmas, lres, marker="o", linestyle=ls, color=LRES_C,
                        label=f"zero in_lres · {rname}")
                ax.plot(sigmas, hres, marker="s", linestyle=ls, color=HRES_C,
                        label=f"zero in_hres · {rname}")
                if has_noisy:
                    ny = [r["ablate_noisy_hres"]["region_per_target_mse"][rname][target]
                          for r in data["sigma_results"]]
                    ax.plot(sigmas, ny, marker="^", linestyle=ls, color=BOTH_C,
                            label=f"zero noisy_hres · {rname}")
            ax.set_title(f"target = {target}  ({', '.join(plot_regions)})")
        else:
            name_to_idx = {"10u": 0, "10v": 1, "2t": 3, "msl": 4, "tp": None}
            idx = name_to_idx[target]
            no_lres = [r["ablate_lres"]["per_var_mse"][idx] for r in data["sigma_results"]]
            no_hres = [r["ablate_hres"]["per_var_mse"][idx] for r in data["sigma_results"]]
            third_key = "ablate_noisy_hres" if "ablate_noisy_hres" in data["sigma_results"][0] else "ablate_both"
            third_lbl = "zero noisy_hres" if third_key == "ablate_noisy_hres" else "zero both"
            no_third = [r[third_key]["per_var_mse"][idx] for r in data["sigma_results"]]
            ax.plot(sigmas, no_lres, marker="o", color=LRES_C, label="zero in_lres")
            ax.plot(sigmas, no_hres, marker="s", color=HRES_C, label="zero in_hres")
            ax.plot(sigmas, no_third, marker="^", color=BOTH_C, label=third_lbl)
            ax.set_title(f"target = {target}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("sigma"); ax.set_ylabel("MSE(ablated, full)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=7, ncol=2)

    # Correlation summary panel (global, all channels)
    ax = axes[len(targets) // cols, len(targets) % cols]
    # Use the new ablate_noisy_hres key if present; fall back to ablate_both for legacy JSONs
    third = "ablate_noisy_hres" if "ablate_noisy_hres" in data["sigma_results"][0] else "ablate_both"
    third_label = "zero noisy_hres" if third == "ablate_noisy_hres" else "zero both"
    for label, key, marker in [("zero in_lres", "ablate_lres", "o"),
                                ("zero in_hres", "ablate_hres", "s"),
                                (third_label, third, "^")]:
        corrs = [r[key]["correlation_with_full"] for r in data["sigma_results"]]
        ax.plot(sigmas, corrs, marker=marker, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("sigma"); ax.set_ylabel("correlation(ablated, full)")
    ax.set_title("correlation summary (all channels, global)")
    ax.grid(True, which="both", alpha=0.3); ax.legend(loc="best", fontsize=8)

    # Hide unused
    for k in range(n_panels, rows * cols):
        axes[k // cols, k % cols].axis("off")

    title_suffix = " · " + ", ".join(regions) if has_region else ""
    fig.suptitle(_title(ckpt_id, "conditioning ablation (zero lres / hres / both vs full)"
                                  + title_suffix), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_pdf)
    plt.close(fig)
    LOGGER.info("  wrote %s", out_pdf.name)


# ---------------------------------------------------------------------------
# activation_profiling: single figure, 3 module panels
# ---------------------------------------------------------------------------

def plot_activation_profiling(json_path: Path, out_pdf: Path, ckpt_id: str):
    data = json.loads(json_path.read_text())
    sigmas_raw = sorted(data["profiles"].keys(), key=lambda s: float(s))
    sigmas = [float(s) for s in sigmas_raw]
    modules = ["encoder", "processor", "decoder"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, module in zip(axes, modules):
        l2 = [data["profiles"][s][module]["l2_norm"] for s in sigmas_raw]
        std = [data["profiles"][s][module]["std"] for s in sigmas_raw]
        max_abs = [data["profiles"][s][module]["max_abs"] for s in sigmas_raw]
        ax.plot(sigmas, l2, marker="o", label="L2 norm")
        ax.plot(sigmas, std, marker="s", label="std")
        ax.plot(sigmas, max_abs, marker="^", label="max |·|")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("sigma"); ax.set_ylabel("activation magnitude (log)"); ax.set_title(module)
        ax.grid(True, which="both", alpha=0.3); ax.legend(loc="best", fontsize=8)
    fig.suptitle(_title(ckpt_id, "activation profiling (encoder · processor · decoder)"),
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_pdf)
    plt.close(fig)
    LOGGER.info("  wrote %s", out_pdf.name)


# ---------------------------------------------------------------------------
# CKA: 5 heatmaps + 1 chunk summary in a 2×3 grid, single PDF
# ---------------------------------------------------------------------------

def plot_cka(json_path: Path, out_pdf: Path, ckpt_id: str):
    data = json.loads(json_path.read_text())
    sigmas = data["sigmas"]
    sigma_keys = [s for s in sigmas if str(s) in data["cka_matrices"]]
    n_heat = len(sigma_keys)
    cols = 3
    rows = (n_heat + 1 + cols - 1) // cols  # +1 for chunk summary
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5 * rows))
    axes = np.atleast_2d(axes).reshape(rows, cols)

    last_im = None
    for i, sig in enumerate(sigma_keys):
        ax = axes[i // cols, i % cols]
        m = np.array(data["cka_matrices"][str(sig)]["matrix"])
        names = data["cka_matrices"][str(sig)]["layer_names"]
        last_im = ax.imshow(m, vmin=0.0, vmax=1.0, cmap="RdBu_r", origin="lower")
        ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90, fontsize=6)
        ax.set_yticklabels(names, fontsize=6)
        if len(names) == 16:
            ax.axhline(7.5, color="black", linewidth=0.6)
            ax.axvline(7.5, color="black", linewidth=0.6)
        ax.set_title(f"σ = {sig:g}", fontsize=10)
    if last_im is not None:
        # single shared colorbar on the heatmaps row would be ideal but for simplicity
        # we let matplotlib place per-subplot colorbars only on the last one
        fig.colorbar(last_im, ax=axes[(n_heat - 1) // cols, (n_heat - 1) % cols],
                     label="CKA", shrink=0.8)

    # Chunk summary
    ax = axes[n_heat // cols, n_heat % cols]
    wa, wb, cr = [], [], []
    for sig in sigma_keys:
        m = np.array(data["cka_matrices"][str(sig)]["matrix"])
        if m.shape[0] < 16:
            continue
        ca = m[:8, :8]; cb = m[8:, 8:]; cross = m[:8, 8:]
        mask_a = ~np.eye(ca.shape[0], dtype=bool); mask_b = ~np.eye(cb.shape[0], dtype=bool)
        wa.append(ca[mask_a].mean()); wb.append(cb[mask_b].mean()); cr.append(cross.mean())
    x = sigma_keys[:len(wa)]
    ax.plot(x, wa, marker="o", label="within chunk-A (L0-7)")
    ax.plot(x, wb, marker="s", label="within chunk-B (L8-15)")
    ax.plot(x, cr, marker="^", label="cross-chunk")
    ax.set_xscale("log"); ax.set_xlabel("sigma"); ax.set_ylabel("mean CKA")
    ax.set_title("chunk summary")
    ax.set_ylim(0.5, 1.0); ax.grid(True, which="both", alpha=0.3); ax.legend(loc="best", fontsize=8)

    # Hide unused
    for k in range(n_heat + 1, rows * cols):
        axes[k // cols, k % cols].axis("off")

    fig.suptitle(_title(ckpt_id, "CKA layer similarity (16 processor blocks)"), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_pdf)
    plt.close(fig)
    LOGGER.info("  wrote %s", out_pdf.name)


# ---------------------------------------------------------------------------
# activation_patching: per target, grid_region (block × sigma) + stage (stage × sigma)
# ---------------------------------------------------------------------------

def plot_activation_patching(json_path: Path, out_pdf: Path, ckpt_id: str):
    data = json.loads(json_path.read_text())
    targets = data["surface_targets"]
    sigmas = [r["sigma"] for r in data["sigma_results"]]
    block_names = data["block_names"]
    chunk_boundary = data.get("chunk_boundary", len(block_names) // 2)
    stage_names = data.get("stage_names", [])
    storm = data.get("storm", {})

    has_gr = any("grid_region" in r for r in data["sigma_results"])
    has_st = any("stage" in r for r in data["sigma_results"])
    gr_rows = block_names + ["chunk-A", "chunk-B"]
    LOGGER.info("activation_patching: sigmas=%s targets=%s grid_region=%s stage=%s",
                sigmas, targets, has_gr, has_st)

    # Diverging colormap centered at 0 (white = no effect), recovery in [-0.2, 1.0].
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-0.2, vmax=1.0)
    n = len(targets)
    ncols = max((1 if has_gr else 0) + (1 if has_st else 0), 1)
    col_gr = 0
    col_st = 1 if has_gr else 0
    fig, axes = plt.subplots(n, ncols, figsize=(6.4 * ncols, 3.3 * n), squeeze=False)

    for r, t in enumerate(targets):
        if has_gr:
            M = np.full((len(gr_rows), len(sigmas)), np.nan)
            for j, res in enumerate(data["sigma_results"]):
                gr = res.get("grid_region")
                if not gr:
                    continue
                for i, name in enumerate(block_names):
                    M[i, j] = gr["recovery_per_block"][name].get(t, np.nan)
                M[len(block_names), j] = gr["recovery_per_chunk"]["A"].get(t, np.nan)
                M[len(block_names) + 1, j] = gr["recovery_per_chunk"]["B"].get(t, np.nan)
            ax = axes[r][col_gr]
            im = ax.imshow(M, aspect="auto", cmap="RdBu_r", norm=norm, origin="upper")
            ax.set_xticks(range(len(sigmas)))
            ax.set_xticklabels([f"{s:g}" for s in sigmas])
            ax.set_yticks(range(len(gr_rows)))
            ax.set_yticklabels(gr_rows, fontsize=6)
            ax.axhline(chunk_boundary - 0.5, color="black", linewidth=0.6)
            ax.axhline(len(block_names) - 0.5, color="grey", linewidth=1.2, linestyle="--")
            ax.set_xlabel("sigma")
            ax.set_title(f"grid_region · {t}")
            fig.colorbar(im, ax=ax, label="recovery", shrink=0.8)
        if has_st:
            M = np.full((len(stage_names), len(sigmas)), np.nan)
            for j, res in enumerate(data["sigma_results"]):
                sg = res.get("stage")
                if not sg:
                    continue
                for i, sn in enumerate(stage_names):
                    M[i, j] = sg["recovery_per_stage"][sn].get(t, np.nan)
            ax = axes[r][col_st]
            im = ax.imshow(M, aspect="auto", cmap="RdBu_r", norm=norm, origin="upper")
            ax.set_xticks(range(len(sigmas)))
            ax.set_xticklabels([f"{s:g}" for s in sigmas])
            ax.set_yticks(range(len(stage_names)))
            ax.set_yticklabels(stage_names, fontsize=8)
            ax.set_xlabel("sigma")
            ax.set_title(f"stage · {t}")
            fig.colorbar(im, ax=ax, label="recovery", shrink=0.8)

    storm_txt = "%s @ (%.1f,%.1f) r=%sdeg" % (
        storm.get("label", "?"), storm.get("center_lat", 0.0),
        storm.get("center_lon", 0.0), storm.get("radius_deg", "?"))
    fig.suptitle(_title(ckpt_id,
                        f"activation patching ({data.get('corruption', '?')}) — {storm_txt}"),
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_pdf)
    plt.close(fig)
    LOGGER.info("  wrote %s", out_pdf.name)


# ---------------------------------------------------------------------------
# integrated_gradients: storm-centered "what drives the extreme" maps + coherence
# ---------------------------------------------------------------------------

try:
    import cartopy.crs as ccrs
    from cartopy.geodesic import Geodesic
    _HAS_CARTOPY = True
except Exception:  # pragma: no cover
    _HAS_CARTOPY = False


def _to180(x):
    return ((np.asarray(x, dtype=float) + 180.0) % 360.0) - 180.0


def _geo_panel(fig, nrows, ncols, pos, lat, lon, vals, *, title, diverging,
               center=None, rings=(200, 500)):
    """One geographic scatter panel; coastlines best-effort (offline-safe)."""
    lon = _to180(lon)
    if _HAS_CARTOPY:
        ax = fig.add_subplot(nrows, ncols, pos, projection=ccrs.PlateCarree())
        tr = {"transform": ccrs.PlateCarree()}
    else:
        ax = fig.add_subplot(nrows, ncols, pos)
        tr = {}
    if diverging:
        vmax = float(np.percentile(np.abs(vals), 99)) or 1e-12
        kw = dict(cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        cbl = "signed attribution"
    else:
        vmax = float(np.percentile(vals, 99)) or 1e-12
        kw = dict(cmap="magma", vmin=0.0, vmax=vmax)
        cbl = "|attribution|"
    sc = ax.scatter(lon, lat, c=vals, s=5, **kw, **tr)
    if _HAS_CARTOPY:
        try:
            ax.coastlines(resolution="110m", linewidth=0.5)
        except Exception:
            pass
        pad = 1.0
        ax.set_extent([lon.min() - pad, lon.max() + pad, lat.min() - pad, lat.max() + pad],
                      crs=ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True, linewidth=0.2, alpha=0.4)
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = {"size": 6}
    else:
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
    if center is not None:
        clon, clat = _to180(center[1]), center[0]
        ax.plot([clon], [clat], marker="*", color="cyan", ms=15, mec="k", **tr)
        if _HAS_CARTOPY:
            geod = Geodesic()
            for rk in rings:
                circ = np.asarray(geod.circle(lon=float(clon), lat=float(clat),
                                              radius=rk * 1000.0, n_samples=120))
                ax.plot(circ[:, 0], circ[:, 1], "k--", lw=0.7, transform=ccrs.PlateCarree())
    ax.set_title(title, fontsize=8)
    fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02, label=cbl)
    return ax


def plot_integrated_gradients(json_path: Path, out_pdf: Path, ckpt_id: str):
    """6-page report focused on the eye extreme:
       1-4: one page per surface target, 4 signed driver-variable maps (zoomed).
       5  : locality (fraction of influence within 500 km of the eye) vs sigma.
       6  : top-12 driver variables per target (eye), bar chart.
    """
    data = json.loads(json_path.read_text())
    targets = data["surface_targets"]
    functionals = data["functionals"]
    sigmas = sorted({e["sigma"] for e in data["results"]})
    map_sigma = data.get("map_sigma")
    disp_sigma = map_sigma if (map_sigma in sigmas) else sigmas[-1]
    idx = {(e["functional"], e["target"], e["sigma"]): e for e in data["results"]}
    FKEY = "eye"  # report focuses on the extreme-core functional only
    if FKEY not in functionals:
        LOGGER.warning("integrated_gradients: 'eye' functional missing — nothing to render")
        return
    LOGGER.info("integrated_gradients: targets=%s disp_sigma=%g cartopy=%s",
                targets, disp_sigma, _HAS_CARTOPY)

    with PdfPages(out_pdf) as pdf:
        # ---- pages 1..N_targets : one per target, top-4 signed driver-variable maps ----
        for t in targets:
            e = idx.get((FKEY, t, disp_sigma))
            if e is None or "zoom" not in e:
                continue
            z = e["zoom"]
            center = (z["center_lat"], z["center_lon"])
            lat, lon = np.asarray(z["lat"]), np.asarray(z["lon"])
            varmaps = list(z["vars"].items())[:4]
            ncols = len(varmaps)
            fig = plt.figure(figsize=(4.8 * ncols, 4.8))
            for j, (vname, vals) in enumerate(varmaps):
                _geo_panel(fig, 1, ncols, 1 + j, lat, lon, np.asarray(vals),
                           title=f"{vname}  (signed)", diverging=True, center=center)
            loc = e.get("coherence", {}).get("frac_within_500km")
            locstr = f"{loc*100:.0f}% of influence within 500 km of eye" if loc is not None else ""
            fig.suptitle(_title(ckpt_id, f"eye — what drives the {t} extreme",
                                f"σ={disp_sigma:g}", locstr), fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, 0.92))
            pdf.savefig(fig); plt.close(fig)

        # ---- page : locality vs sigma (single metric, all targets) ----
        fig, ax = plt.subplots(1, 1, figsize=(9, 5))
        for t in targets:
            ys = [(idx.get((FKEY, t, s), {}) or {}).get("coherence", {}).get("frac_within_500km",
                                                                              np.nan)
                  for s in sigmas]
            ax.plot(sigmas, ys, marker="o", label=t, linewidth=2)
        ax.set_xscale("log")
        ax.set_xlabel("sigma (diffusion noise)")
        ax.set_ylabel("fraction of input influence within 500 km of eye")
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="lower left", fontsize=9)
        ax.set_title("Locality of the model's receptive field on the eye, vs σ", fontsize=10)
        fig.suptitle(_title(ckpt_id, "eye · locality vs sigma"), fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(fig); plt.close(fig)

        # ---- page : top-12 driver variables per target (eye, at disp_sigma) ----
        n = len(targets); cols = 2; rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 3.4 * rows), squeeze=False)
        for i, t in enumerate(targets):
            ax = axes[i // cols][i % cols]
            e = idx.get((FKEY, t, disp_sigma))
            if e is None:
                ax.axis("off"); continue
            items = ([(v["name"], "lres", v["mean_abs"], v["signed_mean"]) for v in e["lres"].values()]
                     + [(v["name"], "hres", v["mean_abs"], v["signed_mean"]) for v in e["hres"].values()])
            items.sort(key=lambda x: x[2], reverse=True)
            top = items[:12]
            yy = np.arange(len(top))[::-1]
            ax.barh(yy, [x[2] for x in top],
                    color=["#d62728" if x[3] >= 0 else "#1f77b4" for x in top],
                    edgecolor=["black" if x[1] == "hres" else "none" for x in top], linewidth=1.0)
            ax.set_yticks(yy); ax.set_yticklabels([x[0] for x in top], fontsize=7)
            ax.set_xlabel("mean |attr|"); ax.set_title(t, fontsize=9)
            ax.grid(True, axis="x", alpha=0.3)
        for k in range(n, rows * cols):
            axes[k // cols][k % cols].axis("off")
        fig.suptitle(_title(ckpt_id, f"top driver variables · eye · σ={disp_sigma:g}",
                            "red=+ / blue=− · hres edged"), fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(fig); plt.close(fig)
    LOGGER.info("  wrote %s", out_pdf.name)


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------

RENDERERS = {
    "feature_permutation": plot_feature_permutation,
    "full_sampling": plot_full_sampling,
    "conditioning_ablation": plot_conditioning_ablation,
    "activation_profiling": plot_activation_profiling,
    "cka": plot_cka,
    "activation_patching": plot_activation_patching,
    "integrated_gradients": plot_integrated_gradients,
}


def main():
    parser = argparse.ArgumentParser(description="Interp viz CLI — one PDF per method")
    parser.add_argument("--run-dir", required=True,
                        help="Per-checkpoint root, e.g. ~/perm/interp/59e4_300k")
    parser.add_argument("--tools", default=None,
                        help="Comma-separated subset of tools to render; default all detected")
    parser.add_argument("--no-overwrite", action="store_true",
                        help="Skip plots that already exist")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise SystemExit(f"run-dir does not exist: {run_dir}")
    ckpt_id = run_dir.name

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Rendering interp plots for ckpt_id=%s → %s", ckpt_id, plots_dir)

    only = set(args.tools.split(",")) if args.tools else None

    n_rendered = 0
    for tool, (subdir, json_name, renderer_key) in TOOL_REGISTRY.items():
        if only is not None and tool not in only:
            continue
        json_path = run_dir / subdir / json_name
        if not json_path.exists():
            LOGGER.info("[%s] no JSON at %s — skip", tool, json_path.relative_to(run_dir))
            continue
        out_pdf = plots_dir / f"{tool}.pdf"
        if out_pdf.exists() and args.no_overwrite:
            LOGGER.info("[%s] %s already exists — skip", tool, out_pdf.name)
            continue
        LOGGER.info("[%s] rendering → %s", tool, out_pdf.name)
        RENDERERS[renderer_key](json_path, out_pdf, ckpt_id)
        n_rendered += 1
    LOGGER.info("Done. Rendered %d PDFs in %s", n_rendered, plots_dir)


if __name__ == "__main__":
    main()
