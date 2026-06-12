"""Cross-checkpoint comparison from run-dir JSONs (no model needed).

Distills each run dir (~/perm/interp/<ckpt_id>) into a handful of summary
scalars per surface target, prints/writes one table, and overlays the two most
diagnostic curves across checkpoints. Open the per-run report.pdf only when a
scalar moves.

Scalars
-------
  perm_top5_share   : top-5 inputs' share of total permutation importance
                      (high = few dominant drivers)
  tail_ratio        : tail vs region paired-MSE importance ratio of the top
                      input (tail-conditioned runs only)
  lres_commit_sigma : first sigma (descending) where corr(zero_lres, full) < 0.5
                      ("below this noise level the model stops copying lres")
  ig_locality_500   : fraction of IG influence within 500 km of the storm probe
  stage_enc_hidden  : stage-patch recovery of enc_hidden at the smallest sigma
  cka_cross_chunk   : mean cross-chunk CKA at the smallest sigma

Usage
-----
    python -m interp compare --run-dirs ~/perm/interp/59e4_300k ~/perm/interp/cfec_200k \
        --output-dir ~/perm/interp/compare_59e4_vs_cfec
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np

from interp.viz.report import find_json

LOGGER = logging.getLogger(__name__)


def _load(run_dir: Path, tool: str) -> dict | None:
    p = find_json(run_dir, tool)
    if p is None:
        return None
    return json.loads(p.read_text())


def summarize_run(run_dir: Path) -> dict:
    """{(scalar, target): value} for one run dir."""
    out: dict[tuple[str, str], float] = {}

    perm = _load(run_dir, "feature_permutation")
    if perm:
        res = perm["sigma_results"][len(perm["sigma_results"]) // 2]  # mid sigma
        targets = res.get("surface_targets", [])
        for t in targets:
            vals = sorted((info["paired_mse_per_target"].get(t, 0.0)
                           for src in ("lres_importance", "hres_importance")
                           for info in res[src].values()), reverse=True)
            total = sum(vals)
            if total > 0:
                out[("perm_top5_share", t)] = sum(vals[:5]) / total
            # tail ratio of the single most important input (if tail data present)
            entries = list(res["lres_importance"].values()) + list(
                res["hres_importance"].values())
            entries.sort(key=lambda i: i["paired_mse_per_target"].get(t, 0.0),
                         reverse=True)
            top = entries[0] if entries else None
            ext = (top or {}).get("extreme_paired_mse_per_target") or {}
            reg = (top or {}).get("region_paired_mse_per_target") or {}
            for rname in ext:
                e, r = ext[rname].get(t), reg.get(rname, {}).get(t)
                if e is not None and r:
                    out[("tail_ratio", t)] = e / r
                    break

    abl = _load(run_dir, "conditioning_ablation")
    if abl:
        sigmas = [r["sigma"] for r in abl["sigma_results"]]
        corrs = [r["ablate_lres"]["correlation_with_full"] for r in abl["sigma_results"]]
        commit = next((s for s, c in sorted(zip(sigmas, corrs), reverse=True)
                       if c < 0.5), None)
        if commit is not None:
            out[("lres_commit_sigma", "all")] = commit

    ig = _load(run_dir, "integrated_gradients")
    if ig:
        # Prefer "eye" so the locality scalar compares the same probe across runs.
        spatial = sorted((f for f in ig["functionals"]
                          if f == "eye" or f.startswith("box:")),
                         key=lambda f: f != "eye")
        if spatial:
            fkey = spatial[0]
            map_sigma = ig.get("map_sigma")
            for e in ig["results"]:
                if e["functional"] == fkey and e["sigma"] == map_sigma:
                    loc = e.get("coherence", {}).get("frac_within_500km")
                    if loc is not None:
                        out[("ig_locality_500", e["target"])] = loc

    patch = _load(run_dir, "activation_patching")
    if patch and patch["sigma_results"]:
        res = min(patch["sigma_results"], key=lambda r: r["sigma"])
        sg = res.get("stage")
        if sg:
            for t, v in sg["recovery_per_stage"].get("enc_hidden", {}).items():
                out[("stage_enc_hidden", t)] = v

    cka = _load(run_dir, "cka")
    if cka and cka["cka_matrices"]:
        sig = min(cka["cka_matrices"], key=float)
        m = np.array(cka["cka_matrices"][sig]["matrix"])
        if m.shape[0] >= 16:
            out[("cka_cross_chunk", "all")] = float(m[:8, 8:].mean())
    return out


def overlay_curves(run_dirs: list[Path], out_pdf: Path):
    """One figure: ablation lres-correlation vs sigma + IG locality vs sigma,
    overlaid across checkpoints."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for rd in run_dirs:
        abl = _load(rd, "conditioning_ablation")
        if abl:
            sigmas = [r["sigma"] for r in abl["sigma_results"]]
            corrs = [r["ablate_lres"]["correlation_with_full"]
                     for r in abl["sigma_results"]]
            ax1.plot(sigmas, corrs, marker="o", label=rd.name)
        ig = _load(rd, "integrated_gradients")
        if ig:
            spatial = sorted((f for f in ig["functionals"]
                              if f == "eye" or f.startswith("box:")),
                             key=lambda f: f != "eye")
            if spatial:
                by_sig = {}
                for e in ig["results"]:
                    if e["functional"] == spatial[0]:
                        loc = e.get("coherence", {}).get("frac_within_500km")
                        if loc is not None:
                            by_sig.setdefault(e["sigma"], []).append(loc)
                xs = sorted(by_sig)
                ax2.plot(xs, [float(np.mean(by_sig[s])) for s in xs], marker="o",
                         label=rd.name)
    for ax, ylab, title in ((ax1, "corr(zero_lres, full)", "lres reliance vs σ"),
                            (ax2, "frac influence ≤ 500 km", "IG locality vs σ")):
        ax.set_xscale("log")
        ax.set_xlabel("sigma")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description="Compare interp runs across checkpoints")
    p.add_argument("--run-dirs", nargs="+", required=True)
    p.add_argument("--output-dir", default=None,
                   help="Where to write compare.csv + compare.pdf (default: stdout only)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run_dirs = [Path(d).expanduser().resolve() for d in args.run_dirs]
    summaries = {rd.name: summarize_run(rd) for rd in run_dirs}
    all_keys = sorted({k for s in summaries.values() for k in s})

    # Table: one row per (scalar, target), one column per run.
    names = list(summaries)
    header = f"{'scalar':<20s} {'target':<6s}" + "".join(f" {n:>16s}" for n in names)
    print("\n" + header)
    print("-" * len(header))
    rows = []
    for key in all_keys:
        scalar, target = key
        cells = [summaries[n].get(key) for n in names]
        rows.append((scalar, target, *cells))
        print(f"{scalar:<20s} {target:<6s}"
              + "".join(f" {('%16.4f' % c) if c is not None else ' ' * 15 + '—'}"
                        for c in cells))

    if args.output_dir:
        out = Path(args.output_dir).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "compare.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scalar", "target", *names])
            w.writerows(rows)
        overlay_curves(run_dirs, out / "compare.pdf")
        LOGGER.info("Wrote %s/compare.csv and compare.pdf", out)


if __name__ == "__main__":
    main()
