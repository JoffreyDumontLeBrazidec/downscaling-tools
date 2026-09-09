"""Score the trained station head and its three controls with one piece of code.

Every number this script writes is a fair kernel continuous ranked probability
score, computed by the same function, `fair_crps` of `head.py`, on exactly the same
rows. The four things scored are:

* `head`, the trained station head fed the predicted 9 km neighbourhood and the
  interpolated AIFS input;
* `nearest`, the current scorecard rule, which is the nearest output point of each
  member with the 0.0065 K per metre lapse-rate correction for 2 m temperature only
  and no correction for dewpoint or wind speed;
* `head_xinterp`, the same head trained the same way but fed only the interpolated
  AIFS input, which answers what the 9 km field adds over its own input;
* `analysis`, the operational analysis at the nearest point, which is deterministic,
  so its fair CRPS is exactly the absolute error. It is the ceiling a gridded
  product can reach and it should be clearly the best of the three forecasts.

The table is produced on every stratum the design note asks for: seen and held-out
stations separately, the stable network and the full network beside each other, all
stations together and then split by terrain class and by region, at each lead
separately. The long-format CSV carries every cell; the markdown file shows the
overall table and the strata that matter most.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import dataset as D  # noqa: E402
import train as T  # noqa: E402
from head import StationHead, fair_crps  # noqa: E402

METHODS = ["head", "head_xinterp", "nearest", "analysis"]


def load_model(ckpt_path: Path, device) -> tuple[StationHead, dict, str]:
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = StationHead(n_neighbourhood=blob["n_neighbourhood"], n_static=blob["n_static"],
                        hidden=blob["hidden"], depth=blob["depth"], n_time=blob["n_time"]).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    stats = {k: (np.asarray(v) if isinstance(v, list) else v) for k, v in blob["stats"].items()}
    return model, stats, blob["features"]


@torch.no_grad()
def members_for(model, stats, features, c: dict, device, batch: int = 8192) -> np.ndarray:
    keep = np.ones(c["y"].size, dtype=bool)
    xs, sts, tds, anc, ys = T.to_tensors(c, keep, stats, device)
    out = np.empty((xs.shape[0], anc.shape[1]), dtype=np.float32)
    for i in range(0, xs.shape[0], batch):
        sl = slice(i, i + batch)
        out[sl] = (anc[sl] + stats["y_scale"] * model(xs[sl], sts[sl], tds[sl])).cpu().numpy()
    return out


def crps_rows(members: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-row fair CRPS, as numpy, for an (n, m) ensemble against an (n,) observation."""
    m = members.shape[1]
    skill = np.abs(members - y[:, None]).mean(axis=1)
    if m > 1:
        pair = np.abs(members[:, :, None] - members[:, None, :]).sum(axis=(1, 2))
        spread = pair / (2.0 * m * (m - 1))
    else:
        spread = np.zeros_like(skill)
    return skill - spread


def score(args) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.run_dir)
    cfg = json.loads((run_dir / "run_config.json").read_text())
    model_full, stats_full, feat_full = load_model(run_dir / "head.pt", device)
    model_int = stats_int = feat_int = None
    if args.xinterp_run_dir:
        model_int, stats_int, feat_int = load_model(Path(args.xinterp_run_dir) / "head.pt", device)

    data_dir = Path(cfg["data_dir"])
    cases = T.list_cases(data_dir)
    which = set(cfg["validation_cases"]) if args.on == "validation" else set(cfg["training_cases"])
    cases = cases[cases["case_id"].isin(which)].reset_index(drop=True)
    print("scoring %s on %d %s cases of %s" % (cfg["target"], len(cases), args.on, data_dir), flush=True)

    records = []
    for r in cases.itertuples():
        c_full = T.load_case(r.path, feat_full)
        per = {}
        per["head"] = crps_rows(members_for(model_full, stats_full, feat_full, c_full, device), c_full["y"])
        per["nearest"] = crps_rows(c_full["ctrl_near"], c_full["y"])
        an = c_full["analysis"]
        per["analysis"] = np.where(np.isfinite(an), np.abs(an - c_full["y"]), np.nan)
        if model_int is not None:
            c_int = T.load_case(r.path, feat_int)
            per["head_xinterp"] = crps_rows(members_for(model_int, stats_int, feat_int, c_int, device),
                                            c_int["y"])
        base = pd.DataFrame({
            "case_id": c_full["case_id"], "lead_h": c_full["lead_h"],
            "stnid": c_full["stnid"], "holdout": c_full["holdout"], "stable": c_full["stable"],
            "terrain": [D.TERRAIN_CLASSES[i] for i in c_full["terrain"]],
            "region": [D.REGIONS[i] for i in c_full["region"]],
        })
        for k, v in per.items():
            base[k] = v
        records.append(base)
        print("  %s: %d rows" % (c_full["case_id"], len(base)), flush=True)

    df = pd.concat(records, ignore_index=True)
    methods = [m for m in METHODS if m in df.columns]

    rows = []
    for lead, dl in [("all", df)] + [(int(l), df[df["lead_h"] == l]) for l in sorted(df["lead_h"].unique())]:
        for station_set, ds in [("seen", dl[~dl["holdout"]]), ("holdout", dl[dl["holdout"]]),
                                ("all", dl)]:
            for network, dn in [("stable", ds[ds["stable"]]), ("full", ds)]:
                groups = [("all", "all", dn)]
                groups += [("terrain", t, dn[dn["terrain"] == t]) for t in D.TERRAIN_CLASSES]
                groups += [("region", g, dn[dn["region"] == g]) for g in D.REGIONS]
                for kind, name, dg in groups:
                    if not len(dg):
                        continue
                    for meth in methods:
                        v = dg[meth].to_numpy(dtype=np.float64)
                        ok = np.isfinite(v)
                        if not ok.any():
                            continue
                        rows.append({"target": cfg["target"], "on": args.on, "lead_h": lead,
                                     "station_set": station_set, "network": network,
                                     "group_kind": kind, "group": name, "method": meth,
                                     "n_rows": int(ok.sum()),
                                     "n_stations": int(dg.loc[ok, "stnid"].nunique()),
                                     "fair_crps": float(v[ok].mean())})
    table = pd.DataFrame(rows)

    label = "PIPELINE_TEST_NOT_A_RESULT" if cfg.get("pipeline_test") else "firstcut"
    stem = "scores_%s_%s_%s" % (cfg["target"], args.on, label)
    csv = run_dir / (stem + ".csv")
    table.to_csv(csv, index=False)

    md = run_dir / (stem + ".md")
    write_markdown(md, table, cfg, args, label)
    print("wrote %s and %s" % (csv, md), flush=True)
    return csv


def _md_table(piv: pd.DataFrame) -> str:
    """A markdown table without depending on tabulate being installed."""
    idx_names = [n if n else "" for n in (piv.index.names or [])]
    header = idx_names + [str(c) for c in piv.columns]
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join(["---"] * len(header)) + " |"]
    for key, row in piv.iterrows():
        keys = list(key) if isinstance(key, tuple) else [key]
        cells = [str(k) for k in keys]
        for c in piv.columns:
            v = row[c]
            cells.append("" if pd.isna(v) else (str(int(v)) if c == "n_rows" else "%.4f" % float(v)))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def write_markdown(path: Path, table: pd.DataFrame, cfg: dict, args, label: str) -> None:
    tgt = cfg["target"]
    unit = "m/s" if tgt == "10ff" else "K"
    lines = []
    if cfg.get("pipeline_test"):
        lines += ["# PIPELINE TEST, NOT A RESULT: fair CRPS of the station head and its controls",
                  "",
                  "Every number below was produced only to exercise the stage 2a chain end to end "
                  "on the handful of cases that had finished when it was run. The split between "
                  "the temporary training and check sets is a split of the validation window by "
                  "initialisation date, not the split of the design note, and the head saw far too "
                  "few cases to have learned anything. None of these numbers is a result and none "
                  "may be quoted as one.", ""]
    else:
        lines += ["# Fair CRPS of the station head and its controls, stage 2a first cut", ""]
    lines += [
        "Target: `%s` (%s). Run `%s`, checkpoint `%s`. Scored on the %s cases."
        % (tgt, unit, cfg["run_id"], cfg["checkpoint"], args.on),
        "",
        "Methods: `head` is the trained station head on the 9 km prediction and the interpolated "
        "input; `nearest` is the nearest output point with quaver's 0.0065 K per metre lapse-rate "
        "correction, applied to 2 m temperature only; `head_xinterp` is the same head trained the "
        "same way on the interpolated AIFS input alone; `analysis` is the operational analysis at "
        "the nearest point, which is deterministic so its fair CRPS is its absolute error.",
        "",
    ]

    def block(title: str, sel: pd.DataFrame, index_cols: list[str]) -> None:
        if not len(sel):
            return
        piv = sel.pivot_table(index=index_cols, columns="method", values="fair_crps")
        n = sel.pivot_table(index=index_cols, columns="method", values="n_rows").max(axis=1)
        piv["n_rows"] = n.astype(int)
        lines.append("## " + title)
        lines.append("")
        lines.append(_md_table(piv))
        lines.append("")

    block("Overall, by lead, station set and network",
          table[(table["group_kind"] == "all")],
          ["lead_h", "station_set", "network"])
    block("By terrain class, held-out stations, stable network",
          table[(table["group_kind"] == "terrain") & (table["station_set"] == "holdout")
                & (table["network"] == "stable")],
          ["lead_h", "group"])
    block("By region, held-out stations, stable network",
          table[(table["group_kind"] == "region") & (table["station_set"] == "holdout")
                & (table["network"] == "stable")],
          ["lead_h", "group"])
    block("By terrain class, seen stations, stable network",
          table[(table["group_kind"] == "terrain") & (table["station_set"] == "seen")
                & (table["network"] == "stable")],
          ["lead_h", "group"])
    block("By region, seen stations, stable network",
          table[(table["group_kind"] == "region") & (table["station_set"] == "seen")
                & (table["network"] == "stable")],
          ["lead_h", "group"])
    lines.append("Label: " + label)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="the run folder of the full head")
    ap.add_argument("--xinterp-run-dir", default=None,
                    help="the run folder of the control head fed only x_interp")
    ap.add_argument("--on", default="validation", choices=["validation", "training"])
    args = ap.parse_args()
    score(args)


if __name__ == "__main__":
    main()
