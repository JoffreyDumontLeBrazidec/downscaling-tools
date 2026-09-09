"""Train the station head on the assembled stage 2a cases.

One run trains one head for one target variable. The choice of one head per target
rather than a single head with three outputs is deliberate and worth stating. The
three targets are measured in different units, a kelvin for temperature and dewpoint
and a metre per second for wind speed, so a joint loss would need a weighting
between them that nothing in the design fixes; the set of stations that report each
parameter is different, so a joint head would train on a ragged mask; and the
comparison the epic asks for is made target by target against a control that is
itself target-specific. Three small heads cost minutes each on one GPU, so nothing
is gained by sharing them.

The head predicts a correction rather than the value itself. For each member the
network output is added to the anchor, which is the nearest output point of that
member with quaver's lapse-rate correction for 2 m temperature, and is scaled by the
standard deviation of the anchor's own error on the training cases. This means an
untrained head starts exactly at the nearest-point control, so the training curve
reads directly as "how much the head improves on the control", and the optimiser
never has to discover the mean of a field in kelvin. The control head that is fed
only the interpolated AIFS input uses the interpolated nearest point as its anchor,
so that the difference between the two runs is only ever what the 9 km field carries.

Cases are streamed one file at a time, because the whole first cut does not fit in
memory. An epoch is one pass over a shuffled list of training case files, and within
a case the station rows are shuffled into minibatches. Early stopping watches the
fair CRPS on the validation cases, on seen stations of the stable network.

Everything a later session needs to reproduce the run, that is the seed, the case
lists, the feature names, the standardisation statistics and the hyperparameters, is
written next to the checkpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import dataset as D  # noqa: E402
from head import StationHead, fair_crps  # noqa: E402

RUN_ROOT = Path("/home/ecm5702/perm/station-head-adapter/head")
DEFAULT_VAL_FROM = "2026-08-18"


# ---------------------------------------------------------------- case handling

def list_cases(data_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(data_dir.glob("*.npz")):
        z = np.load(p, allow_pickle=False)
        rows.append({"path": str(p), "case_id": p.stem,
                     "init": pd.Timestamp(str(z["init"])),
                     "valid": pd.Timestamp(str(z["valid"])),
                     "lead_h": int(z["lead_h"]), "n": int(z["y"].size),
                     "n_members": int(z["feat_pred"].shape[1])})
        z.close()
    if not rows:
        raise SystemExit("no assembled case files in " + str(data_dir))
    return pd.DataFrame(rows).sort_values(["init", "lead_h"]).reset_index(drop=True)


def split_cases(cases: pd.DataFrame, val_from: str, val_to: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    lo = pd.Timestamp(val_from)
    hi = pd.Timestamp(val_to) if val_to else None
    is_val = cases["init"] >= lo
    if hi is not None:
        is_val &= cases["init"] <= hi
    return cases[~is_val].reset_index(drop=True), cases[is_val].reset_index(drop=True)


# An in-memory cache of assembled cases. An epoch reads every training case file
# again, and on a busy scratch filesystem that reading, not the arithmetic, is what
# an epoch costs: during the pipeline test of 2026-09-09 an epoch took fourteen
# seconds of computation and about four minutes of waiting for four case files,
# because the inference arrays were writing eight-gigabyte prediction files on the
# same disks. Set the budget with --cache-gb; zero, the default, disables it.
_CACHE: dict = {}
_CACHE_BYTES = 0
_CACHE_BUDGET = 0


def set_cache_budget(gigabytes: float) -> None:
    global _CACHE_BUDGET, _CACHE, _CACHE_BYTES
    _CACHE_BUDGET = int(gigabytes * (1 << 30))
    _CACHE = {}
    _CACHE_BYTES = 0


def _cache_size(c: dict) -> int:
    return sum(int(v.nbytes) for v in c.values() if isinstance(v, np.ndarray))


def load_case(path: str, features: str) -> dict:
    global _CACHE_BYTES
    key = (path, features)
    if _CACHE_BUDGET and key in _CACHE:
        return _CACHE[key]
    c = _load_case_from_disk(path, features)
    if _CACHE_BUDGET:
        size = _cache_size(c)
        if _CACHE_BYTES + size <= _CACHE_BUDGET:
            _CACHE[key] = c
            _CACHE_BYTES += size
    return c


def _load_case_from_disk(path: str, features: str) -> dict:
    z = np.load(path, allow_pickle=False)
    if features == "both":
        x = np.concatenate([z["feat_pred"], z["feat_int"]], axis=2)
        anchor = z["ctrl_near"]
    elif features == "xinterp":
        x = z["feat_int"]
        anchor = z["ctrl_int_near"]
    elif features == "pred":
        x = z["feat_pred"]
        anchor = z["ctrl_near"]
    else:
        raise ValueError(features)
    out = {"x": x.astype(np.float32), "anchor": anchor.astype(np.float32),
           "static": z["static"].astype(np.float32), "tod": z["tod"].astype(np.float32),
           "y": z["y"].astype(np.float32), "analysis": z["analysis"].astype(np.float32),
           "ctrl_near": z["ctrl_near"].astype(np.float32),
           "ctrl_int_near": z["ctrl_int_near"].astype(np.float32),
           "holdout": z["holdout"], "stable": z["stable"],
           "terrain": z["terrain"], "region": z["region"],
           "lead_h": int(z["lead_h"]), "case_id": str(z["case_id"]),
           "stnid": z["stnid"]}
    z.close()
    return out


def row_filter(c: dict, seen: bool | None, stable_only: bool) -> np.ndarray:
    keep = np.ones(c["y"].size, dtype=bool)
    if seen is True:
        keep &= ~c["holdout"]
    elif seen is False:
        keep &= c["holdout"]
    if stable_only:
        keep &= c["stable"]
    return keep


# ---------------------------------------------------------------- standardisation

def feature_statistics(paths: list[str], features: str, max_cases: int, rng: np.random.Generator) -> dict:
    """Streaming mean and standard deviation of every feature over training rows."""
    use = list(paths)
    if len(use) > max_cases:
        use = [use[i] for i in rng.choice(len(use), size=max_cases, replace=False)]
    n = 0
    n_s = 0
    s_x = s_x2 = None
    s_s = s_s2 = None
    resid = []
    for p in use:
        c = load_case(p, features)
        keep = row_filter(c, seen=True, stable_only=False)
        x = c["x"][keep].reshape(-1, c["x"].shape[2]).astype(np.float64)
        st = c["static"][keep].astype(np.float64)
        if s_x is None:
            s_x = np.zeros(x.shape[1]); s_x2 = np.zeros(x.shape[1])
            s_s = np.zeros(st.shape[1]); s_s2 = np.zeros(st.shape[1])
        s_x += x.sum(axis=0); s_x2 += (x * x).sum(axis=0); n += x.shape[0]
        s_s += st.sum(axis=0); s_s2 += (st * st).sum(axis=0); n_s += st.shape[0]
        resid.append((c["y"][keep][:, None] - c["anchor"][keep]).reshape(-1))
    mean_x = s_x / max(n, 1)
    std_x = np.sqrt(np.maximum(s_x2 / max(n, 1) - mean_x ** 2, 0.0))
    mean_s = s_s / max(n_s, 1)
    std_s = np.sqrt(np.maximum(s_s2 / max(n_s, 1) - mean_s ** 2, 0.0))
    std_x[std_x < 1e-6] = 1.0
    std_s[std_s < 1e-6] = 1.0
    r = np.concatenate(resid)
    y_scale = float(np.std(r)) if r.size else 1.0
    if not np.isfinite(y_scale) or y_scale <= 0:
        y_scale = 1.0
    return {"mean_x": mean_x, "std_x": std_x, "mean_s": mean_s, "std_s": std_s,
            "y_scale": y_scale, "n_rows": int(n), "n_cases_used": len(use)}


def to_tensors(c: dict, keep: np.ndarray, stats: dict, device: torch.device):
    x = (c["x"][keep] - stats["mean_x"]) / stats["std_x"]
    st = (c["static"][keep] - stats["mean_s"]) / stats["std_s"]
    return (torch.as_tensor(x, dtype=torch.float32, device=device),
            torch.as_tensor(st, dtype=torch.float32, device=device),
            torch.as_tensor(c["tod"][keep], dtype=torch.float32, device=device),
            torch.as_tensor(c["anchor"][keep], dtype=torch.float32, device=device),
            torch.as_tensor(c["y"][keep], dtype=torch.float32, device=device))


# ---------------------------------------------------------------- evaluation

@torch.no_grad()
def evaluate(model, paths: list[str], features: str, stats: dict, device, stable_only: bool = True,
             seen: bool | None = True, batch: int = 8192) -> tuple[float, int]:
    model.eval()
    total, count = 0.0, 0
    for p in paths:
        c = load_case(p, features)
        keep = row_filter(c, seen=seen, stable_only=stable_only)
        if not keep.any():
            continue
        xs, sts, tds, anc, ys = to_tensors(c, keep, stats, device)
        for i in range(0, xs.shape[0], batch):
            sl = slice(i, i + batch)
            pred = anc[sl] + stats["y_scale"] * model(xs[sl], sts[sl], tds[sl])
            v = fair_crps(pred, ys[sl])
            k = int(ys[sl].shape[0])
            total += float(v) * k
            count += k
        del xs, sts, tds, anc, ys
    model.train()
    return (total / count if count else float("nan")), count


# ---------------------------------------------------------------- training

def train(args) -> Path:
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    set_cache_budget(getattr(args, "cache_gb", 0.0))
    data_dir = Path(args.data_dir)
    cases = list_cases(data_dir)
    tr, va = split_cases(cases, args.val_init_from, args.val_init_to)
    if not len(tr) or not len(va):
        raise SystemExit("empty split: %d training and %d validation cases" % (len(tr), len(va)))
    print("training cases %d (%s .. %s), validation cases %d (%s .. %s)"
          % (len(tr), tr["init"].min(), tr["init"].max(), len(va), va["init"].min(), va["init"].max()),
          flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device " + str(device), flush=True)

    stats = feature_statistics(list(tr["path"]), args.features, args.stats_cases, rng)
    print("standardisation from %d cases and %d rows; anchor error scale %.4f"
          % (stats["n_cases_used"], stats["n_rows"], stats["y_scale"]), flush=True)

    n_neigh = int(np.load(tr["path"].iloc[0])["feat_pred"].shape[2])
    n_in_neigh = 2 * n_neigh if args.features == "both" else n_neigh
    model = StationHead(n_neighbourhood=n_in_neigh, n_static=D.N_STATIC,
                        hidden=args.hidden, depth=args.depth, n_time=D.N_TIME).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print("head: %d neighbourhood features, %d static, %d parameters"
          % (n_in_neigh, D.N_STATIC, n_par), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_dir = RUN_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / "head.pt"
    curve_rows = []
    best, best_epoch, since = float("inf"), -1, 0

    val0, nval0 = evaluate(model, list(va["path"]), args.features, stats, device)
    print("epoch 0 (untrained, equals the anchor control): validation fair CRPS %.4f on %d rows"
          % (val0, nval0), flush=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        order = rng.permutation(len(tr))
        run_loss, run_n = 0.0, 0
        for ci in order:
            c = load_case(tr["path"].iloc[int(ci)], args.features)
            keep = row_filter(c, seen=True, stable_only=args.train_on_stable_only)
            if not keep.any():
                continue
            xs, sts, tds, anc, ys = to_tensors(c, keep, stats, device)
            perm = torch.randperm(xs.shape[0], device=device)
            for i in range(0, xs.shape[0], args.batch):
                sel = perm[i:i + args.batch]
                pred = anc[sel] + stats["y_scale"] * model(xs[sel], sts[sel], tds[sel])
                loss = fair_crps(pred, ys[sel])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                opt.step()
                run_loss += float(loss.detach()) * int(sel.shape[0])
                run_n += int(sel.shape[0])
            del xs, sts, tds, anc, ys
        train_loss = run_loss / max(run_n, 1)
        val_loss, nval = evaluate(model, list(va["path"]), args.features, stats, device)
        curve_rows.append({"epoch": epoch, "train_crps": train_loss, "val_crps": val_loss,
                           "train_rows": run_n, "val_rows": nval,
                           "seconds": time.time() - t0})
        print("epoch %3d  train %.4f  validation %.4f  (%d train rows, %.0f s)"
              % (epoch, train_loss, val_loss, run_n, time.time() - t0), flush=True)
        if val_loss < best - args.min_delta:
            best, best_epoch, since = val_loss, epoch, 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "val_crps": val_loss, "n_neighbourhood": n_in_neigh,
                        "n_static": D.N_STATIC, "n_time": D.N_TIME,
                        "hidden": args.hidden, "depth": args.depth,
                        "features": args.features,
                        "stats": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                  for k, v in stats.items()}}, ckpt)
        else:
            since += 1
            if since >= args.patience:
                print("early stop at epoch %d; best was epoch %d with %.4f"
                      % (epoch, best_epoch, best), flush=True)
                break

    curve = pd.DataFrame(curve_rows)
    curve.to_csv(run_dir / "training_curve.csv", index=False)
    config = {
        "run_id": args.run_id,
        "target": args.target,
        "features": args.features,
        "seed": args.seed,
        "data_dir": str(data_dir),
        "val_init_from": args.val_init_from,
        "val_init_to": args.val_init_to,
        "training_cases": list(tr["case_id"]),
        "validation_cases": list(va["case_id"]),
        "hyperparameters": {"hidden": args.hidden, "depth": args.depth, "lr": args.lr,
                            "weight_decay": args.weight_decay, "batch": args.batch,
                            "epochs": args.epochs, "patience": args.patience,
                            "clip": args.clip, "train_on_stable_only": args.train_on_stable_only},
        "anchor": ("nearest predicted point with the lapse-rate correction for 2t"
                   if args.features != "xinterp" else
                   "nearest interpolated-input point with the lapse-rate correction for 2t"),
        "standardisation": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in stats.items()},
        "feature_names": json.loads((data_dir / "features.json").read_text())["feature_names"]
        if (data_dir / "features.json").exists() else None,
        "untrained_validation_crps": val0,
        "best_epoch": best_epoch,
        "best_validation_crps": best,
        "checkpoint": str(ckpt),
        "pipeline_test": bool(args.pipeline_test),
        "note": ("PIPELINE TEST, NOT A RESULT" if args.pipeline_test else
                 "stage 2a first cut"),
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2))
    print("wrote " + str(run_dir / "run_config.json"), flush=True)
    print("best validation fair CRPS %.4f at epoch %d (untrained anchor was %.4f)"
          % (best, best_epoch, val0), flush=True)
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=D.TARGETS)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--features", default="both", choices=["both", "pred", "xinterp"])
    ap.add_argument("--val-init-from", default=DEFAULT_VAL_FROM)
    ap.add_argument("--val-init-to", default=None)
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--stats-cases", type=int, default=24)
    ap.add_argument("--cache-gb", type=float, default=0.0,
                    help="hold this many gigabytes of assembled cases in memory, so "
                         "that an epoch does not read every case file from disk again")
    ap.add_argument("--train-on-stable-only", action="store_true",
                    help="train only on the stable network; the default trains on every "
                         "seen station and reports on both networks")
    ap.add_argument("--pipeline-test", action="store_true",
                    help="label this run as a pipeline test and not a result")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
