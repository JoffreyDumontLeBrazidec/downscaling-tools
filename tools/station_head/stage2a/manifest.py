"""Say which stage 2a cases have a finished prediction and a finished gather.

The first cut is 252 cases and the inference runs as SLURM array jobs, so the
question "what is done" has to be answerable at any moment without reading job
logs. This script walks the case list, looks for the prediction file each case
should have written on scratch and for the gathered station table each case should
have written on perm, records the size and the modification time of both, and
writes one CSV to perm. It is safe to run again at any time; it only reads.

A prediction is called finished when its file exists, is larger than a floor that
a truncated write could not reach, and opens as NetCDF with the expected number of
members. That last check is what distinguishes a finished file from one a job was
still writing when the manifest was refreshed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

CASES = "/home/ecm5702/agent-work/20260909-station-head-adapter/notes/cases_firstcut.csv"
PRED_ROOT = Path("/home/ecm5702/scratch/eval/station_head_2a/predictions")
GATHER_ROOT = Path("/home/ecm5702/perm/station-head-adapter/gathered")
BUNDLE_ROOT = Path("/home/ecm5702/scratch/eval/station_head_2a/bundles")
OUT = Path("/home/ecm5702/perm/station-head-adapter/manifests/firstcut_manifest.csv")
MIN_BYTES = 100 * 1024 * 1024


def probe(path: Path, expect_members: int) -> tuple[bool, int, str]:
    if not path.exists():
        return False, 0, "missing"
    size = path.stat().st_size
    if size < MIN_BYTES:
        return False, size, "too small"
    try:
        import netCDF4

        ds = netCDF4.Dataset(str(path))
        n = ds.dimensions["ensemble_member"].size
        ds.close()
    except Exception as exc:
        return False, size, "unreadable: " + str(exc)[:80]
    if n < expect_members:
        return False, size, "only %d of %d members" % (n, expect_members)
    return True, size, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--no-open", action="store_true",
                    help="skip opening the NetCDF files; report on size alone")
    args = ap.parse_args()

    cases = pd.read_csv(CASES)
    rows = []
    for r in cases.itertuples():
        init_key = "%s%02d" % (r.date, int(r.time))
        pred = PRED_ROOT / init_key / "predictions" / ("predictions_%s_step%03d.nc" % (r.date, int(r.lead_h)))
        if args.no_open:
            ok = pred.exists() and pred.stat().st_size >= MIN_BYTES
            size = pred.stat().st_size if pred.exists() else 0
            why = "ok" if ok else "missing or too small"
        else:
            ok, size, why = probe(pred, int(r.n_members))
        gathered = GATHER_ROOT / (r.case_id + ".parquet")
        n_bundles = len(list((BUNDLE_ROOT / init_key).glob("*_step%03dh_input_bundle.nc" % int(r.lead_h)))) \
            if (BUNDLE_ROOT / init_key).exists() else 0
        rows.append({
            "case_id": r.case_id,
            "split": r.split,
            "init": r.init,
            "lead_h": int(r.lead_h),
            "valid": r.valid,
            "synthetic_index": int(r.synthetic_index),
            "n_members_wanted": int(r.n_members),
            "n_bundles": n_bundles,
            "prediction_path": str(pred),
            "prediction_done": ok,
            "prediction_bytes": size,
            "prediction_note": why,
            "gathered_path": str(gathered),
            "gathered_done": gathered.exists(),
            "gathered_bytes": gathered.stat().st_size if gathered.exists() else 0,
        })

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("wrote " + str(out))
    print(df.groupby("split").agg(cases=("case_id", "size"),
                                  bundles_complete=("n_bundles", "sum"),
                                  predictions_done=("prediction_done", "sum"),
                                  gathered_done=("gathered_done", "sum")).to_string())
    bad = df[(~df["prediction_done"]) & (df["prediction_note"] != "missing")]
    if len(bad):
        print("\ncases whose prediction file exists but is not usable:")
        print(bad[["case_id", "prediction_bytes", "prediction_note"]].to_string(index=False))


if __name__ == "__main__":
    main()
