"""Build the stage 2a case list from the stage 1 pairing manifest.

A case is one (initialisation, lead) pair, which is one synthetic index of the
AIFS ENS version 2 store and one valid time. The split is the one fixed in
section 10 of the design note: training initialisations from 12 May 2026 12 UTC
to 17 August 2026 12 UTC, validation initialisations from 18 to 31 August 2026.
The first cut takes every second training initialisation with members 1 to 5 and
every validation initialisation with members 1 to 10.

Writes a CSV with one row per case and prints a short summary.
"""
from __future__ import annotations

import sys

import pandas as pd

MANIFEST = "/home/ecm5702/perm/station-head-adapter/stage1_20260909/outputs/pairing_manifest.parquet"
VAL_START = pd.Timestamp("2026-08-18 00:00:00")


def build(first_cut: bool = True) -> pd.DataFrame:
    df = pd.read_parquet(MANIFEST)
    df["init"] = pd.to_datetime(df["init"])
    df["valid"] = pd.to_datetime(df["valid"])
    df["split"] = ["validation" if t >= VAL_START else "training" for t in df["init"]]

    inits = sorted(df["init"].unique())
    train_inits = [t for t in inits if t < VAL_START]
    # every second training initialisation, in chronological order
    cut_train = set(train_inits[::2])
    val_inits = set(t for t in inits if t >= VAL_START)

    def keep(row):
        if row["split"] == "validation":
            return True
        return row["init"] in cut_train

    def nmem(row):
        return 10 if row["split"] == "validation" else 5

    if first_cut:
        df = df[df.apply(keep, axis=1)].copy()
    df["n_members"] = df.apply(nmem, axis=1)
    df["date"] = df["init"].dt.strftime("%Y%m%d")
    df["time"] = df["init"].dt.strftime("%H")
    df["case_id"] = df["date"] + df["time"] + "_step" + df["lead_h"].astype(int).astype(str).str.zfill(3)
    return df.sort_values(["split", "init", "lead_h"]).reset_index(drop=True)


if __name__ == "__main__":
    out = sys.argv[1]
    d = build(first_cut=True)
    d.to_csv(out, index=False)
    print(f"wrote {len(d)} cases to {out}")
    print(d.groupby("split").agg(cases=("case_id", "size"),
                                 inits=("init", "nunique"),
                                 members=("n_members", "max")))
    print("first:", d["init"].min(), " last:", d["init"].max())
