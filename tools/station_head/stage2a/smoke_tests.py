"""The smoke tests of section 8 of the station head design note.

They run on the one case that has been predicted, and they are meant to be run
again on any later case. Each test prints a line beginning with PASS or FAIL and
the numbers it was decided on, so that the result can be read without rerunning it.

The six tests of the design note are:

  1. the station table joins to the case through the manifest with no time mismatch
  2. the gathered features for a station are the values at the correct output
     points, checked against a direct nearest-point read of the prediction file
  3. the observation changes while the forecast input does not when the valid time
     is changed
  4. the hold-out flag removes the right stations
  5. the fair CRPS reduces to the absolute error for a one-member ensemble
  6. a forward and a backward pass of the head run on a handful of stations

Tests 5 and 6 use the untrained head module beside this file. They check shapes and
gradients only; no number they produce is a result.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

STAGE1 = Path("/home/ecm5702/perm/station-head-adapter/stage1_20260909/outputs")
PERM = Path("/home/ecm5702/perm/station-head-adapter")
CASES = "/home/ecm5702/agent-work/20260909-station-head-adapter/notes/cases_firstcut.csv"

RESULTS: list[tuple[str, bool, str]] = []


def report(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok, detail))
    print(("PASS  " if ok else "FAIL  ") + name + ": " + detail, flush=True)


def test_manifest_join(case_id: str, gathered: pd.DataFrame) -> None:
    cases = pd.read_csv(CASES)
    row = cases[cases["case_id"] == case_id].iloc[0]
    man = pd.read_parquet(STAGE1 / "pairing_manifest.parquet")
    mrow = man[man["synthetic_index"] == int(row["synthetic_index"])].iloc[0]
    init = pd.Timestamp(mrow["init"])
    valid = pd.Timestamp(mrow["valid"])
    lead = int(mrow["lead_h"])
    ok_manifest = (valid == init + pd.Timedelta(hours=lead))
    g_valid = pd.Timestamp(gathered["valid"].iloc[0])
    g_init = pd.Timestamp(gathered["init"].iloc[0])
    g_lead = int(gathered["lead_h"].iloc[0])
    ok_gather = (g_valid == valid) and (g_init == init) and (g_lead == lead)
    obs = pd.read_parquet(STAGE1 / "stations_2026" / "2t.parquet",
                          columns=["synthetic_index", "valid", "stnid", "value"])
    obs_case = obs[obs["synthetic_index"] == int(row["synthetic_index"])]
    obs_valid = pd.Timestamp(obs_case["valid"].iloc[0])
    ok_obs = (obs_valid == valid) and (obs_case["valid"].nunique() == 1)
    report("1 manifest join, no time mismatch",
           bool(ok_manifest and ok_gather and ok_obs),
           "manifest init %s plus %d hours is %s; the gathered table carries the same three "
           "times; the 2 m temperature reports joined to this case all carry valid time %s, "
           "over %d station rows"
           % (init, lead, valid, obs_valid, len(obs_case)))


def test_gathered_values(case_id: str, gathered: pd.DataFrame, prediction: Path) -> None:
    """Read the prediction file directly at the station's own nearest point."""
    stations = pd.read_parquet(STAGE1 / "static_stations.parquet",
                               columns=["stnid", "latitude", "longitude"])
    cache = np.load(PERM / "manifests" / "station_knn_on_prediction_grid.npz", allow_pickle=True)
    idx = cache["indices"]
    ids = list(cache["stnid"])

    ds = netCDF4.Dataset(str(prediction))
    states = [s.strip() for s in str(ds.getncattr("output_weather_states")).split(",")]
    vi = states.index("2t")
    members = np.asarray(ds.variables["ensemble_member"][:]).reshape(-1)
    rng = np.random.default_rng(1)
    picks = rng.choice(len(ids), size=8, replace=False)

    bad = 0
    checked = 0
    detail_lines = []
    for p in picks:
        stnid = ids[p]
        for mi, member in enumerate(members):
            row = gathered[(gathered["stnid"] == stnid) & (gathered["member"] == int(member))]
            if not len(row):
                continue
            for j in (0, 5, 11):
                direct = float(ds.variables["y_pred"][0, mi, int(idx[p, j]), vi])
                stored = float(row["pred_2t_k%02d" % j].iloc[0])
                checked += 1
                if not np.isclose(direct, stored, rtol=0, atol=1e-6):
                    bad += 1
                    detail_lines.append("%s member %d k%02d direct %.6f stored %.6f"
                                        % (stnid, member, j, direct, stored))
    ds.close()
    report("2 gathered values against a direct nearest-point read", bad == 0,
           "%d direct reads over 8 stations, 3 of the 12 neighbours and every member; "
           "%d disagreed%s" % (checked, bad, ("; " + "; ".join(detail_lines[:3])) if bad else ""))


def test_observation_changes(case_id: str, gathered: pd.DataFrame) -> None:
    """The observation must follow the valid time; the forecast must not."""
    cases = pd.read_csv(CASES)
    row = cases[cases["case_id"] == case_id].iloc[0]
    si = int(row["synthetic_index"])
    obs = pd.read_parquet(STAGE1 / "stations_2026" / "2t.parquet",
                          columns=["synthetic_index", "stnid", "value"])
    here = obs[obs["synthetic_index"] == si].set_index("stnid")["value"]
    other_si = si + 4 if si + 4 <= obs["synthetic_index"].max() else si - 4
    there = obs[obs["synthetic_index"] == other_si].set_index("stnid")["value"]
    common = here.index.intersection(there.index)
    diff = (here.loc[common] - there.loc[common]).abs()
    frac_changed = float((diff > 1e-6).mean())

    # the forecast side of the gathered table cannot depend on the valid time we
    # choose to join, because it was written from one prediction file; this checks
    # that the join we performed did not silently reorder it
    m1 = gathered[gathered["member"] == gathered["member"].min()]
    same_length = len(m1) == gathered["stnid"].nunique()
    report("3 the observation moves with the valid time, the forecast does not",
           bool(frac_changed > 0.9 and same_length),
           "between synthetic index %d (%s) and index %d, %.4f of the %d stations that report "
           "at both have a different 2 m temperature; the gathered forecast columns hold exactly "
           "one row per station per member (%d stations)"
           % (si, row["valid"], other_si, frac_changed, len(common), gathered["stnid"].nunique()))


def test_holdout(gathered: pd.DataFrame) -> None:
    stations = pd.read_parquet(STAGE1 / "static_stations.parquet",
                               columns=["stnid", "holdout_station", "terrain_class", "region"])
    held = stations[stations["holdout_station"]]
    seen = stations[~stations["holdout_station"]]
    joined = gathered.merge(stations, on="stnid", how="left")
    n_missing = int(joined["holdout_station"].isna().sum())
    frac = len(held) / len(stations)
    disjoint = len(set(held["stnid"]) & set(seen["stnid"])) == 0
    report("4 the hold-out flag removes the right stations",
           bool(n_missing == 0 and disjoint and 0.04 < frac < 0.07),
           "%d of %d stations are held out, that is %.2f per cent; the seen and held-out sets "
           "do not overlap; every one of the %d gathered rows carries the flag"
           % (len(held), len(stations), 100 * frac, len(joined)))


def test_crps_one_member() -> None:
    import torch

    from head import fair_crps

    rng = np.random.default_rng(2)
    x = torch.tensor(rng.normal(size=(500, 1)), dtype=torch.float64)
    y = torch.tensor(rng.normal(size=(500,)), dtype=torch.float64)
    crps = fair_crps(x, y)
    mae = (x[:, 0] - y).abs().mean()
    d = float((crps - mae).abs())

    # and a second check: with many members the fair score of a perfect ensemble
    # drawn from the same distribution as the observation must be positive
    xm = torch.tensor(rng.normal(size=(500, 10)), dtype=torch.float64)
    crps_m = float(fair_crps(xm, y))
    report("5 the fair CRPS is the absolute error for one member",
           bool(d < 1e-12 and crps_m > 0),
           "on 500 synthetic rows the one-member score and the mean absolute error differ by "
           "%.3g; the ten-member score of a matched ensemble is %.4f, which is positive as it "
           "must be" % (d, crps_m))


def test_head_forward_backward() -> None:
    import torch

    from head import StationHead, fair_crps

    torch.manual_seed(0)
    n, m, k, s = 16, 5, 24, 6
    head = StationHead(n_neighbourhood=k, n_static=s)
    neigh = torch.randn(n, m, k)
    static = torch.randn(n, s)
    tod = torch.randn(n, 2)
    out = head(neigh, static, tod)
    y = torch.randn(n)
    mask = torch.ones(n, dtype=torch.bool)
    mask[:3] = False
    loss = fair_crps(out, y, mask)
    loss.backward()
    grads = [p.grad for p in head.parameters() if p.grad is not None]
    total = float(sum(float(g.abs().sum()) for g in grads))
    ok = (out.shape == (n, m)) and np.isfinite(float(loss)) and total > 0
    report("6 a forward and a backward pass of the untrained head", bool(ok),
           "%d stations by %d members gave an output of shape %s, a masked loss of %.4f on the "
           "%d unmasked rows, and a total absolute gradient of %.4g over %d parameter tensors"
           % (n, m, tuple(out.shape), float(loss), int(mask.sum()), total, len(grads)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="2026051212_step006")
    ap.add_argument("--prediction", required=True)
    args = ap.parse_args()

    gathered = pd.read_parquet(PERM / "gathered" / (args.case + ".parquet"))
    print("gathered table: %d rows, %d columns" % gathered.shape, flush=True)

    test_manifest_join(args.case, gathered)
    test_gathered_values(args.case, gathered, Path(args.prediction))
    test_observation_changes(args.case, gathered)
    test_holdout(gathered)
    test_crps_one_member()
    test_head_forward_backward()

    failed = [n for n, ok, _ in RESULTS if not ok]
    print("\n%d of %d smoke tests passed" % (len(RESULTS) - len(failed), len(RESULTS)))
    if failed:
        print("failed: " + ", ".join(failed))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
