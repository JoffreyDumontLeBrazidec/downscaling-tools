"""backfill_native_x on tiny synthetic ENS-member files and zarr member stores."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")
zarr = pytest.importorskip("zarr")

from eval.tools import backfill_native_x as bf  # noqa: E402

N_M = 4
HDATE, LEAD, FAKE = "20241101", 12, "1900-08-08T01:00:00"


def _grids():
    lla, llo = np.meshgrid(np.arange(30.0, 60.01, 1.0), np.arange(-20.0, 20.01, 1.0), indexing="ij")
    hla, hlo = np.meshgrid(np.arange(31.0, 59.01, 0.5), np.arange(-19.0, 19.01, 0.5), indexing="ij")
    return lla.ravel(), llo.ravel(), hla.ravel(), hlo.ravel()


def _member_fields(lat, lon, rng_seed=0):
    """msl, 2t, tp per member: a shared field plus a smooth member-specific perturbation."""
    rng = np.random.default_rng(rng_seed)
    la, lo = np.radians(lat), np.radians(lon)
    out = {"msl": [], "2t": [], "tp": []}
    for k in range(N_M):
        a, b, c = rng.uniform(1, 4, 3)
        pert = np.sin(a * la * 3 + b) * np.cos(c * lo * 3 + k)
        out["msl"].append(101300.0 + 800.0 * np.cos(2 * la) + 300.0 * pert)
        out["2t"].append(280.0 + 10.0 * np.cos(la) + 2.0 * pert)
        out["tp"].append(np.clip(0.002 * pert, -1e-5, None))          # metres, a few tiny negatives
    return {k: np.stack(v) for k, v in out.items()}


def _make_case(tmp_path: Path, *, separate_tp: bool, swap: bool = False, tp_scale: float = 1.0,
               clip_attr: bool = True, swap_tp: bool = False, msl_noise: float = 0.0, dup_row: bool = False,
               store_fake: str | None = FAKE) -> Path:
    lla, llo, hla, hlo = _grids()
    nat = _member_fields(lla, llo)
    variables = ["2t", "msl", "z_500"] + ([] if separate_tp else ["tp"])
    data = np.zeros((2, len(variables), N_M, lla.size), np.float32)      # row 1 is the case
    for j, v in enumerate(variables):
        if v in nat:
            data[1, j] = nat[v] * (tp_scale if v == "tp" else 1.0)
    if swap:
        data[1] = data[1][:, [1, 0, 2, 3], :]
    if swap_tp and not separate_tp:   # the rain of members 0 and 1 exchanged, msl untouched
        jt = variables.index("tp")
        data[1, jt] = data[1, jt][[1, 0, 2, 3]]
    row_map = [{"hdate": "20241031" if not dup_row else HDATE, "lead": 12, "fake": "1900-08-07T23:00:00"},
               {"hdate": HDATE, "lead": LEAD, **({"fake": store_fake} if store_fake else {})}]

    def store(path, vars_, arr):
        g = zarr.open_group(str(path), mode="w")
        g.create_dataset("data", data=arr)
        g.create_dataset("latitudes", data=lla)
        g.create_dataset("longitudes", data=np.mod(llo, 360.0))
        g.attrs.update(variables=vars_, row_map=row_map, members=list(range(1, N_M + 1)))

    store(tmp_path / "in.zarr", variables, data)
    if separate_tp:
        tpd = np.zeros((2, 2, N_M, lla.size), np.float32)
        tpd[1, 0] = nat["tp"] * tp_scale
        store(tmp_path / "tp.zarr", ["tp", "cp"], tpd)

    # output-grid file: x_interp_members = each member's native field at the nearest native point
    from scipy.spatial import cKDTree
    _, nn = cKDTree(bf._xyz(lla, llo)).query(bf._xyz(hla, hlo))
    states = ["2t", "msl", "tp"]
    xim = np.stack([np.clip(nat[s], 0, None) if s == "tp" else nat[s] for s in states], -1)[:, nn, :]
    if msl_noise:   # x_interp_members msl less like the native members: same-member correlation drops
        xim[:, :, 1] += msl_noise * np.random.default_rng(5).standard_normal(xim[:, :, 1].shape)
    pred = tmp_path / "pred" / "ROW_L012"
    pred.mkdir(parents=True)
    path = pred / f"predictions_{HDATE}_step{LEAD:03d}.nc"
    with netCDF4.Dataset(str(path), "w") as d:
        d.createDimension("sample", 1)
        d.createDimension("ensemble_member", N_M)
        d.createDimension("grid_point_hres", hla.size)
        d.createDimension("weather_state", len(states))
        d.createVariable("lat_hres", "f8", ("grid_point_hres",))[:] = hla
        d.createVariable("lon_hres", "f8", ("grid_point_hres",))[:] = hlo
        v = d.createVariable("weather_state", str, ("weather_state",))
        for i, s in enumerate(states):
            v[i] = s
        d.createVariable("ensemble_member", "i4", ("ensemble_member",))[:] = np.arange(1, N_M + 1)
        d.createVariable("y", "f4", ("sample", "grid_point_hres", "weather_state"))[:] = xim.mean(0)[None]
        d.createVariable("x_interp", "f4", ("sample", "grid_point_hres", "weather_state"))[:] = xim[0][None]
        d.createVariable("y_pred", "f4", ("sample", "ensemble_member", "grid_point_hres", "weather_state"))[:] = \
            (xim + 0.01)[None]
        d.createVariable("x_interp_members", "f4", ("sample", "ensemble_member", "grid_point_hres",
                                                    "weather_state"))[:] = xim[None]
        d.member_store = str(tmp_path / "in.zarr")
        if separate_tp:
            d.member_tp_store = str(tmp_path / "tp.zarr")
        d.fake_date = FAKE
        d.lead_step_hours = LEAD
        if clip_attr:
            d.precip_clipped_input_members = json.dumps({"tp": [0, 0.0]})
    return path


def _snapshot(path: Path) -> dict:
    with netCDF4.Dataset(str(path)) as d:
        return {k: np.array(v[:]) for k, v in d.variables.items()}, {k: d.getncattr(k) for k in d.ncattrs()}


@pytest.mark.parametrize("separate_tp", [False, True])
def test_dry_run_checks_and_writes_nothing(tmp_path: Path, separate_tp: bool) -> None:
    path = _make_case(tmp_path, separate_tp=separate_tp)
    before = _snapshot(path)
    assert bf.main([str(path.parent)]) == 0
    r = bf.process(path, do_apply=False, store=None, tp_store=None)
    assert r["action"].startswith("dry-run")
    assert r["pairing"]["ok"] and r["pairing"]["diag_min"] > 0.99
    assert all(u["ok"] for u in r["units"].values())
    assert r["state_source"]["tp"] == ("tp_store" if separate_tp else "store")
    after = _snapshot(path)
    assert before[0].keys() == after[0].keys() and before[1] == after[1]


def test_apply_appends_only_and_then_skips(tmp_path: Path) -> None:
    path = _make_case(tmp_path, separate_tp=True)
    vars_before, attrs_before = _snapshot(path)
    assert bf.main([str(path), "--apply", "--report", str(tmp_path / "r.json")]) == 0
    vars_after, attrs_after = _snapshot(path)
    for k, v in vars_before.items():                                     # existing variables unchanged
        np.testing.assert_array_equal(vars_after[k], v)
    assert set(vars_after) - set(vars_before) == {"x", "lat_lres", "lon_lres"}
    lla, llo, _, _ = _grids()
    nat = _member_fields(lla, llo)
    with netCDF4.Dataset(str(path)) as d:
        assert d["x"].dimensions == ("sample", "ensemble_member", "grid_point_lres", "weather_state")
        x = np.array(d["x"][0])
        np.testing.assert_allclose(d["lat_lres"][:], lla)
        meta = json.loads(d.native_x_backfill)
        assert d.native_x_backfill_status == "complete"
    np.testing.assert_allclose(x[2, :, 1], nat["msl"][2].astype(np.float32))          # draw k <- member k
    np.testing.assert_allclose(x[1, :, 2], np.clip(nat["tp"][1], 0, None).astype(np.float32))  # clipped like the file
    assert meta["row"] == 1 and meta["hdate"] == HDATE and meta["lead"] == LEAD
    assert meta["store"].endswith("in.zarr") and meta["tp_store"].endswith("tp.zarr")
    assert meta["precip_clipped"]["tp"] > 0
    report = json.loads((tmp_path / "r.json").read_text())
    assert report[0]["action"] == "written"
    assert bf.process(path, do_apply=True, store=None, tp_store=None)["action"] == "skip"


def test_swapped_members_abort_the_file(tmp_path: Path) -> None:
    path = _make_case(tmp_path, separate_tp=False, swap=True)
    before = _snapshot(path)
    r = bf.process(path, do_apply=True, store=None, tp_store=None)
    assert r["action"] == "abort" and "pairing on msl failed" in r["why"]
    assert _snapshot(path)[0].keys() == before[0].keys()


def test_tp_in_the_wrong_unit_aborts_the_file(tmp_path: Path) -> None:
    path = _make_case(tmp_path, separate_tp=True, tp_scale=1000.0)       # mm in the store, m in the file
    r = bf.process(path, do_apply=True, store=None, tp_store=None)
    assert r["action"] == "abort" and "units differ" in r["why"] and "tp" in r["why"]


def test_help_runs() -> None:
    out = subprocess.run([sys.executable, "-m", "eval.tools.backfill_native_x", "--help"],
                         capture_output=True, text=True, cwd=Path(__file__).resolve().parents[3])
    assert out.returncode == 0 and "--apply" in out.stdout


def test_weak_same_member_msl_correlation_aborts(tmp_path: Path) -> None:
    """Best match on the diagonal is not enough: the same-member msl correlation must be at least 0.98."""
    path = _make_case(tmp_path, separate_tp=False, msl_noise=250.0)
    r = bf.process(path, do_apply=True, store=None, tp_store=None)
    assert r["action"] == "abort" and "pairing on msl failed" in r["why"]
    p = json.loads(r["why"].split("failed: ", 1)[1])
    assert p["best_match_ok"] and p["diag_min"] < 0.98


def test_rain_of_swapped_members_aborts(tmp_path: Path) -> None:
    """msl pairs, but the rain of members 0 and 1 is exchanged in the store: the per-member tp check refuses."""
    path = _make_case(tmp_path, separate_tp=False, swap_tp=True)
    r = bf.process(path, do_apply=True, store=None, tp_store=None)
    assert r["action"] == "abort" and "per-member tp correlation" in r["why"]
    ok = bf.process(_make_case(tmp_path / "b", separate_tp=False), do_apply=False, store=None, tp_store=None)
    assert ok["rain_check"]["made"] and ok["rain_check"]["corr_min"] > 0.9


def test_repeated_row_keys_abort(tmp_path: Path) -> None:
    path = _make_case(tmp_path, separate_tp=False, dup_row=True)
    r = bf.process(path, do_apply=False, store=None, tp_store=None)
    assert r["action"] == "abort" and "repeated (hdate, lead)" in r["why"]


def test_missing_fake_date_fails_unless_allowed(tmp_path: Path) -> None:
    path = _make_case(tmp_path, separate_tp=False, store_fake=None)
    r = bf.process(path, do_apply=False, store=None, tp_store=None)
    assert r["action"] == "abort" and "fake-date check cannot be made" in r["why"]
    r = bf.process(path, do_apply=False, store=None, tp_store=None, allow_missing_fake_date=True)
    assert r["action"].startswith("dry-run") and r["fake_date_check"].startswith("not made")
    assert bf.main([str(path), "--allow-missing-fake-date"]) == 0 and bf.main([str(path)]) == 1
