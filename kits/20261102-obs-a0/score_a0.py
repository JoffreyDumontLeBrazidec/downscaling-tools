"""A0: station sanity check and tp audits on the seed-locked rows (rule 31). Jupiter, CPU, reads prediction files only.

Rows scored against STVL stations (fetch_obs_atos.py output), per variable (2t, 10ff, msl, tp), lead (+12..+72 h),
region, date:
  input   x_interp (O320 interpolated to O1280; one field)
  truth   y (ENS hindcast control at O1280; one field)
  hres    operational HRES at the station from STVL (when the fetch found it; one field)
  CA300, DB300   10-member ensembles, seed-locked (seedfix_wide_20261030)
Station value of a gridded field = nearest O1280 point of the E2 box; stations farther than 15 km are dropped.
2t: raw, and corrected to station height with 0.0065 K/m (quaver): truth / ensembles use the O1280 orography,
input the O320 orography (nearest O320 point; approximates the interpolated input's height), hres STVL's own height.
Metrics per date: n, bias, MAE (ensemble mean for ensembles), member-mean MAE, RMSE, corr, fair CRPS (MAE for one
field), spread (ddof 1), spread/RMSE corrected by sqrt((m+1)/m).
tp audits: (a) every field also smoothed over a disc of radius 18 km and 36 km around the station's grid point
(double-penalty probe); (b) window check: corr of truth AND input with the 12 h window ending at valid vs the one
before; (c) truth vs hres at the stations (corr, MAE); (e) member-mean MAE next to ensemble-mean MAE.
Paired differences (truth - input, DB300 - CA300, ...) are summarised as mean and SE over dates.
Usage: python score_a0.py <obs dir> <pred root> <out dir> [--dates N]
"""
import argparse
import sys
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

DATES = ["20230101", "20230111", "20230122", "20230201", "20230212", "20230226", "20230304", "20230314", "20230325",
         "20230404", "20230415", "20230425", "20230506", "20230516", "20230527", "20230606", "20230617", "20230627",
         "20230708", "20230718", "20230729", "20230808", "20230822", "20230829", "20230909", "20230919", "20230930",
         "20231010", "20231021", "20231031"]
LEADS = (12, 24, 36, 48, 60, 72)
ARMS = ("CA300", "DB300")
REGIONS = {"all": None, "europe": (35, 75, -12.5, 42.5), "40-70N": (40, 70.01, -180, 180),
           "25-40N": (25, 40, -180, 180), "10-25N": (10, 25, -180, 180)}
MAX_KM, R_EARTH, LAPSE = 15.0, 6371.0, 0.0065
RADII_KM = (0.0, 18.0, 36.0)
PAIRS = [("truth", "input"), ("DB300", "CA300"), ("CA300", "input"), ("DB300", "input"), ("CA300", "truth"),
         ("DB300", "truth"), ("hres", "truth"), ("hres", "input")]


def xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def fair_crps(ens, ref):
    m = ens.shape[0]
    t1 = np.abs(ens - ref[None]).mean(0)
    s = np.sort(ens, axis=0)
    w = (2 * np.arange(1, m + 1) - m - 1).astype(np.float64)[:, None]
    return t1 - (w * s).sum(0) / (m * (m - 1))


def read(path):
    """lat, lon, {var: (truth, input, members)} for 2t, 10ff, msl, tp in obs units (tp mm, clipped at 0)."""
    with netCDF4.Dataset(str(path)) as ds:
        ds.set_auto_mask(False)
        ws = [str(s) for s in ds["weather_state"][:]]
        lat = np.asarray(ds["lat_hres"][:], np.float64)
        lon = np.asarray(ds["lon_hres"][:], np.float64)
        n = lat.size

        def get(name, var):
            a = np.asarray(ds[name][0][..., ws.index(var)], np.float64).reshape(-1, n)
            return a if name == "y_pred" else a[0]

        out = {}
        for v in ("2t", "msl", "tp"):
            if v in ws:
                out[v] = [get("y", v), get("x_interp", v), get("y_pred", v)]
        if "10u" in ws and "10v" in ws:
            out["10ff"] = [np.hypot(get(k, "10u"), get(k, "10v")) for k in ("y", "x_interp", "y_pred")]
        if "tp" in out:
            out["tp"] = [np.clip(a, 0, None) * 1000.0 for a in out["tp"]]
    lon = np.where(lon > 180, lon - 360, lon)
    return lat, lon, out


def scores(f, ref, ens=None):
    d = f - ref
    rmse = float(np.sqrt((d ** 2).mean()))
    r = dict(n=int(ref.size), bias=float(d.mean()), mae=float(np.abs(d).mean()), rmse=rmse,
             corr=float(np.corrcoef(f, ref)[0, 1]) if f.std() > 0 and ref.std() > 0 else np.nan)
    if ens is None:
        r.update(crps=r["mae"], member_mae=r["mae"], spread=0.0, ssr_corr=np.nan)
    else:
        m = ens.shape[0]
        spread = float(np.sqrt(ens.var(0, ddof=1).mean()))
        r.update(crps=float(fair_crps(ens, ref).mean()), member_mae=float(np.abs(ens - ref[None]).mean()),
                 spread=float(ens.std(0, ddof=1).mean()), ssr_corr=spread / rmse * np.sqrt((m + 1) / m) if rmse > 0 else np.nan)
    return r


def region_masks(la, lo):
    return {k: np.ones(la.size, bool) if b is None else (la >= b[0]) & (la < b[1]) & (lo >= b[2]) & (lo < b[3])
            for k, b in REGIONS.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("obs"); ap.add_argument("pred"); ap.add_argument("out")
    ap.add_argument("--dates", type=int, default=len(DATES))
    a = ap.parse_args()
    obsd, pred, out = Path(a.obs), Path(a.pred), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    obs = {v: pd.read_csv(obsd / f"obs_{v}.csv", dtype={"valid": str}) for v in ("2t", "10ff", "msl", "tp")}
    hres = {v: pd.read_csv(obsd / f"hres_{v}.csv", dtype={"date": str}) for v in obs if (obsd / f"hres_{v}.csv").exists()}
    rows, win, log = [], [], []
    tree = grid = balls = None
    for date in DATES[:a.dates]:
        for lead in LEADS:
            paths = {arm: pred / f"{arm}_L{lead:03d}" / f"predictions_{date}_step{lead:03d}.nc" for arm in ARMS}
            if not all(p.exists() for p in paths.values()):
                log.append(f"MISSING {date} +{lead}: {[str(p) for p in paths.values() if not p.exists()]}"); continue
            data = {}
            for arm, p in paths.items():
                lat, lon, data[arm] = read(p)
                if grid is None:
                    grid = (lat, lon); tree = cKDTree(xyz(lat, lon))
                assert lat.size == grid[0].size and np.allclose(lat, grid[0]) and np.allclose(lon, grid[1]), p
            for v in data["CA300"]:  # truth and input must be identical between arms
                for i, nm in ((0, "truth"), (1, "input")):
                    dmax = float(np.abs(data["CA300"][v][i] - data["DB300"][v][i]).max())
                    if dmax > 1e-4:
                        log.append(f"WARN {nm} differs between arms {date} +{lead} {v}: max {dmax:.3g}")
            valid = (pd.Timestamp(date) + pd.Timedelta(hours=lead)).strftime("%Y%m%d%H")
            prev = (pd.Timestamp(date) + pd.Timedelta(hours=lead - 12)).strftime("%Y%m%d%H")
            for v, (y, xi, _) in data["CA300"].items():
                o = obs[v][obs[v].valid == valid]
                if o.empty:
                    log.append(f"no obs {v} {valid}"); continue
                dist, idx = tree.query(xyz(o.lat.to_numpy(), o.lon.to_numpy()))
                keep = dist * R_EARTH <= MAX_KM
                o, idx = o[keep].reset_index(drop=True), idx[keep]
                ov = o.obs.to_numpy(np.float64)
                fields = {"truth": y, "input": xi, "CA300": data["CA300"][v][2], "DB300": data["DB300"][v][2]}
                # disc means around each station's grid point (tp audit a); radius 0 = the nearest point
                discs = {r: (None if r == 0 else tree.query_ball_point(xyz(grid[0][idx], grid[1][idx]), r / R_EARTH))
                         for r in RADII_KM} if v == "tp" else {0.0: None}
                hr = None
                if v in hres:
                    h = hres[v][(hres[v].date == date) & (hres[v].lead == lead)].set_index("stnid")
                    h = h[~h.index.duplicated()]
                    hr = o.stnid.map(h.hres).to_numpy(np.float64)
                    if v == "tp" and np.nanmedian(hr) < 0.05 * max(np.nanmedian(ov), 1e-6) and np.nanmax(hr) < 1:
                        log.append(f"WARN hres tp looks like metres {date} +{lead}")
                    if v == "2t":
                        hel = o.stnid.map(h.hres_elevation).to_numpy(np.float64)
                for r, ball in discs.items():
                    def at(a):
                        if a.ndim == 1:
                            return a[idx] if ball is None else np.array([a[b].mean() for b in ball])
                        return a[:, idx] if ball is None else np.stack([a[:, b].mean(1) for b in ball], 1)
                    vals = {k: at(f) for k, f in fields.items()}
                    if hr is not None and r == 0:
                        vals["hres"] = hr
                    variants = [("raw", vals)]
                    if v == "2t" and r == 0:
                        dz = {"truth": o.h_o1280 - o.elevation, "input": o.h_o320 - o.elevation,
                              "CA300": o.h_o1280 - o.elevation, "DB300": o.h_o1280 - o.elevation}
                        if "hres" in vals:
                            dz["hres"] = hel - o.elevation.to_numpy()
                        variants.append(("lapse", {k: x + LAPSE * np.asarray(dz[k], np.float64) for k, x in vals.items()}))
                    for corr_tag, vv in variants:
                        ok = np.isfinite(ov)
                        for x in vv.values():
                            ok &= np.isfinite(x).all(0) if x.ndim == 2 else np.isfinite(x)
                        masks = region_masks(o.lat.to_numpy(), o.lon.to_numpy())
                        hgt = np.abs(o.h_o1280.to_numpy() - o.elevation.to_numpy())
                        masks["all_dz<100m"] = ok & (hgt < 100)
                        masks["all_dz>=300m"] = ok & (hgt >= 300)
                        for reg, m in masks.items():
                            m = m & ok
                            if m.sum() < 30:
                                continue
                            for k, x in vv.items():
                                ens = x[:, m] if x.ndim == 2 else None
                                f = ens.mean(0) if ens is not None else x[m]
                                rows.append(dict(var=v, variant=corr_tag, radius_km=r, region=reg, lead=lead,
                                                 date=date, row=k, **scores(f, ov[m], ens)))
                # tp window audit (b), truth and input, nearest point
                if v == "tp":
                    op = obs[v][obs[v].valid == prev]
                    if len(op):
                        dp, ip = tree.query(xyz(op.lat.to_numpy(), op.lon.to_numpy()))
                        kp = dp * R_EARTH <= MAX_KM
                        opv = op.obs.to_numpy()[kp]; ip = ip[kp]
                        for nm, f in (("truth", y), ("input", xi)):
                            win.append(dict(lead=lead, date=date, row=nm, corr_valid=float(np.corrcoef(f[idx], ov)[0, 1]),
                                            corr_prev=float(np.corrcoef(f[ip], opv)[0, 1]), n=int(ov.size)))
            log.append(f"scored {date} +{lead}"); print(log[-1], flush=True)
    df = pd.DataFrame(rows); df.to_csv(out / "a0_scores_per_date.csv", index=False)
    pd.DataFrame(win).to_csv(out / "a0_tp_window_check.csv", index=False)
    (out / "a0_log.txt").write_text("\n".join(log) + "\n")
    summarise(df, pd.DataFrame(win), out)


def summarise(df, win, out):
    keys = ["var", "variant", "radius_km", "region", "lead"]
    cols = ["bias", "mae", "member_mae", "rmse", "corr", "crps", "spread", "ssr_corr"]
    g = df.groupby(keys + ["row"])
    s = g[cols].mean().join(g[cols].sem().add_suffix("_se")).join(g["n"].mean().rename("mean_n")).join(g.size().rename("dates"))
    s.reset_index().to_csv(out / "a0_summary.csv", index=False)
    # paired differences over dates
    prs = []
    wide = df.pivot_table(index=keys + ["date"], columns="row", values=["mae", "rmse", "corr", "crps"])
    for a, b in PAIRS:
        for met in ("mae", "rmse", "corr", "crps"):
            if (met, a) not in wide.columns or (met, b) not in wide.columns:
                continue
            d = (wide[(met, a)] - wide[(met, b)]).dropna()
            rel = (d / wide[(met, b)].reindex(d.index)).dropna()
            gg = d.groupby(level=keys)
            res = pd.DataFrame({"mean": gg.mean(), "se": gg.sem(), "dates": gg.size(),
                                "rel_mean": rel.groupby(level=keys).mean(), "wins": gg.apply(lambda x: int((x < 0).sum()) if met != "corr" else int((x > 0).sum()))})
            res["pair"] = f"{a}-{b}"; res["metric"] = met
            prs.append(res.reset_index())
    p = pd.concat(prs, ignore_index=True); p.to_csv(out / "a0_paired.csv", index=False)
    lines = ["# A0 station read (mean over dates; SE in brackets; paired = a - b per date)", ""]
    sel = p[(p.region == "all") & (p.variant.isin(["raw", "lapse"]))]
    for (v, var_, r), q in sel.groupby(["var", "variant", "radius_km"]):
        lines += [f"## {v} {var_} radius {r:g} km", "", "| pair | metric | " + " | ".join(f"+{l} h" for l in LEADS) + " |",
                  "|---|---|" + "---|" * len(LEADS)]
        for (pair, met), qq in q.groupby(["pair", "metric"]):
            qq = qq.set_index("lead")
            cells = [f"{qq.loc[l, 'mean']:+.3f} ({qq.loc[l, 'se']:.3f}) w{int(qq.loc[l, 'wins'])}/{int(qq.loc[l, 'dates'])}"
                     if l in qq.index else "" for l in LEADS]
            lines.append(f"| {pair} | {met} | " + " | ".join(cells) + " |")
        lines.append("")
    if len(win):
        lines += ["## tp window check (corr with the 12 h window ending at valid vs the window before)", ""]
        for (lead, row), w in win.groupby(["lead", "row"]):
            lines.append(f"- +{lead} h {row}: valid {w.corr_valid.mean():.3f}, previous {w.corr_prev.mean():.3f} ({len(w)} dates, {w.n.mean():.0f} stations)")
    (out / "a0_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
