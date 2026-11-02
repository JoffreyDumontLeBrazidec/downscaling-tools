"""tc_obs plot(): figures from summary.csv, cases_obs.csv, members_obs.csv and run_meta.json alone."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROW_STYLE = {"model": ("model", "o"), "ens": ("reference", "D"), "truth": ("truth", "s"), "input": ("input", "s"),
             "hres": ("baseline", "^")}
OFFSET = {"model": 0.0, "ens": 1.2, "truth": -1.2, "input": -2.4, "hres": 2.4}


def _label(row, run_label):
    return {"model": f"{run_label} (ensemble)", "ens": "Operational ENS tracker (51 members)",
            "truth": "Truth (single field)", "input": "Input (single field)",
            "hres": "Operational HRES tracker (single field)"}[row]


def _style(row):
    from eval.plotting import role_style

    role, marker = ROW_STYLE[row]
    st = role_style(role)
    st.update(marker=marker)
    return st


def plot_by_lead(results_dir, summ, meta):
    import matplotlib.pyplot as plt
    from eval.plotting import eval_style, save_figure

    s = summ[summ.lead != "all"].copy()
    s["lead"] = s.lead.astype(int)
    leads = sorted(set(s.lead))
    panels = [("pmin_crps", "Central pressure: fair CRPS (hPa)"), ("pmin_err", "Central pressure: mean error (hPa)"),
              ("pos_err_member_km", "Centre: mean member error (km)"), ("pos_spread_km", "Centre: ensemble spread (km)")]
    with eval_style():
        fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.0), squeeze=False)
        for ax, (key, ylabel) in zip(axes[0], panels):
            for row in ("model", "ens", "truth", "input", "hres"):
                q = s[s.row == row].set_index("lead").reindex(leads)
                if q.empty or q[key].isna().all() or (key == "pos_spread_km" and row not in ("model", "ens")):
                    continue
                st = _style(row)
                st["linestyle"] = "-" if row in ("model", "ens") else "none"
                ax.errorbar(np.array(leads) + OFFSET[row], q[key], yerr=q[f"{key}_se"], capsize=3,
                            label=_label(row, meta["run_label"]), **st)
            if key == "pmin_err":
                ax.axhline(0, color="#0b0b0b", lw=1)
            ax.set_ylabel(ylabel)
            ax.set_xticks(leads)
            ax.set_xticklabels([f"+{l}" for l in leads])
            ax.set_xlabel("Lead time (h)")
            n = s[s.row == "model"].set_index("lead").reindex(leads).cases
            ax.set_title("cases per lead: " + ", ".join("-" if not np.isfinite(x) else f"{int(x)}" for x in n), fontsize=9)
        h, lab = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, lab, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.99), fontsize=9)
        fig.suptitle(f"{meta['run_label']} storms against the IBTrACS best track. {meta['note']}",
                     fontweight="bold", y=0.86, fontsize=11)
        fig.text(0.01, 0.0, "Mean over cases at each lead; bars one SE over cases (cases on the same date are not "
                 "independent). Model, truth and input from tc_structure on O1280; trackers on the operational grids.",
                 ha="left", va="top", fontsize=8.5, color="#52514e")
        fig.tight_layout(rect=(0, 0.02, 1, 0.8))
        return save_figure(fig, Path(results_dir) / "tc_obs_by_lead", close=True)


def plot_cases(results_dir, cases, members, meta):
    """Per case: best-track central pressure and every member, with truth, input, HRES and the ENS spread."""
    import matplotlib.pyplot as plt
    from eval.plotting import eval_style, save_figure

    keys = cases[["event", "date", "lead"]].drop_duplicates().sort_values(["date", "event", "lead"])
    labels = [f"{e} {d[4:]} +{l}" for e, d, l in keys.itertuples(index=False)]
    x = np.arange(len(keys))
    with eval_style():
        fig, axes = plt.subplots(2, 1, figsize=(max(9.0, 0.42 * len(keys)), 7.6), sharex=True)
        seen = set()

        def once(name):  # a legend entry for the first case that has the row, not only case 0
            if name in seen:
                return None
            seen.add(name)
            return name

        for ax, (val, mval, ylabel) in zip(axes, (("pmin_mean", "pmin_hpa", "Central pressure (hPa)"),
                                                 ("pos_err_member_km", "pos_err_km", "Centre error vs best track (km)"))):
            for i, (e, d, l) in enumerate(keys.itertuples(index=False)):
                c = cases[(cases.event == e) & (cases.date == d) & (cases.lead == l)].set_index("row")
                mm = members[(members.event == e) & (members.date == d) & (members.lead == l)]
                for row, dx in (("model", -0.15), ("ens", 0.15)):
                    v = mm[mm.row == row][mval]
                    if len(v):
                        st = _style(row)
                        ax.scatter(np.full(len(v), i + dx), v, s=9, color=st["color"], alpha=0.6,
                                   label=once(_label(row, meta["run_label"])))
                for row in ("truth", "input", "hres"):
                    if row in c.index:
                        y = c.loc[row, val] if val == "pmin_mean" else c.loc[row, "pos_err_member_km"]
                        st = _style(row)
                        ax.plot([i], [y], marker=st["marker"], color=st["color"], ls="none", ms=6,
                                label=once(_label(row, meta["run_label"])))
                if val == "pmin_mean" and len(c):
                    ax.plot([i - 0.35, i + 0.35], [c.bt_pmin_hpa.iloc[0]] * 2, color="#7f3b08", lw=2.2,
                            label=once("IBTrACS best track"))
            ax.set_ylabel(ylabel)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
        h, lab = axes[0].get_legend_handles_labels()
        fig.legend(h, lab, loc="upper center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 1.0))
        fig.suptitle(f"{meta['run_label']}: every storm case. {meta['note']}", fontweight="bold", y=0.92, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        return save_figure(fig, Path(results_dir) / "tc_obs_cases", close=True)


def plot(results_dir, lane_config: dict, eval_config: dict, *, output_dir=None, **kwargs):
    import pandas as pd

    results_dir = Path(results_dir)
    meta = json.loads((results_dir / "run_meta.json").read_text())
    if eval_config.get("run_label"):
        meta["run_label"] = str(eval_config["run_label"])
    summ = pd.read_csv(results_dir / "summary.csv", dtype={"lead": str})
    cases = pd.read_csv(results_dir / "cases_obs.csv", dtype={"date": str})
    members = pd.read_csv(results_dir / "members_obs.csv", dtype={"date": str})
    return plot_by_lead(results_dir, summ, meta) + plot_cases(results_dir, cases, members, meta)
