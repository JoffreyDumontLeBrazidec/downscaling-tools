"""Bands, the one-line headline, the SUMMARY text, and the trend diff (review note s4 check 12).

Readings, not gates: every status is a label for the reader; the audit always exits 0.
"""
from __future__ import annotations

import fnmatch
import json
import math
from pathlib import Path

import yaml

from .model_map import role_group

HERE = Path(__file__).resolve().parent
CHECK_TITLES = {
    "c1_zero_content": "1 zero-content heads/units, NaN, EMA aliases",
    "c2_grad_dead": "2 gradient-dead units/heads on K batches",
    "c3_optimizer_dead": "3 gradient-dead from optimizer state",
    "c4_preact": "4 pre-activation saturation",
    "c6_scale": "6 scale ladders",
    "c7_stream": "7 residual-stream budget",
    "c8_graph": "8 graph audit",
    "c9_flow": "9 information flow",
}


def load_bands(path: str | None = None) -> list[dict]:
    with open(path or HERE / "bands.yaml") as fh:
        return yaml.safe_load(fh)["rules"]


def rule_for(name: str, rules: list[dict]) -> dict | None:
    for r in rules:
        if fnmatch.fnmatchcase(name, r["match"]):
            return r
    return None


def status(value, rule: dict | None) -> str:
    if value is None or rule is None or (isinstance(value, float) and math.isnan(value)):
        return "info"
    if "fail_above" in rule and value > rule["fail_above"]:
        return "FAIL"
    if "fail_below" in rule and value < rule["fail_below"]:
        return "FAIL"
    if "warn_above" in rule and value > rule["warn_above"]:
        return "WARN"
    if "warn_below" in rule and value < rule["warn_below"]:
        return "WARN"
    if any(k.startswith(("warn_", "fail_")) for k in rule):
        return "ok"
    return "info"


def flatten(checks: dict) -> dict:
    """{'<check>.<role>.<metric>': (value, '<check>.<role group>.<metric>')} for every per-role scalar.

    The second element is the key the bands match on (role group instead of role)."""
    out = {}
    for cid, c in checks.items():
        for role, metrics in (c.get("per_role") or {}).items():
            for m, v in metrics.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    out[f"{cid}.{role}.{m}"] = (float(v), f"{cid}.{role_group(role)}.{m}")
    return out


def evaluate(checks: dict, rules: list[dict]) -> list[dict]:
    """One row per per-role scalar with its status; sorted FAIL, WARN, then the rest."""
    rows = []
    for k, (v, bk) in flatten(checks).items():
        st = status(v, rule_for(bk, rules))
        rows.append({"metric": k, "value": v, "status": st})
    order = {"FAIL": 0, "WARN": 1, "ok": 2, "info": 3}
    return sorted(rows, key=lambda r: (order[r["status"]], r["metric"]))


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.2f} %"


def headline(audit: dict) -> str:
    """'x % of processor capacity is dead or unread; y; z' (review note s8), plus the counts behind it."""
    cap = audit["capacity"]["processor"]
    o = audit["checks"]["c3_optimizer_dead"]
    win = o.get("window_min_steps")
    when = f"no gradient for >= ~{win:,} steps" if win else "no gradient (window unknown)"
    parts = []
    if cap["units"]:
        how = f"{cap['units_dead_zero']} zero-weight + {cap['units_dead_grad']} gradient-dead ({when})"
        if not cap["optimizer_state"]:
            how = f"{cap['units_dead_zero']} zero-weight; gradient-dead not read (no optimizer state)"
        parts.append(f"processor MLP units dead {_pct(cap['unit_dead_frac'])} ({how} of {cap['units']})")
    if cap["heads"]:
        parts.append(
            f"heads dead {cap['heads_dead_zero'] + cap['heads_dead_grad']}/{cap['heads']} "
            f"({cap['heads_dead_zero']} zero-weight, {cap['heads_dead_grad']} gradient-dead)"
        )
    for g in ("encoder", "decoder"):
        c = audit["capacity"].get(g) or {}
        if c.get("units"):
            parts.append(f"{g} units dead {c['units_dead_zero'] + c['units_dead_grad']}/{c['units']}, "
                         f"heads {c['heads_dead_zero'] + c['heads_dead_grad']}/{c['heads']}")
    if o.get("mapping") not in (None, "exact", "none"):
        parts.append(f"optimizer mapping {o['mapping']}: gradient-dead counts unproven")
    c2 = audit["checks"].get("c2_grad_dead")
    if c2 and c2.get("totals"):
        t = c2["totals"]
        parts.append(f"K={t['K']} batches: {t['dead_units']} units dead, {t['dormant_units']} dormant")
    c7 = audit["checks"].get("c7_stream")
    if c7 and c7.get("per_role"):
        low = [r for r, m in c7["per_role"].items() if role_group(r) == "processor" and m.get("update_ratio", 1) < 0.01]
        parts.append(f"{len(low)} processor blocks add < 1 % to the stream")
    geo = [m.get("edge_logit_scale.max") for m in audit["checks"]["c6_scale"]["per_role"].values()]
    geo = [g for g in geo if g is not None]
    if geo:
        parts.append(f"max edge-logit scale {max(geo):.3g} (s11: healthy < 0.2)")
    return "; ".join(parts)


def summary_text(audit: dict) -> str:
    m = audit["meta"]
    lines = [
        f"HEALTH {m['ckpt_id']}  step {m.get('global_step')}  {m['kind']}  optimizer {audit['checks']['c3_optimizer_dead'].get('optimizer')}",
        f"  {audit['headline']}",
    ]
    rows = audit["bands"]
    for cid, title in CHECK_TITLES.items():
        if cid not in audit["checks"]:
            continue
        mine = [r for r in rows if r["metric"].startswith(cid + ".")]
        n_f = sum(r["status"] == "FAIL" for r in mine)
        n_w = sum(r["status"] == "WARN" for r in mine)
        worst = "FAIL" if n_f else "WARN" if n_w else "ok"
        skipped = audit["checks"][cid].get("skipped")
        tail = f"skipped: {skipped}" if skipped else f"{n_f} FAIL, {n_w} WARN of {len(mine)} readings"
        lines.append(f"  [{worst:4s}] {title}: {tail}")
        for r in [r for r in mine if r["status"] in ("FAIL", "WARN")][:8]:
            lines.append(f"         {r['status']} {r['metric']} = {r['value']:.4g}")
    c1 = audit["checks"]["c1_zero_content"]
    nf = c1.get("nonfinite") or {}
    bad = {k: v["values"] for k, v in nf.items() if v.get("values")}
    al = {k: v["mismatched"] for k, v in (c1.get("alias_mismatch") or {}).items() if v.get("mismatched")}
    lines.append(f"  non-finite values: {bad or 'none'}; EMA alias mismatches: {al or 'none'}")
    o = audit["checks"]["c3_optimizer_dead"]
    if o.get("states"):
        lines.append(
            f"  optimizer map: {o['mapped']}/{o['states']} states named, {o.get('entries')} candidate tensors: {o.get('mapping')}"
            + ("" if o.get("mapping") == "exact" else " (names after an unassigned same-shape tensor may be shifted; see 'unmapped')")
        )
        if o.get("window_min_steps"):
            lines.append(f"  gradient-dead window: a zero moment = no gradient for at least ~{o['window_min_steps']:,} steps (beta2 and the block's live moments)")
    nh = c1.get("blocks_without_heads") or []
    if nh:
        lines.append(f"  head checks skipped (head count not inferable: no q_norm, no config): {nh}")
    if audit.get("revive"):
        lines.append("  revival (by hand, never automatic):")
        lines += [f"    {c}" for c in audit["revive"]]
    for label, d in (audit.get("diffs") or {}).items():
        lines.append(f"  check 12 vs {label} ({d['against']}): {len(d['moved'])} readings moved > {d['threshold']:.0%}")
        for mv in d["moved"][:10]:
            lines.append(f"    {mv['metric']}: {mv['before']:.4g} -> {mv['after']:.4g}")
    return "\n".join(lines) + "\n"


def diff(a: dict, b: dict, rules: list[dict], default_threshold: float = 0.25) -> dict:
    """Check 12: readings of audit b vs audit a (same roles). Exact zeros that appear or vanish always listed."""
    fa, fb = flatten(a["checks"]), flatten(b["checks"])
    moved = []
    for k in sorted(set(fa) & set(fb)):
        (x, bk), (y, _) = fa[k], fb[k]
        r = rule_for(bk, rules) or {}
        thr = r.get("change_warn", default_threshold)
        zero_flip = (x == 0) != (y == 0)
        rel = abs(y - x) / max(abs(x), 1e-12)
        if zero_flip or rel > thr:
            moved.append({"metric": k, "before": x, "after": y, "rel": rel if math.isfinite(rel) else None})
    moved.sort(key=lambda m: -(m["rel"] or 0))
    return {
        "against": a["meta"]["ckpt_id"],
        "threshold": default_threshold,
        "moved": moved,
        "only_before": sorted(set(fa) - set(fb))[:50],
        "only_after": sorted(set(fb) - set(fa))[:50],
    }


def write(audit: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit.json").write_text(json.dumps(audit, indent=1, default=str))
    (out_dir / "SUMMARY.txt").write_text(summary_text(audit))
    return out_dir
