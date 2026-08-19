"""tccompare runner: resolve source roots -> lazy-parse -> score -> plot.

Source spec grammar (``--sources``), comma-separated ``role=value`` items:

- ``model=j9f3``              rd expver under ``<scratch>/eval/<lane_short>/tctracker/j9f3``
- ``target=od:enfo:0001``     operational ref in the shared cache
                              ``<scratch>/eval/tcrefs/od_enfo_0001_o<grid>``
- ``input=/abs/path``         explicit tracker run root
- ``target`` / ``input``      bare role: spec taken from the lane defaults
                              (lane ``tctracker.sources`` else ``prepml`` blocks)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from eval._backends.tctracker.pipeline import _lane_short_name
from eval._backends.tctracker.sources import (
    default_source_specs,
    refs_cache_root,
    source_id_for,
)
from eval._backends.tctracker.tables import parse_run_root

from . import scorer

LOG = logging.getLogger(__name__)


def resolve_source_roots(
    sources_arg: str,
    lane_name: str,
    lane_config: dict[str, Any],
    host_config: dict[str, Any],
) -> dict[str, tuple[str, Path]]:
    """{role: (source_id, run_root)} from the --sources spec."""
    tcfg = lane_config.get("tctracker") or {}
    grid = int(tcfg.get("grid", 320))
    lane_short = _lane_short_name(lane_name, lane_config)
    lane_root = Path(host_config["scratch_root"]) / "eval" / lane_short / "tctracker"
    defaults = default_source_specs(lane_config)

    out: dict[str, tuple[str, Path]] = {}
    for token in str(sources_arg or "").split(","):
        token = token.strip()
        if not token:
            continue
        role, _, value = token.partition("=")
        role = role.strip()
        value = value.strip()
        if value.startswith("/"):
            out[role] = (value.rstrip("/").rsplit("/", 1)[-1], Path(value))
            continue
        if not value:
            spec = defaults.get(role)
            if not spec or not spec.get("expver"):
                raise SystemExit(
                    f"--sources role {role!r} has no value and no lane default "
                    f"(tctracker.sources.{role} / prepml input+output blocks)"
                )
            sid = source_id_for(role, spec)
            out[role] = (sid, refs_cache_root(host_config) / f"{sid}_o{grid}")
            continue
        if ":" in value:
            fdb_class, stream, expver = (value.split(":") + ["0001"])[:3]
            spec = {"class": fdb_class, "stream": stream, "expver": expver}
            sid = source_id_for(role, spec)
            out[role] = (sid, refs_cache_root(host_config) / f"{sid}_o{grid}")
            continue
        out[role] = (value, lane_root / value)
    if not out:
        raise SystemExit("--sources resolved to nothing; pass e.g. model=j9f3,ctrl=j95z,target,input")
    return out


def load_sources(
    roots: dict[str, tuple[str, Path]],
    *,
    reparse: bool = False,
) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    for role, (sid, run_root) in roots.items():
        parsed = run_root / "parsed"
        if reparse or not (parsed / "provenance.json").exists():
            if not (run_root / "tars").exists():
                raise SystemExit(
                    f"source {role}={sid}: no parsed tables and no tars/ under {run_root} "
                    f"— run `eval.cli tctracker` for this source first"
                )
            LOG.info("parsing %s (%s) from tars under %s", role, sid, run_root)
            parse_run_root(run_root, role=role, source_id=sid)
        sources[role] = scorer.load_source(parsed)
        n = sources[role]["provenance"].get("n_forecasts_present")
        LOG.info("loaded %s (%s): %d track rows, %s forecasts",
                 role, sid, len(sources[role]["records"]), n)
    return sources


def run(
    *,
    sources_arg: str,
    months: list[str],
    basins: list[str],
    lane_name: str,
    lane_config: dict[str, Any],
    host_config: dict[str, Any],
    out_dir: Path,
    dates: list[str] | None = None,
    reparse: bool = False,
    no_plots: bool = False,
    top_k_cases: int = 3,
) -> dict[str, Any]:
    roots = resolve_source_roots(sources_arg, lane_name, lane_config, host_config)
    sources = load_sources(roots, reparse=reparse)
    if dates:
        # Paired-window pinning: restrict every source to a common init-date
        # set so the compared distributions sample the SAME weather.
        pinned = set(str(d) for d in dates)
        for role, src in sources.items():
            for key in ("records", "summary", "forecasts"):
                df = src[key]
                if not df.empty:
                    src[key] = df[df["init_date"].astype(str).isin(pinned)]
            src["provenance"]["dates_pinned"] = sorted(pinned)
            LOG.info("%s: pinned to %d dates -> %d track rows",
                     role, len(pinned), len(src["records"]))
    metrics = scorer.score_sources(sources, months, basins)
    metrics["dates_pinned"] = sorted(dates) if dates else None
    metrics["cases"] = {
        basin: scorer.select_cases(sources, months, basin, top_k=top_k_cases)
        for basin in basins
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "tc_tracks_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8")
    LOG.info("metrics written to %s", metrics_path)

    import pandas as pd
    flat = pd.DataFrame(metrics["metrics"]).drop(columns=["step_intensity"], errors="ignore")
    flat.to_csv(out_dir / "tc_tracks_metrics.csv", index=False)

    if not no_plots:
        from . import plotter
        paths = plotter.render_all(sources, metrics, months, basins, out_dir)
        LOG.info("rendered %d figures under %s", len(paths), out_dir)
    return metrics
