"""Sweep the TC min/max emergence probe over good/bad checkpoint ladders."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import re
import sys
from pathlib import Path
from types import SimpleNamespace

_DT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_DT_ROOT) not in sys.path:
    sys.path.insert(0, str(_DT_ROOT))

from interp.cli import setup_logging
LOGGER = logging.getLogger(__name__)

from interp.tools.tc_emergence import (
    CASE_PRESETS,
    _case_to_dict,
    _checkpoint_step,
    _resolve_cases,
    add_probe_args,
    run_tc_emergence,
)


def _parse_run(spec: str) -> tuple[str, str]:
    if '=' not in spec:
        raise argparse.ArgumentTypeError('--run must be label=/path/glob')
    label, pattern = spec.split('=', 1)
    if not label or not pattern:
        raise argparse.ArgumentTypeError('--run must be label=/path/glob')
    return label, pattern


def _glob_checkpoints(pattern: str, limit: int | None = None) -> list[str]:
    paths = [Path(p) for p in glob.glob(pattern)]
    paths = sorted(paths, key=lambda p: (_checkpoint_step(str(p)) is None, _checkpoint_step(str(p)) or -1, str(p)))
    if limit is not None:
        paths = paths[:limit]
    return [str(p) for p in paths]


def _safe_label(text: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', text).strip('_') or 'run'


ROW_FIELDNAMES = [
    'run_label',
    'checkpoint',
    'checkpoint_step',
    'case_label',
    'case_date',
    'case_member',
    'case_step',
    'mode',
    'sigma',
    'truth_mslp_min_hpa',
    'input_mslp_min_hpa',
    'model_mslp_min_hpa',
    'mslp_min_gap_closed',
    'truth_wind10m_speed_max_ms',
    'input_wind10m_speed_max_ms',
    'model_wind10m_speed_max_ms',
    'wind10m_speed_max_gap_closed',
]

SUMMARY_FIELDNAMES = [
    'run_label',
    'checkpoint',
    'checkpoint_step',
    'mean_free_mslp_min_gap_closed',
    'mean_free_wind10m_speed_max_gap_closed',
    'tc_minmax_score',
    'mean_teacher_mslp_min_gap_closed',
    'mean_teacher_wind10m_speed_max_gap_closed',
    'mean_mslp_commitment_gap',
    'mean_wind_commitment_gap',
    'tc_ready',
]

BY_SIGMA_FIELDNAMES = [
    'run_label',
    'checkpoint',
    'checkpoint_step',
    'mode',
    'sigma',
    'mean_mslp_min_gap_closed',
    'mean_wind10m_speed_max_gap_closed',
    'mean_tc_minmax_score',
    'n_cases',
]

RANKED_SUMMARY_FIELDNAMES = [
    'rank',
    'run_label',
    'checkpoint',
    'checkpoint_step',
    'tc_rank_score',
    'mean_free_mslp_gap_closed_eligible_clipped',
    'mean_free_wind_gap_closed_eligible_clipped',
    'median_free_mslp_gap_closed_eligible',
    'median_free_wind_gap_closed_eligible',
    'mean_teacher_mslp_gap_closed_eligible_clipped',
    'mean_teacher_wind_gap_closed_eligible_clipped',
    'mean_commitment_mslp_gap_closed_eligible_clipped',
    'mean_commitment_wind_gap_closed_eligible_clipped',
    'n_mslp_gap_eligible',
    'n_wind_gap_eligible',
    'n_both_gap_eligible',
    'n_mslp_gap_inverted',
    'n_wind_gap_inverted',
    'gap_denominator_threshold',
    'gap_clip_min',
    'gap_clip_max',
]

RANKED_BY_SIGMA_FIELDNAMES = [
    'run_label',
    'checkpoint',
    'checkpoint_step',
    'mode',
    'sigma',
    'mean_mslp_gap_closed_eligible_clipped',
    'mean_wind_gap_closed_eligible_clipped',
    'mean_tc_rank_score',
    'n_mslp_gap_eligible',
    'n_wind_gap_eligible',
    'n_both_gap_eligible',
    'gap_denominator_threshold',
    'gap_clip_min',
    'gap_clip_max',
]


def _row_dict(run_label: str, result: dict, case: dict, probe: dict) -> dict:
    metrics = probe['metrics']
    gaps = probe['gap_closed']
    return {
        'run_label': run_label,
        'checkpoint': result['checkpoint'],
        'checkpoint_step': result.get('checkpoint_step'),
        'case_label': case['case']['label'],
        'case_date': case['case']['date'],
        'case_member': case['case']['member'],
        'case_step': case['case']['step'],
        'mode': probe['mode'],
        'sigma': probe['sigma'],
        'truth_mslp_min_hpa': case['truth'].get('mslp_min_hpa'),
        'input_mslp_min_hpa': case['x_interp'].get('mslp_min_hpa'),
        'model_mslp_min_hpa': metrics.get('mslp_min_hpa'),
        'mslp_min_gap_closed': gaps.get('mslp_min_hpa'),
        'truth_wind10m_speed_max_ms': case['truth'].get('wind10m_speed_max_ms'),
        'input_wind10m_speed_max_ms': case['x_interp'].get('wind10m_speed_max_ms'),
        'model_wind10m_speed_max_ms': metrics.get('wind10m_speed_max_ms'),
        'wind10m_speed_max_gap_closed': gaps.get('wind10m_speed_max_ms'),
    }


def _summary_row(run_label: str, result: dict, mslp_threshold: float, wind_threshold: float) -> dict:
    summary = result['summary']
    free = summary.get('mean_free_gap_closed', {})
    mslp = free.get('mslp_min_hpa')
    wind = free.get('wind10m_speed_max_ms')
    ready = mslp is not None and wind is not None and mslp >= mslp_threshold and wind >= wind_threshold
    return {
        'run_label': run_label,
        'checkpoint': result['checkpoint'],
        'checkpoint_step': result.get('checkpoint_step'),
        'mean_free_mslp_min_gap_closed': mslp,
        'mean_free_wind10m_speed_max_gap_closed': wind,
        'tc_minmax_score': summary.get('tc_minmax_score'),
        'mean_teacher_mslp_min_gap_closed': summary.get('mean_teacher_gap_closed', {}).get('mslp_min_hpa'),
        'mean_teacher_wind10m_speed_max_gap_closed': summary.get('mean_teacher_gap_closed', {}).get('wind10m_speed_max_ms'),
        'mean_mslp_commitment_gap': summary.get('mean_commitment_gap', {}).get('mslp_min_hpa'),
        'mean_wind_commitment_gap': summary.get('mean_commitment_gap', {}).get('wind10m_speed_max_ms'),
        'tc_ready': int(bool(ready)),
    }


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        if not rows:
            return
        fieldnames = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _float_or_none(value):
    if value is None or value == '':
        return None
    return float(value)


def _mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def _median(vals: list[float]) -> float | None:
    if not vals:
        return None
    ordered = sorted(vals)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _gap_denominator(row: dict, metric: str) -> float | None:
    if metric == 'mslp_min_hpa':
        x_interp = _float_or_none(row.get('input_mslp_min_hpa'))
        truth = _float_or_none(row.get('truth_mslp_min_hpa'))
        if x_interp is None or truth is None:
            return None
        return x_interp - truth
    if metric == 'wind10m_speed_max_ms':
        x_interp = _float_or_none(row.get('input_wind10m_speed_max_ms'))
        truth = _float_or_none(row.get('truth_wind10m_speed_max_ms'))
        if x_interp is None or truth is None:
            return None
        return truth - x_interp
    raise KeyError(metric)


def _gap_field(metric: str) -> str:
    if metric == 'mslp_min_hpa':
        return 'mslp_min_gap_closed'
    if metric == 'wind10m_speed_max_ms':
        return 'wind10m_speed_max_gap_closed'
    raise KeyError(metric)


def _model_field(metric: str) -> str:
    if metric == 'mslp_min_hpa':
        return 'model_mslp_min_hpa'
    if metric == 'wind10m_speed_max_ms':
        return 'model_wind10m_speed_max_ms'
    raise KeyError(metric)


def _case_key(row: dict) -> tuple:
    return (
        row['run_label'],
        row['checkpoint'],
        row['checkpoint_step'],
        row['case_label'],
        row['case_date'],
        row['case_member'],
        row['case_step'],
    )


def _checkpoint_key(row: dict) -> tuple:
    return (row['run_label'], row['checkpoint'], row['checkpoint_step'])


def _best_case_probe(rows: list[dict], metric: str, mode: str) -> dict | None:
    mode_rows = [row for row in rows if row['mode'] == mode and row.get(_model_field(metric)) not in (None, '')]
    if not mode_rows:
        return None
    if metric == 'mslp_min_hpa':
        return min(mode_rows, key=lambda row: float(row[_model_field(metric)]))
    return max(mode_rows, key=lambda row: float(row[_model_field(metric)]))


def _eligible_gap(row: dict, metric: str, min_denominator: float, clip_min: float, clip_max: float) -> float | None:
    denom = _gap_denominator(row, metric)
    gap = _float_or_none(row.get(_gap_field(metric)))
    if denom is None or gap is None or denom < min_denominator:
        return None
    return _clip(gap, clip_min, clip_max)


def _ranked_summary_rows(rows: list[dict], min_denominator: float, clip_min: float, clip_max: float) -> list[dict]:
    by_case: dict[tuple, list[dict]] = {}
    for row in rows:
        by_case.setdefault(_case_key(row), []).append(row)

    by_checkpoint: dict[tuple, list[list[dict]]] = {}
    for case_rows in by_case.values():
        by_checkpoint.setdefault(_checkpoint_key(case_rows[0]), []).append(case_rows)

    out = []
    for (run_label, checkpoint, checkpoint_step), case_groups in sorted(
        by_checkpoint.items(), key=lambda item: (item[0][0], int(item[0][2]) if item[0][2] not in (None, '') else -1)
    ):
        row = {
            'rank': None,
            'run_label': run_label,
            'checkpoint': checkpoint,
            'checkpoint_step': checkpoint_step,
            'gap_denominator_threshold': min_denominator,
            'gap_clip_min': clip_min,
            'gap_clip_max': clip_max,
        }
        free_means = {}
        for metric, short in (('mslp_min_hpa', 'mslp'), ('wind10m_speed_max_ms', 'wind')):
            denominators = [_gap_denominator(group[0], metric) for group in case_groups]
            denominators = [d for d in denominators if d is not None]
            row[f'n_{short}_gap_eligible'] = sum(1 for d in denominators if d >= min_denominator)
            row[f'n_{short}_gap_inverted'] = sum(1 for d in denominators if d < 0.0)
            for mode in ('free', 'teacher'):
                vals = []
                raw_vals = []
                for group in case_groups:
                    best = _best_case_probe(group, metric, mode)
                    if best is None:
                        continue
                    clipped = _eligible_gap(best, metric, min_denominator, clip_min, clip_max)
                    raw_gap = _float_or_none(best.get(_gap_field(metric)))
                    if clipped is not None:
                        vals.append(clipped)
                    if raw_gap is not None and _gap_denominator(best, metric) is not None and _gap_denominator(best, metric) >= min_denominator:
                        raw_vals.append(raw_gap)
                prefix = f'{mode}_{short}'
                row[f'mean_{prefix}_gap_closed_eligible_clipped'] = _mean(vals)
                if mode == 'free':
                    row[f'median_{prefix}_gap_closed_eligible'] = _median(raw_vals)
                    free_means[short] = row[f'mean_{prefix}_gap_closed_eligible_clipped']

            commitments = []
            for group in case_groups:
                free = _best_case_probe(group, metric, 'free')
                teacher = _best_case_probe(group, metric, 'teacher')
                if free is None or teacher is None:
                    continue
                denom = _gap_denominator(free, metric)
                if denom is None or denom < min_denominator:
                    continue
                free_gap = _float_or_none(free.get(_gap_field(metric)))
                teacher_gap = _float_or_none(teacher.get(_gap_field(metric)))
                if free_gap is not None and teacher_gap is not None:
                    commitments.append(_clip(teacher_gap - free_gap, clip_min, clip_max))
            row[f'mean_commitment_{short}_gap_closed_eligible_clipped'] = _mean(commitments)

        row['n_both_gap_eligible'] = 0
        for group in case_groups:
            mslp_denom = _gap_denominator(group[0], 'mslp_min_hpa')
            wind_denom = _gap_denominator(group[0], 'wind10m_speed_max_ms')
            if mslp_denom is not None and wind_denom is not None and mslp_denom >= min_denominator and wind_denom >= min_denominator:
                row['n_both_gap_eligible'] += 1
        mslp = free_means.get('mslp')
        wind = free_means.get('wind')
        row['tc_rank_score'] = 0.5 * mslp + 0.5 * wind if mslp is not None and wind is not None else None
        out.append(row)

    ranked = sorted(out, key=lambda row: (row['tc_rank_score'] is None, -(row['tc_rank_score'] or -1.0e99), row['run_label'], int(row['checkpoint_step'] or -1)))
    for rank, row in enumerate(ranked, start=1):
        row['rank'] = rank
    return ranked


def _ranked_by_sigma_rows(rows: list[dict], min_denominator: float, clip_min: float, clip_max: float) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (
            row['run_label'],
            row['checkpoint'],
            row['checkpoint_step'],
            row['mode'],
            row['sigma'],
        )
        groups.setdefault(key, []).append(row)

    out = []
    for (run_label, checkpoint, checkpoint_step, mode, sigma), group in sorted(
        groups.items(), key=lambda item: (item[0][0], int(item[0][2]) if item[0][2] not in (None, '') else -1, item[0][3], float(item[0][4]))
    ):
        mslp_vals = [_eligible_gap(row, 'mslp_min_hpa', min_denominator, clip_min, clip_max) for row in group]
        wind_vals = [_eligible_gap(row, 'wind10m_speed_max_ms', min_denominator, clip_min, clip_max) for row in group]
        mslp_vals = [v for v in mslp_vals if v is not None]
        wind_vals = [v for v in wind_vals if v is not None]
        both = 0
        for row in group:
            mslp_denom = _gap_denominator(row, 'mslp_min_hpa')
            wind_denom = _gap_denominator(row, 'wind10m_speed_max_ms')
            if mslp_denom is not None and wind_denom is not None and mslp_denom >= min_denominator and wind_denom >= min_denominator:
                both += 1
        mslp = _mean(mslp_vals)
        wind = _mean(wind_vals)
        out.append({
            'run_label': run_label,
            'checkpoint': checkpoint,
            'checkpoint_step': checkpoint_step,
            'mode': mode,
            'sigma': sigma,
            'mean_mslp_gap_closed_eligible_clipped': mslp,
            'mean_wind_gap_closed_eligible_clipped': wind,
            'mean_tc_rank_score': 0.5 * mslp + 0.5 * wind if mslp is not None and wind is not None else None,
            'n_mslp_gap_eligible': len(mslp_vals),
            'n_wind_gap_eligible': len(wind_vals),
            'n_both_gap_eligible': both,
            'gap_denominator_threshold': min_denominator,
            'gap_clip_min': clip_min,
            'gap_clip_max': clip_max,
        })
    return out


def _by_sigma_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (
            row['run_label'],
            row['checkpoint'],
            row['checkpoint_step'],
            row['mode'],
            row['sigma'],
        )
        groups.setdefault(key, []).append(row)
    out = []
    for (run_label, checkpoint, checkpoint_step, mode, sigma), group in sorted(groups.items(), key=lambda item: (item[0][0], item[0][2] or -1, item[0][3], float(item[0][4]))):
        mslp_vals = [float(row['mslp_min_gap_closed']) for row in group if row['mslp_min_gap_closed'] is not None]
        wind_vals = [float(row['wind10m_speed_max_gap_closed']) for row in group if row['wind10m_speed_max_gap_closed'] is not None]
        mslp = sum(mslp_vals) / len(mslp_vals) if mslp_vals else None
        wind = sum(wind_vals) / len(wind_vals) if wind_vals else None
        score = 0.5 * mslp + 0.5 * wind if mslp is not None and wind is not None else None
        out.append({
            'run_label': run_label,
            'checkpoint': checkpoint,
            'checkpoint_step': checkpoint_step,
            'mode': mode,
            'sigma': sigma,
            'mean_mslp_min_gap_closed': mslp,
            'mean_wind10m_speed_max_gap_closed': wind,
            'mean_tc_minmax_score': score,
            'n_cases': len(group),
        })
    return out


def run_sweep(args):
    runs = [_parse_run(spec) for spec in args.run]
    output_dir = Path(args.output_dir)
    if (getattr(args, 'case_cache', None) is None and getattr(args, 'case_preset', None)
            and int(getattr(args, 'strong_input_top_k', 0) or 0) > 0):
        args.case_cache = str(output_dir / 'tc_emergence_selected_cases.json')
    rows = []
    summary_rows = []
    first_ready = {}
    bundle_dir, cases = _resolve_cases(args)
    resolved_cases = [_case_to_dict(case) for case in cases]

    if args.dry_run:
        from interp.tools.tc_emergence import _case_bundle_path

        plan = {
            'bundle_dir': bundle_dir,
            'case_preset': args.case_preset,
            'case_cache': getattr(args, 'case_cache', None),
            'strong_input_top_k': args.strong_input_top_k,
            'sigmas': [float(s) for s in args.sigmas],
            'modes': list(args.modes),
            'cases': [{**_case_to_dict(case), 'bundle_path': _case_bundle_path(bundle_dir, case)} for case in cases],
            'runs': [],
        }
        for run_label, pattern in runs:
            checkpoints = _glob_checkpoints(pattern, limit=args.limit_per_run)
            if not checkpoints:
                raise SystemExit(f'No checkpoints matched {run_label}={pattern}')
            plan['runs'].append({
                'label': run_label,
                'pattern': pattern,
                'n_checkpoints': len(checkpoints),
                'checkpoints': checkpoints,
            })
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'tc_emergence_sweep_dry_run.json', 'w') as fh:
            json.dump(plan, fh, indent=2)
        print(json.dumps(plan, indent=2))
        return plan

    for run_label, pattern in runs:
        checkpoints = _glob_checkpoints(pattern, limit=args.limit_per_run)
        if not checkpoints:
            raise SystemExit(f'No checkpoints matched {run_label}={pattern}')
        for checkpoint in checkpoints:
            step = _checkpoint_step(checkpoint)
            step_label = f'step_{step:06d}' if step is not None else _safe_label(Path(checkpoint).stem)
            out = output_dir / _safe_label(run_label) / step_label
            LOGGER.info('tc_emergence_sweep: probing run=%s checkpoint=%s', run_label, checkpoint)
            probe_args = SimpleNamespace(
                checkpoint=checkpoint,
                output_dir=str(out),
                device=args.device,
                precision=args.precision,
                event=None,
                bundle_dir=bundle_dir,
                dates=None,
                members=None,
                steps=None,
                case_preset=None,
                tc_case=cases,
                strong_input_top_k=0,
                auto_window=None,
                sigmas=args.sigmas,
                modes=args.modes,
                noise_seed=args.noise_seed,
                eye_radius_km=args.eye_radius_km,
                case_cache=None,
            )
            result = run_tc_emergence(probe_args)
            for case in result['cases']:
                for probe in case['probes']:
                    rows.append(_row_dict(run_label, result, case, probe))
            summary = _summary_row(run_label, result, args.mslp_gap_threshold, args.wind_gap_threshold)
            summary_rows.append(summary)
            if summary['tc_ready'] and run_label not in first_ready:
                first_ready[run_label] = {
                    'checkpoint_step': summary['checkpoint_step'],
                    'checkpoint': summary['checkpoint'],
                }

    output_dir.mkdir(parents=True, exist_ok=True)
    by_sigma_rows = _by_sigma_rows(rows)
    ranked_summary_rows = _ranked_summary_rows(
        rows,
        args.gap_denominator_threshold,
        args.gap_clip_min,
        args.gap_clip_max,
    )
    ranked_by_sigma_rows = _ranked_by_sigma_rows(
        rows,
        args.gap_denominator_threshold,
        args.gap_clip_min,
        args.gap_clip_max,
    )
    _write_csv(output_dir / 'tc_emergence_sweep_rows.csv', rows, ROW_FIELDNAMES)
    _write_csv(output_dir / 'tc_emergence_sweep_by_sigma.csv', by_sigma_rows, BY_SIGMA_FIELDNAMES)
    _write_csv(output_dir / 'tc_emergence_sweep_summary.csv', summary_rows, SUMMARY_FIELDNAMES)
    _write_csv(output_dir / 'tc_emergence_sweep_ranked.csv', ranked_summary_rows, RANKED_SUMMARY_FIELDNAMES)
    _write_csv(output_dir / 'tc_emergence_sweep_by_sigma_ranked.csv', ranked_by_sigma_rows, RANKED_BY_SIGMA_FIELDNAMES)
    summary = {
        'tool': 'tc_emergence_sweep',
        'runs': [{'label': label, 'pattern': pattern} for label, pattern in runs],
        'bundle_dir': bundle_dir,
        'case_preset': args.case_preset,
        'case_cache': getattr(args, 'case_cache', None),
        'resolved_cases': resolved_cases,
        'sigmas': [float(s) for s in args.sigmas],
        'modes': list(args.modes),
        'thresholds': {
            'mslp_min_gap_closed': args.mslp_gap_threshold,
            'wind10m_speed_max_gap_closed': args.wind_gap_threshold,
        },
        'ranked_diagnostic_rule': {
            'gap_denominator_threshold': args.gap_denominator_threshold,
            'gap_clip_min': args.gap_clip_min,
            'gap_clip_max': args.gap_clip_max,
            'meaning': 'Ranked diagnostic filters cases where target is stronger than input by at least the denominator threshold, clips gap-closure outliers, and ranks checkpoints by mean free MSLP/wind closure. It is a rank, not a hard readiness gate.',
        },
        'first_ready': first_ready,
        'summary_rows': summary_rows,
        'by_sigma_rows': by_sigma_rows,
        'ranked_summary_rows': ranked_summary_rows,
        'ranked_by_sigma_rows': ranked_by_sigma_rows,
    }
    with open(output_dir / 'tc_emergence_sweep.json', 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({'first_ready': first_ready, 'n_probe_rows': len(rows), 'n_summary_rows': len(summary_rows)}, indent=2))
    return summary


def add_case_args_no_event(p: argparse.ArgumentParser):
    g = p.add_argument_group('TC cases')
    g.add_argument('--case-preset', default='o96_o320_idalia_franklin', choices=sorted(CASE_PRESETS),
                   help='Default: two o96->o320 validation TCs: Franklin + Idalia')
    g.add_argument('--tc-case', action='append', default=None,
                   help='Explicit case label:date:member:step:lat0,lat1,lon0,lon1. Repeatable.')
    g.add_argument('--strong-input-top-k', type=int, default=20,
                   help='Select top K cases per TC by input MSLP-min and top K by input wind-speed-max. Use 0 for fallback hand-picked cases.')
    g.add_argument('--case-cache', default=None,
                   help='JSON cache for resolved preset TC cases. Defaults to output-dir/tc_emergence_selected_cases.json.')
    return p


def main(argv=None):
    p = argparse.ArgumentParser(description='Sweep fixed-sigma TC min/max probe over checkpoint ladders')
    p.add_argument('--run', action='append', required=True,
                   help='Run ladder spec label=/path/to/anemoi-by_step-*.ckpt. Repeat for good and poor runs.')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    p.add_argument('--precision', default='fp32', choices=['fp32', 'fp16', 'bf16'])
    p.add_argument('--limit-per-run', type=int, default=None,
                   help='Debug limiter: only first N checkpoints per run after step-sort')
    p.add_argument('--dry-run', action='store_true',
                   help='Only resolve checkpoint globs and TC case bundles; do not load models')
    p.add_argument('--mslp-gap-threshold', type=float, default=0.5,
                   help='Prototype readiness threshold for mean free MSLP gap closure')
    p.add_argument('--wind-gap-threshold', type=float, default=0.5,
                   help='Prototype readiness threshold for mean free wind-max gap closure')
    p.add_argument('--gap-denominator-threshold', type=float, default=1.0,
                   help='Minimum input-to-truth TC-intensity gap required for ranked diagnostic eligibility')
    p.add_argument('--gap-clip-min', type=float, default=-1.0,
                   help='Lower clip for ranked gap-closure diagnostics')
    p.add_argument('--gap-clip-max', type=float, default=2.0,
                   help='Upper clip for ranked gap-closure diagnostics')
    p.add_argument('--bundle-dir', default=None)
    add_case_args_no_event(p)
    add_probe_args(p)
    args = p.parse_args(argv)
    setup_logging()
    return run_sweep(args)


if __name__ == '__main__':
    main()
