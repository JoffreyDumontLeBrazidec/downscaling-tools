"""Fast fixed-sigma TC min-MSLP / max-10m-wind probe.

This tool is deliberately narrow: around fixed TC cases, at fixed sigma noise
levels, it logs only two physical extremes:

* ``mslp_min_hpa``
* ``wind10m_speed_max_ms``

The intended calibration is a good-TC o96->o320 run and a poor-TC o96->o320
run, each swept over its by-step checkpoints. The companion
``tc_emergence_sweep`` tool automates that ladder comparison.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_DT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_DT_ROOT) not in sys.path:
    sys.path.insert(0, str(_DT_ROOT))

from interp.cli import add_event_args, add_model_args, setup_logging

LOGGER = logging.getLogger(__name__)
PA_TO_HPA = 1.0 / 100.0
MSLP_PHYS_FLOOR_HPA = 870.0
DEFAULT_TC_SIGMAS = [120.0, 80.0, 50.0, 20.0, 10.0, 5.0, 1.0]
CASE_SELECTOR_CACHE_VERSION = 1

CASE_PRESETS = {
    'o96_o320_idalia_franklin': {
        'bundle_dir': '/home/ecm5702/hpcperm/data/input_data/o96_o320/idalia',
        # Fallback cases used only with --strong-input-top-k 0.
        'cases': [
            'franklin:20230826:01:072:10,45,280,320',
            'idalia:20230829:01:024:20,35,270,290',
        ],
        # Date windows are the available validation bundle windows for the two TCs.
        # The selector ranks bundles by input-field extremes inside these windows.
        'tc_windows': [
            {
                'label': 'franklin',
                'date_ranges': [('20230816', '20230823')],
                'window': (10.0, 45.0, 280.0, 320.0),
            },
            {
                'label': 'idalia',
                'date_ranges': [('20230826', '20230830')],
                'window': (20.0, 35.0, 270.0, 290.0),
            },
        ],
    },
}


@dataclass(frozen=True)
class TCCase:
    label: str
    date: str
    member: str
    step: str
    window: tuple[float, float, float, float]
    source: str = 'manual'


def _normalise_step(step: str) -> str:
    return f'{int(step):03d}'


def parse_tc_case(spec: str) -> TCCase:
    """Parse ``label:date:member:step:lat0,lat1,lon0,lon1``."""
    parts = spec.split(':')
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            '--tc-case must be label:date:member:step:lat0,lat1,lon0,lon1'
        )
    label, date, member, step, window_s = parts
    window_parts = [float(x) for x in window_s.split(',')]
    if len(window_parts) != 4:
        raise argparse.ArgumentTypeError('tc-case window must be lat0,lat1,lon0,lon1')
    return TCCase(label=label, date=date, member=member, step=_normalise_step(step), window=tuple(window_parts))


def case_to_spec(case: TCCase) -> str:
    window = ','.join(f'{x:g}' for x in case.window)
    return f'{case.label}:{case.date}:{case.member}:{case.step}:{window}'


def _case_to_dict(case: TCCase) -> dict:
    return {
        'label': case.label,
        'date': case.date,
        'member': case.member,
        'step': case.step,
        'window': list(case.window),
        'source': case.source,
    }


def _case_from_dict(row: dict) -> TCCase:
    return TCCase(
        label=str(row['label']),
        date=str(row['date']),
        member=str(row['member']),
        step=_normalise_step(str(row['step'])),
        window=tuple(float(x) for x in row['window']),
        source=str(row.get('source', 'cache')),
    )


def _selector_key(preset_name: str, bundle_dir: str, preset: dict, top_k: int) -> dict:
    return {
        'version': CASE_SELECTOR_CACHE_VERSION,
        'preset': preset_name,
        'bundle_dir': str(bundle_dir),
        'top_k': int(top_k),
        'tc_windows': [
            {
                'label': tc['label'],
                'date_ranges': [list(pair) for pair in tc['date_ranges']],
                'window': list(tc['window']),
            }
            for tc in preset.get('tc_windows', [])
        ],
    }


def _bundle_meta_from_name(path: str) -> tuple[str, str, str]:
    name = Path(path).name
    match = re.search(r'date(\d+).*mem(\d+).*step(\d+)h', name)
    if not match:
        raise ValueError(f'cannot parse date/member/step from {name}')
    date, member, step = match.groups()
    return date, member, _normalise_step(step)


def _date_in_ranges(date: str, ranges: list[tuple[str, str]]) -> bool:
    return any(start <= date <= end for start, end in ranges)


def _window_mask(lat, lon, window: tuple[float, float, float, float]):
    import numpy as np

    lat = np.asarray(lat)
    lon = np.asarray(lon) % 360.0
    lat0, lat1, lon0, lon1 = window
    lat_sel = (lat >= lat0) & (lat <= lat1)
    lon_sel = (lon >= lon0) & (lon <= lon1) if lon0 <= lon1 else ((lon >= lon0) | (lon <= lon1))
    return lat_sel & lon_sel


def _scan_input_extremes(path: str, window: tuple[float, float, float, float]) -> dict[str, float]:
    import numpy as np
    from netCDF4 import Dataset

    with Dataset(path, 'r') as ds:
        mask = _window_mask(ds.variables['lat_lres'][:], ds.variables['lon_lres'][:], window)
        if not mask.any():
            raise ValueError(f'input window {window} selects no lres cells for {path}')
        mslp = np.asarray(ds.variables['in_lres_msl'][:], dtype='float64')[mask] * PA_TO_HPA
        u10 = np.asarray(ds.variables['in_lres_10u'][:], dtype='float64')[mask]
        v10 = np.asarray(ds.variables['in_lres_10v'][:], dtype='float64')[mask]
        wind = np.sqrt(u10 * u10 + v10 * v10)
    return {
        'input_mslp_min_hpa': float(np.nanmin(mslp)),
        'input_wind10m_speed_max_ms': float(np.nanmax(wind)),
    }


def _select_strong_input_cases(bundle_dir: str, preset: dict, top_k: int) -> list[TCCase]:
    if top_k <= 0:
        return []
    selected: dict[tuple, list[str]] = {}
    bundle_paths = []
    for path in sorted(glob.glob(str(Path(bundle_dir) / '*input_bundle.nc'))):
        try:
            date, member, step = _bundle_meta_from_name(path)
        except ValueError:
            LOGGER.warning('skipping bundle with unparseable name: %s', path)
            continue
        bundle_paths.append((path, date, member, step))
    if not bundle_paths:
        raise SystemExit(f'no input_bundle.nc files found in {bundle_dir}')

    for tc in preset.get('tc_windows', []):
        rows = []
        for path, date, member, step in bundle_paths:
            if not _date_in_ranges(date, tc['date_ranges']):
                continue
            extremes = _scan_input_extremes(path, tc['window'])
            rows.append({
                'path': path,
                'date': date,
                'member': member,
                'step': step,
                **extremes,
            })
        if not rows:
            raise SystemExit(f'no bundles found for TC selector {tc["label"]}')
        LOGGER.info('TC selector %s: scanned %d candidate bundles', tc['label'], len(rows))
        mslp_rows = sorted(rows, key=lambda row: row['input_mslp_min_hpa'])[:top_k]
        wind_rows = sorted(rows, key=lambda row: row['input_wind10m_speed_max_ms'], reverse=True)[:top_k]
        for metric_name, metric_rows in (('input_mslp_min_hpa', mslp_rows), ('input_wind10m_speed_max_ms', wind_rows)):
            for rank, row in enumerate(metric_rows, start=1):
                key = (tc['label'], row['date'], row['member'], row['step'], tuple(tc['window']))
                selected.setdefault(key, []).append(f'{metric_name}_rank{rank:02d}')
    cases = []
    for (label, date, member, step, window), sources in sorted(selected.items()):
        cases.append(TCCase(label=label, date=date, member=member, step=step, window=window, source=';'.join(sources)))
    LOGGER.info('selected %d strong-input TC cases from %s (top_k=%d)', len(cases), bundle_dir, top_k)
    return cases


def _load_case_cache(cache_path: str | None, selector_key: dict) -> list[TCCase] | None:
    if not cache_path:
        return None
    path = Path(cache_path)
    try:
        with open(path) as fh:
            payload = json.load(fh)
    except FileNotFoundError:
        return None
    except Exception as exc:
        LOGGER.warning('ignoring unreadable TC case cache %s: %s', path, exc)
        return None
    if payload.get('selector') != selector_key:
        LOGGER.info('ignoring stale TC case cache %s', path)
        return None
    try:
        cases = [_case_from_dict(row) for row in payload.get('cases', [])]
    except Exception as exc:
        LOGGER.warning('ignoring malformed TC case cache %s: %s', path, exc)
        return None
    LOGGER.info('reused %d TC cases from %s', len(cases), path)
    return cases


def _write_case_cache(cache_path: str | None, selector_key: dict, cases: list[TCCase]) -> None:
    if not cache_path:
        return
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'tool': 'tc_emergence_case_selector',
        'selector': selector_key,
        'cases': [_case_to_dict(case) for case in cases],
    }
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w') as fh:
        json.dump(payload, fh, indent=2)
    tmp.replace(path)
    LOGGER.info('wrote TC case cache %s', path)


def _resolve_cases(args) -> tuple[str, list[TCCase]]:
    preset_name = getattr(args, 'case_preset', None)
    preset = CASE_PRESETS.get(preset_name) if preset_name else None
    if preset_name and preset is None:
        raise SystemExit(f'unknown --case-preset {preset_name!r}; options: {sorted(CASE_PRESETS)}')
    explicit_cases = getattr(args, 'tc_case', None) or []
    strong_top_k = int(getattr(args, 'strong_input_top_k', 20) or 0)
    if explicit_cases:
        cases = list(explicit_cases) if all(isinstance(case, TCCase) for case in explicit_cases) else [parse_tc_case(spec) for spec in explicit_cases]
        bundle_dir = getattr(args, 'bundle_dir', None) or (preset or {}).get('bundle_dir')
    elif preset:
        bundle_dir = getattr(args, 'bundle_dir', None) or preset['bundle_dir']
        if strong_top_k > 0:
            selector_key = _selector_key(preset_name, bundle_dir, preset, strong_top_k)
            cases = _load_case_cache(getattr(args, 'case_cache', None), selector_key)
            if cases is None:
                cases = _select_strong_input_cases(bundle_dir, preset, strong_top_k)
                _write_case_cache(getattr(args, 'case_cache', None), selector_key, cases)
        else:
            cases = [parse_tc_case(spec) for spec in preset['cases']]
    else:
        from interp.core.data import resolve_event_args

        bundle_dir, dates, members, steps, label = resolve_event_args(args)
        window = tuple(float(x) for x in (getattr(args, 'auto_window', None) or '10,45,280,320').split(','))
        cases = [
            TCCase(label=label, date=date, member=member, step=_normalise_step(step), window=window, source='event')
            for date in dates
            for member in members
            for step in steps
        ]
    if not bundle_dir:
        raise SystemExit('need --case-preset, --event, or --bundle-dir with --tc-case')
    return bundle_dir, cases


def _checkpoint_step(path: str) -> int | None:
    matches = re.findall(r'step[_-](\d+)', Path(path).name)
    return int(matches[-1]) if matches else None


def _fixed_noise(shape, *, seed: int, sigma: float, mode: str, device: str, dtype):
    import torch

    mode_offset = 0 if mode == 'free' else 100_000
    sigma_key = int(round(float(sigma) * 1000.0))
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) + mode_offset + sigma_key)
    return torch.randn(shape, generator=gen, device=device, dtype=dtype)


def _gap_closed(metric: str, model: float | None, x_interp: float | None, truth: float | None) -> float | None:
    if model is None or x_interp is None or truth is None:
        return None
    if metric == 'mslp_min_hpa':
        denom = x_interp - truth
        return None if abs(denom) < 1.0e-12 else (x_interp - model) / denom
    if metric == 'wind10m_speed_max_ms':
        denom = truth - x_interp
        return None if abs(denom) < 1.0e-12 else (model - x_interp) / denom
    raise KeyError(metric)


def _reduce_minmax(field5d, indices: dict[str, int], box_t) -> dict[str, float]:
    import torch

    fb = field5d[:, :, :, box_t, :][0, 0, 0]
    out: dict[str, float] = {}
    if 'msl' in indices:
        mslp = fb[:, indices['msl']].float() * PA_TO_HPA
        physically_valid = mslp[mslp >= MSLP_PHYS_FLOOR_HPA]
        out['mslp_min_hpa'] = float(physically_valid.min() if physically_valid.numel() else mslp.min())
    if '10u' in indices and '10v' in indices:
        u = fb[:, indices['10u']]
        v = fb[:, indices['10v']]
        out['wind10m_speed_max_ms'] = float(torch.sqrt(u * u + v * v).max())
    return out


def _x_interp_minmax(x_interp_raw, box_t, target_indices: dict[str, int], name2in: dict[str, int]) -> dict[str, float]:
    import torch

    fb = x_interp_raw[:, :, :, box_t, :][0, 0, 0]
    out: dict[str, float] = {}
    if 'msl' in target_indices and 'msl' in name2in:
        mslp = fb[:, name2in['msl']].float() * PA_TO_HPA
        physically_valid = mslp[mslp >= MSLP_PHYS_FLOOR_HPA]
        out['mslp_min_hpa'] = float(physically_valid.min() if physically_valid.numel() else mslp.min())
    if '10u' in target_indices and '10v' in target_indices and '10u' in name2in and '10v' in name2in:
        u = fb[:, name2in['10u']]
        v = fb[:, name2in['10v']]
        out['wind10m_speed_max_ms'] = float(torch.sqrt(u * u + v * v).max())
    return out


def _mode_residual(mode: str, y_residual):
    import torch

    if mode == 'teacher':
        return y_residual
    if mode == 'free':
        return torch.zeros_like(y_residual)
    raise ValueError(f'unknown probe mode {mode!r}')


def _reconstruct_phys_box(bundle, x_interp_box, residual, box_t):
    from interp.core.model import is_dict_api

    inner = bundle.inner_model
    ppt = getattr(bundle.model, 'post_processors_tendencies', None)
    res_box = residual[:, :, :, box_t, :].to(x_interp_box.dtype)
    if is_dict_api(inner):
        return inner.add_interp_to_state(
            x_interp_box,
            res_box,
            bundle.post_processors,
            ppt,
            target_dataset='out_hres',
            source_dataset='in_lres',
        )
    dp = getattr(inner, 'direct_prediction_indices', None)
    return inner.add_interp_to_state(
        x_interp_box,
        res_box,
        bundle.post_processors,
        ppt,
        direct_prediction_indices=dp,
    )


def _prepare_one(bundle, eb):
    from interp.core.model import is_dict_api, prepare_batch

    device = bundle.device
    x_lres = eb.x_lres.to(device)
    x_hres = eb.x_hres.to(device)
    y = eb.y.to(device)
    inner = bundle.inner_model

    if is_dict_api(inner):
        batch = {'in_lres': x_lres, 'in_hres': x_hres}
        (x_interp_cond, x_hres_cond), _ = inner._before_sampling(batch, bundle.pre_processors, 1)
        x_interp_raw = inner.apply_interpolate_to_high_res(x_lres[:, 0, ...])[:, None, ...]
        prt = getattr(bundle.model, 'pre_processors_tendencies', None)
        y_residual = inner.compute_residuals(
            y,
            x_interp_raw,
            bundle.pre_processors['out_hres'],
            prt['out_hres'],
            target_dataset='out_hres',
        )
        recon_state = x_interp_cond
    else:
        prepared = prepare_batch(bundle, x_lres, x_hres, y)
        x_interp_cond = prepared['x_interp']
        x_hres_cond = prepared['x_hres']
        x_interp_raw = prepared['x_interp_raw']
        y_residual = prepared['y_residual']
        recon_state = x_interp_raw

    return {
        'x_interp_cond': x_interp_cond,
        'x_hres_cond': x_hres_cond,
        'x_interp_raw': x_interp_raw,
        'y_residual': y_residual,
        'recon_state': recon_state,
        'target': y,
    }


def _case_bundle_path(bundle_dir: str, case: TCCase) -> str:
    pattern = f'*date{case.date}*mem{case.member}*step{case.step}h*input_bundle.nc'
    hits = sorted(glob.glob(str(Path(bundle_dir) / pattern)))
    if not hits:
        raise FileNotFoundError(f'No bundle matching {pattern} in {bundle_dir}')
    return hits[0]


def _summarize_case(case_result: dict) -> dict:
    summary = {'metrics': {}}
    for metric in ('mslp_min_hpa', 'wind10m_speed_max_ms'):
        truth = case_result['truth'].get(metric)
        x_interp = case_result['x_interp'].get(metric)
        metric_summary = {'truth': truth, 'x_interp': x_interp, 'modes': {}}
        for mode in case_result['modes']:
            rows = [row for row in case_result['probes'] if row['mode'] == mode and metric in row['metrics']]
            if not rows:
                continue
            if metric == 'mslp_min_hpa':
                best = min(rows, key=lambda row: row['metrics'][metric])
            else:
                best = max(rows, key=lambda row: row['metrics'][metric])
            metric_summary['modes'][mode] = {
                'best_value': best['metrics'][metric],
                'best_sigma': best['sigma'],
                'best_gap_closed': best['gap_closed'].get(metric),
            }
        if 'free' in metric_summary['modes'] and 'teacher' in metric_summary['modes']:
            free_gap = metric_summary['modes']['free'].get('best_gap_closed')
            teacher_gap = metric_summary['modes']['teacher'].get('best_gap_closed')
            if free_gap is not None and teacher_gap is not None:
                metric_summary['commitment_gap'] = teacher_gap - free_gap
        summary['metrics'][metric] = metric_summary
    return summary


def _summarize_run(cases: list[dict]) -> dict:
    out = {'mean_free_gap_closed': {}, 'mean_teacher_gap_closed': {}, 'mean_commitment_gap': {}}
    for metric in ('mslp_min_hpa', 'wind10m_speed_max_ms'):
        for mode, key in (('free', 'mean_free_gap_closed'), ('teacher', 'mean_teacher_gap_closed')):
            vals = []
            for case in cases:
                mode_summary = case['summary']['metrics'].get(metric, {}).get('modes', {}).get(mode, {})
                if mode_summary.get('best_gap_closed') is not None:
                    vals.append(float(mode_summary['best_gap_closed']))
            if vals:
                out[key][metric] = sum(vals) / len(vals)
        gaps = []
        for case in cases:
            case_gap = case['summary']['metrics'].get(metric, {}).get('commitment_gap')
            if case_gap is not None:
                gaps.append(float(case_gap))
        if gaps:
            out['mean_commitment_gap'][metric] = sum(gaps) / len(gaps)
    mslp = out['mean_free_gap_closed'].get('mslp_min_hpa')
    wind = out['mean_free_gap_closed'].get('wind10m_speed_max_ms')
    if mslp is not None and wind is not None:
        out['tc_minmax_score'] = 0.5 * mslp + 0.5 * wind
    elif mslp is not None:
        out['tc_minmax_score'] = mslp
    elif wind is not None:
        out['tc_minmax_score'] = wind
    return out


def run_tc_emergence(args):
    import torch
    from interp.core.data import load_single_bundle
    from interp.core.geometry import box_mask_km, detect_min_center
    from interp.core.model import denoise_at_sigma, get_surface_target_indices, get_variable_names, load_model
    from interp.core.runmeta import ckpt_id_from_path, write_run_meta

    out_path = Path(args.output_dir)
    if (getattr(args, 'case_cache', None) is None and getattr(args, 'case_preset', None)
            and int(getattr(args, 'strong_input_top_k', 0) or 0) > 0):
        args.case_cache = str(out_path / 'tc_emergence_selected_cases.json')
    bundle_dir, cases = _resolve_cases(args)
    LOGGER.info('tc_emergence: loading checkpoint %s on %s precision=%s', args.checkpoint, args.device, args.precision)
    bundle = load_model(args.checkpoint, args.device, args.precision)
    target_indices = get_surface_target_indices(bundle)
    LOGGER.info('tc_emergence: loaded checkpoint %s with target indices %s', args.checkpoint, target_indices)
    if 'msl' not in target_indices:
        raise SystemExit('tc_emergence needs an msl output channel')
    if '10u' not in target_indices or '10v' not in target_indices:
        raise SystemExit('tc_emergence needs 10u and 10v output channels')
    vn = get_variable_names(bundle)
    name2in = {name: idx for idx, name in vn.get('input_lres', {}).items()}

    case_results = []
    for case in cases:
        path = _case_bundle_path(bundle_dir, case)
        LOGGER.info('tc_emergence: loading case %s from %s', case_to_spec(case), path)
        eb = load_single_bundle(bundle, path)
        prep = _prepare_one(bundle, eb)
        lat_h, lon_h = eb.coords[2], eb.coords[3]
        y0 = prep['target']
        mslp_field = y0[0, 0, 0, :, target_indices['msl']].detach().cpu().numpy() * PA_TO_HPA
        clat, clon = detect_min_center(mslp_field, lat_h, lon_h, window=case.window)
        box_np = box_mask_km(lat_h, lon_h, clat, clon, args.eye_radius_km)
        box_t = torch.as_tensor(box_np, device=bundle.device, dtype=torch.bool)
        truth = _reduce_minmax(y0, target_indices, box_t)
        x_interp = _x_interp_minmax(prep['x_interp_raw'], box_t, target_indices, name2in)
        case_result = {
            'case': case.__dict__,
            'bundle_path': path,
            'storm_box': {
                'lat': float(clat),
                'lon': float(clon % 360.0),
                'radius_km': float(args.eye_radius_km),
                'n_cells': int(box_np.sum()),
            },
            'truth': truth,
            'x_interp': x_interp,
            'modes': list(args.modes),
            'probes': [],
        }
        for mode in args.modes:
            for sigma in args.sigmas:
                noise = _fixed_noise(
                    prep['y_residual'].shape,
                    seed=args.noise_seed + len(case_results) * 10_000,
                    sigma=float(sigma),
                    mode=mode,
                    device=bundle.device,
                    dtype=prep['y_residual'].dtype,
                )
                D = denoise_at_sigma(
                    bundle,
                    prep['x_interp_cond'],
                    prep['x_hres_cond'],
                    _mode_residual(mode, prep['y_residual']),
                    float(sigma),
                    noise=noise,
                )
                phys_box = _reconstruct_phys_box(bundle, prep['recon_state'][:, :, :, box_t, :], D, box_t)
                metrics = _reduce_minmax(
                    phys_box,
                    target_indices,
                    torch.ones(phys_box.shape[-2], device=bundle.device, dtype=torch.bool),
                )
                gaps = {
                    metric: _gap_closed(metric, metrics.get(metric), x_interp.get(metric), truth.get(metric))
                    for metric in ('mslp_min_hpa', 'wind10m_speed_max_ms')
                }
                case_result['probes'].append({
                    'mode': mode,
                    'sigma': float(sigma),
                    'metrics': metrics,
                    'gap_closed': gaps,
                })
                LOGGER.info(
                    '%s mode=%s sigma=%.3g mslp_min=%.2f wind10m_max=%.2f',
                    case.label,
                    mode,
                    sigma,
                    metrics.get('mslp_min_hpa', float('nan')),
                    metrics.get('wind10m_speed_max_ms', float('nan')),
                )
        case_result['summary'] = _summarize_case(case_result)
        case_results.append(case_result)

    result = {
        'tool': 'tc_emergence',
        'checkpoint': args.checkpoint,
        'checkpoint_step': _checkpoint_step(args.checkpoint),
        'ckpt_id': ckpt_id_from_path(args.checkpoint),
        'bundle_dir': bundle_dir,
        'case_preset': getattr(args, 'case_preset', None),
        'case_cache': getattr(args, 'case_cache', None),
        'resolved_cases': [_case_to_dict(case) for case in cases],
        'sigmas': [float(s) for s in args.sigmas],
        'modes': list(args.modes),
        'noise_seed': int(args.noise_seed),
        'metric_rule': {
            'mslp_min_hpa': 'minimum MSLP in storm-centered box, hPa, with sub-870 hPa artifacts ignored',
            'wind10m_speed_max_ms': 'maximum sqrt(10u^2 + 10v^2) in storm-centered box, m/s',
            'gap_closed': 'input->model->truth, sign-adjusted; 1 means truth reached, 0 means stuck at input',
        },
        'cases': case_results,
        'summary': _summarize_run(case_results),
    }
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / 'tc_emergence.json', 'w') as fh:
        json.dump(result, fh, indent=2)
    write_run_meta(out_path, 'tc_emergence', args)
    _print_summary(result)
    LOGGER.info('tc_emergence saved to %s', out_path / 'tc_emergence.json')
    return result


def _print_summary(result: dict) -> None:
    print('\n' + '=' * 78)
    print('TC EMERGENCE - fixed-sigma MSLP-min / 10m-wind-max probe')
    print('=' * 78)
    print(json.dumps(result['summary'], indent=2, sort_keys=True))


def add_case_args(p: argparse.ArgumentParser):
    g = p.add_argument_group('TC cases')
    g.add_argument('--case-preset', default='o96_o320_idalia_franklin', choices=sorted(CASE_PRESETS),
                   help='Preset TC case bundle. Default: two o96->o320 validation TCs: Franklin + Idalia.')
    g.add_argument('--tc-case', action='append', default=None,
                   help='Explicit case label:date:member:step:lat0,lat1,lon0,lon1. Repeatable. Overrides preset cases; uses --bundle-dir unless preset supplies one.')
    g.add_argument('--strong-input-top-k', type=int, default=20,
                   help='For presets with TC windows, select top K cases per TC by input MSLP-min and top K by input wind-speed-max. Use 0 for fallback hand-picked cases.')
    g.add_argument('--auto-window', default=None,
                   help='Fallback window for --event mode: lat0,lat1,lon0,lon1')
    g.add_argument('--case-cache', default=None,
                   help='JSON cache for resolved preset TC cases. Defaults to output-dir/tc_emergence_selected_cases.json when strong preset selection is active.')
    return p


def add_probe_args(p: argparse.ArgumentParser):
    p.add_argument('--sigmas', nargs='+', type=float, default=list(DEFAULT_TC_SIGMAS),
                   help='Fixed sigma values to probe')
    p.add_argument('--modes', nargs='+', default=['free', 'teacher'], choices=['free', 'teacher'],
                   help='free = no target residual signal; teacher = true residual + noise ceiling')
    p.add_argument('--noise-seed', type=int, default=12345,
                   help='Base seed for deterministic probe noise')
    p.add_argument('--eye-radius-km', type=float, default=500.0,
                   help='Storm-core box radius around target MSLP minimum')
    return p


def main(argv=None):
    p = argparse.ArgumentParser(description='Fast fixed-sigma TC min-MSLP / max-10m-wind probe')
    add_model_args(p)
    add_event_args(p)
    add_case_args(p)
    add_probe_args(p)
    args = p.parse_args(argv)
    setup_logging()
    return run_tc_emergence(args)


if __name__ == '__main__':
    main()
