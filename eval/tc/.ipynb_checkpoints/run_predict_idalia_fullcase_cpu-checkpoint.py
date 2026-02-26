import os
import sys
import numpy as np
import torch
import earthkit.data as ekd

sys.path.append('/home/ecm5702/dev/downscaling-tools')
from get_objects_from_ckpt import (
    ObjectFromCheckpointLoader,
    get_checkpoint,
    instantiate_config,
    adapt_config_hpc,
)

DIR_EXP = '/home/ecm5702/scratch/aifs/checkpoint'
NAME_EXP = '6c984778157e434ab6ad75f2d8d2d8db'
NAME_CKPT = 'anemoi-by_time-epoch_021-step_100000.ckpt'

SFC_GRIB = '/home/ecm5702/hpcperm/data/tc/idalia/full_fields_eefo_o96_global/eefo_o96_full_EEFO_O96_0001_date20230829_time0000_mem26_step024h_sfc.grib'
PL_GRIB = '/home/ecm5702/hpcperm/data/tc/idalia/full_fields_eefo_o96_global/eefo_o96_full_EEFO_O96_0001_date20230829_time0000_mem26_step024h_pl.grib'

# Idalia region
LAT_MIN, LAT_MAX = 10.0, 40.0
LON_MIN, LON_MAX = 260.0, 290.0

def load_case_fields():
    ds_sfc = ekd.from_source('file', SFC_GRIB).to_xarray(engine='cfgrib')
    ds_pl = ekd.from_source('file', PL_GRIB).to_xarray(engine='cfgrib')

    fields = {
        '10u': np.asarray(ds_sfc['u10'].values, dtype=np.float32).squeeze(),
        '10v': np.asarray(ds_sfc['v10'].values, dtype=np.float32).squeeze(),
        '2d': np.asarray(ds_sfc['d2m'].values, dtype=np.float32).squeeze(),
        '2t': np.asarray(ds_sfc['t2m'].values, dtype=np.float32).squeeze(),
        'msl': np.asarray(ds_sfc['msl'].values, dtype=np.float32).squeeze(),
        'skt': np.asarray(ds_sfc['skt'].values, dtype=np.float32).squeeze(),
        'sp': np.asarray(ds_sfc['sp'].values, dtype=np.float32).squeeze(),
        'tcw': np.asarray(ds_sfc['tcw'].values, dtype=np.float32).squeeze(),
    }

    levels = [50, 100, 200, 300, 400, 500, 700, 850, 925, 1000]
    for base in ['q', 't', 'u', 'v', 'w', 'z']:
        for lev in levels:
            key = f'{base}_{lev}'
            arr = np.asarray(ds_pl[base].sel(isobaricInhPa=lev).values, dtype=np.float32).squeeze()
            fields[key] = arr

    lat = np.asarray(ds_sfc['latitude'].values)
    lon = np.asarray(ds_sfc['longitude'].values)
    return fields, lat, lon

def idalia_mask(lat, lon):
    lon_360 = np.mod(lon, 360.0)
    return (lat >= LAT_MIN) & (lat <= LAT_MAX) & (lon_360 >= LON_MIN) & (lon_360 <= LON_MAX)

def metrics(arr_msl, arr_u10, arr_v10, mask):
    wind = np.sqrt(arr_u10**2 + arr_v10**2)
    return {
        'min_msl_global': float(np.nanmin(arr_msl)),
        'max_wind_global': float(np.nanmax(wind)),
        'min_msl_idalia': float(np.nanmin(arr_msl[mask])),
        'max_wind_idalia': float(np.nanmax(wind[mask])),
    }

def main():
    device = 'cpu'
    torch.set_grad_enabled(False)

    loader = ObjectFromCheckpointLoader(DIR_EXP, NAME_EXP, NAME_CKPT)
    checkpoint, cfg_ckpt = get_checkpoint(DIR_EXP, NAME_EXP, NAME_CKPT)
    cfg = instantiate_config()
    cfg_ckpt = adapt_config_hpc(cfg_ckpt, cfg)
    loader.config_checkpoint = cfg_ckpt
    loader.config_for_datamodule.dataloader.validation.frequency = '50h'
    loader.load()

    datamodule = loader.datamodule
    interface = loader.interface.to(device)

    batch = next(iter(datamodule.val_dataloader()))
    x_in, x_in_hres, y = [t.to(device) for t in batch]

    # x_in shape expected [1,1,1,40320,68]
    x_in_case = x_in.clone()

    fields, lat_lres_case, lon_lres_case = load_case_fields()
    mask_in = idalia_mask(lat_lres_case, lon_lres_case)

    name_to_idx = datamodule.data_indices.data.input[0].name_to_index
    missing = [k for k in name_to_idx.keys() if k not in fields]
    if missing:
        raise KeyError(f'Missing fields for input channels: {missing}')

    for name, idx in name_to_idx.items():
        x_in_case[0, 0, 0, :, idx] = torch.from_numpy(fields[name]).to(device)

    in_stats = metrics(fields['msl'], fields['10u'], fields['10v'], mask_in)

    noise_scheduler_params = {
        'schedule_type': 'karras',
        'sigma_max': 80.0,
        'sigma_min': 0.03,
        'rho': 7.0,
        'num_steps': 3,
    }
    sampler_params = {
        'sampler': 'heun',
        'S_churn': 0.0,
        'S_min': 0.0,
        'S_max': 80.0,
        'S_noise': 1.0,
    }

    print('Running predict_step on CPU with num_steps=3 ...')
    pred = interface.predict_step(
        x_in_case[0:1],
        x_in_hres[0:1],
        noise_scheduler_params=noise_scheduler_params,
        sampler_params=sampler_params,
    )

    out = pred.detach().cpu().numpy()
    out = np.squeeze(out)
    if out.ndim != 2:
        out = out.reshape(-1, out.shape[-1])

    out_idx = datamodule.data_indices.model.output.name_to_index
    i_msl = out_idx['msl']
    i_u = out_idx['10u']
    i_v = out_idx['10v']

    lat_hres = np.asarray(datamodule.ds_valid.data.latitudes[2])
    lon_hres = np.asarray(datamodule.ds_valid.data.longitudes[2])
    mask_out = idalia_mask(lat_hres, lon_hres)

    out_stats = metrics(out[:, i_msl], out[:, i_u], out[:, i_v], mask_out)

    print('\nInput (injected EEFO case) extremes:')
    for k, v in in_stats.items():
        print(f'  {k}: {v:.3f}')

    print('\nOutput (model prediction) extremes:')
    for k, v in out_stats.items():
        print(f'  {k}: {v:.3f}')

if __name__ == '__main__':
    main()
