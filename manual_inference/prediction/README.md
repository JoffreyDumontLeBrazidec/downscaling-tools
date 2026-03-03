**Prediction Workflow**
This folder produces `predictions.nc` from a checkpoint using either:
1. the Anemoi dataloader, or
2. a prebuilt **input bundle** (`.nc`) made from MARS/GRIB.

**Simple notebooks**
- `manual_inference/notebooks/01_prediction_from_dataloader.ipynb`
- `manual_inference/notebooks/02_prediction_from_bundle.ipynb`
- `manual_inference/notebooks/03_build_bundle.ipynb`

**Default Output Location**
If `--out` is not set, predictions are written to:
`/home/ecm5702/hpcperm/experiments/<name_exp>/predictions.nc`

**Commands**

1. From dataloader:
```bash
python -m manual_inference.prediction.predict from-dataloader \
  --name-ckpt <exp_or_ckpt> \
  --idx 0 --n-samples 1 --members 0 \
  --extra-args-json '{"num_steps":40,"sigma_max":1000.0,"sigma_min":0.03,"rho":7.0,"sampler":"heun"}'
```

2. From input bundle (preferred for MARS data):
```bash
python -m manual_inference.prediction.predict from-bundle \
  --name-ckpt <exp_or_ckpt> \
  --bundle-nc /home/ecm5702/hpcperm/data/input_data/o96/<bundle>.nc \
  --extra-args-json '{"num_steps":40,"sigma_max":1000.0,"sigma_min":0.03,"rho":7.0,"sampler":"heun"}'
```

3. Build bundle (GRIB → bundle), if you need it:
```bash
python -m manual_inference.prediction.predict build-bundle \
  --lres-sfc-grib /path/lres_sfc.grib \
  --lres-pl-grib  /path/lres_pl.grib \
  --hres-grib     /path/hres_static.grib \
  --target-sfc-grib /path/hres_target_sfc.grib \
  --target-pl-grib  /path/hres_target_pl.grib \
  --out /home/ecm5702/hpcperm/data/input_data/o96/<bundle>.nc
```

**Notes**
- This workflow is **xarray-first**. Use MARS → xarray and build bundles from GRIB only when needed.
- The `--name-ckpt` can be a checkpoint name under the default root, or a full path.
