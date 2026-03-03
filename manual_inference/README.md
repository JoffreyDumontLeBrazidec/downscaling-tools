# Manual Inference

This package produces `predictions.nc` from a checkpoint using either:
- the Anemoi dataloader
- a prebuilt input bundle (`.nc`) from MARS/GRIB

## Notebooks (Super Simple)
- `manual_inference/notebooks/00_manual_inference_overview.ipynb`
- `manual_inference/notebooks/01_prediction_from_dataloader.ipynb`
- `manual_inference/notebooks/02_prediction_from_bundle.ipynb`
- `manual_inference/notebooks/03_build_bundle.ipynb`

## Entry Points
- `manual_inference/prediction/predict.py` (CLI module)
- `manual_inference/input_data_construction/bundle.py` (GRIB → bundle)

## Common Commands
From dataloader:
```bash
python -m manual_inference.prediction.predict from-dataloader \
  --name-ckpt <exp_or_ckpt> \
  --idx 0 --n-samples 1 --members 0 \
  --extra-args-json '{"num_steps":40,"sigma_max":1000.0,"sigma_min":0.03,"rho":7.0,"sampler":"heun"}'
```

From input bundle:
```bash
python -m manual_inference.prediction.predict from-bundle \
  --name-ckpt <exp_or_ckpt> \
  --bundle-nc /home/ecm5702/hpcperm/data/input_data/o96/<bundle>.nc \
  --extra-args-json '{"num_steps":40,"sigma_max":1000.0,"sigma_min":0.03,"rho":7.0,"sampler":"heun"}'
```

Build bundle:
```bash
python -m manual_inference.prediction.predict build-bundle \
  --lres-sfc-grib /path/lres_sfc.grib \
  --lres-pl-grib  /path/lres_pl.grib \
  --hres-grib     /path/hres_static.grib \
  --target-sfc-grib /path/hres_target_sfc.grib \
  --target-pl-grib  /path/hres_target_pl.grib \
  --out /home/ecm5702/hpcperm/data/input_data/o96/<bundle>.nc
```

## Output
If `--out` is not set, predictions are written to:
`/home/ecm5702/hpcperm/experiments/<name_exp>/predictions.nc`
