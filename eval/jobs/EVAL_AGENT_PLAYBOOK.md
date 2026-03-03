# Eval Agent Playbook (Parallel Bundle)

Use this as the default when a user asks to evaluate a run.

## Required Behavior

- Launch independent eval components in parallel.
- Keep local plotting to one representative date (5 lead times) unless the user asks for all dates.
- Use short quaver windows (5 days) by default.
- Every `tc_normed_pdfs*.pdf` generation now also writes a sibling `*.stats.json` file in the same output directory.
- For Idalia, `*.stats.json` now includes an `extreme_tail` section with default thresholds:
  - `MSLP in [980, 990] hPa`
  - `10m wind > 25 m/s`
  - plus an `extreme_score` ranking that combines both tails.
- For all TC requests, agents must explicitly report these extreme-tail values for the target run:
  - `extreme_score`, `mslp_980_990_fraction`, `wind_gt_25_fraction`
  - plus the thresholds used in that file.
- For all TC requests, agents must also compare the target run against all previously evaluated experiments currently available in `/home/ecm5702/perm/eval` by scanning:
  - `tc_extreme_tail_*.json`
  - `tc_normed_pdfs_*from_predictions.stats.json`
  - and return a ranked cross-experiment table (highest `extreme_score` first).
- Record all submitted jobs and monitor commands in `in_progress/tasks/<task>.md`.

## A) Predictions Run (`run-id`) - Eval Only

Assumes predictions already exist at:
`/home/ecm5702/perm/eval/<RUN_ID>/predictions/predictions_*.nc`

1) One-date local plots (5 files, example date `20230826`):

```bash
cat > /tmp/eval_one_date_<RUN_ID>.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=e5_<RUN_ID>_d20230826
#SBATCH --qos=nf
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/ecm5702/perm/eval/<RUN_ID>/logs/eval5_20230826_<RUN_ID>_%j.out
set -euo pipefail
source /home/ecm5702/dev/.ds-dyn/bin/activate
export PYTHONPATH="/etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools:${PYTHONPATH:-}"
cd /etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools
count=0
for f in /home/ecm5702/perm/eval/<RUN_ID>/predictions/predictions_20230826_step*.nc; do
  [ -f "$f" ] || continue
  base=$(basename "$f" .nc)
  python -m eval.run --eval-root /home/ecm5702/perm/eval/<RUN_ID>/eval_one_date predictions --predictions-nc "$f" --run-name "$base"
  count=$((count+1))
  echo "evaluated ${count} file(s): $f"
done
if [ "$count" -ne 5 ]; then
  echo "Expected 5 prediction files for 20230826 but evaluated $count" >&2
  exit 3
fi
EOF
sbatch /tmp/eval_one_date_<RUN_ID>.sbatch
```

2) TC plots in parallel:

```bash
sbatch --job-name=tcplot_<RUN_ID> --qos=nf --time=03:00:00 --cpus-per-task=4 --mem=32G \
  --output=/home/ecm5702/perm/eval/<RUN_ID>/logs/tc_plot_<RUN_ID>_%j.out \
  --wrap='set -euo pipefail; source /home/ecm5702/dev/.ds-dyn/bin/activate; module unload ifs || true; module load ecmwf-toolbox; export PYTHONPATH="/etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools:${PYTHONPATH:-}"; cd /etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools; python -m eval.tc.plot_members_tc --expver 0001 --outdir /home/ecm5702/perm/eval/<RUN_ID>; python -m eval.tc.plot_pdf_tc_from_predictions --predictions-dir /home/ecm5702/perm/eval/<RUN_ID>/predictions --outdir /home/ecm5702/perm/eval/<RUN_ID> --run-label <RUN_ID> --out-name tc_normed_pdfs_idalia_<RUN_ID>_from_predictions.pdf; cat /home/ecm5702/perm/eval/<RUN_ID>/tc_normed_pdfs_idalia_<RUN_ID>_from_predictions.pdf > /home/ecm5702/perm/eval/<RUN_ID>/tc_normed_pdfs_all_events_<RUN_ID>.pdf'
```
This plotting path always includes `ip6y` (`ENFO_O320_ip6y`) as an additional fixed reference.

3) Reference spectra generation in parallel:

```bash
sbatch --export=ALL,TARGETS=enfo_o1280,eefo_o320,DATE_RANGE=20230826/to/20230827/by/1,STEPS=144,NUMBERS=1/2,DATE_START=2023-08-26,DATE_END=2023-08-27,STEP_LIST=144,NUMBERS_LIST=1,2 \
  /home/ecm5702/dev/post_prepml/spectra/generate_reference_spectra.sbatch
```

4) Write run manifest (required):

```bash
cat > /home/ecm5702/perm/eval/<RUN_ID>/EXPERIMENT_CONFIG.yaml <<'EOF'
run_id: <RUN_ID>
purpose: "Prediction-run eval bundle"
predictions:
  source_dir: /home/ecm5702/perm/eval/<RUN_ID>/predictions
  file_pattern: predictions_YYYYMMDD_stepXXX.nc
  sampling_parameters:
    source_attribute: sampling_config_json
    parsed: {}
tc_evaluation:
  method: prediction-driven
  script: /etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools/eval/tc/plot_pdf_tc_from_predictions.py
  canonical_pdf: /home/ecm5702/perm/eval/<RUN_ID>/tc_normed_pdfs_all_events_<RUN_ID>.pdf
  canonical_stats_json: /home/ecm5702/perm/eval/<RUN_ID>/tc_normed_pdfs_all_events_<RUN_ID>.stats.json
EOF
```

## B) Expver Run (Full Eval Family)

Use existing launcher and let it submit components in parallel:

```bash
/home/ecm5702/dev/downscaling-tools/eval/jobs/launch_full_eval_suite.sh --expver <EXPVER>
```

Recommended short-window flags:

```bash
/home/ecm5702/dev/downscaling-tools/eval/jobs/launch_full_eval_suite.sh \
  --expver <EXPVER> \
  --quaver-first-date 20230826 \
  --quaver-last-date 20230830 \
  --spectra-date 20230826/to/20230830/by/1
```

## Monitoring

```bash
squeue -j <JOB1>,<JOB2>,<JOB3>
sacct -j <JOB1>,<JOB2>,<JOB3> --format=JobID,JobName%30,QOS,State,ExitCode,Elapsed,Timelimit,NodeList,Reason -n -P
```

## TC Extreme Comparison (Required Reporting)

Use this when summarizing TC results so users always get cross-run extreme-tail context:

```bash
find /home/ecm5702/perm/eval -type f \( -name 'tc_extreme_tail_*.json' -o -name 'tc_normed_pdfs_*from_predictions.stats.json' \) | sort
```

Extract/merge rows and rank by `extreme_score` descending. Include per-row:
- `exp`
- `extreme_score`
- `mslp_980_990_fraction`
- `wind_gt_25_fraction`
- source file path

## Notes

- If a previous all-dates eval job is running and user requests one-date-only local plots, cancel the broader job and replace it with one-date eval-only submission.
- For checkpoint mode (`python -m eval.run checkpoint ...`), keep sigma evaluator enabled unless the user explicitly asks to skip.
- Do not use `python -m eval.tc.plot_pdf_tc --expver 0001` as the ML curve for run-id evaluations; this duplicates the ENFO reference and is not run-specific.
