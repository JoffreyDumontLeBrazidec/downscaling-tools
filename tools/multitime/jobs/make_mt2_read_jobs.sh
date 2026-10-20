#!/bin/bash
# mt2_e2_20k read kit (2026-10-20): write the job files of reads A and B into <work>/jobs/gen. Nothing is submitted.
#   mt2read_smoke.sbatch        1 GPU: EMA copy of the 20k inference file, then 20230226 stride 1 member 1, converter checks
#   mt2read_s<1|2>_<date>.sbatch 1 GPU each: one date, 10 members (seeds 1234-1243), stride 1 (heun_window averaged, the
#                                method) or 2 (isolated windows), then the eval.cli files (template: the TD-228 kit files)
#   do0r20_ind.sbatch           1 node x 4 GPU: reference do0_e2_50k_ag step 20,000 EMA-2000 with the E2 kit (predict_e2.py,
#                                profile temporal_sample, independent noise per lead, like mt2's one draw over all leads)
#   mt2score_a.sbatch           CPU: eval.cli judge at +24 h (mt2 stride 2 vs do0 20k; stride 1 vs stride 2) and +72 h
#   mt2score_b.sbatch           CPU: TD-227 temporal read per run and date, merged against each other and the TD-228 pair,
#                                and temporal_consistency_metrics.py on the npz trajectories (stride 1 vs stride 2)
# Usage: bash make_mt2_read_jobs.sh <work folder>   (the work folder holds tools/, predict/, jobs/env.sh, code ->,
#        downscaling-tools/ at 26c6f89; see its README.md)
set -euo pipefail
W=${1:?work folder}
S=/home/ecm5702/scratch/sandbox/20261016-multitime-e2
DTB=$S/code/downscaling-tools/tools/multitime/jobs
RUN=$S/checkpoints/mt2_e2_20k/bdedf02a66914fbb839c5ad049a523e9
TRAIN=$RUN/anemoi-by_step-epoch_019-step_020000.ckpt
EMA=$W/checkpoints/inference-ema2000-anemoi-by_step-epoch_019-step_020000.ckpt
CFG=$S/configs/mt2_e2_20k.rendered.yaml
TPL=/home/ecm5702/scratch/agent-work/20261016-temporal-td227/runs/temporal_td228_independent/predictions
DO0=/home/ecm5702/scratch/pl25_ac/checkpoints/do0_e2_50k_ag/77c7a1fafa7346b39b5b4c36712f94eb/anemoi-by_step-epoch_019-step_020000.ckpt
DATES="20230226 20230822 20230829 20230912 20230919"
G=$W/jobs/gen; mkdir -p "$G" "$W/logs"

gpu_header() {  # name hours gpus cpus mem
  cat <<EOF
#!/bin/bash
#SBATCH --job-name=$1
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --account=ecrdmomu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=$3
#SBATCH --cpus-per-task=$4
#SBATCH --hint=nomultithread
#SBATCH --mem=$5
#SBATCH --time=$2
#SBATCH --output=$W/logs/%x_%j.out
#SBATCH --error=$W/logs/%x_%j.err
EOF
}
cpu_header() {  # name hours cpus mem
  cat <<EOF
#!/bin/bash
#SBATCH --job-name=$1
#SBATCH --qos=nf
#SBATCH --account=ecrdmomu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$3
#SBATCH --mem=$4
#SBATCH --time=$2
#SBATCH --output=$W/logs/%x_%j.out
#SBATCH --error=$W/logs/%x_%j.err
EOF
}
common="MT_CKPT=$EMA MT_TRAIN_CKPT=$TRAIN MT_CONFIG=$CFG MT_TEMPLATE=$TPL MT_SEED=1234"

{ gpu_header mt2read_smoke 01:30:00 1 16 120G
  echo "# mt2_e2_20k read kit smoke (2026-10-20): EMA copy, then 20230226 stride 1 member 1; converter in check-only mode."
  echo "export $common MT_OUT=$W/smoke/mt2_s1 MT_HDATE=20230226 MT_STRIDE=1 MT_MEMBERS=1 MT_CONVERT=check MT_LABEL=smoke_mt2_s1"
  echo "exec bash $DTB/mt2_read_body.sh"; } > "$G/mt2read_smoke.sbatch"

for st in 1 2; do
  hours=06:00:00; [[ $st == 2 ]] && hours=04:00:00
  for d in $DATES; do
    { gpu_header "mt2read_s${st}_$d" $hours 1 16 120G
      echo "# mt2_e2_20k step 20,000 EMA-2000, $d leads 12-72 h, 10 members, stride $st (1 = averaged windows, 2 = isolated)."
      echo "export $common MT_OUT=$W/runs/mt2_s$st MT_HDATE=$d MT_STRIDE=$st MT_MEMBERS=10 MT_CONVERT=yes MT_LABEL=mt2_e2_20k_s$st MT_GATE=$W/smoke/SMOKE_OK"
      echo "exec bash $DTB/mt2_read_body.sh"; } > "$G/mt2read_s${st}_$d.sbatch"
  done
done

{ gpu_header do0r20_ind 04:00:00 4 32 240G
  echo "# Reference do0_e2_50k_ag step 20,000 EMA-2000 on the E2 kit (copy of the TD-227 kit in this folder), profile"
  echo "# temporal_sample (5 dates x 12-72 h x seeds 1234-1243), fp32, 30-step churn-off sampler, independent noise per lead."
  echo "export WK_LABEL=do0r20_ind WK_CKPT=$DO0 WK_WEIGHTS=ema2000 WK_NOISE=independent"
  echo "exec bash $W/jobs/wide4_body.sh"; } > "$G/do0r20_ind.sbatch"

{ cpu_header mt2score_a 08:00:00 32 120G
  echo "export MT_W=$W"
  echo "exec bash $DTB/mt2_score_a_body.sh"; } > "$G/mt2score_a.sbatch"
{ cpu_header mt2score_b 08:00:00 32 120G
  echo "export MT_W=$W"
  echo "exec bash $DTB/mt2_score_b_body.sh"; } > "$G/mt2score_b.sbatch"
ls -1 "$G"
