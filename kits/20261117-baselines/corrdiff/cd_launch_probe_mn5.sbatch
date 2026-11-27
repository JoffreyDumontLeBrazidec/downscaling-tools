#!/bin/bash
# CPU probe (GPP, 2 nodes, minutes) of what the GPU jobs rely on from a compute node: the srun line of cd_srun (4 tasks
# per node, torchrun's variables per task, MASTER_ADDR = the first node) joining one torch.distributed group across both
# nodes (gloo; the GPU jobs use NCCL over the same rendezvous), squeue time limit, sacct accounting of this job through
# cd_gpuh.py, and sbatch --test-only. Prints LAUNCH_PROBE_OK when every part works. Submits nothing.
#   sbatch cd_launch_probe_mn5.sbatch      (or --qos=gp_debug)
#SBATCH --job-name=cd_m5_probe
#SBATCH --account=ehpc01
#SBATCH --qos=gp_ehpc
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --time=00:10:00
#SBATCH --output=/gpfs/projects/ehpc01/ecm800825/sandbox/20261127-corrdiff-mn5/logs/%x_%j.out
set -uo pipefail
source /gpfs/projects/ehpc01/ecm800825/sandbox/20261127-corrdiff-mn5/kit/cd_env_mn5.sh
ok=1
export MASTER_PORT=$(( 20000 + RANDOM % 20000 )) MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
echo "nodes $SLURM_JOB_NODELIST master $MASTER_ADDR:$MASTER_PORT"
srun --kill-on-bad-exit=1 --nodes=$SLURM_NNODES --ntasks-per-node=4 --cpus-per-task=20 --cpu-bind=none \
  bash -c 'export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID; exec python "$@"' _ -c '
import os, socket, datetime, torch, torch.distributed as d
d.init_process_group("gloo", timeout=datetime.timedelta(seconds=120))
t = torch.tensor([float(d.get_rank())]); d.all_reduce(t)
w = d.get_world_size(); lr = os.environ["LOCAL_RANK"]
print(f"rank {d.get_rank()} of {w} local {lr} on {socket.gethostname()}: sum of ranks {t.item():.0f} (want {w*(w-1)//2})", flush=True)
assert t.item() == w * (w - 1) // 2
' > $CD/logs/probe_ranks_$SLURM_JOB_ID.log 2>&1 || ok=0
sort -n -k2 $CD/logs/probe_ranks_$SLURM_JOB_ID.log | grep "^rank" ; grep -c "^rank" $CD/logs/probe_ranks_$SLURM_JOB_ID.log | grep -qx 8 || ok=0
echo "squeue time limit: $(squeue -h -j $SLURM_JOB_ID -o %l)" ; [[ -n "$(squeue -h -j $SLURM_JOB_ID -o %l)" ]] || ok=0
echo "$SLURM_JOB_ID seg 1 probe" > $TMPDIR/probe_jobs_$SLURM_JOB_ID.txt
sleep 5; echo "sacct: $(sacct -X -n -P -j $SLURM_JOB_ID -o JobID,ElapsedRaw,AllocTRES)"
G=$(python $CD_KIT/cd_gpuh.py used --jobs $TMPDIR/probe_jobs_$SLURM_JOB_ID.txt --verbose) || ok=0; echo "cd_gpuh.py used -> $G (a CPU job: node count x 4 GPUs by the fallback)"
sbatch --test-only -A ehpc01 -p acc -q acc_ehpc -N 1 --gres=gpu:4 -c 80 -t 00:10:00 --wrap hostname 2>&1 | sed 's/^/[sbatch test] /'
(( ok == 1 )) && echo LAUNCH_PROBE_OK || echo LAUNCH_PROBE_FAIL
