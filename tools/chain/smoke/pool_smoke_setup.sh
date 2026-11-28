#!/bin/bash
# Pool smoke (CPU, real Slurm): 3 chains share pilots. Run once in the smoke folder on the login node:
#   bash pool_smoke_setup.sh <mn5|leo>     then: sbatch pool_pilot_<cluster>.sbatch
# Audit after: for c in pa pb pc; do python3 overlap_smoke_audit.py pool/$c 800; done; cat pool/POOL.log
set -eo pipefail
cl="$1"; [[ "$cl" == mn5 || "$cl" == leo ]] || { echo "usage: pool_smoke_setup.sh mn5|leo" >&2; exit 2; }
here="$(cd "$(dirname "$0")" && pwd)"; mkdir -p "$here/pool" "$here/logs"
: > "$here/pool/POOL.tsv"
for c in pa pb pc; do
  python3 "$here/chain/chain_state.py" init --root "$here/pool/$c" --note "pool smoke 28 Nov"
  echo pool > "$here/pool/$c/CHAIN.pool"
  printf '%s\t800\t%s %s 800\n' "$here/pool/$c" "$here/overlap_smoke_body.sh" "$here/pool/$c" >> "$here/pool/POOL.tsv"
done
src="$here/overlap_smoke_${cl}.sbatch"
sed -e 's/--job-name=ovl_smoke/--job-name=pool_smoke/' \
    -e '$d' "$src" > "$here/pool_pilot_${cl}.sbatch"
echo 'CHAIN_PY=python3 CHAIN_TOOLS="$SLURM_SUBMIT_DIR/chain" bash "$SLURM_SUBMIT_DIR/chain/pilot_job.sh" "$SLURM_SUBMIT_DIR/pool/POOL.tsv" 0' >> "$here/pool_pilot_${cl}.sbatch"
cat "$here/pool/POOL.tsv"; echo "wrote $here/pool_pilot_${cl}.sbatch"
