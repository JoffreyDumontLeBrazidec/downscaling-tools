#!/bin/bash
# Regression baseline sandbox: no automatic reads (the XSET-1 and S25 rows are submitted by hand from the XSET kit,
# rows REG_GL512_20k / REG_GL512_30k of jupiter/xset_runs.sh). Called by reg_job.sh; does nothing.
echo "[read_hook] $(date -u +%FT%TZ) ${1:-} no reads configured in this sandbox"
