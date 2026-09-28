# station_head — building code for the station head adapter (AIFS-CRPS epic)

Design and state: `epics/downscaling-aifs-crps/in-progress/20260909_station_head_adapter_design.md` in the docs repo.
`stage1/` holds the scripts that built the 2026 station tables, the pairing manifest and the static station table
(run under `module load python3 vtb ecmwf-toolbox`, as SLURM jobs on the nf queue), plus the coverage report.
Data products live on `/home/ecm5702/perm/station-head-adapter/` (durable) and working copies on scratch.
