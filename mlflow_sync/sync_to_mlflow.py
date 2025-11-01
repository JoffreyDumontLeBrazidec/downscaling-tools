from pathlib import Path
import logging
import re
import os
from icecream import ic
import subprocess
import mlflow

from mlflow.tracking import MlflowClient


class MLFlowSyncer:
    """Syncs MLFlow logs from HPC to MLFlow server."""

    def __init__(self, HPC, runs_id_to_sync):
        self.HPC = HPC
        self.runs_id_to_sync = runs_id_to_sync

    def sync_to_mlflow(self):

        if self.HPC == "atos":
            mlflow_log_dir = "/home/ecm5702/scratch/aifs/logs/mlflow"
        elif self.HPC == "leo":
            mlflow_log_dir = (
                "/leonardo_work/DestE_340_25/output/jdumontl/downscaling/logs/mlflow"
            )
        else:
            raise ValueError(f"Unknown HPC: {self.HPC}")

        # Sync each run in runs_id_to_sync
        for run_id in self.runs_id_to_sync:
            sync_cmd = (
                f"anemoi-training mlflow sync -r {run_id} "
                f"-s {mlflow_log_dir} -d https://mlflow.ecmwf.int -e ds-diffusion -a"
            )
            logging.info(f"Executing: {sync_cmd}")
            os.system(sync_cmd)


if __name__ == "__main__":
    HPC = os.environ["HPC"]

    runs_id_to_sync = [
        "18b751f42d7d4d75be787d090fc0c9dd",
        "d52a3a102e5f487abc61f65651eb2ccb",
    ]

    syncer = MLFlowSyncer(HPC, runs_id_to_sync)
    syncer.sync_to_mlflow()
