from pathlib import Path
import os, re, time, logging, subprocess
from datetime import datetime, timedelta


class MLFlowAutoSyncer:
    def __init__(self, HPC, days=3):
        logging.info(
            "Initializing MLFlowAutoSyncer with HPC: %s and days: %d", HPC, days
        )
        if HPC == "atos":
            self.root = Path("/home/ecm5702/scratch/aifs/logs/mlflow")
        elif HPC == "leo":
            self.root = Path(
                "/leonardo_work/DestE_340_25/output/jdumontl/downscaling/logs/mlflow"
            )
        else:
            raise ValueError(f"Unknown HPC: {HPC}")
        self.cutoff = time.time() - days * 86400
        self.rx = re.compile(r"^[0-9a-f]{32}$")
        logging.info("Root directory set to: %s", self.root)
        logging.info("Cutoff time set to: %s", datetime.fromtimestamp(self.cutoff))

    def _dir_latest_mtime(self, p):
        logging.debug("Calculating latest modification time for directory: %s", p)
        latest = p.stat().st_mtime
        for root, dirs, files in os.walk(p):
            for n in files:
                t = Path(root, n).stat().st_mtime
                if t > latest:
                    latest = t
        logging.debug("Latest modification time for directory %s: %s", p, latest)
        return latest

    def _dir_non_empty(self, p):
        logging.debug("Checking if directory is non-empty: %s", p)
        for _, _, files in os.walk(p):
            if files:
                logging.debug("Directory %s is non-empty", p)
                return True
        logging.debug("Directory %s is empty", p)
        return False

    def find_recent_runs(self):
        logging.info("Finding recent runs in root directory: %s", self.root)
        runs = set()
        for root, dirs, files in os.walk(self.root):
            for d in dirs:
                if self.rx.match(d):
                    p = Path(root, d)
                    if (
                        self._dir_non_empty(p)
                        and self._dir_latest_mtime(p) >= self.cutoff
                    ):
                        logging.info("Recent run found: %s", d)
                        runs.add(d)
        logging.info("Total recent runs found: %d", len(runs))
        return sorted(runs)

    def sync(self, run_ids):
        logging.info("Starting sync for %d runs", len(run_ids))
        for r in run_ids:
            cmd = [
                "anemoi-training",
                "mlflow",
                "sync",
                "-r",
                r,
                "-s",
                str(self.root),
                "-d",
                "https://mlflow.ecmwf.int",
                "-e",
                "ds-diffusion",
                "-a",
            ]
            logging.info("Executing: %s", " ".join(cmd))
            subprocess.run(cmd, check=False)
        logging.info("Sync completed for all runs")


if __name__ == "__main__":
    HPC = os.environ.get("HPC")
    days = int(os.environ.get("DAYS", "1"))
    s = MLFlowAutoSyncer(HPC, days)
    s.sync(s.find_recent_runs())
