from pathlib import Path
import os, re, time, logging, subprocess
from datetime import datetime, timedelta


class MLFlowAutoSyncer:
    def __init__(self, HPC, days=3):
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

    def _dir_latest_mtime(self, p):
        latest = p.stat().st_mtime
        for root, dirs, files in os.walk(p):
            for n in files:
                t = Path(root, n).stat().st_mtime
                if t > latest:
                    latest = t
        return latest

    def _dir_non_empty(self, p):
        for _, _, files in os.walk(p):
            if files:
                return True
        return False

    def find_recent_runs(self):
        runs = set()
        for root, dirs, files in os.walk(self.root):
            for d in dirs:
                if self.rx.match(d):
                    p = Path(root, d)
                    if (
                        self._dir_non_empty(p)
                        and self._dir_latest_mtime(p) >= self.cutoff
                    ):
                        runs.add(d)
        return sorted(runs)

    def sync(self, run_ids):
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


if __name__ == "__main__":
    HPC = os.environ.get("HPC")
    days = int(os.environ.get("DAYS", "2"))
    s = MLFlowAutoSyncer(HPC, days)
    s.sync(s.find_recent_runs())
