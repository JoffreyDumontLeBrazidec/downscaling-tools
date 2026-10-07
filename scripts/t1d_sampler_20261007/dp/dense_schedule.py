"""Write the dense log-uniform schedule as a `schedule_type: custom` noise-scheduler block.

  python -m scripts.t1d_sampler_20261007.dp.dense_schedule --n 240 --out dense240.json
Pass the file's content to `trajectory --noise-scheduler-json` together with --num-steps <n>.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import custom_scheduler_json, dense_levels  # type: ignore
else:
    from .common import custom_scheduler_json, dense_levels


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    blk = custom_scheduler_json(dense_levels(a.n))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(blk, separators=(",", ":")) + "\n")
    print(f"{a.out}: {a.n} levels {blk['sigmas'][0]} .. {blk['sigmas'][-1]}, {2 * a.n - 1} Heun calls")


if __name__ == "__main__":
    main()
