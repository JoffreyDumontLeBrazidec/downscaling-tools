#!/usr/bin/env python3
"""o2560 box read: write checks/MANIFEST.json with the sha256 of every checkpoint, config, graph file and matrix the read
uses, the code pins (git HEAD and clean state), the dates, leads, members, seed formula and sampler. Run on Jupiter."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kit"))
import o2560_common as c  # noqa: E402

J = c.KIT_ROOT
S = "/e/data1/jureap-data/ecmwf/users/jdumont/db_o2560_e2_50k_v2_20261030"
BR = f"{S}/ckpt_root/168c43551e3a414ead0f39a25db7cf22"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def git(root):
    head = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip()
    return {"path": root, "head": head, "clean": not dirty}


files = {
    "A100_ema10000": f"{J}/ckpt/A100k/inference-ema10000-anemoi-by_step-epoch_198-step_100000.ckpt",
    "A100_raw_inference": f"{J}/ckpt/A100k/inference-anemoi-by_step-epoch_198-step_100000.ckpt",
    "B24_ema10000": f"{BR}/inference-ema10000-anemoi-by_step-epoch_036-step_024000.ckpt",
    "B24_raw_inference": f"{BR}/inference-anemoi-by_step-epoch_036-step_024000.ckpt",
    "B50_ema10000": f"{BR}/inference-ema10000-anemoi-by_step-epoch_076-step_050000.ckpt",
    "B50_raw_inference": f"{BR}/inference-anemoi-by_step-epoch_076-step_050000.ckpt",
    "config_read": f"{J}/configs/db_o2560_read.yaml",
    "config_B_chain": f"{S}/configs/db_o2560_e2_50k_v2.yaml",
    "config_A_chain_AG": f"{J}/configs/db_o2560_crop_100k_AG.yaml",
}
for p in sorted(Path(f"{S}/assets/graph").glob("*")) + sorted(Path(f"{S}/assets/inter_mat").glob("*.npz")):
    files[f"asset_{p.name}"] = str(p)
man = {"built_utc": subprocess.run(["date", "-u", "+%FT%TZ"], capture_output=True, text=True).stdout.strip(),
       "files": {k: {"path": v, "sha256": sha(v), "bytes": Path(v).stat().st_size} for k, v in files.items()},
       "a100_atos_source_sha256_ema10000": "53ef84ac6966e1ba031acddee9b4284c66f1fb608804cf4fac1cdcbd8a693f2a",
       "code": {"A100": git(f"{J}/code/anemoi-core-4ad32838c"), "B24_B50": git(f"{S}/code/anemoi-core"),
                "anemoi_datasets": git("/e/home/jusers/dumontlebrazidec1/jupiter/dev/anemoi-datasets-hres-precip-ru"),
                "downscaling_tools_scoring": git(f"{J}/dstools/downscaling-tools")},
       "gh200_hours": {"A100": 234.1, "B24": 229.9, "B50": 477.0},
       "dates": c.DATES12, "leads_default": c.LEADS_DEFAULT, "leads_possible": c.LEADS_ALL, "members": c.MEMBERS,
       "seed_base": c.SEED_BASE, "seed_formula": c.SEED_FORMULA,
       "seed_check": {"(1234,20230101,12,0)": c.derive_draw_seed(1234, 20230101, 12, 0), "(1234,20230101,12,1)": c.derive_draw_seed(1234, 20230101, 12, 1)},
       "sampler": {**c.PW_SCHED, **c.PW_SAMPLER}, "outputs": list(c.OUTPUT_VARS), "inference_num_chunks": 8, "model_ranks_per_job": 4,
       "out_root": c.OUT_ROOT}
# provenance of each chain's training inputs (scripts/provenance_side.py run on Atos for A and on Jupiter for B)
man["training_inputs"] = {side: json.loads((Path(J) / "checks" / f"provenance_{side}.json").read_text())
                          for side in ("atos_A", "jupiter_B")}
out = Path(J) / "checks" / "MANIFEST.json"
out.write_text(json.dumps(man, indent=1) + "\n")
print("MANIFEST_DONE", out, len(files), "files")
