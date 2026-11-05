"""o2560 box read kit tests (2 Nov 2026). Run on the Jupiter login node from the kit folder:
    source kit/rows.sh B24 && source kit/env.sh && PYTHONPATH=$PYTHONPATH:dstools/downscaling-tools python -m pytest -q tests
Tests that need a Jupiter path skip with the reason when the path is absent (for example on a laptop)."""
import os
import sys
from pathlib import Path

import pytest

KIT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KIT_DIR / "kit"))
J = "/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261102-o2560-box-read"
BR = "/e/data1/jureap-data/ecmwf/users/jdumont/db_o2560_e2_50k_v2_20261030/ckpt_root/168c43551e3a414ead0f39a25db7cf22"
CKPTS = {
    "A100": f"{J}/ckpt/A100k/inference-ema10000-anemoi-by_step-epoch_198-step_100000.ckpt",
    "B24": f"{BR}/inference-ema10000-anemoi-by_step-epoch_036-step_024000.ckpt",
    "B50": f"{BR}/inference-ema10000-anemoi-by_step-epoch_076-step_050000.ckpt",
}
STORE = "/e/data1/jureap-data/ecmwf/users/jdumont/datasets/o1280_o2560_europe_e2"
GRAPH_OUT = "/e/data1/jureap-data/ecmwf/users/jdumont/db_o2560_e2_50k_v2_20261030/assets/graph/europe_e2_o2560_m600r9rm_graph_out_hres.pt"
MAT = ("/e/data1/jureap-data/ecmwf/users/jdumont/db_o2560_e2_50k_v2_20261030/assets/inter_mat/"
       "interpol_O1280_to_O2560_linear-europe_e2_10_70N_80W_40E_halo1p5.mat.npz")
CFG_A = f"{J}/configs/db_o2560_crop_100k_AG.yaml"
CFG_READ = f"{J}/configs/db_o2560_read.yaml"


def need(path):
    if not os.path.exists(path):
        pytest.skip(f"not on Jupiter: {path} absent")
    return path
