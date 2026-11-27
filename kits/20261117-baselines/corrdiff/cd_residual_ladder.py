#!/usr/bin/env python3
"""CorrDiff baseline: residual ladder of regression checkpoints. For each checkpoint (eval mode, fp32, TF32 off, the
model-loading and regression_step pattern of cd_predict.py) and the same N training rows and N validation rows (default
48 each, N/6 per lead 12..72 h, spread evenly in valid time over the split; deterministic), the root mean square of the residual
target - regression output in normalised units (the field the diffusion learns; its sigma_data is 0.5), per output
variable: training rows, validation rows, and validation rows per lead time.

Checkpoints with or without the time inputs: the input channel count of the checkpoint's saved arguments
(img_in_channels - N_grid_channels = 46 or 51) chooses the dataset variant (cd_dataset.py time_inputs).
Restartable and shardable: one part file per checkpoint (OUT/parts/<name>.json, skipped when it exists or when OUT/
ladder.json already holds the checkpoint); --shard i/n takes every n-th checkpoint (one process per GPU); `report`
merges the parts into OUT/ladder.json and OUT/ladder.md.

    python3 cd_residual_ladder.py run --cfg CONF.yaml --out OUT --ckpts "A.mdlus B.mdlus ..." [--n 48] [--shard 0/4]
    python3 cd_residual_ladder.py report --out OUT
Run from the PhysicsNeMo example folder (examples/weather/corrdiff), like cd_predict.py.
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
LEADS = [12, 24, 36, 48, 60, 72]


def ckpt_list(spec):
    out = []
    for s in re.split(r"[,\s]+", spec.strip()):
        if s:
            hits = sorted(glob.glob(s)) if any(c in s for c in "*?[") else [s]
            out += hits
    return out


def samples_of(p):
    m = re.search(r"\.(\d+)(\.w\d+)?\.mdlus$", Path(p).name)
    return int(m.group(1)) if m else -1


def part_name(p):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", f"{Path(p).parent.parent.name}__{Path(p).parent.name}__{Path(p).name}") + ".json"


def pick(ds, n):
    """n rows of ds: n / 6 per lead, evenly spread in valid time over the split's rows of that lead, each lead offset by
    a sixth of the spacing (so the leads do not share their start dates)."""
    per = n // len(LEADS)
    idx = []
    for j, L in enumerate(LEADS):
        rows = sorted((i for i, s in enumerate(ds.steps) if s == L), key=lambda i: ds.valid[i])
        step = len(rows) / per
        idx += [rows[min(int((k + j / len(LEADS)) * step), len(rows) - 1)] for k in range(per)]
    return sorted(set(idx))


def cmd_run(a):
    import torch
    import yaml
    from physicsnemo import Module
    from physicsnemo.diffusion.generate import regression_step
    from cd_dataset import E2v4Box
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(a.out)
    (out / "parts").mkdir(parents=True, exist_ok=True)
    done = set(json.load(open(out / "ladder.json"))["checkpoints"]) if (out / "ladder.json").exists() else set()
    si, sn = (int(x) for x in a.shard.split("/"))
    todo = [p for k, p in enumerate(ckpt_list(a.ckpts)) if k % sn == si]
    todo = [p for p in todo if str(Path(p).resolve()) not in done and not (out / "parts" / part_name(p)).exists()]
    print(f"[ladder {si}/{sn}] {len(todo)} checkpoints to do: {[Path(p).name for p in todo]}", flush=True)
    if not todo:
        return
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dcfg = dict(yaml.safe_load(open(a.cfg))["dataset"])
    dcfg.pop("type")
    dcfg.pop("time_inputs", None)
    dcfg.pop("crop", None)   # full box (a regression config of the crop runs has dataset.crop)
    dsets = {(sp, ti): E2v4Box(**dict(dcfg, split=sp, max_rows=0, time_inputs=ti))
             for sp in ("train", "valid") for ti in (False, True)}
    rows = {}
    t0 = time.time()
    for sp in ("train", "valid"):
        ds = dsets[(sp, False)]
        sel = pick(ds, a.n)
        items = []
        for i in sel:
            r = int(ds.rows[i])
            xc = np.asarray(ds.root["input"][r], np.float32)
            y = ds.normalize_output(np.asarray(ds.root["output"][r], np.float32)[ds.oi])
            items.append(dict(i=i, row=r, step=ds.steps[i], valid=ds.valid[i], xc=xc, y=y))
        rows[sp] = items
        hours = sorted({it["valid"].hour for it in items})
        print(f"[ladder] {sp}: {len(items)} rows, leads {np.bincount([it['step'] // 12 for it in items])[1:].tolist()} "
              f"(12..72 h), valid hours {hours}, valid {min(it['valid'] for it in items):%Y-%m-%d}.."
              f"{max(it['valid'] for it in items):%Y-%m-%d}", flush=True)
    print(f"[ladder] rows read in {time.time() - t0:.0f} s", flush=True)
    names = dsets[("valid", False)].output_variables
    for p in todo:
        t0 = time.time()
        m = Module.from_checkpoint(str(p), override_args={"use_apex_gn": False})
        nin = int(m.img_in_channels) - int(getattr(m.model, "N_grid_channels", 4))
        m.use_fp16 = False
        m = m.eval().to(dev).to(memory_format=torch.channels_last)
        if hasattr(m, "amp_mode"):
            m.amp_mode = False
        ti = nin == len(dsets[("valid", True)].input_channels())
        assert ti or nin == len(dsets[("valid", False)].input_channels()), f"{p}: {nin} input channels"
        res = {"checkpoint": str(Path(p).resolve()), "samples": samples_of(p), "dataset_channels": nin, "time_inputs": ti,
               "variables": names}
        for sp in ("train", "valid"):
            ds = dsets[(sp, ti)]
            ss = np.zeros(len(names))
            by_lead = {L: np.zeros(len(names)) for L in LEADS}
            n_lead = {L: 0 for L in LEADS}
            for it in rows[sp]:
                x = ds.coarse_input(it["xc"], it["valid"])
                img_lr = torch.from_numpy(x[None]).to(dev).to(memory_format=torch.channels_last)
                mean = regression_step(net=m, img_lr=img_lr, latents_shape=(1, len(names), *ds.image_shape()))
                r2 = ((it["y"] - mean[0].float().cpu().numpy()) ** 2).reshape(len(names), -1).mean(1)
                ss += r2
                by_lead[it["step"]] += r2
                n_lead[it["step"]] += 1
            res[sp] = dict(zip(names, np.sqrt(ss / len(rows[sp])).round(5).tolist()))
            res[sp + "_rows"] = [it["row"] for it in rows[sp]]
            if sp == "valid":
                res["valid_by_lead"] = {str(L): dict(zip(names, np.sqrt(by_lead[L] / n_lead[L]).round(5).tolist()))
                                        for L in LEADS if n_lead[L]}
        res["seconds"] = round(time.time() - t0, 1)
        tmp = out / "parts" / (part_name(p) + ".tmp")
        tmp.write_text(json.dumps(res, indent=1))
        os.replace(tmp, out / "parts" / part_name(p))
        print(f"[ladder] {Path(p).name} ({nin} channels, time inputs {ti}): valid "
              + " ".join(f"{v} {res['valid'][v]:.3f}" for v in names) + f" | {res['seconds']:.0f} s", flush=True)
        del m
        torch.cuda.empty_cache()


def cmd_report(a):
    out = Path(a.out)
    prev = json.load(open(out / "ladder.json")) if (out / "ladder.json").exists() else {"checkpoints": {}}
    ck = dict(prev["checkpoints"])
    for f in sorted((out / "parts").glob("*.json")):
        r = json.load(open(f))
        ck[r["checkpoint"]] = r
    order = sorted(ck, key=lambda k: (Path(k).parent.parent.name, ck[k]["samples"], k))
    (out / "ladder.json").write_text(json.dumps({"checkpoints": {k: ck[k] for k in order}}, indent=1))
    if not order:
        print("no results")
        return
    names = ck[order[0]]["variables"]

    def label(k):
        return f"{Path(k).parent.parent.name}/{ck[k]['samples']}{' +t' if ck[k]['time_inputs'] else ''}"
    md = ["# CorrDiff regression residual ladder", "",
          "Root mean square of target minus regression output, normalised units (sigma_data of the diffusion = 0.5). "
          f"Rows: {len(ck[order[0]]['train_rows'])} training, {len(ck[order[0]]['valid_rows'])} validation, "
          "equal numbers per lead. `+t` = with time inputs.", ""]
    for sp, title in (("valid", "Validation rows (2022)"), ("train", "Training rows (2003-2021)")):
        md += [f"## {title}", "", "| checkpoint | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
        md += ["| " + label(k) + " | " + " | ".join(f"{ck[k][sp][v]:.3f}" for v in names) + " |" for k in order]
        md.append("")
    last = order[-1]
    md += [f"## Validation rows per lead, {label(last)}", "", "| lead h | " + " | ".join(names) + " |",
           "|---" * (len(names) + 1) + "|"]
    md += ["| " + L + " | " + " | ".join(f"{d[v]:.3f}" for v in names) + " |" for L, d in ck[last]["valid_by_lead"].items()]
    (out / "ladder.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"LADDER_REPORT {len(order)} checkpoints -> {out / 'ladder.json'}, {out / 'ladder.md'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--cfg")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpts", default="")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    {"run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
