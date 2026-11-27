#!/usr/bin/env python3
"""GPU-hours used by a CorrDiff training chain on MN5 (the job ids recorded in state/chain_*/jobs.txt), from sacct:
sum over the jobs of ElapsedRaw (s, the running job up to now, failed and cancelled segments included) x the GPUs in
AllocTRES (gres/gpu=N; a job that never started has none and counts 0). Used by cd_train_mn5.sbatch for the ceiling.

    python3 cd_gpuh.py used --jobs STATE/jobs.txt [--exclude JOBID]     -> prints the GPU-h (float)
    python3 cd_gpuh.py selftest                                         -> parser on fake sacct lines
"""
import argparse
import re
import subprocess
import sys


def job_ids(path, exclude=None):
    ids = []
    try:
        for line in open(path):
            w = line.split()
            if w and w[0].isdigit() and w[0] != str(exclude) and w[0] not in ids:
                ids.append(w[0])
    except FileNotFoundError:
        pass
    return ids


def parse(lines):
    """sacct -X -n -P -o JobID,ElapsedRaw,AllocTRES lines -> {jobid: gpu_hours}."""
    out = {}
    for line in lines:
        f = line.strip().split("|")
        if len(f) < 3 or not f[0]:
            continue
        jid = f[0].split(".")[0]
        try:
            sec = int(f[1] or 0)
        except ValueError:
            sec = 0
        m = re.search(r"gres/gpu(?::[^=,]+)?=(\d+)", f[2])
        gpus = int(m.group(1)) if m else 0
        out[jid] = max(out.get(jid, 0.0), sec * gpus / 3600.0)
    return out


def used(ids):
    if not ids:
        return 0.0, {}
    r = subprocess.run(["sacct", "-X", "-n", "-P", "-j", ",".join(ids), "-o", "JobID,ElapsedRaw,AllocTRES"],
                       capture_output=True, text=True, check=True)
    d = parse(r.stdout.splitlines())
    return sum(d.values()), d


def selftest():
    fake = ["49500001|43200|billing=320,cpu=80,gres/gpu=4,mem=480G,node=1",
            "49500002|3600|billing=640,cpu=160,gres/gpu=8,mem=960G,node=2",
            "49500003|0||",                                   # pending, never started
            "49500004|1800|cpu=80,gres/gpu:h100=4,node=1",    # typed gres
            "49500005|120|cpu=80,mem=10G,node=1"]             # CPU job
    d = parse(fake)
    assert abs(d["49500001"] - 48.0) < 1e-9 and abs(d["49500002"] - 8.0) < 1e-9, d
    assert d["49500003"] == 0.0 and abs(d["49500004"] - 2.0) < 1e-9 and d["49500005"] == 0.0, d
    assert abs(sum(d.values()) - 58.0) < 1e-9
    print("GPUH_SELFTEST_OK", d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["used", "selftest"])
    ap.add_argument("--jobs")
    ap.add_argument("--exclude")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
        return
    tot, d = used(job_ids(a.jobs, a.exclude))
    if a.verbose:
        print(" ".join(f"{k}:{v:.2f}" for k, v in d.items()), file=sys.stderr)
    print(f"{tot:.3f}")


if __name__ == "__main__":
    main()
