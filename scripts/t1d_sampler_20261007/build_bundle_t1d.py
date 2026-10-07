#!/usr/bin/env python3
"""Build the docs bundle of campaign T1d stage A (2026-10-07) on hpc-login, after every stage A job finished.

Lays out <staging>/ (default $T1D_W/bundle/stageA) for docs/epics/fast-generative-downscaling/in-progress/
20261007_T1d_sampler_beat_1p2M_results/stageA/:
  <run>/evaluators/<evaluator>/metrics.json   six evaluators per run (+ probabilistic/scores_by_lead.csv, which the
                                               read uses for its per-case signs); the layout read_stage1.py reads
  <run>/sampler.json                           the resolved sampler block the job passed (from <run root>/meta)
  tc_intensity_stageA.tsv, _checks.txt, _summary.txt
  spectra_v3/                                  CSVs, summary.md, run.log and PNGs of the v3 box spectra
  read_stageA/stage1_read.{json,md}            the paired read (baseline p12m_pw30_c0 s756, replicate s757000)
  gates/                                       G1 / G2 outputs saved under $T1D_W/notes/gates (see LAUNCH.md)
  jobs.tsv                                     every job id (gates and stage A)
  timings.tsv                                  sacct of every job + seconds per draw and peak GPU memory per run
  logs_rc.txt                                  the host/rc/summary lines of every job log
  lanes/                                       the base lane and the four arm lanes as resolved files
Files over 1 MB (the docs pre-commit limit) are listed in OVER_1MB.txt and NOT copied; copy them to the docs media
archive by hand if wanted. Run: python build_bundle_t1d.py [--staging DIR]   (refuses an existing staging dir)
"""
import argparse
import csv
import glob
import os
import re
import shutil
import subprocess
import sys

W = os.environ.get("T1D_W", "/home/ecm5702/work-t1d-20261007")
CODE = os.environ.get("T1D_CODE", W + "/code/downscaling-tools")
E = os.environ.get("T1D_E", "/home/ecm5702/scratch/eval")
TAG = "20261007"
RUNS = {  # bundle name -> run root
    "p12m_pw30_c0_s756": f"{E}/o320_o1280_p12m_pw30_c0_tc100_{TAG}",
    "p12m_pw30_c0_r757": f"{E}/o320_o1280_p12m_pw30_c0_r757_tc100_{TAG}",
    "p12m_c0_pw16_s1k": f"{E}/o320_o1280_p12m_c0_pw16_s1k_tc100_{TAG}",
    "p12m_st2": f"{E}/o320_o1280_p12m_st2_tc100_{TAG}",
    "p12m_st4": f"{E}/o320_o1280_p12m_st4_tc100_{TAG}",
}
EVALUATORS = ("tc", "texture", "wind_extremes", "probabilistic", "surface", "shape")
LANES = ["tc_o320_o1280_p12m_ctrl", "tc_o320_o1280_p12m_pw30_c0", "tc_o320_o1280_p12m_c0_pw16_s1k",
         "tc_o320_o1280_p12m_st2", "tc_o320_o1280_p12m_st4"]
MB = 1024 * 1024
LOG_PAT = re.compile(r"^(host=|RUNTIME |GPU |dstools=|LANE |CHECKPOINT |APPLIED |PROBE_SCHEDULE .*call=1 |PROBE_SAMPLER .*call=1 |"
                     r"PROBE_PATHS|SE_TC_T1D|T1D_|  \S+ metrics.json|CHECK |rows=|FATAL|Traceback|\[ECMWF-INFO -ecepilog\] "
                     r"(JobID|Start|End|ExitCode|State) )")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", default=f"{W}/bundle/stageA")
    args = ap.parse_args()
    st = args.staging
    if os.path.exists(st):
        sys.exit(f"refusing: {st} exists")
    big, missing = [], []

    def put(src, rel):
        if not os.path.isfile(src):
            missing.append(src)
            return
        if os.path.getsize(src) > MB:
            big.append((src, os.path.getsize(src)))
            return
        dst = os.path.join(st, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

    for name, rr in RUNS.items():
        for ev in EVALUATORS:
            put(f"{rr}/eval_stageA/evaluators/{ev}/metrics.json", f"{name}/evaluators/{ev}/metrics.json")
        put(f"{rr}/eval_stageA/evaluators/probabilistic/scores_by_lead.csv",
            f"{name}/evaluators/probabilistic/scores_by_lead.csv")
        for f in glob.glob(f"{rr}/meta/*.sampler.json"):
            put(f, f"{name}/sampler.json")
    n = f"{W}/stageA/notes"
    for f in ("tc_intensity_stageA.tsv", "tc_intensity_stageA_checks.txt", "tc_intensity_stageA_summary.txt"):
        put(f"{n}/{f}", f)
    for f in sorted(glob.glob(f"{W}/stageA/spectra_v3/**/*", recursive=True)):
        if os.path.isfile(f) and f.endswith((".csv", ".md", ".log", ".png", ".json")):
            put(f, "spectra_v3/" + os.path.relpath(f, f"{W}/stageA/spectra_v3"))
    for f in ("stage1_read.json", "stage1_read.md"):
        put(f"{W}/stageA/read_stageA/{f}", f"read_stageA/{f}")
    for f in sorted(glob.glob(f"{W}/notes/gates/*")):
        put(f, "gates/" + os.path.basename(f))
    for lane in LANES:
        put(f"{CODE}/eval/config/lanes/{lane}.yaml", f"lanes/{lane}.yaml")
    put(f"{W}/notes/jobs.tsv", "jobs.tsv")

    # timings: sacct of every job in jobs.tsv, plus the per-run summary line of each prediction log
    jobs = list(csv.DictReader(open(f"{W}/notes/jobs.tsv"), delimiter="\t")) if os.path.isfile(f"{W}/notes/jobs.tsv") else []
    ids = [j["jobid"] for j in jobs if j["jobid"].isdigit()]
    os.makedirs(st, exist_ok=True)
    if ids:
        fmt = "JobID,JobName%40,Partition,QOS,State,ExitCode,Submit,Start,End,Elapsed,AllocTRES%80,NodeList"
        try:
            r = subprocess.run(["sacct", "-X", "-P", "-j", ",".join(ids), f"--format={fmt}"], capture_output=True, text=True)
            open(f"{st}/sacct.txt", "w").write(r.stdout + r.stderr)
        except FileNotFoundError:
            missing.append("sacct (not on PATH: run on hpc-login)")
    with open(f"{st}/timings.tsv", "w") as fh:
        fh.write("kind\tarm\tseed\tjobid\truntime\trc\tfiles\twall_s\tdraws\ts_per_draw\tpeak_mem_mib\tlog\n")
        for j in jobs:
            if j["kind"] not in ("predict_box", "gate_g1"):
                continue
            logs = glob.glob(f"{W}/logs/*_{j['jobid']}.out")
            line = ""
            for lg in logs:
                for x in open(lg, errors="replace"):
                    if x.startswith("SE_TC_T1D "):
                        line = x
            kv = dict(t.split("=", 1) for t in line.split()[1:] if "=" in t)
            fh.write("\t".join([j["kind"], j["arm"], j["seed"], j["jobid"], kv.get("runtime", ""), kv.get("rc", ""),
                                kv.get("files", ""), kv.get("wall_s", ""), kv.get("draws", ""), kv.get("s_per_draw", ""),
                                kv.get("peak_mem_mib", ""), logs[0] if logs else "missing"]) + "\n")
    with open(f"{st}/logs_rc.txt", "w") as fh:
        for j in jobs:
            for lg in glob.glob(f"{W}/logs/*_{j['jobid']}.out"):
                fh.write(f"== {lg} ({j['kind']} {j['arm']} seed {j['seed']})\n")
                for x in open(lg, errors="replace"):
                    if LOG_PAT.match(x):
                        fh.write(x.rstrip()[:2000] + "\n")
                fh.write("\n")
    if big:
        with open(f"{st}/OVER_1MB.txt", "w") as fh:
            fh.writelines(f"{s}\t{p}\n" for p, s in big)
    files = [(os.path.join(dp, f)) for dp, _, fs in os.walk(st) for f in fs]
    print(f"bundle {st}: {len(files)} files, {sum(os.path.getsize(f) for f in files)} bytes")
    print("over 1 MB, not copied:", big or "none")
    print("missing:", missing or "none")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
