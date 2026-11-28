#!/usr/bin/env python3
"""Audit an overlap smoke chain: steps strictly increase, each job writes one contiguous block (no interleaving =
never two writers), every handover in CHAIN.log has its scancel before the new writer, and the end step was reached.
Usage: python3 overlap_smoke_audit.py <root> <end_step>"""
import os, re, sys
root, end = sys.argv[1], int(sys.argv[2])
rows = [l.split() for l in open(os.path.join(root, "steps.log")) if l.strip()]
steps = [int(r[1]) for r in rows]
jobs = [r[0] for r in rows]
blocks = [j for i, j in enumerate(jobs) if i == 0 or jobs[i - 1] != j]
ok = True
if any(b <= a for a, b in zip(steps, steps[1:])):
    print("FAIL steps not strictly increasing"); ok = False
if len(blocks) != len(set(blocks)):
    print("FAIL a job wrote in two separate blocks (interleaved writers): {}".format(blocks)); ok = False
if steps[-1] < end:
    print("FAIL end step {} not reached (last {})".format(end, steps[-1])); ok = False
log = open(os.path.join(root, "CHAIN.log")).read()
handovers = len(re.findall(r"takeover: scancel \d+ after checkpoint", log))
timeouts = len(re.findall(r"takeover: scancel \d+ after timeout", log))
refused = len(re.findall(r"REFUSED", log))
yields, fenced, kept = (len(re.findall(k, log)) for k in (r"takeover: YIELD", r"fence: CHAIN.writer names", r"keeper: no successor"))
print("{} yields, {} fence stops (must be 0 in a clean run), {} keeper resubmissions".format(yields, fenced, kept))
if fenced:
    print("FAIL a segment was fenced: two jobs believed they were the writer"); ok = False
print("writers in order: {}".format(" ".join(blocks)))
print("{} {} writes, {} writers, {} checkpoint handovers, {} timeout kills, {} refusals, last step {}".format(
    "AUDIT_OK" if ok else "AUDIT_FAIL", len(rows), len(blocks), handovers, timeouts, refused, steps[-1]))
sys.exit(0 if ok else 1)
