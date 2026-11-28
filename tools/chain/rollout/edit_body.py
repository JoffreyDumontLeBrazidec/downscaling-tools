#!/usr/bin/env python3
"""Insert chain_takeover before a line and/or replace the successor call in a job body (exact-match edits only).
Refuses (exit 3) when an anchor is missing or appears more than once, or when the body already has chain_takeover."""
import argparse, sys
p = argparse.ArgumentParser()
p.add_argument("body"); p.add_argument("--out", required=True)
p.add_argument("--takeover-before", help="exact line content (stripped) before which chain_takeover is inserted")
p.add_argument("--root-var", default="CKPT_ROOT")
p.add_argument("--next-from"); p.add_argument("--next-to")
p.add_argument("--insert-after", help="exact line content (stripped) after which --insert-text goes")
p.add_argument("--insert-text")
a = p.parse_args()
lines = open(a.body).read().split("\n")
if any("chain_takeover" in l for l in lines):
    sys.exit("REFUSED: body already calls chain_takeover")
def find(anchor):
    idx = [i for i, l in enumerate(lines) if l.strip() == anchor]
    if len(idx) != 1:
        print("REFUSED: anchor {!r} found {} times".format(anchor, len(idx)), file=sys.stderr); sys.exit(3)
    return idx[0]
if a.next_from:
    i = find(a.next_from); lines[i] = lines[i].replace(a.next_from, a.next_to)
if a.takeover_before:
    i = find(a.takeover_before); ind = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
    lines.insert(i, ind + 'chain_takeover "${}"   # overlap mode: wait for the running writer\'s next checkpoint, '
                 'stop it, then train (tools/chain/chain_handover.py)'.format(a.root_var))
if a.insert_after:
    i = find(a.insert_after)
    for k, t in enumerate(a.insert_text.split("\\n")):
        lines.insert(i + 1 + k, t)
open(a.out, "w").write("\n".join(lines))
