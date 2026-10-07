#!/usr/bin/env python3
"""Gate G2 of campaign T1d stage A (2026-10-07), after the first file of a run lands (the s3_check_attrs.py pattern).

For every prediction file given: the sampling_config_json attribute equals the lane's resolved sampler block
(<run root>/meta/<lane>.sampler.json, written by resolve_lane_t1d.py inside the job), checkpoint_id is the 1.2M
parent 551dfd1e (and checkpoint_path, when recorded, lies in its folder; never the donor 12dcefea), member_ids equal
--members. With --log, the probe lines of the job log are checked too: PROBE_SCHEDULE's levels equal the arm's applied
levels (make_lanes_t1d.applied_sigmas, at the probe's 6 significant digits) and PROBE_SAMPLER's denoiser_calls equal
2N-1 with S_churn 0 applied.
Usage: python check_attrs_t1d.py <sampler json> [--members 1,...,10] [--log <job log>] <file> [<file> ...]
Exit 0 only if everything matches.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import netCDF4
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_lanes_t1d as mk  # noqa: E402

RUN_1P2M = "551dfd1eb2a649d49f099acd30b59483"
DONOR = "12dcefeafa92457aab286d4931ffe63f"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sampler_json")
    ap.add_argument("files", nargs="*")
    ap.add_argument("--members", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()
    want = json.load(open(args.sampler_json))
    sig = mk.applied_sigmas(want)
    print(f"RESOLVED {json.dumps(want)}")
    print(f"APPLIED levels={len(sig)} calls={mk.ml.calls(want)}")
    bad = 0
    for f in args.files:
        with netCDF4.Dataset(f) as d:
            cfg = json.loads(d.getncattr("sampling_config_json"))
            ck = d.getncattr("checkpoint_id")
            cp = d.getncattr("checkpoint_path") if "checkpoint_path" in d.ncattrs() else ""
            mem = d.getncattr("member_ids")
        mem = mem if isinstance(mem, str) else ",".join(str(int(m)) for m in np.atleast_1d(mem))
        ok = cfg == want and ck == RUN_1P2M and mem == args.members and (not cp or (RUN_1P2M in cp and DONOR not in cp))
        bad += not ok
        print(("OK  " if ok else "BAD ") + f + f" ckpt_id={ck} ckpt_path={cp or 'n/a'} members={mem}"
              + ("" if cfg == want else f" got={json.dumps(cfg)}"))
    if args.log:
        text = open(args.log, errors="replace").read()
        m = re.search(r"PROBE_SCHEDULE .*? sigmas=\[([^\]]*)\]", text)
        if not m:
            print("BAD log: no PROBE_SCHEDULE line (was the job run with PROBE=1?)")
            bad += 1
        else:
            got = [float(x) for x in m.group(1).split(",")]
            exp = [float(f"{s:.6g}") for s in sig] + [0.0]
            same = got == exp
            bad += not same
            print(("OK  " if same else "BAD ") + f"probe schedule {len(got) - 1} levels + 0" + ("" if same else f" got={got} expected={exp}"))
        m = re.search(r"PROBE_SAMPLER .*?applied=(\{[^}]*\}).*?denoiser_calls=(\d+)", text)
        if not m:
            print("BAD log: no PROBE_SAMPLER line")
            bad += 1
        else:
            calls_ok = int(m.group(2)) == mk.ml.calls(want)
            churn_ok = re.search(r"'S_churn': (tensor\()?0(\.0*)?[,)}]", m.group(1)) is not None
            bad += not (calls_ok and churn_ok)
            print(("OK  " if calls_ok and churn_ok else "BAD ") + f"probe denoiser_calls={m.group(2)} applied={m.group(1)}")
    print("G2 PASS" if not bad else f"G2 FAIL ({bad})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
