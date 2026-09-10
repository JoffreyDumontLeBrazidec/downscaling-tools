"""Stage 5: gather the per-start records into one manifest per block.

Each stage already writes a small manifest next to its own output.  This module
collects them into manifests/<block>.json, and manifests/campaign.json across
every block, so that a single file answers what exists, how it was produced and
whether it can be reproduced.

Per start and member the manifest records the seed the forecast used, the
checksum of the initial condition it started from, the size and checksum of the
native forecast, the size of the O320 file, the status, and the measured times.
Checksumming the native files is the slow part, so it is optional; without it
the manifest still records sizes and status.
"""

from __future__ import annotations

import argparse
import os
import sys

from . import calendar as cal
from . import gribspec as spec
from .common import (
    CHECKPOINT,
    DEFAULT_ROOT,
    atomic_write_json,
    log,
    read_json,
    sha256_file,
)
from .regrid import derived_root_for, native_root_for
from .assemble import members_root


def build_block_manifest(root: str, block: str, checksums: bool) -> dict:
    starts = cal.block_starts(block, root)
    ic_root = members_root(root, block)
    nat_root = native_root_for(root, block)
    der_root = derived_root_for(root, block)

    record = {
        "block": block,
        "root": root,
        "checkpoint": CHECKPOINT,
        "checkpoint_sha256": sha256_file(CHECKPOINT) if checksums else None,
        "members": spec.MEMBERS,
        "lead_steps": spec.LEAD_STEPS,
        "fields_per_member_initial_condition": spec.FIELDS_PER_MEMBER_FILE,
        "fields_per_step_native": spec.NATIVE_FIELDS_PER_STEP_EXPECTED,
        "fields_per_start_o320": spec.LANE_FIELDS_PER_START,
        "starts": {},
    }

    complete = 0
    for start in starts:
        key = cal.start_key(start)
        ic_manifest = read_json(os.path.join(ic_root, key, "manifest.json"), {}) or {}
        nat_manifest = read_json(os.path.join(nat_root, key, "manifest.json"), {}) or {}
        derived = os.path.join(der_root, f"{key}.grib")

        entry = {"start_iso": start.isoformat(), "members": {}}
        for m in spec.MEMBERS:
            tag = f"m{m:02d}"
            ic = (ic_manifest.get("members") or {}).get(tag, {})
            nat = (nat_manifest.get("members") or {}).get(tag, {})
            nat_path = os.path.join(nat_root, key, f"{tag}.grib")
            e = {
                "seed": nat.get("seed"),
                "initial_condition_status": ic.get("status", "absent"),
                "initial_condition_sha256": ic.get("sha256") or nat.get(
                    "initial_condition_sha256"
                ),
                "native_status": nat.get("status", "absent"),
                "native_bytes": nat.get("bytes"),
                "native_sha256": None,
                "seconds_forecast": nat.get("seconds_forecast"),
                "peak_gpu_gib": nat.get("peak_gpu_gib"),
                "fields_entirely_missing": nat.get("fields_entirely_missing"),
                "missing_values_total": nat.get("missing_values_total"),
            }
            if checksums and os.path.exists(nat_path):
                e["native_sha256"] = sha256_file(nat_path)
            if e["native_status"] == "complete":
                complete += 1
            entry["members"][tag] = e

        entry["o320_path"] = derived if os.path.exists(derived) else None
        entry["o320_bytes"] = os.path.getsize(derived) if os.path.exists(derived) else None
        entry["checkpoint_load_seconds"] = nat_manifest.get("checkpoint_load_seconds")
        record["starts"][key] = entry

    record["members_complete"] = complete
    record["members_expected"] = len(starts) * len(spec.MEMBERS)
    return record


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--block", help="one block; omit to rebuild every block and the campaign file")
    p.add_argument("--root", default=os.environ.get("AIFSENS2_ROOT", DEFAULT_ROOT))
    p.add_argument(
        "--checksums",
        action="store_true",
        help="also checksum the checkpoint and every native file; this is slow",
    )
    a = p.parse_args(argv)

    blocks = [a.block] if a.block else cal.all_blocks()
    out_dir = os.path.join(a.root, "manifests")
    campaign = {"root": a.root, "blocks": {}}

    for b in blocks:
        try:
            starts = cal.block_starts(b, a.root)
        except Exception as exc:
            log(f"skipping block {b}: {exc}")
            continue
        if not starts:
            log(f"skipping block {b}: no starts defined yet")
            continue
        rec = build_block_manifest(a.root, b, a.checksums)
        path = os.path.join(out_dir, f"{b}.json")
        atomic_write_json(path, rec)
        log(
            f"MANIFEST {b}: {rec['members_complete']} of {rec['members_expected']} "
            f"member forecasts complete -> {path}"
        )
        campaign["blocks"][b] = {
            "starts": len(starts),
            "members_complete": rec["members_complete"],
            "members_expected": rec["members_expected"],
            "manifest": path,
        }

    campaign_path = os.path.join(out_dir, "campaign.json")
    if a.block:
        previous = read_json(campaign_path, {"blocks": {}}) or {"blocks": {}}
        previous.setdefault("blocks", {}).update(campaign["blocks"])
        previous["root"] = a.root
        campaign = previous
    atomic_write_json(campaign_path, campaign)
    log(f"MANIFEST campaign -> {campaign_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
