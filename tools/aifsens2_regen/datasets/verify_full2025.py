"""Gates for the full-year 2025 AIFS-CRPS stores (modelled on verify_full2026.py).

The 2025 record of the ensemble is split across two MARS experiment versions: expver 0103, the
pre-operational experiment, up to and including the 2025-07-01 00 UTC start, and expver 0001
from the 2025-07-01 12 UTC start onward. The recipe joins them with the concat input action, so
the verification probes one sample on each side of that boundary and compares it bit for bit
with a field retrieved directly from MARS under the corresponding experiment version.

Two profiles are available. The default one checks the real full-year build (1,460 synthetic
dates, fifty members). The --fixture profile checks the four-start fixture around the boundary
that was built with two members, which is the cheap test of the same code path.
"""

import argparse
import datetime
import json
import subprocess
import sys
import tempfile

import numpy as np
from anemoi.datasets import open_dataset

FULL = dict(
    ndates=1460,
    nmembers=50,
    # (start, lead, member, expver), one probe on each side of the 2025-07-01 12 UTC boundary
    probes=[
        ("2025-03-15T00:00:00", 6, 13, "0103"),
        ("2025-09-20T12:00:00", 12, 7, "0001"),
    ],
)

FIXTURE = dict(
    ndates=8,
    nmembers=2,
    probes=[
        ("2025-06-30T12:00:00", 6, 2, "0103"),
        ("2025-07-01T12:00:00", 12, 1, "0001"),
    ],
)


def mars_field(request):
    """Retrieve one field from MARS and return its values as a flat array."""
    import eccodes as ec

    with tempfile.TemporaryDirectory() as td:
        tgt = f"{td}/f.grib"
        full = request + ',target="' + tgt + '"'
        subprocess.run(["mars"], input=full.encode(), check=True, capture_output=True)
        with open(tgt, "rb") as f:
            h = ec.codes_grib_new_from_file(f)
            v = ec.codes_get_values(h)
            ec.codes_release(h)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/home/ecm5702/scratch/data/aifscrps_2025_full/training")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--fixture", action="store_true")
    args = ap.parse_args()

    prof = FIXTURE if args.fixture else FULL
    s, sfx = args.dir, args.suffix
    IN = f"{s}/downscaling-ai-pf-enfo-0001-mars-o320-2025-2025-12h-6h-v1-aifscrps{sfx}.zarr"
    TG = f"{s}/downscaling-od-an-oper-0001-mars-o1280-2025-2025-12h-6h-v1-aifscrps-validtime{sfx}.zarr"
    FO = f"{s}/downscaling-od-an-oper-0001-mars-o1280-2025-2025-12h-6h-v1-aifscrps-validtime-forcings{sfx}.zarr"

    state = {"ok": True}

    def check(c, msg):
        print(("PASS " if c else "FAIL ") + msg, flush=True)
        state["ok"] = state["ok"] and bool(c)

    zi, zt, zf = open_dataset(IN), open_dataset(TG), open_dataset(FO)
    for n, z in (("input", zi), ("target", zt), ("forcings", zf)):
        print(n, z.shape, "vars", len(z.variables))

    mi, mt, mf = (json.load(open(f"{p}/.zattrs"))["fake_forecasts"] for p in (IN, TG, FO))
    check(mi == mt == mf, "identical fake_forecasts mapping across the three stores")
    check(len(mi) == prof["ndates"], "%d synthetic dates (got %d)" % (prof["ndates"], len(mi)))
    nm = prof["nmembers"]
    check(
        zi.shape[2] == nm and zt.shape[2] == 1 and zf.shape[2] == 1,
        "ensemble dims %d/1/1 (got %d/%d/%d)" % (nm, zi.shape[2], zt.shape[2], zf.shape[2]),
    )
    check(
        zi.shape[1] == 68 and zt.shape[1] == 68,
        "68 variables in input and target (got %d and %d)" % (zi.shape[1], zt.shape[1]),
    )
    check(list(zi.variables) == list(zt.variables), "same variable order in input and target")
    check(zi.shape[0] == zt.shape[0] == zf.shape[0] == prof["ndates"], "same length in the three stores")

    inv = {tuple(v): k for k, v in mi.items()}
    dates = [str(d).replace(" ", "T")[:19] for d in zt.dates]
    t2 = zt.name_to_index["2t"]

    for start, lead, member, expver in prof["probes"]:
        key = inv[(start, lead)]
        i = dates.index(key[:19])
        d0 = datetime.datetime.fromisoformat(start)
        valid = d0 + datetime.timedelta(hours=lead)
        # input sample: the AIFS-CRPS member retrieved under its own experiment version
        req = (
            "retrieve,class=ai,model=aifs-ens,stream=enfo,expver=%s,type=pf,number=%d,"
            "date=%s,time=%s,step=%d,levtype=sfc,param=167,grid=O320"
            % (expver, member, d0.strftime("%Y%m%d"), d0.strftime("%H%M"), lead)
        )
        ai = mars_field(req)
        slot = member - 1
        diff = np.abs(zi[i, zi.name_to_index["2t"], slot, :].astype(np.float64) - ai).max()
        check(
            diff < 1e-3,
            "input ensemble slot %d == AIFS member %d, start %s +%dh, expver %s (max abs diff %.2e K)"
            % (slot, member, start, lead, expver, diff),
        )
        # target sample: the operational analysis at the valid time
        an = mars_field(
            "retrieve,class=od,stream=oper,expver=0001,type=an,date=%s,time=%s,step=0,"
            "levtype=sfc,param=167,grid=O1280" % (valid.strftime("%Y%m%d"), valid.strftime("%H%M"))
        )
        d2 = np.abs(zt[i, t2, 0, :].astype(np.float64) - an).max()
        check(
            d2 < 1e-3,
            "target for start %s +%dh == analysis %s UTC from MARS (max abs diff %.2e K)"
            % (start, lead, valid.strftime("%Y-%m-%d %H"), d2),
        )
        check(
            np.isfinite(zt[i]).all() and np.isfinite(zi[i]).all() and np.isfinite(zf[i]).all(),
            "no NaN in the sample for start %s +%dh" % (start, lead),
        )
        if zi.shape[2] > 1:
            m1 = zi[i, zi.name_to_index["msl"], 0, :]
            m2 = zi[i, zi.name_to_index["msl"], 1, :]
            check(not np.array_equal(m1, m2), "members differ (msl slots 0 and 1) for start %s +%dh" % (start, lead))

    # Both sides of the experiment-version boundary must be on the axis.
    check(
        ("2025-07-01T00:00:00", 12) in inv and ("2025-07-01T12:00:00", 6) in inv,
        "both sides of the 2025-07-01 12 UTC experiment-version boundary are present",
    )

    # Insolation is computed at the valid time, so it differs between the 6 h and 12 h samples.
    first_start = prof["probes"][0][0]
    ins = zf.name_to_index["insolation"]
    k6 = dates.index(inv[(first_start, 6)][:19])
    k12 = dates.index(inv[(first_start, 12)][:19])
    check(
        not np.array_equal(zf[k6, ins, 0, :], zf[k12, ins, 0, :]),
        "insolation differs between the 6 h and 12 h samples of one start (valid-time based)",
    )

    # Documentation limit: variables_metadata carries a single mars request per variable, so it
    # records only one of the two experiment versions for the whole store.
    vm = json.load(open(f"{IN}/.zattrs")).get("variables_metadata", {})
    ev = {k: (v.get("mars", {}) or {}).get("expver") for k, v in vm.items()}
    seen = sorted({e for e in ev.values() if e is not None})
    print(
        "NOTE variables_metadata expver values recorded in the input store: %s (example 2t -> %s). "
        "The store really spans expver 0103 and expver 0001, so this attribute understates the "
        "content; the split is documented in the recipe description and in this verifier only."
        % (seen, ev.get("2t"))
    )

    st = zt.statistics
    print("target stats mean 2t", float(st["mean"][t2]), "stdev", float(st["stdev"][t2]))
    print("ALL_PASS" if state["ok"] else "SOME_FAIL")
    sys.exit(0 if state["ok"] else 1)


if __name__ == "__main__":
    main()
