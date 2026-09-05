"""Put the high-resolution truth back into a prediction made from stripped bundles.

Predicting on a remote machine means shipping bundles without their
``target_hres_*`` fields, because the truth is 88% of a bundle and the remote
host has no use for it. Such a run is written with ``--allow-missing-target-unsafe``,
so its ``y`` is all-NaN and it carries
``missing_target_policy = all_nan_due_to_allow_missing_target_unsafe``. The
evaluators need a real ``y``, and the truth never left ECMWF, so this reattaches
it in place from the original truth-bearing bundles.

Each prediction records the bundle it used per member in ``source_bundle``, so
the mapping is by basename: only the directory differs. The channel mapping is
delegated to ``extract_target_from_bundle_dataset``, the same function the
prediction path itself uses, so level fields such as t_850 are resolved
identically rather than re-derived here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import netCDF4
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from manual_inference.input_data_construction.bundle import (  # noqa: E402
    extract_target_from_bundle_dataset,
)

STRIPPED_POLICY = "all_nan_due_to_allow_missing_target_unsafe"


def reattach(pred_path: Path, truth_dir: Path, *, dry_run: bool = False) -> tuple[int, int]:
    """Fill ``y`` from the truth bundles; return (members filled, channels found)."""
    mode = "r" if dry_run else "a"
    with netCDF4.Dataset(pred_path, mode) as ds:
        policy = getattr(ds, "missing_target_policy", "")
        if policy != STRIPPED_POLICY:
            raise SystemExit(
                f"{pred_path.name}: missing_target_policy is {policy!r}, not {STRIPPED_POLICY!r}. "
                "Refusing to overwrite y in a prediction that already carries truth."
            )
        states = ds.getncattr("output_weather_states").split(",")
        # source_bundle is (sample, ensemble_member); take the member axis.
        sources = [str(x) for x in np.asarray(ds.variables["source_bundle"][:]).reshape(-1)]
        y = ds.variables["y"]
        y.set_auto_mask(False)
        n_members = y.shape[1]
        if len(sources) != n_members:
            raise SystemExit(f"{pred_path.name}: {len(sources)} source bundles for {n_members} members")

        filled = 0
        found_min = len(states)
        for m, src in enumerate(sources):
            truth = truth_dir / Path(src.strip()).name
            if not truth.exists():
                raise SystemExit(f"{pred_path.name}: truth bundle not found: {truth}")
            with xr.open_dataset(truth) as bundle:
                target, found = extract_target_from_bundle_dataset(bundle, states)
            if target is None:
                raise SystemExit(f"{truth.name}: no target fields found for {states}")
            found_min = min(found_min, found)
            if not dry_run:
                y[0, m, :, :] = np.asarray(target, dtype=np.float32)
            filled += 1

        if not dry_run:
            ds.missing_target_policy = (
                f"truth reattached from {truth_dir} after a prediction made from stripped bundles"
            )
            ds.truth_reattached_from = str(truth_dir)
    return filled, found_min


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred-dir", required=True, help="directory of predictions_*.nc to fix in place")
    p.add_argument("--truth-dir", required=True, help="directory of the ORIGINAL truth-bearing bundles")
    p.add_argument("--pattern", default="predictions_*.nc")
    p.add_argument("--dry-run", action="store_true", help="check mapping without writing")
    args = p.parse_args(argv)

    preds = sorted(Path(args.pred_dir).glob(args.pattern))
    if not preds:
        raise SystemExit(f"no files matching {args.pattern} in {args.pred_dir}")
    truth_dir = Path(args.truth_dir)

    for f in preds:
        filled, found = reattach(f, truth_dir, dry_run=args.dry_run)
        verb = "would fill" if args.dry_run else "filled"
        print(f"  {f.name}: {verb} {filled} members, {found} of the expected channels found")
    print(f"{'checked' if args.dry_run else 'reattached'} {len(preds)} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
