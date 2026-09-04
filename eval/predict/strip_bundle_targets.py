"""Copy an input bundle without its high-resolution target fields.

A truth-aware bundle carries the O1280 truth in ``target_hres_*`` variables. For
prediction those fields are dead weight: the model never reads them, and every
evaluator runs at ECMWF where the original bundle still lives. On the
o320->o1280 lane the targets are about 88% of a 2.06 GB bundle, so stripping
them is what makes it practical to ship inputs to a remote machine.

The stripped bundle is only valid for prediction, and prediction must then be
run with ``--allow-missing-target-unsafe``, which writes an all-NaN ``y`` and
stamps the output so a truth-free prediction cannot be mistaken for a scored
one. Score against predictions produced from the original bundle instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import netCDF4

TARGET_PREFIXES = ("target_", "out_hres_", "y_hres_")
# Copy in slabs so a 6.6M-point field never lands in memory whole.
_SLAB = 1_000_000


def _is_target(name: str) -> bool:
    return name.startswith(TARGET_PREFIXES)


def strip_bundle(src: Path, dst: Path, *, overwrite: bool = False) -> tuple[int, int, list[str]]:
    """Write ``src`` to ``dst`` minus target fields; return (src bytes, dst bytes, dropped)."""
    if dst.exists() and not overwrite:
        raise SystemExit(f"refusing to overwrite {dst} (pass --overwrite)")

    with netCDF4.Dataset(src) as ds_in, netCDF4.Dataset(dst, "w", format=ds_in.data_model) as ds_out:
        dropped = sorted(v for v in ds_in.variables if _is_target(v))
        keep = [v for v in ds_in.variables if not _is_target(v)]

        # Only carry over dimensions the kept variables actually use, so an
        # orphaned target_level does not survive.
        used = {d for v in keep for d in ds_in.variables[v].dimensions}
        for name, dim in ds_in.dimensions.items():
            if name in used:
                ds_out.createDimension(name, None if dim.isunlimited() else len(dim))

        ds_out.setncatts({k: ds_in.getncattr(k) for k in ds_in.ncattrs()})
        ds_out.setncattr(
            "target_fields_stripped",
            "target_*/out_hres_*/y_hres_* removed for transfer; "
            "predict with --allow-missing-target-unsafe and score at the source",
        )

        for name in keep:
            v_in = ds_in.variables[name]
            v_out = ds_out.createVariable(
                name,
                v_in.datatype,
                v_in.dimensions,
                fill_value=getattr(v_in, "_FillValue", None),
            )
            v_out.setncatts({k: v_in.getncattr(k) for k in v_in.ncattrs() if k != "_FillValue"})
            if v_in.ndim == 0:
                v_out[...] = v_in[...]
                continue
            # Slab along the last axis, which is the grid dimension in these bundles.
            n = v_in.shape[-1]
            for start in range(0, n, _SLAB):
                stop = min(start + _SLAB, n)
                v_out[..., start:stop] = v_in[..., start:stop]

    return src.stat().st_size, dst.stat().st_size, dropped


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src-dir", required=True, help="directory of *_input_bundle.nc files")
    p.add_argument("--dst-dir", required=True, help="where to write the stripped copies")
    p.add_argument("--pattern", default="*_input_bundle.nc")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    src_dir, dst_dir = Path(args.src_dir), Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(src_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"no files matching {args.pattern} in {src_dir}")

    tot_in = tot_out = 0
    for f in files:
        b_in, b_out, dropped = strip_bundle(f, dst_dir / f.name, overwrite=args.overwrite)
        tot_in += b_in
        tot_out += b_out
        print(f"  {f.name}: {b_in/1e9:.3f} -> {b_out/1e9:.3f} GB  ({len(dropped)} fields dropped)")
    print(
        f"total: {tot_in/1e9:.2f} -> {tot_out/1e9:.2f} GB "
        f"({100*tot_out/tot_in:.1f}% of original, {len(files)} files)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
