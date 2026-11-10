#!/usr/bin/env python3
"""Refuse to train or read from a zarr store that has lost chunk files.

Why (2026-11-09): the Atos scratch purge deletes each file about 30 days after its last access. Inside a zarr
store that removes single row chunks, and with fill value 0.0 a missing row reads back as zeros without any
error. The E2-v4 box stores lost 353 row chunks per store, the E2 v3 forcings store 479, plus small metadata
arrays (count/, has_nans/) that are only ever opened. A job reading such a store trains or scores on zeros.

The guard checks every array of every store a job names: `.zarray` present, and the number of chunk files equal
to the number the array's shape and chunks imply. It reads only directory listings and the small `.zarray`/
`.zgroup` files, never a chunk, so it does not refresh the access time of the data.

    store_guard.py check STORE.zarr [STORE.zarr ...]      # exit 0 if complete, 49 if not
    store_guard.py check --config rendered.yaml           # every `dataset: <...>.zarr` path in a config
    store_guard.py missing STORE.zarr [--array data]      # print the missing chunk row indices (repair lists)

`STORE_GUARD_ALLOW_INCOMPLETE=1` turns a refusal into a warning (exit 0), for timing smokes only; the report is
printed either way. Python 3.6 compatible (the Atos login nodes), stdlib only.
"""
import argparse
import json
import os
import re
import sys

EXIT_INCOMPLETE = 49
ALLOW_ENV = "STORE_GUARD_ALLOW_INCOMPLETE"
DATASET_RE = re.compile(r"""^\s*-?\s*dataset\s*:\s*['"]?([^'"\s#]+\.zarr)/?['"]?\s*(#.*)?$""")


def _ceil_div(a, b):
    return -(-a // b)


def expected_chunks(meta):
    """Number of chunk files and the grid of chunk counts per dimension, from a .zarray dict."""
    shape, chunks = meta["shape"], meta["chunks"]
    if not shape:  # a 0-d array is stored as one chunk "0"
        return 1, []
    grid = [_ceil_div(s, c) if s else 0 for s, c in zip(shape, chunks)]
    n = 1
    for g in grid:
        n *= g
    return n, grid


def _chunk_keys(array_dir, sep):
    """Chunk keys present in an array folder (dot-files and rsync temp files are not chunks)."""
    key_re = re.compile(r"^\d+(\.\d+)*$")
    keys = set()
    if sep == "/":
        for dirpath, dirnames, filenames in os.walk(array_dir):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            rel = os.path.relpath(dirpath, array_dir)
            for f in filenames:
                if f.startswith("."):
                    continue
                key = f if rel == "." else rel.replace(os.sep, ".") + "." + f
                if key_re.match(key):
                    keys.add(key)
    else:
        for entry in os.scandir(array_dir):
            if not entry.name.startswith(".") and key_re.match(entry.name):
                keys.add(entry.name)
    return keys


def _arrays(store):
    """Every array folder in a zarr v2 group, by the folders that are expected to hold one.

    A folder counts as an array if it holds a `.zarray`, or if it is a direct child of a group (a folder with
    `.zgroup`) and holds no `.zgroup` itself: a purged array keeps its folder but can lose its `.zarray`.
    """
    found = []
    for dirpath, dirnames, filenames in os.walk(store):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        is_group = ".zgroup" in filenames
        if ".zarray" in filenames:
            found.append(dirpath)
            dirnames[:] = []  # chunk sub-folders (dimension_separator "/") are not arrays
            continue
        if is_group:
            for d in dirnames:
                sub = os.path.join(dirpath, d)
                try:
                    names = set(os.listdir(sub))
                except OSError:
                    continue
                if ".zarray" not in names and ".zgroup" not in names:
                    found.append(sub)  # an array that lost its .zarray (or an empty folder)
    return sorted(set(found))


def check_store(store):
    """Return (problems, summary lines) for one store."""
    problems, lines = [], []
    if not os.path.isdir(store):
        return ["%s: not a directory" % store], lines
    if not os.path.isfile(os.path.join(store, ".zgroup")):
        problems.append("%s: no .zgroup" % store)
    for arr in _arrays(store):
        name = os.path.relpath(arr, store)
        zpath = os.path.join(arr, ".zarray")
        if not os.path.isfile(zpath):
            problems.append("%s/%s: .zarray missing" % (os.path.basename(store), name))
            continue
        with open(zpath) as fh:
            meta = json.load(fh)
        n_exp, _ = expected_chunks(meta)
        n_have = len(_chunk_keys(arr, meta.get("dimension_separator") or "."))
        lines.append("%s/%s: %d/%d chunks" % (os.path.basename(store), name, n_have, n_exp))
        if n_have != n_exp:
            problems.append("%s/%s: %d of %d chunk files present (%d missing)"
                            % (os.path.basename(store), name, n_have, n_exp, n_exp - n_have))
    return problems, lines


def missing_rows(store, array="data"):
    """First-axis indices with at least one missing chunk in one array (assumes one row per chunk on axis 0)."""
    arr = os.path.join(store, array)
    with open(os.path.join(arr, ".zarray")) as fh:
        meta = json.load(fh)
    sep = meta.get("dimension_separator") or "."
    _, grid = expected_chunks(meta)
    per_row = 1
    for g in grid[1:]:
        per_row *= g
    counts = {}
    for key in _chunk_keys(arr, sep):
        i = int(key.split(".")[0])
        counts[i] = counts.get(i, 0) + 1
    step = meta["chunks"][0]
    return [i * step for i in range(grid[0]) if counts.get(i, 0) != per_row]


def stores_from_config(path):
    """Every `dataset: <path>.zarr` in a YAML config (rendered or not), in order, without duplicates."""
    out = []
    with open(path) as fh:
        for line in fh:
            m = DATASET_RE.match(line)
            if m and m.group(1) not in out:
                out.append(m.group(1))
    return out


def _ranges(idx):
    out, start, prev = [], None, None
    for i in idx:
        if start is None:
            start = prev = i
        elif i == prev + 1:
            prev = i
        else:
            out.append(str(start) if start == prev else "%d-%d" % (start, prev))
            start = prev = i
    if start is not None:
        out.append(str(start) if start == prev else "%d-%d" % (start, prev))
    return " ".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd")
    c = sub.add_parser("check", help="exit 49 if any named store lost chunk or metadata files")
    c.add_argument("stores", nargs="*")
    c.add_argument("--config", action="append", default=[], help="take the stores from a YAML config")
    c.add_argument("--quiet", action="store_true", help="print only problems and the verdict")
    m = sub.add_parser("missing", help="print the missing row indices of one array")
    m.add_argument("store")
    m.add_argument("--array", default="data")
    m.add_argument("--ranges", action="store_true", help="compact ranges instead of one index per line")
    a = p.parse_args(argv)

    if a.cmd == "missing":
        rows = missing_rows(a.store, a.array)
        if a.ranges:
            print("%d missing: %s" % (len(rows), _ranges(rows)))
        else:
            for r in rows:
                print(r)
        return 0

    if a.cmd != "check":
        p.print_help()
        return 2
    stores = list(a.stores)
    for cfg in a.config:
        stores += [s for s in stores_from_config(cfg) if s not in stores]
    if not stores:
        print("[store_guard] no store named (no `dataset: ...zarr` in the config?)", file=sys.stderr)
        return 2
    problems = []
    for s in stores:
        probs, lines = check_store(s)
        problems += probs
        if not a.quiet:
            for ln in lines:
                print("[store_guard] %s" % ln)
    if not problems:
        print("[store_guard] STORES_COMPLETE: %d store(s)" % len(stores))
        return 0
    for pr in problems:
        print("[store_guard] INCOMPLETE %s" % pr, file=sys.stderr)
    if os.environ.get(ALLOW_ENV) == "1":
        print("[store_guard] WARNING: %s=1, continuing on an incomplete store (missing rows read as the fill value)"
              % ALLOW_ENV, file=sys.stderr)
        return 0
    print("[store_guard] REFUSED: %d problem(s); repair the store or set %s=1 for a timing smoke"
          % (len(problems), ALLOW_ENV), file=sys.stderr)
    return EXIT_INCOMPLETE


if __name__ == "__main__":
    sys.exit(main())
