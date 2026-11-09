"""store_guard.py on synthetic zarr v2 stores (no zarr needed): complete, purged chunks, purged metadata."""
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
GUARD = os.path.join(HERE, "..", "store_guard.py")
sys.path.insert(0, os.path.join(HERE, ".."))
import store_guard  # noqa: E402


def _array(path, shape, chunks, sep=None, write=True):
    os.makedirs(path)
    meta = {"shape": shape, "chunks": chunks, "zarr_format": 2, "dtype": "<f4", "fill_value": 0.0}
    if sep:
        meta["dimension_separator"] = sep
    with open(os.path.join(path, ".zarray"), "w") as fh:
        json.dump(meta, fh)
    if not write:
        return
    grid = [-(-s // c) for s, c in zip(shape, chunks)] or []
    keys = [()]
    for g in grid:
        keys = [k + (i,) for k in keys for i in range(g)]
    for k in keys:
        name = ".".join(str(i) for i in k) if k else "0"
        if sep == "/":
            p = os.path.join(path, *name.split("."))
            os.makedirs(os.path.dirname(p), exist_ok=True)
        else:
            p = os.path.join(path, name)
        open(p, "wb").close()


def _store(tmp_path, rows=12):
    s = str(tmp_path / "box.zarr")
    os.makedirs(s)
    open(os.path.join(s, ".zgroup"), "w").write('{"zarr_format": 2}')
    _array(os.path.join(s, "data"), [rows, 3, 1, 50], [1, 3, 1, 50])
    _array(os.path.join(s, "dates"), [rows], [rows])
    _array(os.path.join(s, "count"), [3], [3])
    _array(os.path.join(s, "scalar"), [], [])
    return s


def _run(*args, env=None):
    e = dict(os.environ)
    e.pop(store_guard.ALLOW_ENV, None)
    e.update(env or {})
    return subprocess.run([sys.executable, GUARD] + list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True, env=e)


def test_complete_store_passes(tmp_path):
    s = _store(tmp_path)
    r = _run("check", s)
    assert r.returncode == 0, r.stderr
    assert "STORES_COMPLETE" in r.stdout
    assert "box.zarr/data: 12/12 chunks" in r.stdout


def test_purged_rows_refuse_and_are_listed(tmp_path):
    s = _store(tmp_path)
    for i in (3, 4, 9):
        os.remove(os.path.join(s, "data", "%d.0.0.0" % i))
    open(os.path.join(s, "data", ".5.0.0.0.AbCd12"), "w").close()  # rsync temp file: not a chunk
    r = _run("check", s)
    assert r.returncode == store_guard.EXIT_INCOMPLETE
    assert "9 of 12 chunk files present (3 missing)" in r.stderr
    assert store_guard.missing_rows(s) == [3, 4, 9]
    r = _run("missing", s, "--ranges")
    assert r.stdout.strip() == "3 missing: 3-4 9"


def test_purged_metadata_array_refuses(tmp_path):
    s = _store(tmp_path)
    os.remove(os.path.join(s, "count", "0"))
    os.remove(os.path.join(s, "count", ".zarray"))  # the folder stays, as after the Atos purge
    r = _run("check", s)
    assert r.returncode == store_guard.EXIT_INCOMPLETE
    assert "count: .zarray missing" in r.stderr


def test_allow_env_downgrades_to_warning(tmp_path):
    s = _store(tmp_path)
    os.remove(os.path.join(s, "data", "0.0.0.0"))
    r = _run("check", s, env={store_guard.ALLOW_ENV: "1"})
    assert r.returncode == 0
    assert "WARNING" in r.stderr and "INCOMPLETE" in r.stderr


def test_config_paths_and_nested_separator(tmp_path):
    s = _store(tmp_path)
    s2 = str(tmp_path / "nested.zarr")
    os.makedirs(s2)
    open(os.path.join(s2, ".zgroup"), "w").write("{}")
    _array(os.path.join(s2, "data"), [4, 2], [1, 2], sep="/")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("dataloader:\n  training:\n    in_lres:\n      dataset: %s\n    out_hres:\n"
                   "      - dataset: '%s/'  # target\n    again:\n      dataset: %s\n" % (s, s2, s))
    assert store_guard.stores_from_config(str(cfg)) == [s, s2]
    assert _run("check", "--config", str(cfg)).returncode == 0
    os.remove(os.path.join(s2, "data", "2", "0"))
    assert store_guard.missing_rows(s2) == [2]
    assert _run("check", "--config", str(cfg)).returncode == store_guard.EXIT_INCOMPLETE


def test_missing_store_refuses(tmp_path):
    assert _run("check", str(tmp_path / "nope.zarr")).returncode == store_guard.EXIT_INCOMPLETE


@pytest.mark.parametrize("shape,chunks,n", [([17280, 70, 1, 814810], [1, 70, 1, 814810], 17280),
                                            ([7441, 70, 1, 30, 9], [1, 70, 1, 1, 9], 7441 * 30),
                                            ([5], [2], 3), ([], [], 1)])
def test_expected_chunks(shape, chunks, n):
    assert store_guard.expected_chunks({"shape": shape, "chunks": chunks})[0] == n
