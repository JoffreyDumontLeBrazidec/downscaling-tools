import re
from pathlib import Path

from conftest import KIT_DIR
from o2560_common import KIT_ROOT, OUT_ROOT

FDB = re.compile(r"grib2fdb|fdb_write|pyfdb|fdb-write|fdb write|eval\.predict\.fdb")
# the kit's own code only: on Jupiter the folder also holds the anemoi-core worktree (code/) and downscaling-tools (dstools/)
FILES = [p for d in ("kit", "scripts") for p in (KIT_DIR / d).rglob("*") if p.is_file() and p.suffix in (".py", ".sh", ".sbatch")]


def test_kit_files_found():
    assert len(FILES) >= 10, FILES


def test_no_fdb_call_anywhere():
    hits = [(str(p), i + 1, ln) for p in FILES for i, ln in enumerate(p.read_text().splitlines()) if FDB.search(ln)]
    assert not hits, hits


def test_outputs_not_under_home():
    for root in (KIT_ROOT, OUT_ROOT):
        assert root.startswith("/e/data1/") and not root.startswith(("/home", "/e/home"))
    pat = re.compile(r"(?:\bOUT|OUTDIR|OUTPUT|EVAL_ROOT|--output)=?\s*=?\s*([^\s;\"']+)")
    for p in FILES:
        for ln in p.read_text().splitlines():
            for m in pat.finditer(ln):
                v = m.group(1)
                if v.startswith("/"):
                    assert not v.startswith(("/home", "/e/home")), (str(p), ln)


def test_no_cancel_or_delete_commands():
    bad = re.compile(r"\bscancel\b|\brm\s+-|\bshutil\.rmtree\b|\bos\.remove\(")
    hits = [(str(p), ln) for p in FILES for ln in p.read_text().splitlines() if bad.search(ln) and not ln.lstrip().startswith("#")]
    assert not hits, hits
