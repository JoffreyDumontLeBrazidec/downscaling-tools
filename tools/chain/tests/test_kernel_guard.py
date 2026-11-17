"""kernel_guard: the old Triton kernel stops the process at import; the fixed one passes; ImportError fallbacks
cannot swallow the refusal; the job-start check reads what the interpreter would import."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GUARD = HERE.parent / "kernel_guard.py"
sys.path.insert(0, str(HERE.parent))
import kernel_guard  # noqa: E402

OLD = HERE / "data" / "gt_old_2791a56f.py.txt"
FIXED = HERE / "data" / "gt_fixed_88b03616.py.txt"


def _tree(root, src):
    """root/anemoi/models/triton/gt.py with the given source; the kernel body is never executed by the tests
    that refuse it, and the fixed one is replaced by a stub that keeps the same bytes check (imports stubbed)."""
    pkg = root / "anemoi" / "models" / "triton"
    pkg.mkdir(parents=True)
    (root / "anemoi" / "models" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    shutil.copyfile(src, pkg / "gt.py")
    return pkg / "gt.py"


def _stub_triton(root):
    """Minimal torch/triton stand-ins so the fixed gt.py can execute its module body on a CPU box."""
    stub = root / "stubs"
    (stub / "triton").mkdir(parents=True)
    (stub / "triton" / "__init__.py").write_text(
        "def jit(f=None, **k):\n    return f if f else (lambda g: g)\nclass _L:\n    def __getattr__(self, n):\n"
        "        return _L()\n    def __call__(self, *a, **k):\n        return _L()\n")
    (stub / "triton" / "language.py").write_text(
        "class constexpr:\n    def __init__(self, v=None):\n        self.v = v\nfloat16 = bfloat16 = float32 = object()\n"
        "def __getattr__(n):\n    return object()\n")
    (stub / "torch").mkdir()
    (stub / "torch" / "__init__.py").write_text(
        "class _F:\n    pass\nclass autograd:\n    Function = _F\nfloat16 = bfloat16 = float32 = object()\n"
        "def __getattr__(n):\n    return object()\n")
    return stub


def _run(code, *paths, env_extra=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in paths)
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)


def test_blob_matches_git():
    assert kernel_guard.blob_sha1(str(OLD)) == "2791a56f548deae860e2120e87c778f8f70e450a"
    assert kernel_guard.blob_sha1(str(FIXED)) == "88b036167a14068cc18b2a91f12151b1aff495fa"
    assert kernel_guard.classify(str(OLD))[1] == "OLD"
    assert kernel_guard.classify(str(FIXED))[1] == "FIXED"


def test_hook_refuses_old_kernel_even_inside_except_importerror(tmp_path):
    _tree(tmp_path / "old", OLD)
    code = ("import kernel_guard; kernel_guard.install_import_hook()\n"
            "try:\n    import anemoi.models.triton.gt\nexcept ImportError:\n    print('FELL BACK')\n"
            "except Exception:\n    print('CAUGHT')\nprint('SURVIVED')\n")
    r = _run(code, GUARD.parent, tmp_path / "old")
    assert r.returncode == 97, (r.stdout, r.stderr)
    assert "REFUSED" in r.stderr and "2791a56f" in r.stderr
    assert "SURVIVED" not in r.stdout and "FELL BACK" not in r.stdout


def test_hook_lets_fixed_kernel_import(tmp_path):
    _tree(tmp_path / "fixed", FIXED)
    stub = _stub_triton(tmp_path)
    code = ("import kernel_guard; kernel_guard.install_import_hook()\n"
            "import anemoi.models.triton.gt as g; print('OK', g.__file__)\n")
    r = _run(code, GUARD.parent, tmp_path / "fixed", stub)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_pth_install_arms_every_interpreter_and_first_on_path_wins(tmp_path):
    site_dir = tmp_path / "site"
    site_dir.mkdir()
    kernel_guard.install_hook_into(str(site_dir))
    assert (site_dir / "_ds_kernel_guard.pth").exists()
    _tree(tmp_path / "old", OLD)
    _tree(tmp_path / "fixed", FIXED)
    stub = _stub_triton(tmp_path)
    arm = "import site; site.addsitedir(%r)\n" % str(site_dir)
    # old checkout first on the path: refused
    r = _run(arm + "import anemoi.models.triton.gt\nprint('SURVIVED')\n", tmp_path / "old", tmp_path / "fixed", stub)
    assert r.returncode == 97 and "SURVIVED" not in r.stdout, (r.stdout, r.stderr)
    # fixed checkout first: passes
    r = _run(arm + "import anemoi.models.triton.gt\nprint('SURVIVED')\n", tmp_path / "fixed", tmp_path / "old", stub)
    assert r.returncode == 0 and "SURVIVED" in r.stdout, r.stderr


def test_check_cli(tmp_path):
    _tree(tmp_path / "old", OLD)
    _tree(tmp_path / "fixed", FIXED)
    stub = _stub_triton(tmp_path)
    env = dict(os.environ)
    for tree, rc in (("old", 97), ("fixed", 0)):
        env["PYTHONPATH"] = os.pathsep.join([str(tmp_path / tree), str(stub)])
        r = subprocess.run([sys.executable, str(GUARD), "check"], env=env, capture_output=True, text=True)
        assert r.returncode == rc, (tree, r.stdout, r.stderr)
    # no anemoi at all: nothing to check, pass
    env["PYTHONPATH"] = str(stub)
    r = subprocess.run([sys.executable, str(GUARD), "check"], env=env, capture_output=True, text=True)
    assert r.returncode == 0 and "nothing to check" in r.stdout


def test_already_imported_old_module_is_caught(tmp_path):
    _tree(tmp_path / "old", OLD)
    stub = _stub_triton(tmp_path)
    # import the old module first with no guard (needs the stubs to execute), then arm: refused at arming
    code = "import anemoi.models.triton.gt\nimport kernel_guard; kernel_guard.install_import_hook()\nprint('SURVIVED')\n"
    r = _run(code, GUARD.parent, tmp_path / "old", stub)
    assert r.returncode == 97 and "SURVIVED" not in r.stdout, (r.stdout, r.stderr)
