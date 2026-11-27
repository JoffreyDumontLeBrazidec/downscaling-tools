#!/usr/bin/env python3
"""git_hash patch of the installed nvidia-physicsnemo 2.2.2 (physicsnemo/core/version_check.py,
get_physicsnemo_pkg_info): git_hash = None on ANY git failure, not only InvalidGitRepositoryError. Without it the first
checkpoint save on a node without a git binary dies with GitCommandNotFound (Jupiter segments 3023870/3033020/3038907,
21 Nov; baselines note, section "What is compared"). Same patch as the Jupiter venv (backup .bak_20261121_git there).
Idempotent; keeps a backup <file>.bak_20261127_git. Run with the venv's python:
    $CD_VENV/bin/python cd_patch_version_check.py
"""
import shutil
import sys
from importlib.util import find_spec
from pathlib import Path

OLD = """    import git

    version = get_installed_version("nvidia-physicsnemo")

    try:
        git_hash = git.Repo(search_parent_directories=True).head.commit.hexsha
    except git.InvalidGitRepositoryError:
        git_hash = None
"""
NEW = """    version = get_installed_version("nvidia-physicsnemo")

    try:  # patched (CorrDiff kit, cd_patch_version_check.py): no git binary on compute nodes
        import git

        git_hash = git.Repo(search_parent_directories=True).head.commit.hexsha
    except Exception:
        git_hash = None
"""


def main():
    spec = find_spec("physicsnemo")
    f = Path(spec.origin).parent / "core" / "version_check.py"
    s = f.read_text()
    if NEW in s:
        print(f"PATCH_PRESENT {f}")
        return 0
    if s.count(OLD) != 1:
        print(f"PATCH_FAIL {f}: original block not found once")
        return 1
    shutil.copy2(f, str(f) + ".bak_20261127_git")
    f.write_text(s.replace(OLD, NEW))
    print(f"PATCH_APPLIED {f} (backup {f}.bak_20261127_git)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
