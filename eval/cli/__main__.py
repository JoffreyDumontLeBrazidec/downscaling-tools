"""Lets ``python -m eval.cli`` run the command line."""
import importlib.util
import os

# Rule 47: refuse the old Triton kernel even when the repo is not on PYTHONPATH (sitecustomize not loaded).
_kg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools", "chain", "kernel_guard.py")
_spec = importlib.util.spec_from_file_location("_ds_kernel_guard_cli", _kg)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_mod.install_import_hook()

from eval.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
