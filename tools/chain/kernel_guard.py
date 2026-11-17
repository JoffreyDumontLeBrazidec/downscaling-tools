#!/usr/bin/env python3
"""Refuse the old (Dec-2025) Triton GraphTransformer kernel in every Python process.

Background: the anemoi-models Triton kernel `anemoi/models/triton/gt.py` before upstream bbca1748 ("Triton GT
stability", #1172, 11 Jun 2026) saves the forward output and D = <d_out, out> in bf16; its query gradient is
wrong (probe note s16/s20: 64/128 sign agreement, corr -0.16, 2.6x too large) and drives the attention blow-up.
Owner rule 26 (22 Oct 2026) and rule 47 (17 Nov 2026): no job ever imports it again.

The check is a hash ALLOWLIST: the git blob sha1 of the gt.py file that Python would import must be one of
ALLOWED_GT_BLOBS (all carry `out_saved` in fp32 and D in fp32). Any other file - an old kernel, a hand edit, an
unknown port - stops the process. A new fixed variant is added here only after its blob passes the marker
check (`kernel_guard.py classify FILE`) and the one-hot bf16 gradient test (anemoi-core
models/tests/models/test_encoder_trap.py).

Three ways to use it (stdlib only, Python >= 3.6):

  python kernel_guard.py check            # job start: exit 0 if the importable gt.py is fixed (or absent), 97 if not
  python kernel_guard.py file PATH...     # hash and classify files (inventory)
  python kernel_guard.py install-hook SITE_PACKAGES_DIR
                                          # every later interpreter of that venv refuses an old kernel at import
  import kernel_guard; kernel_guard.install_import_hook()   # in-process (chain_guard_launch.py, eval.cli)

The import hook runs before gt.py executes. On refusal it prints one line to stderr and calls os._exit(97):
not an ImportError, because anemoi falls back to another attention backend on ImportError, and a silent backend
switch is as bad as the old kernel.
"""
import hashlib
import os
import sys

TARGET = "anemoi.models.triton.gt"
EXIT_CODE = 97

# git blob sha1 of models/src/anemoi/models/triton/gt.py -> where it comes from (anemoi-core fork, 17 Nov 2026)
ALLOWED_GT_BLOBS = {
    "88b036167a14068cc18b2a91f12151b1aff495fa": "ds-trunk / f1659385f port of bbca1748 on the encoder-trap line (softcap)",
    "5f765a15fbdb8d798eb0f0c2a652f57a17c4869e": "upstream bbca1748 on the Dec-2025 base (2791a56f + fix)",
    "7efd61a7ba2fa28c9c2fb7b8bb8585deebbfa8bb": "upstream main after bbca1748 (2025-2026 licence header)",
    "ef054ed33e132235cbf2505b3563d6d0be7ba287": "upstream ec11bf28 torch.compile custom-op wrapper + bbca1748",
    "fc35ccd53c54b2a74df4544dfb8b29ce175a85f7": "8a0906788 cosine-attention line (KEYNORM) + bbca1748",
}

# known OLD blobs (for clearer messages only; the allowlist decides)
KNOWN_OLD_GT_BLOBS = {
    "2791a56f548deae860e2120e87c778f8f70e450a": "Dec-2025 kernel (0608acf0f), pristine / active / most old branches",
    "0e83c63b1aa2be2d7d0d0f26f0627ebf868d8b70": "Oct-2025 kernel, Jupiter branch ds 2553fb109",
    "3e8bedce82deacdd8d33c4756f680d69b5bae1e7": "0ea80441a encoder-trap options (softcap), no fix",
    "765fa8bc826fbc29551b9991b05465fa910b42a2": "765abee01 cosine attention, no fix",
    "4b6e6f7cae2614589d0749a1117670d6b2d9cb9c": "Nov-2025 kernel (b40b6c610)",
    "2f40684c726bfbf10806c242f00f3df4652db4f3": "Dec-2025 kernel (00b38c9fd)",
    "c4a994e7cb26c6cf7f99b4dfb086e5ba891d1afe": "Jan-2026 kernel (60cde78c8)",
    "6360f3c1a3b1bf5b09cb65effddeeaec9ecdefe2": "Oct-2025 kernel (0a5d19243)",
    "5da8ec80378237f7a170fdc58c8e5860ba1efc3c": "Mar-2026 kernel (cf59b7c1f)",
}


def blob_sha1(path):
    """git hash-object of a file (what `git rev-parse HEAD:path` prints for the same content)."""
    with open(path, "rb") as fh:
        data = fh.read()
    h = hashlib.sha1()
    h.update(b"blob %d\0" % len(data))
    h.update(data)
    return h.hexdigest()


def markers(path):
    """(has fp32 saved output, has fp32 D): the two numeric changes of bbca1748."""
    with open(path, "r", errors="replace") as fh:
        src = fh.read()
    out_saved = "out_saved" in src and "dtype=torch.float32" in src
    d_fp32 = "D = torch.empty((N_dst, H), device=q.device, dtype=torch.float32)" in src
    return out_saved, d_fp32


def classify(path):
    b = blob_sha1(path)
    if b in ALLOWED_GT_BLOBS:
        return b, "FIXED", ALLOWED_GT_BLOBS[b]
    if b in KNOWN_OLD_GT_BLOBS:
        return b, "OLD", KNOWN_OLD_GT_BLOBS[b]
    o, d = markers(path)
    if o and d:
        return b, "UNLISTED-FIX-MARKERS", "has both bbca1748 markers but is not on the allowlist (refused until added)"
    return b, "OLD", "unknown blob without the bbca1748 markers (out_saved fp32: %s, D fp32: %s)" % (o, d)


def _refuse(path, blob, why):
    msg = ("[kernel-guard] REFUSED: {} resolves to {} (git blob {}: {}). Only the fixed Triton kernel may run "
           "(rule 47; allowlist in downscaling-tools tools/chain/kernel_guard.py). Put a checkout with the "
           "fixed gt.py (ds-trunk) first on PYTHONPATH, or patch this file. pid {} host {} job {}\n").format(
        TARGET, path, blob, why, os.getpid(), os.uname()[1], os.environ.get("SLURM_JOB_ID", "-"))
    try:
        sys.stderr.write(msg)
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(EXIT_CODE)


def verify_path(path):
    """Return the blob if allowed, otherwise stop the process."""
    if not path or not os.path.isfile(path):
        _refuse(path, "-", "no source file (sourceless or namespace module)")
    blob = blob_sha1(path)
    if blob not in ALLOWED_GT_BLOBS:
        _refuse(path, blob, classify(path)[2])
    return blob


class _GtGuardFinder(object):
    """sys.meta_path entry: resolves anemoi.models.triton.gt through the other finders and checks the file
    before the module body runs."""

    _ds_kernel_guard = True

    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        spec = None
        for finder in list(sys.meta_path):
            if getattr(finder, "_ds_kernel_guard", False):
                continue
            fs = getattr(finder, "find_spec", None)
            if fs is None:
                continue
            try:
                spec = fs(fullname, path, target)
            except TypeError:
                spec = fs(fullname, path)
            if spec is not None:
                break
        if spec is None:
            return None
        verify_path(spec.origin)
        if os.environ.get("KERNEL_GUARD_VERBOSE") and os.environ.get("SLURM_PROCID", "0") == "0":
            sys.stderr.write("[kernel-guard] ok: {} = {}\n".format(TARGET, spec.origin))
        return spec

    # Python < 3.4 protocol, never used here
    def find_module(self, fullname, path=None):
        return None


def install_import_hook():
    """Idempotent. Also checks a gt module that is already imported."""
    mod = sys.modules.get(TARGET)
    if mod is not None:
        verify_path(getattr(mod, "__file__", None))
    if not any(getattr(f, "_ds_kernel_guard", False) for f in sys.meta_path):
        sys.meta_path.insert(0, _GtGuardFinder())


def importable_gt():
    """Path of the gt.py this interpreter would import, or None if anemoi.models is not installed."""
    import importlib.util
    try:
        spec = importlib.util.find_spec(TARGET)
    except ModuleNotFoundError:
        return None
    except ImportError as exc:  # parent package present but broken: treat as a failure, not as absent
        sys.stderr.write("[kernel-guard] cannot resolve {}: {}\n".format(TARGET, exc))
        os._exit(EXIT_CODE)
    return spec.origin if spec is not None else None


def check():
    """Job-start check in the job's own interpreter (same venv, same PYTHONPATH)."""
    install_import_hook()
    path = importable_gt()
    if path is None:
        sys.stdout.write("[kernel-guard] ok: anemoi.models not importable here; nothing to check\n")
        return 0
    blob = verify_path(path)
    sys.stdout.write("[kernel-guard] ok: {} = {} (fixed, blob {})\n".format(TARGET, path, blob[:12]))
    return 0


HOOK_MODULE = "_ds_kernel_guard"
PTH_NAME = "_ds_kernel_guard.pth"


def install_hook_into(site_dir):
    """Copy this file into a venv's site-packages as _ds_kernel_guard.py and add a .pth that arms it at start-up."""
    import shutil
    if not os.path.isdir(site_dir):
        raise SystemExit("not a directory: {}".format(site_dir))
    dst = os.path.join(site_dir, HOOK_MODULE + ".py")
    shutil.copyfile(os.path.abspath(__file__), dst)
    with open(os.path.join(site_dir, PTH_NAME), "w") as fh:
        # one line, executed by site.py at every interpreter start of this venv (not with python -S)
        fh.write("import _ds_kernel_guard; _ds_kernel_guard.install_import_hook()\n")
    sys.stdout.write("[kernel-guard] hook installed in {} (blob of guard {})\n".format(site_dir, blob_sha1(dst)[:12]))
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        sys.stdout.write(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "check":
        return check()
    if cmd == "file":
        bad = 0
        for p in rest:
            b, state, why = classify(p)
            sys.stdout.write("{}\t{}\t{}\t{}\n".format(state, b, p, why))
            bad += state != "FIXED"
        return 1 if bad else 0
    if cmd == "install-hook":
        for d in rest:
            install_hook_into(d)
        return 0
    raise SystemExit("unknown command {!r}; see --help".format(cmd))


if __name__ == "__main__":
    sys.exit(main())
