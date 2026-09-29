"""Tombstone for ``python -m eval.tools.videogen``: prints where the code went and exits 1."""
import sys

print("ERROR: videogen was retired on 2026-09-29; the code is in eval/_quarantine/20260929/videogen/.",
      file=sys.stderr)
raise SystemExit(1)
