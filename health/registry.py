"""The reference registry: one clean checkpoint per lane, replaced when a cleaner one lands (review note s8).

Every audit diffs against its lane's reference (check 12). Edit registry.yaml by hand; say in `why` what makes the
entry the cleanest on file, with the audit headline that shows it.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def load(path: str | None = None) -> dict:
    with open(path or HERE / "registry.yaml") as fh:
        return (yaml.safe_load(fh) or {}).get("lanes", {}) or {}


def reference_audit(lane: str | None) -> str | None:
    if not lane:
        return None
    e = load().get(lane)
    if not e or not e.get("audit_json"):
        return None
    return os.path.expanduser(e["audit_json"])
