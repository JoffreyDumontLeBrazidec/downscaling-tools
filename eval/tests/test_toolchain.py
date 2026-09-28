"""Tests for the single toolchain definition.

These run anywhere: they check what is rendered, not what is installed.
"""
from __future__ import annotations

import pytest

from eval._backends.env import toolchain


def test_both_spectra_toolchains_are_defined() -> None:
    assert toolchain.available() == ["gptosp", "metview"]


def test_unknown_toolchain_names_the_defined_ones() -> None:
    with pytest.raises(KeyError, match="gptosp, metview"):
        toolchain.recipe("nope")


@pytest.mark.parametrize("name", ["gptosp", "metview"])
def test_module_loads_are_not_silenced(name: str) -> None:
    """A failing `module load` must stop the script, not be swallowed.

    Callers used to write `module load X 2>/dev/null || true`, which hid the
    reason a module failed and let the failure surface much later as a
    confusing downstream error.  `module unload` keeps its guard, because
    unloading something that was never loaded is genuinely harmless.
    """
    for line in toolchain.render_module_block(name).splitlines():
        if line.startswith("module load "):
            assert "2>/dev/null" not in line, line
            assert "|| true" not in line, line


@pytest.mark.parametrize(
    ("name", "probe"),
    [("gptosp", "gptosp.ser"), ("metview", "metview")],
)
def test_block_ends_in_a_positive_probe(name: str, probe: str) -> None:
    block = toolchain.render_module_block(name)
    last = block.splitlines()[-1]
    assert last.startswith(f"command -v {probe} >/dev/null")
    assert "exit 1" in last
    # The hostname must be left for the shell: the block is rendered on one host
    # and often runs on another.
    assert '"$(hostname)"' in last


def test_metview_toolbox_version_is_pinned() -> None:
    """An unpinned ecmwf-toolbox lets the Metview version drift under cached spectra."""
    block = toolchain.render_module_block("metview")
    assert "module load ecmwf-toolbox/" in block
    assert "module load ecmwf-toolbox\n" not in block


def test_metview_block_carries_its_startup_workarounds() -> None:
    """Without these, `import metview` hangs on node-local /tmp or times out."""
    block = toolchain.render_module_block("metview")
    assert "METVIEW_TMPDIR" in block
    assert "METVIEW_PYTHON_START_TIMEOUT" in block


def test_gptosp_unloads_the_toolbox_first() -> None:
    """eclib/pifsenv/ifs and ecmwf-toolbox conflict, so the order matters."""
    lines = toolchain.render_module_block("gptosp").splitlines()
    unload = next(i for i, ln in enumerate(lines) if ln.startswith("module unload ecmwf-toolbox"))
    load_ifs = next(i for i, ln in enumerate(lines) if ln == "module load ifs")
    assert unload < load_ifs
