"""Tests for PrepML CLI integration."""
from __future__ import annotations

import pytest


def test_parser_accepts_mode_prepml():
    from eval.cli import build_parser
    parser = build_parser()
    args = parser.parse_args([
        "predict", "--mode", "prepml", "--checkpoint", "/path/ckpt",
        "--lane", "o96_o320", "--expver", "j2pw",
    ])
    assert args.mode == "prepml"
    assert args.expver == "j2pw"


def test_parser_mode_defaults_to_manual():
    from eval.cli import build_parser
    parser = build_parser()
    args = parser.parse_args([
        "predict", "--checkpoint", "/path/ckpt", "--lane", "o96_o320",
    ])
    assert args.mode == "manual"
    assert args.expver is None


def test_parser_run_accepts_mode():
    from eval.cli import build_parser
    parser = build_parser()
    args = parser.parse_args([
        "run", "--mode", "prepml", "--checkpoint", "/path/ckpt",
        "--lane", "o96_o320", "--expver", "test",
    ])
    assert args.mode == "prepml"


def test_parser_invalid_mode_rejected():
    from eval.cli import build_parser
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "predict", "--mode", "invalid", "--checkpoint", "/x",
            "--lane", "o96_o320",
        ])
