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


def test_cmd_predict_prepml_sets_bundle_dir_as_input_root(tmp_path):
    from unittest.mock import patch

    from eval import cli as eval_cli

    bundles = tmp_path / "bundles_with_y"
    bundles.mkdir()
    lane_config = {
        "predict": {
            "members": [1],
            "steps": [24],
            "dates": ["20250926"],
            "num_gpus_per_model": 4,
        },
        "prepare": {"bundle_filename_tpl": "bundle.nc"},
        "prepml": {"debug_expvers": ["test"]},
    }
    host_config = {"environment_setup": {"exports": {}}}
    args = type("Args", (), {})()
    args.checkpoint = "/tmp/fake.ckpt"
    args.source_grib_root = None
    args.bundle_dir = str(bundles)
    args.mode = "prepml"
    args.expver = "test"
    args.prepml_runner = None
    captured = {}

    def fake_prepml_predict(**kwargs):
        captured.update(kwargs)

    with patch("eval.predict.prepml.prepml_predict", side_effect=fake_prepml_predict):
        eval_cli.cmd_predict(args, lane_config, host_config, tmp_path)

    assert captured["lane_config"]["predict"]["input_root"] == str(bundles)


def test_cmd_predict_prepml_requires_bundle_input(tmp_path):
    from eval import cli as eval_cli

    lane_config = {
        "predict": {
            "members": [1],
            "steps": [24],
            "dates": ["20250926"],
            "num_gpus_per_model": 4,
        },
        "prepare": {"bundle_filename_tpl": "bundle.nc"},
        "prepml": {"debug_expvers": ["test"]},
    }
    host_config = {"environment_setup": {"exports": {"DATA_DIR": "/data/raw"}}}
    args = type("Args", (), {})()
    args.checkpoint = "/tmp/fake.ckpt"
    args.source_grib_root = None
    args.bundle_dir = None
    args.mode = "prepml"
    args.expver = "test"
    args.prepml_runner = None

    with pytest.raises(SystemExit, match="requires truth-aware bundles"):
        eval_cli.cmd_predict(args, lane_config, host_config, tmp_path)
