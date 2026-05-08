"""eval.predict — Modular prediction generation from input bundles.

Replaces generate_predictions_25_files.py with clean module boundaries.
See README.md for usage.
"""

from .bundle_manager import BUNDLE_RE, date_str_to_datetime64, discover_bundles, parse_bundle_key, resolve_date_step_pairs
from .dataset_builder import build_prediction_dataset, validate_predictions_dataset
from .distributed_io import Rank0FileWriter, _destroy_process_group, _distributed_barrier
from .inference_engine import predict_ensemble_members, predict_single_bundle
from .main import create_parser, main
from .mars_retrieve import (
    assemble_predictions_file,
    build_prediction_request,
    group_weather_states_for_mars,
    retrieve_predictions_from_mars,
    weather_state_to_mars,
)
from .model_loader import load_inference_model, setup_distributed
from .output_writer import prediction_output_path, write_predictions_file
from .prepml import prepml_predict, resolve_expver
from .prepml_config import generate_prepml_config, write_prepml_config
from .types import BundleKey, EnsemblePrediction, PredictionConfig, PredictionMetadata, PredictionResult

__all__ = [
    "BUNDLE_RE",
    "BundleKey",
    "EnsemblePrediction",
    "PredictionConfig",
    "PredictionMetadata",
    "PredictionResult",
    "Rank0FileWriter",
    "_destroy_process_group",
    "_distributed_barrier",
    "assemble_predictions_file",
    "build_prediction_dataset",
    "build_prediction_request",
    "create_parser",
    "date_str_to_datetime64",
    "discover_bundles",
    "generate_prepml_config",
    "group_weather_states_for_mars",
    "load_inference_model",
    "main",
    "parse_bundle_key",
    "predict_ensemble_members",
    "predict_single_bundle",
    "prediction_output_path",
    "prepml_predict",
    "resolve_date_step_pairs",
    "resolve_expver",
    "retrieve_predictions_from_mars",
    "setup_distributed",
    "validate_predictions_dataset",
    "weather_state_to_mars",
    "write_prepml_config",
    "write_predictions_file",
]
