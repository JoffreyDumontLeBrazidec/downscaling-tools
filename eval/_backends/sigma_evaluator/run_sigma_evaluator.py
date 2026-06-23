from __future__ import annotations

import argparse
import gc
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from eval._backends.checkpoint_utils import infer_lane_from_config

from manual_inference.checkpoints import (
    ObjectFromCheckpointLoader,
    adapt_config_hpc,
    get_checkpoint,
    instantiate_config,
)
from manual_inference.prediction.predict import (
    _get_parallel_info,
    _init_model_comm_group,
    _resolve_device,
    _rewrite_dataset_paths_in_place,
)

from .sigma_evaluator import SigmaEvaluator
from .sigmas import sigmas

logger = logging.getLogger(__name__)


def _normalize_cfg_for_lane_inference(cfg):
    if hasattr(cfg, "model_dump"):
        cfg = cfg.model_dump()
    try:
        from omegaconf import OmegaConf  # pylint: disable=import-outside-toplevel

        if OmegaConf.is_config(cfg):
            cfg = OmegaConf.to_container(cfg, resolve=False)
    except Exception:
        pass
    return cfg


def _maybe_fix_missing_residual_statistics(cfg, fallback_name: str = "") -> Path | None:
    try:
        residual_dir = getattr(cfg.hardware.paths, "residual_statistics")
        residual_file = getattr(cfg.hardware.files, "residual_statistics")
    except AttributeError:
        return None

    if not residual_dir or not residual_file:
        return None

    current_path = Path(residual_dir) / residual_file
    if current_path.exists():
        return None

    if not fallback_name or fallback_name == residual_file:
        return None

    fallback_path = Path(residual_dir) / fallback_name
    if not fallback_path.exists():
        return None

    cfg.hardware.files.residual_statistics = fallback_name
    return fallback_path


def _destroy_process_group() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _distributed_barrier(*, device: str, local_rank: int) -> None:
    if not dist.is_available() or not dist.is_initialized():
        return
    sync_device_ids = [local_rank] if str(device).startswith("cuda") and torch.cuda.is_available() else None
    dist.barrier(device_ids=sync_device_ids)


def _inject_minimal_hardware_config(config_checkpoint, host_config) -> None:
    """Fallback when adapt_config_hpc fails (e.g. missing hardware key).

    Copies hardware paths from host_config and sets safe defaults so that
    the sigma evaluator can proceed on a single GPU.
    """
    from omegaconf import OmegaConf, DictConfig  # pylint: disable=import-outside-toplevel

    hw_dict: dict = {}
    try:
        hw_dict["paths"] = OmegaConf.to_container(host_config.hardware.paths, resolve=True)
    except Exception:
        hw_dict["paths"] = {
            "data": os.environ.get("DATA_DIR", ""),
            "grid": os.environ.get("GRID_DIR", ""),
            "residual_statistics": os.environ.get("RESIDUAL_STATISTICS_DIR", ""),
        }
    hw_dict.setdefault("num_gpus_per_model", 1)

    if isinstance(config_checkpoint, DictConfig):
        config_checkpoint.hardware = OmegaConf.create(hw_dict)
    else:
        from types import SimpleNamespace  # pylint: disable=import-outside-toplevel
        config_checkpoint.hardware = SimpleNamespace(**hw_dict)


def _set_nested_config_value(cfg, path: tuple[str, ...], value) -> None:
    if cfg is None:
        return
    try:
        from omegaconf import OmegaConf, open_dict  # pylint: disable=import-outside-toplevel

        if OmegaConf.is_config(cfg):
            with open_dict(cfg):
                node = cfg
                for part in path[:-1]:
                    child = OmegaConf.select(node, part, default=None)
                    if child is None:
                        setattr(node, part, OmegaConf.create({}))
                        child = getattr(node, part)
                    node = child
                setattr(node, path[-1], value)
            return
    except Exception:
        logger.debug("Falling back to object/dict config mutation", exc_info=True)

    node = cfg
    for part in path[:-1]:
        if isinstance(node, dict):
            node = node.setdefault(part, {})
        else:
            child = getattr(node, part, None)
            if child is None:
                child = SimpleNamespace()
                setattr(node, part, child)
            node = child
    if isinstance(node, dict):
        node[path[-1]] = value
    else:
        setattr(node, path[-1], value)


def _get_nested_config_value(cfg, path: tuple[str, ...]):
    if cfg is None:
        return None
    try:
        from omegaconf import OmegaConf  # pylint: disable=import-outside-toplevel

        if OmegaConf.is_config(cfg):
            return OmegaConf.select(cfg, ".".join(path), default=None)
    except Exception:
        logger.debug("Falling back to object/dict config lookup", exc_info=True)

    node = cfg
    for part in path:
        if isinstance(node, dict):
            node = node.get(part)
        else:
            node = getattr(node, part, None)
        if node is None:
            return None
    return node


def _apply_checkpoint_compat_profile(cfg, profile: str) -> None:
    profile = (profile or "").strip()
    if profile:
        _set_nested_config_value(cfg, ("model", "model", "compatibility_profile"), profile)


def _localize_external_checkpoint_paths(cfg) -> list[tuple[str, str]]:
    rewrites: list[tuple[str, str]] = []

    inter_mat_dir = Path(os.environ.get("INTER_MAT_DIR", "/home/ecm5702/hpcperm/data/inter_mat"))
    inter_mat_paths = (
        ("model", "residual", "in_lres", "interpolation_file_path"),
        ("system", "input", "truncation"),
        ("system", "input", "truncation_inv"),
    )
    for path in inter_mat_paths:
        current = _get_nested_config_value(cfg, path)
        if not isinstance(current, str) or not current:
            continue
        local_path = inter_mat_dir / Path(current).name
        if local_path.exists() and str(local_path) != current:
            _set_nested_config_value(cfg, path, str(local_path))
            rewrites.append((current, str(local_path)))

    return rewrites


def _get_output_name_to_index_from_data_indices(data_indices) -> dict[str, int]:
    if data_indices is None:
        return {}
    candidates = []
    if isinstance(data_indices, dict):
        candidates.extend(
            data_indices[key]
            for key in ("out_hres", "output", "target")
            if key in data_indices
        )
        candidates.extend(data_indices.values())
    else:
        candidates.append(data_indices)

    for candidate in candidates:
        try:
            name_to_index = candidate.model.output.name_to_index
            if name_to_index:
                return dict(name_to_index)
        except AttributeError:
            continue
    return {}


class _BundleSigmaDataset(Dataset):
    def __init__(self, bundle_paths: list[Path], data_indices) -> None:
        self.bundle_paths = bundle_paths
        self.data_indices = data_indices
        self.name_to_idx_lres = data_indices["in_lres"].data.input.name_to_index
        self.name_to_idx_hres = data_indices["in_hres"].data.input.name_to_index
        self.output_name_to_index = _get_output_name_to_index_from_data_indices(data_indices)
        if not self.output_name_to_index:
            raise ValueError("Could not resolve output name_to_index from checkpoint data_indices.")
        self.weather_states = list(self.output_name_to_index.keys())

    def __len__(self) -> int:
        return len(self.bundle_paths)

    def __getitem__(self, idx: int):
        from manual_inference.input_data_construction.bundle import (  # pylint: disable=import-outside-toplevel
            extract_target_from_bundle_dataset,
            load_inputs_from_bundle_numpy,
            open_bundle_dataset,
        )

        bundle = open_bundle_dataset(self.bundle_paths[idx])
        try:
            x_lres_np, x_hres_np, *_ = load_inputs_from_bundle_numpy(
                bundle,
                self.name_to_idx_lres,
                self.name_to_idx_hres,
            )
            y_np, found_target_channels = extract_target_from_bundle_dataset(
                bundle,
                self.weather_states,
            )
            if y_np is None:
                raise ValueError(f"Bundle has no target_hres_* truth fields: {self.bundle_paths[idx]}")
            if found_target_channels < len(self.weather_states):
                raise ValueError(
                    f"Bundle target coverage is incomplete for sigma evaluation: "
                    f"{found_target_channels}/{len(self.weather_states)} in {self.bundle_paths[idx]}"
                )

            # Native Anemoi batches are dates x ensemble x grid x variables; the
            # DataLoader adds the leading batch dimension.
            return (
                torch.from_numpy(x_lres_np)[None, None, ...],
                torch.from_numpy(x_hres_np)[None, None, ...],
                torch.from_numpy(y_np)[None, None, ...],
            )
        finally:
            try:
                bundle.close()
            except Exception:
                pass


class _BundleSigmaDataModule:
    def __init__(self, bundle_paths: list[Path], data_indices) -> None:
        if not bundle_paths:
            raise FileNotFoundError("No *_input_bundle.nc files found for bundle-root sigma evaluation.")
        self.bundle_paths = bundle_paths
        self.data_indices = data_indices
        self._dataset = _BundleSigmaDataset(bundle_paths, data_indices)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._dataset, batch_size=1, shuffle=False, num_workers=0)


def _make_bundle_sigma_datamodule(*, bundle_root: str, data_indices, n_samples: int):
    root = Path(bundle_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Bundle root does not exist: {root}")
    bundle_paths = sorted(root.glob("*_input_bundle.nc"))
    if n_samples > 0:
        bundle_paths = bundle_paths[:n_samples]
    return _BundleSigmaDataModule(bundle_paths, data_indices)


def _resolve_downscaler_cls():
    import importlib  # pylint: disable=import-outside-toplevel

    candidates = (
        ("anemoi.training.train.tasks", "GraphDiffusionDownscaler"),
        ("anemoi.training.train.tasks.diffusiondownscaler", "GraphDiffusionDownscaler"),
        ("anemoi.training.train.tasks.single_step", "GraphDownscaler"),
        ("anemoi.training.train.downscaler", "GraphDownscaler"),
    )
    errors = []
    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as exc:
            errors.append(f"{module_name}.{class_name}: {exc}")
    raise ImportError("Could not import a downscaler task class. Tried: " + "; ".join(errors))


def _load_downscaler_from_checkpoint_metadata(
    *,
    ckpt_root: str,
    name_exp: str,
    name_ckpt: str,
    checkpoint: dict,
    config_checkpoint,
):
    downscaler_cls = _resolve_downscaler_cls()
    hyper_parameters = checkpoint.get("hyper_parameters", {})
    kwargs = {
        "config": config_checkpoint,
        "data_indices": hyper_parameters["data_indices"],
        "graph_data": hyper_parameters["graph_data"],
        "metadata": hyper_parameters["metadata"],
        "statistics": hyper_parameters["statistics"],
    }
    if hyper_parameters.get("supporting_arrays") is not None:
        kwargs["supporting_arrays"] = hyper_parameters["supporting_arrays"]
    if hyper_parameters.get("statistics_tendencies") is not None:
        kwargs["statistics_tendencies"] = hyper_parameters["statistics_tendencies"]
    if hyper_parameters.get("truncation_data") is not None:
        kwargs["truncation_data"] = hyper_parameters["truncation_data"]

    return downscaler_cls.load_from_checkpoint(
        str(Path(ckpt_root) / name_exp / name_ckpt),
        strict=False,
        weights_only=False,
        **kwargs,
    )


def _resolve_output_name_to_index(downscaler, datamodule) -> dict[str, int]:
    """Extract model output name_to_index from downscaler or datamodule."""
    for source_name, obj in [("downscaler", downscaler), ("datamodule", datamodule)]:
        try:
            nti = _get_output_name_to_index_from_data_indices(obj.data_indices)
            if nti:
                logger.info("Resolved output name_to_index from %s data_indices (%d vars)", source_name, len(nti))
                return nti
        except AttributeError:
            pass
        try:
            nti = obj.data_indices.model.output.name_to_index
            if nti is not None and len(nti) > 0:
                logger.info("Resolved output name_to_index from %s (%d vars)", source_name, len(nti))
                return dict(nti)
        except AttributeError:
            continue
    logger.warning("Could not resolve output name_to_index — per-field metrics will be NaN")
    return {}


def _setup_model_comm_group(downscaler, model_comm_group, global_rank, world_size) -> None:
    """Configure model-parallel communication group on the downscaler.

    Sets model_comm_group, model_comm_group_size, reader_group_rank, and
    grid shard attributes so that multi-GPU sharding (used for o320->o1280
    and o1280->o2560) works correctly during sigma evaluation.
    """
    if model_comm_group is None or world_size <= 1:
        return

    if hasattr(downscaler, "set_model_comm_group"):
        downscaler.set_model_comm_group(
            model_comm_group,
            model_comm_group_id=0,
            model_comm_group_rank=global_rank,
            model_comm_num_groups=1,
            model_comm_group_size=world_size,
        )
    else:
        downscaler.model_comm_group = model_comm_group
        downscaler.model_comm_group_size = world_size

    downscaler.reader_group_rank = global_rank
    if (
        getattr(downscaler, "keep_batch_sharded", False)
        and hasattr(downscaler, "lres_grid_indices")
        and hasattr(downscaler, "hres_grid_indices")
    ):
        downscaler.lres_grid_shard_shapes = downscaler.lres_grid_indices.shard_shapes
        downscaler.hres_grid_shard_shapes = downscaler.hres_grid_indices.shard_shapes
        downscaler.lres_grid_shard_slice = downscaler.lres_grid_indices.get_shard_slice(global_rank)
        downscaler.hres_grid_shard_slice = downscaler.hres_grid_indices.get_shard_slice(global_rank)
        if hasattr(downscaler, "grid_indices"):
            downscaler.grid_shard_shapes = downscaler.grid_indices.shard_shapes
            downscaler.grid_shard_slice = downscaler.grid_indices.get_shard_slice(global_rank)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run sigma evaluator.")
    parser.add_argument(
        "--name_exp", type=str, required=True, help="Name of the experiment."
    )
    parser.add_argument(
        "--name_ckpt", type=str, required=True, help="Name of the checkpoint file."
    )
    parser.add_argument(
        "--ckpt-root",
        type=str,
        default="/home/ecm5702/scratch/aifs/checkpoint",
        help="Checkpoint root directory.",
    )
    parser.add_argument("--out_file", type=str, default="sigma_eval_table.csv")
    parser.add_argument(
        "--out_csv",
        type=str,
        default="",
        help="Optional full output CSV path. If unset, uses <ckpt-root>/<name_exp>/<out_file>.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Execution device.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=0,
        help="Model-parallel width. Use 0 to infer the canonical value from the checkpoint lane.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="Number of validation batches to evaluate.",
    )
    parser.add_argument(
        "--validation_frequency",
        type=str,
        default="50h",
        help="Validation dataloader frequency override.",
    )
    parser.add_argument(
        "--sigmas",
        type=str,
        default="",
        help="Comma-separated sigma list override, e.g. '0.02,0.5,2'.",
    )
    parser.add_argument(
        "--run_pure_noise",
        action="store_true",
        help="Also evaluate pure-noise target mode.",
    )
    parser.add_argument(
        "--run_noised",
        action="store_true",
        help="Evaluate noised-target mode. If neither mode flag is set, this is enabled by default.",
    )
    parser.add_argument(
        "--residual-statistics-fallback",
        type=str,
        default="",
        help="Fallback residual statistics filename when the configured file is missing.",
    )
    parser.add_argument(
        "--checkpoint-compat-profile",
        type=str,
        default="",
        help="Optional checkpoint compatibility profile, e.g. jupiter_ln_proof_20260622.",
    )
    parser.add_argument(
        "--bundle-root",
        type=str,
        default="",
        help="Optional directory of *_input_bundle.nc files to use instead of the checkpoint validation zarr dataloader.",
    )
    return parser


def _resolve_out_csv(args: argparse.Namespace) -> Path:
    if args.out_csv:
        return Path(args.out_csv)
    return Path(args.ckpt_root) / args.name_exp / args.out_file


def run_sigma_evaluator(args: argparse.Namespace) -> Path:
    out_csv = _resolve_out_csv(args)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print("ckpt used is", os.path.join(args.ckpt_root, args.name_exp, args.name_ckpt))
    print(f"Output CSV will be saved to: {out_csv}")

    global_rank, local_rank, world_size = _get_parallel_info()

    checkpoint, config_checkpoint = get_checkpoint(args.ckpt_root, args.name_exp, args.name_ckpt)
    config = instantiate_config()
    try:
        config_checkpoint = adapt_config_hpc(config_checkpoint, config)
    except Exception as exc:
        logger.warning("adapt_config_hpc failed (%s), injecting minimal hardware config", exc)
        _inject_minimal_hardware_config(config_checkpoint, config)

    try:
        object_loader = ObjectFromCheckpointLoader(args.ckpt_root, args.name_exp, args.name_ckpt)
    except Exception as loader_exc:
        logger.warning("ObjectFromCheckpointLoader init failed (%s), constructing manually", loader_exc)
        from manual_inference.checkpoints import to_omegaconf  # pylint: disable=import-outside-toplevel
        object_loader = ObjectFromCheckpointLoader.__new__(ObjectFromCheckpointLoader)
        object_loader.dir_exp = args.ckpt_root
        object_loader.name_exp = args.name_exp
        object_loader.name_ckpt = args.name_ckpt
        object_loader.checkpoint = checkpoint
        object_loader.config_checkpoint = config_checkpoint
        object_loader.config_for_datamodule = to_omegaconf(config_checkpoint)

    object_loader.config_checkpoint = config_checkpoint
    # Some checkpoints were produced on external paths (e.g. /leonardo_work/...).
    # Rewrite known dataset prefixes to local mirrors when present.
    object_loader.config_for_datamodule = _rewrite_dataset_paths_in_place(
        object_loader.config_for_datamodule
    )
    object_loader.config_for_datamodule.dataloader.validation.frequency = (
        args.validation_frequency
    )
    if hasattr(object_loader.config_for_datamodule.dataloader.validation, "num_workers"):
        object_loader.config_for_datamodule.dataloader.validation.num_workers = 0
    # Newer anemoi configs store num_workers at dataloader.num_workers.{split},
    # not at dataloader.validation.num_workers. Cap at 1 (not 0: setting 0
    # triggers a prefetch_factor validation error in the anemoi datamodule).
    # 1 worker × 4 ranks = 4 workers instead of 5×4=20, preventing OOM.
    _nw = getattr(object_loader.config_for_datamodule.dataloader, "num_workers", None)
    if _nw is not None and hasattr(_nw, "validation"):
        _nw.validation = 1

    inferred_lane = infer_lane_from_config(_normalize_cfg_for_lane_inference(config_checkpoint))
    requested = int(args.num_gpus_per_model)
    if requested <= 0:
        try:
            requested = int(config_checkpoint.hardware.num_gpus_per_model)
        except (AttributeError, TypeError):
            requested = 1
    requested_model_parallel_gpus = requested
    if requested_model_parallel_gpus > 1 and world_size != requested_model_parallel_gpus:
        raise RuntimeError(
            f"Expected world_size={requested_model_parallel_gpus} for sigma evaluation, "
            f"got {world_size}. Launch with srun/torchrun across "
            f"{requested_model_parallel_gpus} tasks."
        )

    if hasattr(object_loader.config_checkpoint, "hardware"):
        object_loader.config_checkpoint.hardware.num_gpus_per_model = requested_model_parallel_gpus
    if hasattr(object_loader.config_checkpoint.dataloader, "read_group_size"):
        object_loader.config_checkpoint.dataloader.read_group_size = requested_model_parallel_gpus
    if hasattr(object_loader.config_for_datamodule, "hardware"):
        object_loader.config_for_datamodule.hardware.num_gpus_per_model = requested_model_parallel_gpus
    if hasattr(object_loader.config_for_datamodule.dataloader, "read_group_size"):
        object_loader.config_for_datamodule.dataloader.read_group_size = requested_model_parallel_gpus

    fallback_residuals = None
    for cfg_candidate in (
        object_loader.config_checkpoint,
        object_loader.config_for_datamodule,
    ):
        repaired = _maybe_fix_missing_residual_statistics(cfg_candidate, args.residual_statistics_fallback)
        if repaired is not None and fallback_residuals is None:
            fallback_residuals = repaired
    if fallback_residuals is not None:
        print(f"Using fallback residual statistics file: {fallback_residuals}")

    checkpoint_compat_profile = (getattr(args, "checkpoint_compat_profile", "") or "").strip()
    if checkpoint_compat_profile:
        print(f"Applying checkpoint compatibility profile: {checkpoint_compat_profile}")
        _apply_checkpoint_compat_profile(object_loader.config_checkpoint, checkpoint_compat_profile)
        _apply_checkpoint_compat_profile(object_loader.config_for_datamodule, checkpoint_compat_profile)

    localized_paths = []
    for cfg_candidate in (
        object_loader.config_checkpoint,
        object_loader.config_for_datamodule,
    ):
        localized_paths.extend(_localize_external_checkpoint_paths(cfg_candidate))
    for old_path, new_path in localized_paths:
        print(f"Localized checkpoint path: {old_path} -> {new_path}")

    if args.device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        requested_device = args.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but torch.cuda.is_available() is False.")

    device = _resolve_device(requested_device, local_rank)
    if str(device).startswith("cuda"):
        torch.cuda.set_device(int(str(device).split(":")[1]))

    try:
        model_comm_group = _init_model_comm_group(device, global_rank, world_size)
        bundle_root = (getattr(args, "bundle_root", "") or "").strip()
        if bundle_root:
            print(f"Using bundle-root sigma inputs: {bundle_root}")
            datamodule = _make_bundle_sigma_datamodule(
                bundle_root=bundle_root,
                data_indices=checkpoint["hyper_parameters"]["data_indices"],
                n_samples=args.n_samples,
            )
            interface = None
            downscaler = _load_downscaler_from_checkpoint_metadata(
                ckpt_root=args.ckpt_root,
                name_exp=args.name_exp,
                name_ckpt=args.name_ckpt,
                checkpoint=checkpoint,
                config_checkpoint=object_loader.config_checkpoint,
            )
        else:
            object_loader.load()

            datamodule = object_loader.datamodule
            interface = object_loader.interface
            downscaler = object_loader.downscaler
        _ = checkpoint  # keep behavior; checkpoint is loaded for config compatibility.

        if interface is not None:
            interface = interface.to(device)
        downscaler = downscaler.to(device)

        _setup_model_comm_group(downscaler, model_comm_group, global_rank, world_size)

        print(
            f"Running sigma evaluator on device: {device} "
            f"(lane={inferred_lane}, num_gpus_per_model={requested_model_parallel_gpus}, "
            f"keep_batch_sharded={getattr(downscaler, 'keep_batch_sharded', 'N/A')}, "
            f"lres_shard_shapes={getattr(downscaler, 'lres_grid_shard_shapes', 'N/A') is not None})"
        )

        if args.sigmas.strip():
            sigma_values = [float(x.strip()) for x in args.sigmas.split(",") if x.strip()]
        else:
            sigma_values = sigmas

        run_noised = args.run_noised or (not args.run_noised and not args.run_pure_noise)
        run_pure_noise = args.run_pure_noise
        name_to_index = _resolve_output_name_to_index(downscaler, datamodule)
        sigma_evaluator = SigmaEvaluator(downscaler, datamodule, args.n_samples, name_to_index)

        def _run_one(sigma: float, prediction_on_pure_noise: bool) -> dict:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            loss, metrics = sigma_evaluator.evaluate_sigma(sigma, prediction_on_pure_noise)
            row = {
                "sigma": float(sigma),
                "prediction_on_pure_noise": bool(prediction_on_pure_noise),
                "loss": float(loss),
                "diff_all_var_non_weighted": float(metrics["diff_all_var_non_weighted"]),
            }
            if torch.cuda.is_available():
                row["cuda_max_memory_allocated_GB"] = float(
                    torch.cuda.max_memory_allocated() / 1e9
                )
            for k, v in metrics.items():
                try:
                    row[f"metric__{k}"] = float(v)
                except Exception:
                    row[f"metric__{k}"] = v
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return row

        rows = []
        for sigma in sigma_values:
            if run_noised:
                print(f"Evaluating sigma {sigma} with noisy output")
                rows.append(_run_one(sigma, prediction_on_pure_noise=False))
            if run_pure_noise:
                print(f"Evaluating sigma {sigma} with pure noise")
                rows.append(_run_one(sigma, prediction_on_pure_noise=True))

        if global_rank == 0:
            df = pd.DataFrame(rows)
            base_cols = ["sigma", "prediction_on_pure_noise", "loss"]
            metric_cols = sorted([c for c in df.columns if c.startswith("metric__")])
            other_cols = [c for c in df.columns if c not in base_cols + metric_cols]
            df = df[[c for c in base_cols if c in df.columns] + other_cols + metric_cols]
            if out_csv.exists():
                out_csv.unlink()
            df.to_csv(out_csv, index=False)

        _distributed_barrier(device=device, local_rank=local_rank)
        return out_csv
    finally:
        _destroy_process_group()


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    out_csv = run_sigma_evaluator(args)
    print(f"Saved sigma table: {out_csv}")


if __name__ == "__main__":
    main()
