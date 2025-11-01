import os
import sys
import torch
import numpy as np
from pathlib import Path
import xarray as xr
from einops import rearrange
from icecream import ic
from anemoi.training.distributed.strategy import DDPGroupStrategy

import argparse
import torch.distributed as dist
import torch.multiprocessing as mp  # For launching processes
from torch.nn.parallel import DistributedDataParallel as DDP
import pytorch_lightning as pl
from tqdm import tqdm
import datetime
import socket
from dataclasses import dataclass


import time
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

import subprocess

from local_plots.plot_predictions import LocalInferencePlotter
from distributed.utils import (
    get_parallel_info,
    init_parallel,
    init_network,
)
from data_processing import (
    WeatherDataBatch,
    process_residuals,
    tensors_to_numpy,
    create_xarray_dataset,
)
from omegaconf import OmegaConf

import logging

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def prepare_config_from_ckpt(
    dir_exp,
    name_exp,
    name_ckpt,
    config_dir=os.path.join(os.environ["HOME"], "dev", "anemoi-config"),
    config_name="hindcast_o320",
):
    logging.info(f"Preparing configs ...")
    checkpoint = torch.load(
        os.path.join(dir_exp, name_exp, name_ckpt),
        map_location=torch.device("cuda"),
        weights_only=False,
    )
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, job_name="compose_config"):
        cfg = compose(config_name=config_name)
    cfg_ckpt = checkpoint["hyper_parameters"]["config"]
    cfg_ckpt.hardware.paths = OmegaConf.to_container(cfg.hardware.paths, resolve=True)

    return cfg_ckpt, checkpoint, cfg


def prepare_datamodule(config_checkpoint, graph_data):
    logging.info(f"Preparing datamodule ...")

    from anemoi.training.data.datamodule import DownscalingAnemoiDatasetsDataModule

    datamodule = DownscalingAnemoiDatasetsDataModule(config_checkpoint, graph_data)
    return datamodule


@dataclass
class SampleSaver:
    dir_exp: "/home/ecm5702/scratch/aifs/checkpoint/"
    name_exp: "099e7dcdeca248198373d7397127edd5"
    name_ckpt: "last.ckpt"
    N_members: 3
    N_samples: 2
    idx: 0
    return_intermediate: False

    def __post_init__(self) -> None:
        ### Prepare sharding
        self.global_rank, self.local_rank, self.world_size = get_parallel_info()
        self.device = f"cuda:{self.local_rank}"
        print(
            f"Running on global rank {self.global_rank} and local rank {self.local_rank} out of {self.world_size}"
        )
        torch.cuda.set_device(self.local_rank)

        self.model_comm_group = init_parallel(
            device=self.device, global_rank=self.global_rank, world_size=self.world_size
        )

        ### Checkpoint, graph, datamodule, model
        self.config_checkpoint, self.checkpoint, self.new_config = (
            prepare_config_from_ckpt(self.dir_exp, self.name_exp, self.name_ckpt)
        )
        self.graph_data = torch.load(
            os.path.join(
                os.environ["OUTPUT"],
                "graphs",
                "o96_o320_icosahedral_r6_multiscale_h1_s6-1encoder.pt",
            ),
            weights_only=False,
        )
        self.datamodule = prepare_datamodule(self.new_config, self.graph_data)
        self.interface = torch.load(
            os.path.join(self.dir_exp, self.name_exp, "inference-" + self.name_ckpt),
            map_location=torch.device("cuda"),
            weights_only=False,
        )

        ### Prepare data batch
        self.data_batch = WeatherDataBatch(self.datamodule.ds_valid)
        self.data_batch.prepare(idx=0, N_samples=self.N_samples)
        self.data_batch.prepare_miscellaneous()

    def sample(self, noise_scheduler_params=None, sampler_params=None):
        d_noise = {
            "schedule_type": "karras",
            "sigma_max": 100000.0,
            "sigma_min": 0.02,
            "rho": 7.0,
            "num_steps": 50,
        }
        d_samp = {
            "sampler": "heun",
            "S_churn": 0.0,
            "S_min": 0.0,
            "S_max": 100000,
            "S_noise": 1.0,
        }
        if noise_scheduler_params:
            d_noise.update(noise_scheduler_params)
        if sampler_params:
            d_samp.update(sampler_params)
        self.samples = [None] * self.N_samples
        with torch.autocast(device_type="cuda"):
            for idx_sample in range(self.N_samples):
                logging.info(
                    f"Predicting sample {idx_sample}, Current GPU memory usage: {torch.cuda.memory_allocated()/1024**3:.2f} GB"
                )
                self.samples[idx_sample] = [None] * self.N_members
                for idx_member in range(self.N_members):
                    logging.info(f"Sample {idx_sample}: Predicting member {idx_member}")
                    logging.info(
                        f"Input tensor lres shape {self.data_batch.x_in[idx_sample].shape}"
                    )
                    logging.info(
                        f"Input tensor hres shape {self.data_batch.x_in_hres[idx_sample].shape}"
                    )
                    y_pred = self.interface.predict_step(
                        self.data_batch.x_in[idx_sample].clone().to(self.device),
                        self.data_batch.x_in_hres[idx_sample].clone().to(self.device),
                        noise_scheduler_params=d_noise,
                        sampler_params=d_samp,
                        model_comm_group=self.model_comm_group,
                    )

                    self.samples[idx_sample][idx_member] = {"y_pred": y_pred}

    def save_sampling(self, name_predictions_file):

        self.data_batch.x_in = self.data_batch.x_in.cpu()
        self.data_batch.y = self.data_batch.y.cpu()

        self.data_batch.y_residuals, self.samples = tensors_to_numpy(
            process_residuals(
                self.interface,
                self.data_batch.x_in,
                self.data_batch.y,
                self.samples,
                self.N_samples,
                self.N_members,
            )
        )

        self.data_batch.y_pred = np.array(
            [
                [member["y_pred"].squeeze() for member in sample]
                for sample in self.samples
            ]
        )
        self.data_batch.y_pred_residuals = np.array(
            [
                [member["y_pred_residuals"].squeeze() for member in sample]
                for sample in self.samples
            ]
        )

        """
        if self.return_intermediate:
            data_batch.intermediates = np.array(
                [
                    [
                        np.stack(
                            [step.squeeze() for step in member["intermediate_states"]],
                            axis=0,
                        )
                        for member in sample
                    ]
                    for sample in samples
                ]
            )
        """

        self.data_batch.prepare_miscellaneous()

        ds = create_xarray_dataset(
            self.data_batch,
            self.N_samples,
            self.N_members,
            self.config_checkpoint,
            self.datamodule.data_indices,
            # return_intermediate=return_intermediate,
        )

        # Add synchronization barrier before saving
        if self.model_comm_group is not None:
            torch.distributed.barrier(group=self.model_comm_group)

        # Only save from rank 0 process to avoid conflicts
        if self.global_rank == 0:
            predictions_path = os.path.join(
                self.dir_exp, self.name_exp, name_predictions_file
            )
            if os.path.exists(predictions_path):
                os.remove(predictions_path)
            ds.to_netcdf(predictions_path)
            logging.info(f"Predictions saved at {predictions_path}")


if __name__ == "__main__":

    if os.environ["HPC"] == "atos":
        dir_exp = "/home/ecm5702/scratch/aifs/checkpoint/"
    elif os.environ["HPC"] == "leo":
        dir_exp = "/leonardo_work/DestE_340_25/output/jdumontl/downscaling/checkpoint"
    elif os.environ["HPC"] == "marenostrum":
        dir_exp = "/home/ecm/ecm800825/outputs/checkpoint"
    else:
        raise ValueError(f"Unknown HPC: {os.environ['HPC']}")

    parser = argparse.ArgumentParser(description="Run inference and save predictions.")
    parser.add_argument(
        "--name_exp", type=str, required=True, help="Name of the experiment."
    )
    parser.add_argument(
        "--N_members", type=int, default=3, help="Number of ensemble members."
    )
    parser.add_argument(
        "--N_samples", type=int, default=2, help="Number of samples to predict."
    )
    parser.add_argument(
        "--idx", type=int, default=0, help="Starting index for samples."
    )
    parser.add_argument(
        "--name_ckpt",
        type=str,
        default="last.ckpt",
        help="Name of the checkpoint file.",
    )
    parser.add_argument(
        "--schedule_type",
        type=str,
        default="karras",
        help="Type of noise schedule (default: karras).",
    )
    parser.add_argument(
        "--sigma_max",
        type=float,
        default=100000.0,
        help="Maximum sigma value for noise schedule (default: 100000.0).",
    )
    parser.add_argument(
        "--sigma_min",
        type=float,
        default=0.02,
        help="Minimum sigma value for noise schedule (default: 0.02).",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=7.0,
        help="Rho value for noise schedule (default: 7.0).",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=50,
        help="Number of steps for noise schedule (default: 50).",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="heun",
        help="Sampler type (default: heun).",
    )
    parser.add_argument(
        "--S_churn",
        type=float,
        default=0.0,
        help="S_churn value for sampler (default: 0.0).",
    )
    parser.add_argument(
        "--S_min",
        type=float,
        default=0.0,
        help="S_min value for sampler (default: 0.0).",
    )
    parser.add_argument(
        "--S_max",
        type=float,
        default=100000.0,
        help="S_max value for sampler (default: 100000.0).",
    )
    parser.add_argument(
        "--S_noise",
        type=float,
        default=1.0,
        help="S_noise value for sampler (default: 1.0).",
    )

    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    logger.info(f"Checkpoint directory: {dir_exp}")

    ### Sampling and saving
    sample_saver = SampleSaver(
        dir_exp=dir_exp,
        name_exp=args.name_exp,
        name_ckpt=args.name_ckpt,
        N_members=args.N_members,
        N_samples=args.N_samples,
        idx=args.idx,
        return_intermediate=False,
    )
    sample_saver.sample(
        noise_scheduler_params={
            "schedule_type": args.schedule_type,
            "sigma_max": args.sigma_max,
            "sigma_min": args.sigma_min,
            "rho": args.rho,
            "num_steps": args.num_steps,
        },
        sampler_params={
            "sampler": args.sampler,
            "S_churn": args.S_churn,
            "S_min": args.S_min,
            "S_max": args.S_max,
            "S_noise": args.S_noise,
        },
    )
    name_predictions_file = "predictions.nc"
    sample_saver.save_sampling(name_predictions_file=name_predictions_file)

    ### Waiting before plotting to make sure all processes are ready and predictions.nc is well saved
    logging.info("Waiting for all processes for 2mn before plotting")
    time.sleep(120)

    ### Plotting
    lip = LocalInferencePlotter(dir_exp, args.name_exp, name_predictions_file)
    lip.save_plot(
        lip.regions,
        list_model_variables=["x", "y", "y_pred_0", "y_pred_1"],
        num_samples_to_plot=args.N_samples,
    )
