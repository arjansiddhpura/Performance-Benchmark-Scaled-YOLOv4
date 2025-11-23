# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import argparse
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Union

import numpy as np

# import poptorch
import torch
import torch.optim.lr_scheduler as lr_scheduler
import wandb
import yacs
from models.detector import Detector
from models.yolov4_p5 import Yolov4P5
from torch.utils.data import DataLoader as torchDataLoader
from torchinfo import summary
from tqdm import tqdm
from utils.config import get_cfg_defaults, override_cfg, save_cfg
from utils.dataset import Dataset
from utils.parse_args import parse_args
from utils.postprocessing import IPUPredictionsPostProcessing, post_processing
from utils.tools import StatRecorder, load_and_fuse_pretrained_weights
from utils.visualization import plotting_tool
from utils.weight_avg import average_model_weights

path_to_detection = Path(__file__).parent.resolve()
os.environ["PYTORCH_APPS_DETECTION_PATH"] = str(path_to_detection)

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger("Detector")


def get_loader(
    opt: argparse.ArgumentParser,
    cfg: yacs.config.CfgNode,
    # ipu_opts: None,
    mode: str,
):
    """Gets a new loader for the model.
    Parameters:
        opt: opt object containing options introduced in the command line
        cfg: yacs object containing the config
        ipu_opts: Options for the IPU configuration
        mode: str indicating 'train', 'test' or 'test_inference'
    Returns:
        model[Detector]: a torch Detector Model
    """
    dataset = Dataset(path=opt.data, cfg=cfg, mode=mode)
    shuffle = mode == "train"

    # Create a standard PyTorch DataLoader
    loader = torchDataLoader(
        dataset,
        batch_size=cfg.model.micro_batch_size,
        shuffle=shuffle,
        num_workers=cfg.system.num_workers,
        pin_memory=True,
    )

    return loader


def get_model_and_loader(
    opt: argparse.ArgumentParser,
    cfg: yacs.config.CfgNode,
    mode: str,
    device: Union[str, torch.device],
):
    """Prepares the model and gets a new loader for the model.
    Parameters:
        opt: opt object containing options introduced in the command line
        cfg: yacs object containing the config
        mode: str indicating 'train', 'test' or 'test_inference'
    Returns:
        model[Detector]: a torch Detector Model
        loader[DataLoader]: a torch or poptorch DataLoader containing the specified dataset on "cfg"
    """

    # Create model
    model = Yolov4P5(cfg)

    # Load weights and fuses some batch normalizations with some convolutions
    if cfg.model.normalization == "batch":
        if opt.weights:
            print("loading pretrained weights")
            model = load_and_fuse_pretrained_weights(
                model, opt.weights, mode != "train"
            )

    if mode == "train":
        model.train()
    else:
        model.optimize_for_inference()
        model.eval()

    if opt.print_summary:
        summary(
            model,
            input_size=(
                cfg.model.micro_batch_size,
                cfg.model.input_channels,
                cfg.model.image_size,
                cfg.model.image_size,
            ),
        )
        print("listing all layers by names")
        named_layers = {name: layer for name, layer in model.named_modules()}
        for layer in named_layers:
            print(layer)

    # Creates the loader
    loader = get_loader(opt, cfg, mode)

    # Speed up convolution autotuning for benchmarking
    torch.backends.cudnn.benchmark = True

    # This replaces the logic that was in the ipu_options function
    logger.info(f"Setting model precision to: {cfg.model.precision}")
    if cfg.model.precision == "half":
        model.half()  # Convert all model parameters to float16
    elif cfg.model.precision == "mixed":
        model.half()  # Convert model to float16
        # As per the original IPU logic, convert specific head layers back to float32
        model.headp3 = model.headp3.float()
        model.headp4 = model.headp4.float()
        model.headp5 = model.headp5.float()
    elif cfg.model.precision == "single":
        # model is already in float32 by default, no action needed.
        pass
    else:
        raise ValueError(
            "Only 'half', 'mixed', or 'single' precision are supported for GPU."
        )

    # Block for GPU execution
    model.to(device)

    if opt.benchmark:
        logger.info("Warming up the GPU for benchmarking...")
        img, _, _, _ = next(iter(loader))
        img = img.to(device)

        if cfg.model.precision == "half" or cfg.model.precision == "mixed":
            img = img.half()

        # Warm-up iterations for accurate timing
        for _ in range(100):
            with torch.no_grad():
                _ = model(img)
        torch.cuda.synchronize()  # Ensure warm-up is complete

    return model, loader


def inference(
    opt: argparse.ArgumentParser,
    cfg: yacs.config.CfgNode,
    model: Union[Detector],
    loader: Union[torchDataLoader],
    stat_recorder: StatRecorder,
    run_coco_eval: bool = False,
    device: torch.device = "cuda",
):
    inference_progress = tqdm(loader)
    inference_progress.set_description("Running inference")
    stat_recorder.reset_eval_stats()

    # For accurate GPU timing
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )

    for batch_idx, (
        transformed_images,
        transformed_labels,
        image_sizes,
        image_indxs,
    ) in enumerate(inference_progress):

        # Start wall time for throughput measurement
        start_time = time.time()

        # Start round-trip latency measurement before any data is sent to the GPU
        starter.record()

        # Move data to the GPU (Host -> Device)
        transformed_images = transformed_images.to(device, non_blocking=True)
        # Ensure correct precision for the input tensor
        if cfg.model.precision == "half":
            transformed_images = transformed_images.half()

        # GPU inference
        with torch.no_grad():
            y = model(transformed_images)

        # Move data back to the CPU to complete the round-trip (Device -> Host)
        y_cpu = tuple(t.cpu() for t in y)

        # End round-trip latency measurement after data is back on the CPU
        ender.record()

        # Wait for all GPU ops to complete
        torch.cuda.synchronize()

        # End wall time for throughput measurement
        inference_step_time = time.time() - start_time

        # Calculate high-precision round-trip latency
        round_trip_latency_sec = (
            starter.elapsed_time(ender) / 1000.0
        )  # Convert ms to seconds
        inference_round_trip_time = (round_trip_latency_sec,) * 3

        processed_batch = post_processing(cfg, y, image_sizes, transformed_labels)

        stat_recorder.record_inference_stats(
            inference_round_trip_time, inference_step_time
        )

        if cfg.inference.plot_output and batch_idx % cfg.inference.plot_step == 0:
            img_paths = plotting_tool(
                cfg,
                processed_batch[0],
                [loader.dataset.get_image(img_idx) for img_idx in image_indxs],
            )
            if opt.wandb:
                wandb.log(
                    {
                        "inference_batch_{}".format(batch_idx): [
                            wandb.Image(path) for path in img_paths
                        ]
                    }
                )

        if opt.benchmark and batch_idx == 100:
            break

        pruned_preds_batch = processed_batch[0]
        processed_labels_batch = processed_batch[1]
        if cfg.eval.metrics:
            for idx, (pruned_preds, processed_labels) in enumerate(
                zip(pruned_preds_batch, processed_labels_batch)
            ):
                stat_recorder.record_eval_stats(
                    processed_labels,
                    pruned_preds,
                    image_sizes[idx],
                    loader.dataset.images_id[image_indxs[idx]],
                    run_coco_eval,
                )

    stat_recorder.logging(print, run_coco_eval)


if __name__ == "__main__":
    opt = parse_args()
    if len(opt.data) > 0 and opt.data[-1] != "/":
        opt.data += "/"

    cfg = get_cfg_defaults()

    cfg.merge_from_file(opt.config)
    cfg = override_cfg(opt, cfg)
    # Force cfg.model.ipu to False to ensure GPU-specific paths are taken
    cfg.defrost()
    cfg.model.ipu = False
    cfg.freeze()

    config_filename = Path(opt.config)
    config_filename = config_filename.with_name(f"override-{config_filename.name}")
    save_cfg(config_filename, cfg)

    if opt.show_config:
        logger.info(f"Model options: \n'{cfg}'")

    if not torch.cuda.is_available():
        logger.error(
            "GPU not available. Please run on a machine with a CUDA-enabled GPU."
        )
        sys.exit(1)

    # Determine the execution device
    device = torch.device("cuda")
    logger.info(f"Using device: {torch.cuda.get_device_name(0)}")

    stat_recorder = StatRecorder(cfg, opt.data, opt.wandb)
    # Pass the device to the model and loader getter
    model, loader = get_model_and_loader(opt, cfg, cfg.model.mode, device)

    if opt.compile_only:
        logger.info(
            "'--compile-only' is an IPU-specific flag. Exiting as no compilation is needed for GPU."
        )
        sys.exit(0)

    if cfg.model.mode == "train":
        logger.error("Training to be implemented!")
    else:
        run_coco_eval = cfg.eval.metrics and not opt.benchmark
        # Pass the device to the inference function
        inference(opt, cfg, model, loader, stat_recorder, run_coco_eval, device)
