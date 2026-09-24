"""Fine-tune released pi05_base on LIBERO with TVM active from the first update."""

import argparse
import dataclasses
import os
import pathlib

from openpi.training import config as training_config
from openpi.training import tvm
from openpi.training import weight_loaders
from scripts import train

RUN_NAME = "pi05_libero_tvm_early_base_seed42_a01_30k_20260924"
OUTPUT_ROOT = "/volt/data/openpi_runs/tvm_early/checkpoints"
ASSETS_ROOT = "/volt/data/openpi/openpi-assets"
BASE_PARAMS = "/volt/data/openpi/openpi-assets/checkpoints/pi05_base/params"


def make_config(*, smoke: bool = False) -> training_config.TrainConfig:
    config = training_config.get_config("pi05_libero_tvm")
    return dataclasses.replace(
        config,
        exp_name=f"{RUN_NAME}_smoke" if smoke else RUN_NAME,
        checkpoint_base_dir=OUTPUT_ROOT,
        assets_base_dir=ASSETS_ROOT,
        weight_loader=weight_loaders.CheckpointWeightLoader(BASE_PARAMS),
        resume_checkpoint_dir=None,
        seed=42,
        num_train_steps=3 if smoke else 30_000,
        tvm=tvm.TVMTrainingConfig(
            enabled=True, warmup_steps=0, ramp_steps=5_000, alpha_final=0.1, fm_loss_weight=1.0
        ),
        checkpoint_completed_steps=() if smoke else (5_000, 10_000, 20_000, 30_000),
        keep_period=None,
        save_interval=30_000,
        log_interval=1 if smoke else 10,
        wandb_enabled=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not pathlib.Path(BASE_PARAMS).is_dir():
        raise FileNotFoundError(f"Released pi05_base params unavailable: {BASE_PARAMS}")
    if not os.environ.get("OPENPI_GIT_SHA"):
        raise ValueError("OPENPI_GIT_SHA must record the committed training code")
    train.main(make_config(smoke=args.smoke))


if __name__ == "__main__":
    main()
