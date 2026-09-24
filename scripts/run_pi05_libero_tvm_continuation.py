"""Run the verified 30k no-TVM to 40k Pi0.5 LIBERO continuation."""

import argparse
import dataclasses
import os
import pathlib

from openpi.training import config as training_config
from openpi.training import tvm
from scripts import train

SOURCE = "/volt/data/openpi_runs/tvm_continuation/source/pi05_libero_base_seed42_30k_20260914"
OUTPUT_ROOT = "/volt/data/openpi_runs/tvm_continuation/checkpoints"
RUN_NAME = "pi05_libero_tvm_no_tvm30k_seed42_alpha01_40k_20260924"


def make_config(*, smoke: bool = False) -> training_config.TrainConfig:
    config = training_config.get_config("pi05_libero_tvm")
    return dataclasses.replace(
        config,
        exp_name=f"{RUN_NAME}_smoke" if smoke else RUN_NAME,
        checkpoint_base_dir=OUTPUT_ROOT,
        assets_base_dir="/volt/data/openpi/openpi-assets",
        resume_checkpoint_dir=SOURCE,
        seed=42,
        num_train_steps=30_002 if smoke else 40_000,
        tvm=tvm.TVMTrainingConfig(
            enabled=True, warmup_steps=30_100, ramp_steps=5_000, alpha_final=0.1, fm_loss_weight=1.0
        ),
        checkpoint_completed_steps=() if smoke else (30_100, 32_500, 35_000, 35_100, 40_000),
        keep_period=100,
        save_interval=1_000,
        log_interval=1 if smoke else 10,
        wandb_enabled=not smoke,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = make_config(smoke=args.smoke)
    if not pathlib.Path(SOURCE, "29999", "train_state").is_dir():
        raise FileNotFoundError("Full no-TVM source state is unavailable")
    if not os.environ.get("OPENPI_GIT_SHA"):
        raise ValueError("OPENPI_GIT_SHA must record the committed training code")
    train.main(config)


if __name__ == "__main__":
    main()
