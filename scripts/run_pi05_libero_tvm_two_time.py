"""Prepare the two-time Pi0.5 TVM LIBERO run from the released base checkpoint."""

import argparse
import dataclasses
import os
import pathlib

from openpi.training import config as training_config
from scripts import run_pi05_libero_tvm_early
from scripts import train

RUN_NAME = "pi05_libero_tvm_two_time_base_seed42_a01_30k_20260926"


def make_config(*, smoke: bool = False) -> training_config.TrainConfig:
    baseline = run_pi05_libero_tvm_early.make_config(smoke=False)
    return dataclasses.replace(
        baseline,
        exp_name=f"{RUN_NAME}_smoke_kernelonly" if smoke else RUN_NAME,
        num_train_steps=4 if smoke else 30_000,
        checkpoint_completed_steps=(4,) if smoke else (5_000, 10_000, 20_000, 30_000),
        save_interval=30_000,
        log_interval=1 if smoke else 10,
        wandb_enabled=not smoke,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not pathlib.Path(run_pi05_libero_tvm_early.BASE_PARAMS).is_dir():
        raise FileNotFoundError(f"Released pi05_base params unavailable: {run_pi05_libero_tvm_early.BASE_PARAMS}")
    if not os.environ.get("OPENPI_GIT_SHA"):
        raise ValueError("OPENPI_GIT_SHA must record the committed training code")
    train.main(make_config(smoke=args.smoke))


if __name__ == "__main__":
    main()
