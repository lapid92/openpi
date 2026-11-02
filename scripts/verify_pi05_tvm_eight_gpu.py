"""Bounded eight-GPU TVM training diagnostic with model-shaped synthetic data.

This checks the optimizer, EMA, active TVM gradients, sharding, and checkpointing.
It does not validate the RoboCasa dataset; use run_pi05_robocasa_tvm.py for that.
"""

import argparse
import dataclasses
from pathlib import Path

import jax

from openpi.training import config as training_config
from openpi.training import weight_loaders
from scripts.train import main as train_main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", choices=("released", "base"), required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--checkpoint-base-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if jax.device_count() != 8 or any(device.platform != "gpu" for device in jax.devices()):
        raise RuntimeError("Expected eight GPUs")
    if not (args.params / "_METADATA").is_file():
        raise FileNotFoundError(args.params)
    name = f"pi05_robocasa_human300_tvm_{args.init}"
    cfg = training_config.get_config(name)
    cfg = dataclasses.replace(
        cfg,
        data=training_config.FakeDataConfig(),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(args.params)),
        exp_name=f"synthetic_active_tvm_{args.init}",
        checkpoint_base_dir=str(args.checkpoint_base_dir),
        assets_base_dir=str(args.checkpoint_base_dir / "assets"),
        batch_size=8,
        gradient_accumulation_steps=1,
        num_train_steps=3 if args.resume else 2,
        checkpoint_completed_steps=(1, 2, 3),
        num_workers=1,
        log_interval=1,
        save_interval=1,
        wandb_enabled=False,
        resume=args.resume,
        tvm=dataclasses.replace(cfg.tvm, warmup_steps=0, ramp_steps=0),
    )
    if cfg.checkpoint_dir.exists() and not args.resume:
        raise FileExistsError(cfg.checkpoint_dir)
    train_main(cfg)


if __name__ == "__main__":
    main()
