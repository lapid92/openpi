"""Launch an eight-GPU Human300 TVM run from released RoboCasa or public pi05_base weights."""

import argparse
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys

from openpi.training import config as training_config
from openpi.training import weight_loaders


CONFIGS = {
    "released": "pi05_robocasa_human300_tvm_released",
    "base": "pi05_robocasa_human300_tvm_base",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", choices=CONFIGS, required=True)
    parser.add_argument("--params", type=Path, required=True, help="Source checkpoint params directory")
    parser.add_argument("--released-assets", type=Path, help="Released checkpoint assets directory; required for --init released")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--checkpoint-base-dir", type=Path, required=True)
    parser.add_argument("--assets-base-dir", type=Path, required=True)
    parser.add_argument("--smoke-task", help="Run two active-TVM updates using one Human300 task")
    parser.add_argument("--resume", action="store_true", help="Resume optimizer and EMA from this experiment's checkpoint")
    args = parser.parse_args()

    if not (args.params / "_METADATA").is_file():
        parser.error(f"Missing Orbax params metadata: {args.params}")
    if args.init == "released" and (args.released_assets is None or not (args.released_assets / "norm_stats.json").is_file()):
        parser.error("--init released requires --released-assets containing norm_stats.json")
    if args.init == "base" and args.released_assets is not None:
        parser.error("--released-assets applies only to --init released")
    if args.smoke_task and args.resume:
        parser.error("Smoke runs start from source weights; use the training config to test resume separately")

    cfg = training_config.get_config(CONFIGS[args.init])
    data = dataclasses.replace(
        cfg.data,
        norm_stats_path=str(args.released_assets) if args.init == "released" else None,
        task_names=(args.smoke_task,) if args.smoke_task else (),
    )
    cfg = dataclasses.replace(
        cfg,
        data=data,
        weight_loader=weight_loaders.CheckpointWeightLoader(str(args.params)),
        exp_name=args.exp_name,
        checkpoint_base_dir=str(args.checkpoint_base_dir),
        assets_base_dir=str(args.assets_base_dir),
        # The provenance file makes the directory exist before Orbax initializes.
        # Orbax treats an empty directory as a fresh run and a saved step as a resume.
        resume=True,
        wandb_enabled=not bool(args.smoke_task),
    )
    if args.smoke_task:
        cfg = dataclasses.replace(
            cfg,
            batch_size=8,
            gradient_accumulation_steps=1,
            num_train_steps=2,
            checkpoint_completed_steps=(1, 2),
            num_workers=1,
            log_interval=1,
            save_interval=1,
            tvm=dataclasses.replace(cfg.tvm, warmup_steps=0, ramp_steps=0),
        )

    run_dir = cfg.checkpoint_dir
    if args.resume and not run_dir.exists():
        parser.error(f"Cannot resume; experiment does not exist: {run_dir}")
    if run_dir.exists() and not args.resume:
        parser.error(f"Experiment already exists: {run_dir}; choose a new name or pass --resume")
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_sha = "unknown"
    provenance = {
        "config": cfg.name,
        "git_sha": git_sha,
        "initialization": args.init,
        "source_repo": "robocasa/robocasa365_checkpoints" if args.init == "released" else "gs://openpi-assets/checkpoints/pi05_base",
        "source_revision": "c484448aba1a9b60a04c9b0ca117241518ea69f3" if args.init == "released" else "public pi05_base",
        "source_params": str(args.params.resolve()),
        "released_assets": str(args.released_assets.resolve()) if args.released_assets else None,
        "human300_tasks": [args.smoke_task] if args.smoke_task else "all 300",
        "smoke": bool(args.smoke_task),
        "effective_batch_size": cfg.batch_size,
        "microbatch_size": cfg.microbatch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "tvm": dataclasses.asdict(cfg.tvm),
        "optimizer": dataclasses.asdict(cfg.optimizer),
        "lr_schedule": dataclasses.asdict(cfg.lr_schedule),
        "fresh_optimizer": not args.resume,
    }
    manifest_path = run_dir / "provenance.json"
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(provenance, indent=2) + "\n")
    os.environ["OPENPI_GIT_SHA"] = git_sha
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

    import jax

    if jax.device_count() != 8 or any(device.platform != "gpu" for device in jax.devices()):
        raise RuntimeError("Human300 TVM recipes require exactly eight visible GPUs")

    # Import after selecting the run directory so the manifest exists even when startup fails.
    from scripts.train import main as train_main

    train_main(cfg)


if __name__ == "__main__":
    sys.exit(main())
