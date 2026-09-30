"""Smoke a bounded adaptive sampler on one LIBERO training demonstration observation."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import time

if os.environ.get("CUDA_VISIBLE_DEVICES") != "3":
    raise RuntimeError("Set CUDA_VISIBLE_DEVICES=3; no other GPU is permitted")

import jax
import jax.numpy as jnp
import numpy as np
from train_velocity_residual import EXPECTED_UUID
from train_velocity_residual import gpu_identity
from train_velocity_residual import make_dataset
import wandb

from openpi.models import model as model_lib
from openpi.models.adaptive_euler import AdaptiveEulerSampler
from openpi.models.adaptive_euler import AdaptiveStepConfig
from openpi.training import config as config_lib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wandb-project", default="pi05-libero-velocity-residual")
    args = parser.parse_args()
    uuid = gpu_identity()
    if uuid != EXPECTED_UUID:
        raise RuntimeError("Wrong GPU")
    with np.load(args.head, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
    checkpoint = Path(metadata["base_checkpoint"])
    config = config_lib.get_config("pi05_libero")
    factory = dataclasses.replace(config.data, assets=config_lib.AssetsConfig(assets_dir=str(checkpoint / "assets")))
    data_config = factory.create(checkpoint / "assets", config.model)
    episode = int(metadata["validation_episodes"][0])
    dataset = make_dataset([episode], data_config, config.model.action_horizon, metadata["dataset_revision"])
    transformed, valid, suite = dataset[0]
    batch = jax.tree.map(lambda value: jnp.asarray(value)[None], transformed)
    observation = model_lib.Observation.from_dict(batch)
    noise = jax.random.normal(jax.random.key(20261003), (1, config.model.action_horizon, config.model.action_dim))
    rule = AdaptiveStepConfig(
        max_velocity_evaluations=10,
        min_step=0.05,
        max_step=0.5,
        reference_step=0.25,
        reference_sigma=0.05,
    )
    sampler = AdaptiveEulerSampler.from_paths(checkpoint, args.head, rule)
    start = time.monotonic()
    first = sampler.sample_actions(jax.random.key(0), observation, noise=noise)
    first_seconds = time.monotonic() - start
    start = time.monotonic()
    second = sampler.sample_actions(jax.random.key(0), observation, noise=noise)
    warm_seconds = time.monotonic() - start
    np.testing.assert_allclose(np.asarray(first.actions), np.asarray(second.actions), atol=0, rtol=0)
    actions = np.asarray(first.actions)
    if not np.all(np.isfinite(actions)) or actions.shape != (1, 10, 32):
        raise RuntimeError("Adaptive action chunk is invalid")
    if not 1 <= first.velocity_evaluations <= rule.max_velocity_evaluations:
        raise RuntimeError("Velocity evaluation budget violated")
    if first.trace[0]["time"] != 1 or first.trace[-1]["next_time"] != 0:
        raise RuntimeError("Adaptive Euler did not integrate backwards to t=0")
    if any(item["next_time"] >= item["time"] for item in first.trace):
        raise RuntimeError("Adaptive time did not decrease")
    result = {
        "episode": episode,
        "suite_index": int(suite),
        "valid_action_steps": int(np.sum(valid)),
        "gpu_uuid": uuid,
        "base_checkpoint_identity": metadata["base_checkpoint_identity"],
        "head_steps": metadata["steps"],
        "velocity_evaluations": first.velocity_evaluations,
        "cold_seconds": first_seconds,
        "warm_seconds": warm_seconds,
        "trace": first.trace,
        "rule": dataclasses.asdict(rule),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    with wandb.init(
        project=args.wandb_project,
        name="pi05-adaptive-sampler-smoke",
        config={k: v for k, v in result.items() if k != "trace"},
    ) as run:
        wandb.log(
            {
                "smoke/velocity_evaluations": first.velocity_evaluations,
                "smoke/cold_seconds": first_seconds,
                "smoke/warm_seconds": warm_seconds,
            }
        )
        print(
            json.dumps(
                {
                    "report": str(output),
                    "wandb_url": run.url,
                    "velocity_evaluations": first.velocity_evaluations,
                    "warm_seconds": warm_seconds,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
