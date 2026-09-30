"""Train a small velocity-residual head against frozen standard π₀.₅ LIBERO."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import time

# Must be set before JAX is imported.
if os.environ.get("CUDA_VISIBLE_DEVICES") != "3":
    raise RuntimeError("Set CUDA_VISIBLE_DEVICES=3; no other GPU is permitted")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from huggingface_hub import HfApi
import jax
import jax.numpy as jnp
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import numpy as np
import optax
import torch
import wandb

from openpi.models import model as model_lib
from openpi.models.velocity_residual import FEATURE_WIDTH
from openpi.models.velocity_residual import checkpoint_identity
from openpi.models.velocity_residual import gaussian_nll
from openpi.models.velocity_residual import init_head
from openpi.models.velocity_residual import load_head
from openpi.models.velocity_residual import predict_log_sigma
from openpi.models.velocity_residual import residual_energy
from openpi.models.velocity_residual import sample_flow
from openpi.models.velocity_residual import save_head
from openpi.models.velocity_residual import split_episodes
from openpi.models.velocity_residual import split_identity
from openpi.shared import download
from openpi.shared import nnx_utils
from openpi.training import config as config_lib
from openpi.training import data_loader
import openpi.transforms as transforms

DATASET = "physical-intelligence/libero"
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
EXPECTED_UUID = "GPU-4779cdde-a260-f8ec-da6a-6fa390bc7fd7"


def gpu_identity():
    uuid = subprocess.check_output(
        ["nvidia-smi", "-i", "3", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip()
    if uuid != EXPECTED_UUID:
        raise RuntimeError(f"GPU 3 identity mismatch: {uuid}")
    devices = jax.devices()
    if len(devices) != 1 or devices[0].id != 0 or devices[0].platform != "gpu":
        raise RuntimeError(f"Expected only masked cuda:0, got {devices}")
    return uuid


class EpisodeDataset(torch.utils.data.Dataset):
    def __init__(self, raw, data_config):
        self.raw = raw
        self.transformed = data_loader.transform_dataset(raw, data_config)

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, index):
        sample = self.raw[index]
        if "actions_is_pad" not in sample:
            raise RuntimeError("LeRobot did not return the action padding mask")
        transformed = self.transformed[index]
        task_index = int(sample["task_index"])
        if not 0 <= task_index < 40:
            raise ValueError(f"Unknown LIBERO task index: {task_index}")
        return transformed, np.asarray(~sample["actions_is_pad"]), np.int32(task_index // 10)


def collate_examples(items):
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def batch_iterator(dataset, batch_size, seed, *, shuffle):
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        generator=generator,
        collate_fn=collate_examples,
    )
    while True:
        yield from loader


def make_dataset(episodes, data_config, horizon, revision):
    if not episodes:
        raise ValueError("No training episodes selected")
    meta = LeRobotDatasetMetadata(DATASET, revision="v2.0")
    raw = LeRobotDataset(
        DATASET,
        episodes=[int(x) for x in episodes],
        revision="v2.0",
        delta_timestamps={"actions": [t / meta.fps for t in range(horizon)]},
    )
    # LeRobot v2.0 stores subset offsets densely but looks them up with global episode IDs.
    index = raw.episode_data_index
    expanded = {name: torch.zeros(max(episodes) + 1, dtype=values.dtype) for name, values in index.items()}
    for local, episode_id in enumerate(episodes):
        for name, values in index.items():
            expanded[name][episode_id] = values[local]
    raw.episode_data_index = expanded
    raw = data_loader.TransformedDataset(raw, [transforms.PromptFromLeRobotTask(meta.tasks)])
    return EpisodeDataset(raw, data_config)


def record_calibration(logs, prefix, energy, sigma, time_values, suites):
    # Transfer these small per-example diagnostics once, outside the model/optimizer JIT.
    energy = np.asarray(energy)
    sigma = np.asarray(sigma)
    time_values = np.asarray(time_values)
    suites = np.asarray(suites)
    for bin_index in range(5):
        upper = time_values <= 1.0 if bin_index == 4 else time_values < (bin_index + 1) / 5
        selected = (time_values >= bin_index / 5) & upper
        if selected.any():
            logs[f"{prefix}/time_{bin_index}/residual_rms"] = float(np.sqrt(np.mean(energy[selected])))
            logs[f"{prefix}/time_{bin_index}/predicted_sigma"] = float(np.mean(sigma[selected]))
            logs[f"{prefix}/time_{bin_index}/count"] = int(selected.sum())
    for suite_index, suite in enumerate(SUITES):
        selected = suites == suite_index
        if selected.any():
            logs[f"{prefix}/{suite}/residual_rms"] = float(np.sqrt(np.mean(energy[selected])))
            logs[f"{prefix}/{suite}/predicted_sigma"] = float(np.mean(sigma[selected]))
            logs[f"{prefix}/{suite}/count"] = int(selected.sum())


def run(args):
    uuid = gpu_identity()
    if args.checkpoint == "official":
        checkpoint = Path(download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero"))
    else:
        checkpoint = Path(args.checkpoint)
    identity = checkpoint_identity(checkpoint)
    train_config = config_lib.get_config("pi05_libero")
    if not train_config.model.pi05 or train_config.model.action_horizon != 10:
        raise RuntimeError("Expected standard π₀.₅ LIBERO configuration")
    data_factory = dataclasses.replace(
        train_config.data,
        assets=config_lib.AssetsConfig(assets_dir=str(checkpoint / "assets")),
    )
    data_config = data_factory.create(checkpoint / "assets", train_config.model)
    if data_config.norm_stats is None:
        raise RuntimeError("Checkpoint normalization statistics are required")

    dataset_info = HfApi().dataset_info(DATASET, revision="v2.0")
    revision = dataset_info.sha
    metadata = LeRobotDatasetMetadata(DATASET, revision="v2.0")
    info = metadata.info
    if info["splits"] != {"train": f"0:{info['total_episodes']}"}:
        raise RuntimeError(f"Unrecognized training split: {info['splits']}")
    episode_ids = np.arange(info["total_episodes"], dtype=np.int64)
    _, _, train_ids, val_ids = split_episodes(episode_ids, args.seed)
    split_hash = split_identity(train_ids, val_ids, revision)
    if args.max_train_episodes:
        train_ids = train_ids[: args.max_train_episodes]
    if args.max_val_episodes:
        val_ids = val_ids[: args.max_val_episodes]
    train_data = make_dataset(train_ids.tolist(), data_config, train_config.model.action_horizon, revision)
    val_data = make_dataset(val_ids.tolist(), data_config, train_config.model.action_horizon, revision)
    train_batches = batch_iterator(train_data, args.batch_size, args.seed, shuffle=True)
    val_batches = batch_iterator(val_data, args.batch_size, args.seed, shuffle=True)

    base = train_config.model.load(model_lib.restore_params(checkpoint / "params", dtype=jnp.bfloat16))
    expert_width = int(base.action_out_proj.kernel.value.shape[0])
    if expert_width != FEATURE_WIDTH:
        raise RuntimeError(f"Checkpoint expert width is {expert_width}, expected {FEATURE_WIDTH}")
    base_projection_before = np.asarray(base.action_out_proj.kernel.value).copy()
    frozen_forward = nnx_utils.module_jit(base.velocity_and_action_features)
    head = init_head(jax.random.key(args.seed), expert_width)
    initial_head = jax.tree.map(lambda x: np.asarray(x).copy(), head)
    optimizer = optax.adam(args.learning_rate)
    opt_state = optimizer.init(head)

    @jax.jit
    def head_step(params, state, features, time_values, energy, valid):
        def objective(p):
            return gaussian_nll(predict_log_sigma(p, features, time_values, valid), energy)

        loss, grads = jax.value_and_grad(objective)(params)
        updates, state = optimizer.update(grads, state, params)
        return optax.apply_updates(params, updates), state, loss

    run = wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config={
            "base_checkpoint": str(checkpoint),
            "base_checkpoint_identity": identity,
            "dataset": DATASET,
            "dataset_revision": revision,
            "split_identity": split_hash,
            "train_episode_count": len(train_ids),
            "validation_episode_count": len(val_ids),
            "train_episodes": train_ids.tolist(),
            "validation_episodes": val_ids.tolist(),
            "time_distribution": "Beta(1.5,1)*0.999+0.001",
            "residual_reduction": "mean over valid horizon steps and 32 action dimensions",
            "log_sigma_bounds": [-8.0, 6.0],
            "gpu_local_index": 3,
            "gpu_uuid": uuid,
            "masked_device": "cuda:0",
            "feature_width": expert_width,
        },
    )
    key = jax.random.key(args.seed + 1)
    start = time.monotonic()
    try:
        for step in range(args.steps):
            batch, mask, suites = next(train_batches)
            observation = model_lib.Observation.from_dict(batch)
            actions = jnp.asarray(batch["actions"], dtype=jnp.float32)
            key, flow_key = jax.random.split(key)
            x_t, flow_time, target = sample_flow(flow_key, actions)
            observation = model_lib.preprocess_observation(None, observation, train=False)
            velocity, features = frozen_forward(observation, x_t, flow_time)
            if features.shape != (len(actions), train_config.model.action_horizon, FEATURE_WIDTH):
                raise RuntimeError(f"Unexpected checkpoint feature shape {features.shape}")
            residual = jax.lax.stop_gradient(target - velocity)
            energy = residual_energy(residual, jnp.asarray(mask))
            head, opt_state, loss = head_step(
                head, opt_state, jax.lax.stop_gradient(features), flow_time, energy, jnp.asarray(mask)
            )
            sigma = jnp.exp(predict_log_sigma(head, features, flow_time, jnp.asarray(mask)))
            elapsed = max(time.monotonic() - start, 1e-6)
            logs = {
                "train/loss": float(loss),
                "train/residual_rms": float(jnp.sqrt(jnp.mean(energy))),
                "train/predicted_sigma": float(jnp.mean(sigma)),
                "train/examples_per_second": (step + 1) * len(actions) / elapsed,
            }
            record_calibration(logs, "calibration", energy, sigma, flow_time, suites)
            if (step + 1) % args.validation_every == 0 or step + 1 == args.steps:
                val_batch, val_mask, val_suites = next(val_batches)
                val_obs = model_lib.preprocess_observation(
                    None, model_lib.Observation.from_dict(val_batch), train=False
                )
                val_actions = jnp.asarray(val_batch["actions"], dtype=jnp.float32)
                key, val_key = jax.random.split(key)
                val_x, val_time, val_target = sample_flow(val_key, val_actions)
                val_velocity, val_features = frozen_forward(val_obs, val_x, val_time)
                val_energy = residual_energy(val_target - val_velocity, jnp.asarray(val_mask))
                val_log_sigma = predict_log_sigma(head, val_features, val_time, jnp.asarray(val_mask))
                val_sigma = jnp.exp(val_log_sigma)
                logs["validation/loss"] = float(gaussian_nll(val_log_sigma, val_energy))
                logs["validation/residual_rms"] = float(jnp.sqrt(jnp.mean(val_energy)))
                logs["validation/predicted_sigma"] = float(jnp.mean(val_sigma))
                record_calibration(logs, "validation_calibration", val_energy, val_sigma, val_time, val_suites)
            wandb.log(logs, step=step + 1)
            if (step + 1) % args.save_every == 0:
                save_head(
                    args.output,
                    head,
                    {
                        "base_checkpoint": str(checkpoint),
                        "base_checkpoint_identity": identity,
                        "model_config": "pi05_libero",
                        "dataset": DATASET,
                        "dataset_revision": revision,
                        "split_identity": split_hash,
                        "train_episodes": train_ids.tolist(),
                        "validation_episodes": val_ids.tolist(),
                        "gpu_uuid": uuid,
                        "steps": step + 1,
                        "requested_steps": args.steps,
                        "wandb_run_url": run.url,
                        "feature_width": expert_width,
                    },
                )
        if not any(
            not np.array_equal(np.asarray(a), np.asarray(b))
            for a, b in zip(jax.tree.leaves(initial_head), jax.tree.leaves(head), strict=True)
        ):
            raise RuntimeError("No head weights changed")

        # Report held-out calibration with enough examples for every suite and time bin.
        calibration_records = []
        for _ in range(args.calibration_batches):
            val_batch, val_mask, val_suites = next(val_batches)
            val_obs = model_lib.preprocess_observation(None, model_lib.Observation.from_dict(val_batch), train=False)
            val_actions = jnp.asarray(val_batch["actions"], dtype=jnp.float32)
            key, val_key = jax.random.split(key)
            val_x, val_time, val_target = sample_flow(val_key, val_actions)
            val_velocity, val_features = frozen_forward(val_obs, val_x, val_time)
            val_energy = residual_energy(val_target - val_velocity, jnp.asarray(val_mask))
            val_sigma = jnp.exp(predict_log_sigma(head, val_features, val_time, jnp.asarray(val_mask)))
            calibration_records.append(
                (np.asarray(val_energy), np.asarray(val_sigma), np.asarray(val_time), np.asarray(val_suites))
            )
        energy_values, sigma_values, time_values, suite_values = (
            np.concatenate(values) for values in zip(*calibration_records, strict=True)
        )
        calibration = {}
        record_calibration(calibration, "heldout_calibration", energy_values, sigma_values, time_values, suite_values)
        calibration["heldout_calibration/overall/count"] = len(energy_values)
        calibration["heldout_calibration/overall/residual_rms"] = float(np.sqrt(np.mean(energy_values)))
        calibration["heldout_calibration/overall/predicted_sigma"] = float(np.mean(sigma_values))
        calibration["heldout_calibration/overall/nll"] = float(
            np.mean(0.5 * energy_values / np.square(sigma_values) + np.log(sigma_values))
        )
        for name, value in calibration.items():
            run.summary[name] = value
        calibration_path = Path(args.output).with_suffix(".calibration.json")
        calibration_path.parent.mkdir(parents=True, exist_ok=True)
        calibration_path.write_text(
            json.dumps({"dataset_revision": revision, "split_identity": split_hash, "metrics": calibration}, indent=2)
            + "\n"
        )
        metadata = {
            "base_checkpoint": str(checkpoint),
            "base_checkpoint_identity": identity,
            "model_config": "pi05_libero",
            "dataset": DATASET,
            "dataset_revision": revision,
            "split_identity": split_hash,
            "train_episodes": train_ids.tolist(),
            "validation_episodes": val_ids.tolist(),
            "gpu_uuid": uuid,
            "steps": args.steps,
            "wandb_run_url": run.url,
            "feature_width": expert_width,
            "heldout_calibration_report": str(calibration_path),
        }
        if not np.array_equal(np.asarray(base.action_out_proj.kernel.value), base_projection_before):
            raise RuntimeError("Frozen π₀.₅ projection weights changed")
        save_head(args.output, head, metadata)
        reloaded, _ = load_head(args.output, identity)
        for original, restored in zip(jax.tree.leaves(head), jax.tree.leaves(reloaded), strict=True):
            np.testing.assert_array_equal(np.asarray(original), np.asarray(restored))
        print(
            json.dumps(
                {
                    "head_checkpoint": str(Path(args.output).resolve()),
                    "wandb_run_url": run.url,
                    "checkpoint_identity": identity,
                    "split_identity": split_hash,
                }
            )
        )
    finally:
        run.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="official")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--calibration-batches", type=int, default=32)
    parser.add_argument("--max-train-episodes", type=int, default=0)
    parser.add_argument("--max-val-episodes", type=int, default=0)
    parser.add_argument("--wandb-project", default="pi05-libero-velocity-residual")
    parser.add_argument("--run-name", default="velocity-residual")
    args = parser.parse_args()
    if (
        args.steps < 1
        or args.batch_size < 1
        or args.validation_every < 1
        or args.save_every < 1
        or args.calibration_batches < 1
    ):
        parser.error("steps, batch-size, validation-every, save-every, and calibration-batches must be positive")
    run(args)


if __name__ == "__main__":
    main()
