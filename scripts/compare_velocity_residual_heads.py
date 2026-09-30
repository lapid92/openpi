"""Score two frozen residual heads on identical held-out LIBERO examples."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path

if os.environ.get("CUDA_VISIBLE_DEVICES") != "3":
    raise RuntimeError("Set CUDA_VISIBLE_DEVICES=3; no other GPU is permitted")

import jax
import jax.numpy as jnp
import numpy as np
from train_velocity_residual import EXPECTED_UUID
from train_velocity_residual import SUITES
from train_velocity_residual import batch_iterator
from train_velocity_residual import checkpoint_identity
from train_velocity_residual import gpu_identity
from train_velocity_residual import make_dataset
from train_velocity_residual import record_calibration
import wandb

from openpi.models import model as model_lib
from openpi.models.velocity_residual import load_head
from openpi.models.velocity_residual import predict_log_sigma
from openpi.models.velocity_residual import residual_energy
from openpi.models.velocity_residual import sample_flow
from openpi.shared import nnx_utils
from openpi.training import config as config_lib


def read_metadata(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(str(archive["metadata"]))


def compare(args):
    uuid = gpu_identity()
    paths = {"head_a": Path(args.head_a), "head_b": Path(args.head_b)}
    metadata = {name: read_metadata(path) for name, path in paths.items()}
    first, second = metadata.values()
    for field in ("base_checkpoint_identity", "dataset_revision", "split_identity", "validation_episodes", "gpu_uuid"):
        if first[field] != second[field]:
            raise ValueError(f"Head metadata differs in {field}")
    if uuid != first["gpu_uuid"] or uuid != EXPECTED_UUID:
        raise ValueError("GPU identity differs from the saved heads")
    checkpoint = Path(first["base_checkpoint"])
    if checkpoint_identity(checkpoint) != first["base_checkpoint_identity"]:
        raise ValueError("Frozen checkpoint identity differs from the saved heads")

    config = config_lib.get_config("pi05_libero")
    factory = dataclasses.replace(config.data, assets=config_lib.AssetsConfig(assets_dir=str(checkpoint / "assets")))
    data_config = factory.create(checkpoint / "assets", config.model)
    validation = make_dataset(
        first["validation_episodes"], data_config, config.model.action_horizon, first["dataset_revision"]
    )
    batches = batch_iterator(validation, args.batch_size, args.seed, shuffle=True)
    base = config.model.load(model_lib.restore_params(checkpoint / "params", dtype=jnp.bfloat16))
    forward = nnx_utils.module_jit(base.velocity_and_action_features)
    heads = {name: load_head(path, first["base_checkpoint_identity"])[0] for name, path in paths.items()}

    records = {name: {"nll": [], "sigma": []} for name in heads}
    common = {"energy": [], "time": [], "suite": []}
    key = jax.random.key(args.seed)
    for _ in range(args.batches):
        batch, mask, suites = next(batches)
        observation = model_lib.preprocess_observation(None, model_lib.Observation.from_dict(batch), train=False)
        actions = jnp.asarray(batch["actions"], dtype=jnp.float32)
        key, flow_key = jax.random.split(key)
        x_t, flow_time, target = sample_flow(flow_key, actions)
        velocity, features = forward(observation, x_t, flow_time)
        energy = np.asarray(residual_energy(target - velocity, jnp.asarray(mask)))
        common["energy"].append(energy)
        common["time"].append(np.asarray(flow_time))
        common["suite"].append(np.asarray(suites))
        for name, head in heads.items():
            log_sigma = np.asarray(predict_log_sigma(head, features, flow_time, jnp.asarray(mask)))
            records[name]["nll"].append(0.5 * energy * np.exp(-2 * log_sigma) + log_sigma)
            records[name]["sigma"].append(np.exp(log_sigma))

    energy, times, suites = (np.concatenate(common[field]) for field in ("energy", "time", "suite"))
    metrics = {"paired/count": len(energy), "paired/residual_rms": float(np.sqrt(energy.mean()))}
    for name, record in records.items():
        nll = np.concatenate(record["nll"])
        sigma = np.concatenate(record["sigma"])
        record["nll"] = nll
        metrics[f"{name}/nll"] = float(nll.mean())
        metrics[f"{name}/predicted_sigma"] = float(sigma.mean())
        record_calibration(metrics, name, energy, sigma, times, suites)
        for index in range(5):
            selected = (times >= index / 5) & ((times <= 1) if index == 4 else (times < (index + 1) / 5))
            if selected.any():
                metrics[f"{name}/time_{index}/nll"] = float(nll[selected].mean())
        for index, suite in enumerate(SUITES):
            selected = suites == index
            if selected.any():
                metrics[f"{name}/{suite}/nll"] = float(nll[selected].mean())
    metrics["paired/nll_b_minus_a"] = float((records["head_b"]["nll"] - records["head_a"]["nll"]).mean())

    report = {
        "head_a": str(paths["head_a"]),
        "head_b": str(paths["head_b"]),
        "base_checkpoint_identity": first["base_checkpoint_identity"],
        "split_identity": first["split_identity"],
        "dataset_revision": first["dataset_revision"],
        "gpu_uuid": uuid,
        "seed": args.seed,
        "batches": args.batches,
        "batch_size": args.batch_size,
        "metrics": metrics,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    with wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config={key: value for key, value in report.items() if key != "metrics"},
    ) as run:
        wandb.log(metrics)
        print(
            json.dumps(
                {
                    "report": str(Path(args.output).resolve()),
                    "wandb_run_url": run.url,
                    "paired_nll_difference": metrics["paired/nll_b_minus_a"],
                }
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-a", required=True)
    parser.add_argument("--head-b", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batches", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--wandb-project", default="pi05-libero-velocity-residual")
    parser.add_argument("--run-name", default="paired-heldout-head-comparison")
    args = parser.parse_args()
    if args.batches < 1 or args.batch_size < 1:
        parser.error("batches and batch-size must be positive")
    compare(args)


if __name__ == "__main__":
    main()
