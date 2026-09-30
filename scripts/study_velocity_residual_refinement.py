"""Offline residual-score and fixed-Euler refinement study on LIBERO training demonstrations."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import time

if os.environ.get("CUDA_VISIBLE_DEVICES") != "3":
    raise RuntimeError("Set CUDA_VISIBLE_DEVICES=3; no other GPU is permitted")

import jax
import jax.numpy as jnp
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import numpy as np
from scipy.stats import rankdata
from scipy.stats import spearmanr
import torch
from train_velocity_residual import DATASET
from train_velocity_residual import EXPECTED_UUID
from train_velocity_residual import SUITES
from train_velocity_residual import checkpoint_identity
from train_velocity_residual import collate_examples
from train_velocity_residual import gpu_identity
from train_velocity_residual import make_dataset
import wandb

from openpi.models import model as model_lib
from openpi.models.velocity_residual import load_head
from openpi.models.velocity_residual import predict_log_sigma
from openpi.models.velocity_residual import residual_energy
from openpi.models.velocity_residual import sample_flow
from openpi.shared import nnx_utils
from openpi.training import config as config_lib

STEPS = (1, 2, 4, 10)
FRAME_FRACTIONS = (0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 0.93, 0.98)
SPLIT_SEED = 20261001


def diagnostic_split(validation_ids, metadata):
    task_index = {prompt: index for index, prompt in metadata.tasks.items()}
    suites = {int(episode): task_index[metadata.episodes[int(episode)]["tasks"][0]] // 10 for episode in validation_ids}
    rng = np.random.default_rng(SPLIT_SEED)
    diagnostic, calibration = [], []
    for suite in range(4):
        ids = np.array([int(episode) for episode in validation_ids if suites[int(episode)] == suite])
        selected = rng.permutation(ids)
        cut = len(selected) // 2
        diagnostic.extend(selected[:cut].tolist())
        calibration.extend(selected[cut:].tolist())
    diagnostic.sort()
    calibration.sort()
    if set(diagnostic) & set(calibration) or set(diagnostic + calibration) != set(validation_ids):
        raise RuntimeError("Diagnostic and rule-calibration trajectories overlap or omit episodes")
    payload = json.dumps({"diagnostic": diagnostic, "rule_calibration": calibration}, sort_keys=True)
    return diagnostic, calibration, hashlib.sha256(payload.encode()).hexdigest(), suites


def selected_frames(dataset, episode_ids, metadata):
    frames = []
    start = 0
    for episode in episode_ids:
        length = int(metadata.episodes[episode]["length"])
        if length < 1:
            raise RuntimeError(f"Empty episode {episode}")
        frames.extend(start + min(length - 1, round((length - 1) * fraction)) for fraction in FRAME_FRACTIONS)
        start += length
    if start != len(dataset):
        raise RuntimeError("Episode lengths do not match the loaded diagnostic subset")
    return frames


class IndexedExamples(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        index = self.indices[i]
        transformed, valid, suite = self.dataset[index]
        episode = int(self.dataset.raw[index]["episode_index"])
        return transformed, valid, suite, np.int32(episode)


def masked_mse(left, right, valid):
    squared = np.square(np.asarray(left, dtype=np.float32) - np.asarray(right, dtype=np.float32))
    weights = np.asarray(valid, dtype=np.float32)
    valid_steps = np.sum(weights, axis=1)
    if np.any(valid_steps == 0):
        raise ValueError("Every example needs at least one valid action step")
    return np.sum(squared * weights[..., None], axis=(1, 2)) / (valid_steps * squared.shape[-1])


def grouped_calibration(logs, prefix, predicted_variance, energy, time_values, suites):
    score_edges = np.quantile(predicted_variance, np.linspace(0, 1, 6))
    groups = {}
    for i in range(5):
        selected = (predicted_variance >= score_edges[i]) & (
            (predicted_variance <= score_edges[i + 1]) if i == 4 else (predicted_variance < score_edges[i + 1])
        )
        groups[f"score_{i}"] = selected
        groups[f"time_{i}"] = (time_values >= i / 5) & ((time_values <= 1) if i == 4 else (time_values < (i + 1) / 5))
    for i, suite in enumerate(SUITES):
        groups[suite] = suites == i
    for label, selected in groups.items():
        if not selected.any():
            continue
        predicted = float(np.mean(predicted_variance[selected]))
        measured = float(np.mean(energy[selected]))
        logs[f"{prefix}/{label}/count"] = int(selected.sum())
        logs[f"{prefix}/{label}/mean_sigma_squared"] = predicted
        logs[f"{prefix}/{label}/mean_residual_squared"] = measured
        logs[f"{prefix}/{label}/variance_ratio"] = predicted / max(measured, 1e-12)
    logs[f"{prefix}/overall/count"] = len(energy)
    logs[f"{prefix}/overall/mean_sigma_squared"] = float(np.mean(predicted_variance))
    logs[f"{prefix}/overall/mean_residual_squared"] = float(np.mean(energy))


def auc(score, positive):
    n_pos = int(np.sum(positive))
    n_neg = len(positive) - n_pos
    if not n_pos or not n_neg:
        return None
    ranks = rankdata(score, method="average")
    return float((np.sum(ranks[positive]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def ranking_metrics(score, benefit, episode, suite, seed=1234, bootstrap=1000):
    if len(score) != len(benefit) or len(score) != len(episode) or len(score) != len(suite) or not len(score):
        raise ValueError("Ranking arrays must be nonempty and aligned")
    if not (np.all(np.isfinite(score)) and np.all(np.isfinite(benefit))):
        raise ValueError("Ranking arrays must be finite")

    def safe_spearman(x, y):
        if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
            return None
        value = float(spearmanr(x, y).statistic)
        return value if np.isfinite(value) else None

    rho = safe_spearman(score, benefit)
    low, high = np.quantile(score, (0.25, 0.75))
    contrast = float(np.mean(benefit[score >= high]) - np.mean(benefit[score <= low]))
    result = {
        "spearman": rho,
        "high_minus_low_benefit": contrast,
        "positive_benefit_fraction": float(np.mean(benefit > 0)),
        "auc_positive_benefit": auc(score, benefit > 0),
    }
    unique = np.unique(episode)
    groups = {int(item): np.flatnonzero(episode == item) for item in unique}
    rng = np.random.default_rng(seed)
    rhos, contrasts = [], []
    for _ in range(bootstrap):
        chosen = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([groups[int(item)] for item in chosen])
        boot_rho = safe_spearman(score[indices], benefit[indices])
        boot_low = benefit[indices][score[indices] <= low]
        boot_high = benefit[indices][score[indices] >= high]
        if boot_rho is not None and len(boot_low) and len(boot_high):
            rhos.append(boot_rho)
            contrasts.append(float(np.mean(boot_high) - np.mean(boot_low)))
    result["spearman_ci95"] = np.quantile(rhos, (0.025, 0.975)).tolist() if rhos else None
    result["high_minus_low_ci95"] = np.quantile(contrasts, (0.025, 0.975)).tolist() if contrasts else None
    for index, name in enumerate(SUITES):
        selected = suite == index
        result[f"{name}/count"] = int(selected.sum())
        result[f"{name}/spearman"] = safe_spearman(score[selected], benefit[selected])
        result[f"{name}/mean_benefit"] = float(np.mean(benefit[selected])) if selected.any() else None
    suite_rho = [result[f"{name}/spearman"] for name in SUITES]
    result["gate_pass"] = bool(
        result["spearman_ci95"] is not None
        and result["high_minus_low_ci95"] is not None
        and result["spearman_ci95"][0] > 0
        and result["high_minus_low_ci95"][0] > 0
        and sum(value is not None and value > 0 for value in suite_rho) >= 3
    )
    return result


def run(args):
    uuid = gpu_identity()
    with np.load(args.head, allow_pickle=False) as archive:
        saved = json.loads(str(archive["metadata"]))
    checkpoint = Path(saved["base_checkpoint"])
    if uuid != EXPECTED_UUID or checkpoint_identity(checkpoint) != saved["base_checkpoint_identity"]:
        raise RuntimeError("GPU or frozen checkpoint identity mismatch")
    head, _ = load_head(args.head, saved["base_checkpoint_identity"])
    metadata = LeRobotDatasetMetadata(DATASET, revision="v2.0")
    diagnostic_ids, rule_ids, diagnostic_hash, episode_suite = diagnostic_split(saved["validation_episodes"], metadata)
    config = config_lib.get_config("pi05_libero")
    factory = dataclasses.replace(config.data, assets=config_lib.AssetsConfig(assets_dir=str(checkpoint / "assets")))
    data_config = factory.create(checkpoint / "assets", config.model)
    dataset = make_dataset(diagnostic_ids, data_config, config.model.action_horizon, saved["dataset_revision"])
    indices = selected_frames(dataset, diagnostic_ids, metadata)
    loader = torch.utils.data.DataLoader(
        IndexedExamples(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_examples,
    )
    base = config.model.load(model_lib.restore_params(checkpoint / "params", dtype=jnp.bfloat16))
    prepare = nnx_utils.module_jit(base.prepare_action_prefix)
    velocity_and_features = nnx_utils.module_jit(base.cached_velocity_and_action_features)
    fixed_sampler = nnx_utils.module_jit(base.sample_actions, static_argnames=("num_steps",))
    key = jax.random.key(args.seed)
    records = []
    start = time.monotonic()
    for batch_index, (batch, valid, suites, episodes) in enumerate(loader):
        raw_observation = model_lib.Observation.from_dict(batch)
        observation = model_lib.preprocess_observation(None, raw_observation, train=False)
        actions = jnp.asarray(batch["actions"], dtype=jnp.float32)
        batch_size = len(actions)
        valid_jax = jnp.asarray(valid)
        context = prepare(observation)
        key, noise_key, flow_key = jax.random.split(key, 3)
        noise = jax.random.normal(noise_key, actions.shape, dtype=jnp.float32)
        interpolation, flow_time, target = sample_flow(flow_key, actions)
        interpolated_velocity, interpolated_features = velocity_and_features(
            observation, interpolation, flow_time, context
        )
        energy = np.asarray(residual_energy(target - interpolated_velocity, valid_jax))
        log_sigma = np.asarray(predict_log_sigma(head, interpolated_features, flow_time, valid_jax))
        outputs, trace, trace_output_disagreement, trace_output_max_abs = {}, {}, {}, {}
        for n in STEPS:
            x = noise
            trace[n] = []
            for i in range(n):
                time_value = 1 - i / n
                t = jnp.full((batch_size,), time_value, dtype=jnp.float32)
                velocity, features = velocity_and_features(observation, x, t, context)
                sigma = np.asarray(jnp.exp(predict_log_sigma(head, features, t)))
                trace[n].append((time_value, sigma))
                x = x - velocity / n
            fixed_output = fixed_sampler(jax.random.key(0), raw_observation, num_steps=n, noise=noise)
            outputs[n] = np.asarray(fixed_output)
            trace_output_disagreement[n] = masked_mse(x, fixed_output, valid)
            trace_output_max_abs[n] = np.max(np.abs(np.asarray(x) - outputs[n]), axis=(1, 2))
        half_time = jnp.full((batch_size,), 0.5, dtype=jnp.float32)
        half_interpolation = 0.5 * noise + 0.5 * actions
        _, half_features = velocity_and_features(observation, half_interpolation, half_time, context)
        sigma_half_interpolation = np.asarray(jnp.exp(predict_log_sigma(head, half_features, half_time)))
        for i in range(batch_size):
            if episode_suite[int(episodes[i])] != int(suites[i]):
                raise RuntimeError("Episode suite metadata mismatch")
            row = {
                "episode": int(episodes[i]),
                "suite": int(suites[i]),
                "valid_steps": int(np.sum(valid[i])),
                "sigma_t1": float(trace[1][0][1][i]),
                "sigma_half_generated": float(trace[2][1][1][i]),
                "sigma_half_interpolation": float(sigma_half_interpolation[i]),
                "calibration_time": float(flow_time[i]),
                "calibration_sigma_squared": float(np.exp(2 * log_sigma[i])),
                "calibration_residual_squared": float(energy[i]),
                "trace": {
                    str(n): [{"time": float(t), "sigma": float(sigma[i])} for t, sigma in trace[n]] for n in STEPS
                },
            }
            for n in STEPS:
                row[f"mse_{n}"] = float(masked_mse(outputs[n][i : i + 1], actions[i : i + 1], valid[i : i + 1])[0])
                row[f"distance_to_10_{n}"] = float(
                    masked_mse(outputs[n][i : i + 1], outputs[10][i : i + 1], valid[i : i + 1])[0]
                )
                row[f"velocity_evaluations_{n}"] = n
                row[f"trace_vs_fixed_mse_{n}"] = float(trace_output_disagreement[n][i])
                row[f"trace_vs_fixed_max_abs_{n}"] = float(trace_output_max_abs[n][i])
            records.append(row)
        if (batch_index + 1) % 10 == 0:
            print(
                f"processed_batches={batch_index + 1} examples={len(records)} elapsed_s={time.monotonic() - start:.1f}",
                flush=True,
            )
        if args.max_batches and batch_index + 1 >= args.max_batches:
            break

    if args.max_batches and len(records) >= args.max_batches * args.batch_size:
        print("diagnostic smoke subset complete", flush=True)
    suite = np.array([row["suite"] for row in records])
    episode = np.array([row["episode"] for row in records])
    score = np.array([row["sigma_t1"] for row in records])
    logs = {}
    grouped_calibration(
        logs,
        "interpolation_calibration",
        np.array([row["calibration_sigma_squared"] for row in records]),
        np.array([row["calibration_residual_squared"] for row in records]),
        np.array([row["calibration_time"] for row in records]),
        suite,
    )
    results = {}
    for n in (1, 2, 4):
        benefit = np.array([row[f"mse_{n}"] - row["mse_10"] for row in records])
        results[f"{n}_to_10"] = ranking_metrics(score, benefit, episode, suite, seed=args.seed + n)
        if results[f"{n}_to_10"]["spearman"] is not None:
            logs[f"refinement/{n}_to_10/spearman"] = results[f"{n}_to_10"]["spearman"]
        logs[f"refinement/{n}_to_10/mean_demo_mse_benefit"] = float(np.mean(benefit))
        disagreement = np.array([row[f"distance_to_10_{n}"] for row in records])
        logs[f"refinement/{n}_to_10/mean_endpoint_disagreement"] = float(np.mean(disagreement))
        if len(score) > 1 and np.any(score != score[0]) and np.any(disagreement != disagreement[0]):
            logs[f"exploratory/{n}_to_10/score_vs_endpoint_disagreement_spearman"] = float(
                spearmanr(score, disagreement).statistic
            )
    generated = np.array([row["sigma_half_generated"] for row in records])
    interpolated = np.array([row["sigma_half_interpolation"] for row in records])
    logs["distribution_shift/t_half_generated_sigma_mean"] = float(generated.mean())
    logs["distribution_shift/t_half_interpolation_sigma_mean"] = float(interpolated.mean())
    logs["distribution_shift/t_half_mean_log_ratio"] = float(np.mean(np.log(generated / interpolated)))
    logs["distribution_shift/t_half_generated_over_interpolation_fraction"] = float(np.mean(generated > interpolated))
    for n in STEPS:
        logs[f"trace_vs_fixed/{n}/mean_mse"] = float(np.mean([row[f"trace_vs_fixed_mse_{n}"] for row in records]))
        logs[f"trace_vs_fixed/{n}/max_abs_all_tokens"] = float(
            np.max([row[f"trace_vs_fixed_max_abs_{n}"] for row in records])
        )
    logs["offline/total_examples"] = len(records)
    logs["offline/diagnostic_episodes"] = len(diagnostic_ids)
    logs["offline/elapsed_seconds"] = time.monotonic() - start
    report = {
        "checkpoint": str(checkpoint),
        "base_checkpoint_identity": saved["base_checkpoint_identity"],
        "head": str(Path(args.head).resolve()),
        "head_steps": saved["steps"],
        "training_split_identity": saved["split_identity"],
        "diagnostic_split_identity": diagnostic_hash,
        "diagnostic_episodes": diagnostic_ids,
        "rule_calibration_episodes": rule_ids,
        "dataset_revision": saved["dataset_revision"],
        "gpu_uuid": uuid,
        "seed": args.seed,
        "frame_fractions": FRAME_FRACTIONS,
        "steps": STEPS,
        "metrics": logs,
        "ranking": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    output.with_suffix(".records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    with wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config={k: v for k, v in report.items() if k not in ("metrics", "ranking")},
    ) as run:
        wandb.log(logs)
        run.summary["primary_gate_pass"] = results["1_to_10"]["gate_pass"]
        run.summary["primary_spearman_ci95"] = results["1_to_10"]["spearman_ci95"]
        print(
            json.dumps(
                {
                    "report": str(output),
                    "wandb_url": run.url,
                    "primary_gate_pass": results["1_to_10"]["gate_pass"],
                    "primary_ranking": results["1_to_10"],
                }
            ),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--wandb-project", default="pi05-libero-velocity-residual")
    parser.add_argument("--run-name", default="offline-refinement-diagnostic")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_batches < 0:
        parser.error("batch-size must be positive and max-batches nonnegative")
    run(args)


if __name__ == "__main__":
    main()
