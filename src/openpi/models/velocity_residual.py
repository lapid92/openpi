"""Frozen π₀.₅ velocity residual predictor."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models.pi0 import posemb_sincos

FEATURE_WIDTH = 1024
TIME_WIDTH = 32
LOG_SIGMA_MIN = -8.0
LOG_SIGMA_MAX = 6.0


def init_head(key, feature_width=FEATURE_WIDTH):
    if feature_width != FEATURE_WIDTH:
        raise ValueError(f"Expected 1024-wide expert, got {feature_width}")
    dims = (1056, 256, 64, 1)
    keys = jax.random.split(key, 3)
    return {
        f"layer_{i}": {
            "kernel": jax.random.normal(keys[i], (dims[i], dims[i + 1]), dtype=jnp.float32) * jnp.sqrt(2 / dims[i]),
            "bias": jnp.zeros((dims[i + 1],), dtype=jnp.float32),
        }
        for i in range(3)
    }


def predict_log_sigma(params, features, time, valid=None):
    if features.ndim != 3 or features.shape[-1] != FEATURE_WIDTH:
        raise ValueError(f"Expected [batch, horizon, 1024] features, got {features.shape}")
    if time.shape != (features.shape[0],):
        raise ValueError(f"Expected [batch] times, got {time.shape}")
    if valid is None:
        pooled = jnp.mean(features.astype(jnp.float32), axis=1)
    else:
        if valid.shape != features.shape[:2]:
            raise ValueError("Action validity mask shape mismatch")
        weights = valid.astype(jnp.float32)
        pooled = jnp.sum(features.astype(jnp.float32) * weights[..., None], axis=1) / jnp.maximum(
            jnp.sum(weights, axis=1, keepdims=True), 1
        )
    hidden = jnp.concatenate(
        [
            pooled,
            posemb_sincos(time.astype(jnp.float32), TIME_WIDTH, 4e-3, 4.0),
        ],
        axis=-1,
    )
    for i in range(3):
        p = params[f"layer_{i}"]
        hidden = hidden @ p["kernel"] + p["bias"]
        if i < 2:
            hidden = jax.nn.silu(hidden)
    return jnp.clip(hidden[:, 0], LOG_SIGMA_MIN, LOG_SIGMA_MAX)


def residual_energy(residual, valid=None):
    squared = jnp.square(residual.astype(jnp.float32))
    if valid is None:
        return jnp.mean(squared, axis=(-2, -1))
    if valid.shape != residual.shape[:-1]:
        raise ValueError("Action validity mask shape mismatch")
    count = jnp.sum(valid.astype(jnp.float32), axis=-1) * residual.shape[-1]
    if bool(jnp.any(count == 0)):
        raise ValueError("Every example must contain a valid action step")
    return jnp.sum(squared * valid[..., None], axis=(-2, -1)) / count


def gaussian_nll(log_sigma, energy):
    log_sigma = jnp.clip(log_sigma.astype(jnp.float32), LOG_SIGMA_MIN, LOG_SIGMA_MAX)
    return jnp.mean(0.5 * energy.astype(jnp.float32) * jnp.exp(-2 * log_sigma) + log_sigma)


def sample_flow(key, actions):
    noise_key, time_key = jax.random.split(key)
    noise = jax.random.normal(noise_key, actions.shape, dtype=jnp.float32)
    time = jax.random.beta(time_key, 1.5, 1.0, (actions.shape[0],)) * 0.999 + 0.001
    x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
    return x_t, time, noise - actions


def split_episodes(episode_ids, seed=42, validation_fraction=0.1):
    ids = np.asarray(episode_ids, dtype=np.int64)
    if ids.ndim != 1 or not len(ids):
        raise ValueError("Need one episode ID per training frame")
    unique = np.unique(ids)
    if len(unique) < 2:
        raise ValueError("Need at least two training episodes")
    shuffled = np.random.default_rng(seed).permutation(unique)
    n_val = max(1, min(len(unique) - 1, round(len(unique) * validation_fraction)))
    val_ids, train_ids = np.sort(shuffled[:n_val]), np.sort(shuffled[n_val:])
    return (np.flatnonzero(np.isin(ids, train_ids)), np.flatnonzero(np.isin(ids, val_ids)), train_ids, val_ids)


def split_identity(train_ids, val_ids, dataset_revision):
    payload = json.dumps(
        {
            "dataset_revision": dataset_revision,
            "train": np.asarray(train_ids).tolist(),
            "validation": np.asarray(val_ids).tolist(),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def save_head(path, params, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"{name}/{field}": np.asarray(value) for name, layer in params.items() for field, value in layer.items()}
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as output:
        np.savez(output, **arrays, metadata=np.array(json.dumps(metadata, sort_keys=True)))
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def load_head(path, checkpoint_identity):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        if metadata["base_checkpoint_identity"] != checkpoint_identity:
            raise ValueError("Wrong frozen checkpoint for predictor")
        params = {
            f"layer_{i}": {field: jnp.asarray(data[f"layer_{i}/{field}"]) for field in ("kernel", "bias")}
            for i in range(3)
        }
    return params, metadata
