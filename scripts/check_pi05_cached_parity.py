"""Record stepwise GPU parity against a source-mirroring fixed-step oracle."""

import os

if os.environ.get("CUDA_VISIBLE_DEVICES") != "3":
    raise RuntimeError("Set CUDA_VISIBLE_DEVICES=3")
import argparse
import dataclasses
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import einops
import jax
import jax.numpy as jnp
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import numpy as np
from study_velocity_residual_refinement import IndexedExamples
from study_velocity_residual_refinement import diagnostic_split
from study_velocity_residual_refinement import selected_frames
import torch
from train_velocity_residual import DATASET
from train_velocity_residual import EXPECTED_UUID
from train_velocity_residual import collate_examples
from train_velocity_residual import gpu_identity
from train_velocity_residual import make_dataset

from openpi.models import model as model_lib
from openpi.models.adaptive_euler import AdaptiveEulerSampler
from openpi.models.adaptive_euler import AdaptiveStepConfig
from openpi.models.adaptive_euler import _compile_adaptive_sample
from openpi.models.pi0 import Pi0
from openpi.models.pi0 import make_attn_mask
from openpi.models.velocity_residual import checkpoint_identity
from openpi.models.velocity_residual import load_head
from openpi.shared import nnx_utils
from openpi.training import config as config_lib

parser = argparse.ArgumentParser()
parser.add_argument("--head", type=Path, default=Path("/volt/data/openpi_velocity_runs/long_3000_20260930/head.npz"))
parser.add_argument(
    "--checkpoint", type=Path, default=Path("/root/.cache/openpi/openpi-assets/checkpoints/pi05_libero")
)
parser.add_argument("--output", type=Path, default=Path("/volt/data/pi05_adaptive/parity_stepwise"))
parser.add_argument("--batch-index", type=int, default=18)
parser.add_argument("--batch-offset", type=int, default=2)
parser.add_argument("--episode", type=int, default=184)
parser.add_argument("--seed", type=int, default=20261002)
args = parser.parse_args()
if args.batch_index < 0 or not 0 <= args.batch_offset < 4:
    parser.error("batch-index must be nonnegative and batch-offset must be in [0, 4)")
if gpu_identity() != EXPECTED_UUID:
    raise RuntimeError("GPU identity mismatch")
checkpoint = args.checkpoint
config = config_lib.get_config("pi05_libero")
factory = dataclasses.replace(config.data, assets=config_lib.AssetsConfig(assets_dir=str(checkpoint / "assets")))
data_config = factory.create(checkpoint / "assets", config.model)
saved = json.loads(str(np.load(args.head, allow_pickle=False)["metadata"]))
metadata = LeRobotDatasetMetadata(DATASET, revision="v2.0")
ids, _, _, _ = diagnostic_split(saved["validation_episodes"], metadata)
dataset = make_dataset(ids, data_config, config.model.action_horizon, saved["dataset_revision"])
indices = selected_frames(dataset, ids, metadata)
loader = torch.utils.data.DataLoader(
    IndexedExamples(dataset, indices), batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_examples
)
key = jax.random.key(args.seed)
for batch_index, item in enumerate(loader):
    key, noise_key, flow_key = jax.random.split(key, 3)
    if batch_index == args.batch_index:
        batch, _, _, episodes = item
        break
if batch_index != args.batch_index:
    raise RuntimeError("Target batch was not found")
if int(episodes[args.batch_offset]) != args.episode:
    raise RuntimeError("Target episode changed")
raw = model_lib.Observation.from_dict(batch)
obs = model_lib.preprocess_observation(None, raw, train=False)
noise = jax.random.normal(noise_key, jnp.asarray(batch["actions"]).shape, dtype=jnp.float32)
base = config.model.load(model_lib.restore_params(checkpoint / "params", dtype=jnp.bfloat16))


def trace_fixed(self, rng, observation, *, num_steps, noise):
    observation = model_lib.preprocess_observation(None, observation, train=False)
    dt = -1.0 / num_steps
    batch_size = observation.state.shape[0]
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
    states = jnp.zeros((num_steps + 1, *noise.shape), dtype=noise.dtype).at[0].set(noise)
    velocities = jnp.zeros((num_steps, *noise.shape), dtype=noise.dtype)
    times = jnp.zeros((num_steps,), dtype=jnp.float32)

    def step(carry):
        x_t, time, i, states, velocities, times = carry
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, batch_size)
        )
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        next_x = x_t + dt * v_t
        states = states.at[i + 1].set(next_x)
        velocities = velocities.at[i].set(v_t)
        times = times.at[i].set(time)
        return next_x, time + dt, i + 1, states, velocities, times

    def cond(carry):
        return carry[1] >= -dt / 2

    return jax.lax.while_loop(cond, step, (noise, 1.0, 0, states, velocities, times))


Pi0.trace_fixed = trace_fixed
fixed = nnx_utils.module_jit(base.sample_actions, static_argnames=("num_steps",))
reference = nnx_utils.module_jit(base.trace_fixed, static_argnames=("num_steps",))
cached = nnx_utils.module_jit(base.cached_fixed_step_trace, static_argnames=("num_steps",))
prepare = nnx_utils.module_jit(base.prepare_action_prefix)
velocity = nnx_utils.module_jit(base.cached_velocity_and_action_features)
context = prepare(obs)
report = {
    "gpu_uuid": EXPECTED_UUID,
    "episode": int(episodes[args.batch_offset]),
    "batch_index": batch_index,
    "batch_offset": args.batch_offset,
    "steps": {},
}
arrays = {}
for n in (1, 2, 4, 10):
    production = np.asarray(fixed(jax.random.key(0), raw, num_steps=n, noise=noise))
    ref_action, _, _, ref_states, ref_velocities, ref_times = reference(
        jax.random.key(0), raw, num_steps=n, noise=noise
    )
    cached_action, cached_states, cached_velocities, _, cached_times = cached(
        jax.random.key(0), raw, num_steps=n, noise=noise
    )
    ref_states, ref_velocities, ref_times = map(np.asarray, (ref_states, ref_velocities, ref_times))
    cached_states, cached_velocities, cached_times = map(np.asarray, (cached_states, cached_velocities, cached_times))
    old_states = [np.asarray(noise)]
    old_velocities = []
    old_times = []
    x = noise
    for i in range(n):
        t = jnp.full((len(noise),), 1 - i / n, dtype=jnp.float32)
        v, _ = velocity(obs, x, t, context)
        old_velocities.append(np.asarray(v))
        old_times.append(float(t[0]))
        x = x - v / n
        old_states.append(np.asarray(x))
    old_states = np.stack(old_states)
    old_velocities = np.stack(old_velocities)
    row = args.batch_offset
    steps = [
        {
            "index": i,
            "reference_time": float(ref_times[i]),
            "cached_time": float(cached_times[i]),
            "old_time": old_times[i],
            "cached_state_max_abs": float(np.max(np.abs(ref_states[i, row] - cached_states[i, row]))),
            "cached_velocity_max_abs": float(np.max(np.abs(ref_velocities[i, row] - cached_velocities[i, row]))),
            "old_state_max_abs": float(np.max(np.abs(ref_states[i, row] - old_states[i, row]))),
            "old_velocity_max_abs": float(np.max(np.abs(ref_velocities[i, row] - old_velocities[i, row]))),
        }
        for i in range(n)
    ]
    item = {
        "reference_vs_production_final_max_abs": float(np.max(np.abs(np.asarray(ref_action) - production))),
        "cached_vs_production_final_max_abs": float(np.max(np.abs(np.asarray(cached_action) - production))),
        "old_vs_production_final_max_abs": float(np.max(np.abs(old_states[-1] - production))),
        "steps": steps,
    }
    report["steps"][str(n)] = item
    for label, value in (
        ("reference_states", ref_states),
        ("reference_velocities", ref_velocities),
        ("cached_states", cached_states),
        ("cached_velocities", cached_velocities),
        ("old_states", old_states),
        ("old_velocities", old_velocities),
    ):
        arrays[f"{n}_{label}"] = value[:, row]
    if item["reference_vs_production_final_max_abs"] > 1e-5:
        raise RuntimeError(f"Reference trace differs from production at {n} steps")
    if item["cached_vs_production_final_max_abs"] > 1e-5:
        raise RuntimeError(f"Cached trace differs from production at {n} steps")
    if any(s["cached_state_max_abs"] > 1e-5 or s["cached_velocity_max_abs"] > 1e-5 for s in steps):
        raise RuntimeError(f"Cached step trace differs at {n} steps")
head, head_metadata = load_head(args.head, checkpoint_identity(checkpoint))
single_observation = jax.tree.map(
    lambda leaf: leaf[row : row + 1] if hasattr(leaf, "shape") and leaf.shape and leaf.shape[0] == len(noise) else leaf,
    raw,
)
report["adaptive_constant_step"] = {}
for n in (1, 2, 4, 10):
    config = AdaptiveStepConfig(n, 1 / n, 1 / n, 1 / n, 0.05)
    sampler = AdaptiveEulerSampler(
        base, head, config, head_metadata=head_metadata, base_checkpoint_identity=checkpoint_identity(checkpoint)
    )
    adaptive = sampler.sample_actions(jax.random.key(0), single_observation, noise=noise[row : row + 1])
    production = fixed(jax.random.key(0), single_observation, num_steps=n, noise=noise[row : row + 1])
    diagnostic_state, diagnostic_sampler = _compile_adaptive_sample(base, config, record_states=True)
    _, _, _, _, _, _, _, adaptive_states, adaptive_velocities = diagnostic_sampler(
        diagnostic_state, head, single_observation, noise[row : row + 1]
    )
    _, _, _, single_states, single_velocities, single_times = reference(
        jax.random.key(0), single_observation, num_steps=n, noise=noise[row : row + 1]
    )
    adaptive_states = np.asarray(adaptive_states)[:, 0]
    adaptive_velocities = np.asarray(adaptive_velocities)[:, 0]
    single_states = np.asarray(single_states)[:, 0]
    single_velocities = np.asarray(single_velocities)[:, 0]
    adaptive_step_errors = [
        {
            "index": i,
            "time": float(single_times[i]),
            "state_max_abs": float(np.max(np.abs(adaptive_states[i] - single_states[i]))),
            "velocity_max_abs": float(np.max(np.abs(adaptive_velocities[i] - single_velocities[i]))),
        }
        for i in range(n)
    ]
    report["adaptive_constant_step"][str(n)] = {
        "velocity_evaluations": adaptive.velocity_evaluations,
        "per_step_errors": adaptive_step_errors,
        "final_max_abs": float(np.max(np.abs(np.asarray(adaptive.actions) - np.asarray(production)))),
        "times": [entry["time"] for entry in adaptive.trace],
        "steps": [entry["step"] for entry in adaptive.trace],
    }
    adaptive_result = report["adaptive_constant_step"][str(n)]
    if adaptive_result["velocity_evaluations"] != n:
        raise RuntimeError(f"Adaptive evaluation count differs at {n} steps")
    if adaptive_result["final_max_abs"] > 1e-5:
        raise RuntimeError(f"Adaptive final action differs from production at {n} steps")
    if any(item["state_max_abs"] > 1e-5 or item["velocity_max_abs"] > 1e-5 for item in adaptive_step_errors):
        raise RuntimeError(f"Adaptive intermediate state or velocity differs at {n} steps")
    if not np.allclose(adaptive_result["times"], np.asarray(single_times), rtol=0, atol=1e-6):
        raise RuntimeError(f"Adaptive flow times differ at {n} steps")
out = args.output
out.parent.mkdir(parents=True, exist_ok=True)
out.with_suffix(".json").write_text(json.dumps(report, indent=2) + chr(10))
np.savez_compressed(out.with_suffix(".npz"), **arrays)
print(
    json.dumps(
        {
            "report": str(out.with_suffix(".json")),
            "npz": str(out.with_suffix(".npz")),
            "old_final_10": report["steps"]["10"]["old_vs_production_final_max_abs"],
        }
    ),
    flush=True,
)
