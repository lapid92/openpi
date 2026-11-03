"""Pinned JAX pi0.5 checkpoint loading and counted sampler."""
from __future__ import annotations
import dataclasses
import hashlib
import pathlib
import time
import types
from .server_common import take_request, finish, write_metadata

def pi05_inference_config(checkpoint):
    """Use supported pre-supplied stats; preserve original factory and transforms."""
    from openpi.shared import normalize
    from openpi.training import config
    checkpoint = pathlib.Path(checkpoint)
    stats_path = checkpoint / "assets/norm_stats.json"
    stats = normalize.load(stats_path.parent)
    original = config.get_config("pi05_pretrain_human300")
    base = dataclasses.replace(original.data.base_config or config.DataConfig(), norm_stats=stats)
    trained = dataclasses.replace(original, data=dataclasses.replace(original.data, base_config=base))
    resolved = trained.data.create(trained.assets_dirs, trained.model)
    if resolved.norm_stats is not stats:
        raise RuntimeError("Data factory did not preserve explicit checkpoint normalization")
    effective_json = normalize.serialize_json(stats)
    evidence = {
        "source": str(stats_path),
        "file_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
        "effective_stats_sha256": hashlib.sha256(effective_json.encode()).hexdigest(),
        "effective_stats_serialization": "openpi.shared.normalize.serialize_json",
        "keys": sorted(stats),
        "use_quantile_norm": resolved.use_quantile_norm,
        "training_data_access": "bypassed via supported base_config.norm_stats injection",
    }
    resolved_config = {
        "train_config_name": trained.name, "model": dataclasses.asdict(trained.model),
        "asset_id": resolved.asset_id, "use_quantile_norm": resolved.use_quantile_norm,
        "data_inputs": [type(x).__name__ for x in resolved.data_transforms.inputs],
        "data_outputs": [type(x).__name__ for x in resolved.data_transforms.outputs],
        "model_inputs": [type(x).__name__ for x in resolved.model_transforms.inputs],
        "model_outputs": [type(x).__name__ for x in resolved.model_transforms.outputs],
    }
    return trained, stats, evidence, resolved_config


def counted_pi05_sample(self, rng, observation, *, num_steps=10, noise=None):
    """Pinned pi0.py sampler with only an integer count added to the loop carry.

    One count increment follows each completed action_out_proj. Prefix encoding
    is not an action-network call. Floating-point loop termination is unchanged.
    """
    import einops
    import jax
    import jax.numpy as jnp
    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask

    observation = _model.preprocess_observation(None, observation, train=False)
    dt = -1.0 / num_steps
    batch_size = observation.state.shape[0]
    if noise is None:
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    def step(carry):
        x_t, timestep, count = carry
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, jnp.broadcast_to(timestep, batch_size)
        )
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        assert full_attn_mask.shape == (
            batch_size, suffix_tokens.shape[1], prefix_tokens.shape[1] + suffix_tokens.shape[1]
        )
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens], mask=full_attn_mask, positions=positions,
            kv_cache=kv_cache, adarms_cond=[None, adarms_cond],
        )
        assert prefix_out is None
        velocity = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        return x_t + dt * velocity, timestep + dt, count + jnp.int32(1)

    def cond(carry):
        _, timestep, _ = carry
        return timestep >= -dt / 2

    actions, _, count = jax.lax.while_loop(cond, step, (noise, 1.0, jnp.int32(0)))
    return actions, count


class Pi05Adapter:
    def __init__(self, checkpoint, steps, verify_sampler=True):
        from openpi.policies import policy_config
        from openpi.shared import nnx_utils
        self.steps = steps
        trained, stats, normalization, resolved_config = pi05_inference_config(checkpoint)
        self.policy = policy_config.create_trained_policy(
            trained, checkpoint, norm_stats=stats,
            sample_kwargs={"num_steps": steps},
        )
        if self.policy._is_pytorch_model:
            raise ValueError("This adapter requires the released JAX checkpoint")
        self.sample = nnx_utils.module_jit(types.MethodType(counted_pi05_sample, self.policy._model))
        self.verify_pending = verify_sampler
        self.warmup_pending = True
        self.metadata = {
            "model": "pi05", "steps": steps, "checkpoint": str(checkpoint),
            "normalization": normalization, "resolved_config": resolved_config,
            "noise_rng": "jax.random.normal(jax.random.key(query_seed))",
            "counter": "actual JAX while-loop carry; excludes prefix encoding",
            "sampler_parity": "pending" if verify_sampler else "not_requested",
        }

    def reset(self):
        pass  # No queued actions or evolving policy RNG in this adapter.

    def infer(self, obs):
        import jax
        import jax.numpy as jnp
        import numpy as np
        from openpi.models import model
        started = time.monotonic()
        obs, seed = take_request(obs)
        inputs = self.policy._input_transform(obs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
        observation = model.Observation.from_dict(inputs)
        key = jax.random.key(seed)
        noise = jax.random.normal(key, (1, self.policy._model.action_horizon, self.policy._model.action_dim))
        jax.block_until_ready((inputs, noise))
        t = time.monotonic()
        actions, count = self.sample(key, observation, num_steps=self.steps, noise=noise)
        jax.block_until_ready((actions, count))
        elapsed = time.monotonic() - t
        count = int(np.asarray(count))
        if count != self.steps:
            raise RuntimeError(f"Actual action-network calls {count} != requested {self.steps}")
        # First request verifies the minimally changed sampler against the exact
        # released sampler on identical inputs/noise. Excluded from model latency.
        if self.verify_pending:
            reference = self.policy._sample_actions(key, observation, num_steps=self.steps, noise=noise)
            reference = np.asarray(reference)
            np.testing.assert_allclose(np.asarray(actions), reference, rtol=1e-5, atol=1e-5)
            self.verify_pending = False
            self.metadata["sampler_parity"] = "passed_first_query_rtol1e-5_atol1e-5"
            if getattr(self, "metadata_path", None):
                write_metadata(self.metadata_path, self.metadata)
        output = self.policy._output_transform({
            "state": np.asarray(inputs["state"][0]), "actions": np.asarray(actions[0])
        })
        result = finish(output["actions"], seed, count, elapsed, started, self.metadata)
        result["warmup_query"] = self.warmup_pending
        self.warmup_pending = False
        return result
