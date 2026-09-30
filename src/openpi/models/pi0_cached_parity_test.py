"""Per-step parity of production fixed Euler semantics and the cached suffix path."""

import einops
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as model_lib
from openpi.models.pi0 import make_attn_mask
from openpi.models.pi0_config import Pi0Config
from openpi.shared import nnx_utils


def fixed_step_trace(base, observation, noise, num_steps):
    """Mirror Pi0.sample_actions step operations while retaining its intermediates.

    This intentionally duplicates the production fixed-step path for a regression
    oracle; it does not call the cached feature helper under test.
    """
    observation = model_lib.preprocess_observation(None, observation, train=False)
    batch_size = observation.state.shape[0]
    dt = -1.0 / num_steps
    prefix_tokens, prefix_mask, prefix_ar_mask = base.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = base.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    action, time = noise, jnp.asarray(1.0, dtype=jnp.float32)
    trace = []
    for _ in range(num_steps):
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = base.embed_suffix(
            observation, action, jnp.broadcast_to(time, batch_size)
        )
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        suffix_prefix_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([suffix_prefix_mask, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (prefix_out, suffix_out), _ = base.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        assert prefix_out is None
        velocity = base.action_out_proj(suffix_out[:, -base.action_horizon :])
        next_action = action + dt * velocity
        trace.append((time, action, velocity, next_action))
        action, time = next_action, time + dt
    return trace, action


def cached_step_trace(base, observation, noise, num_steps):
    run = nnx_utils.module_jit(base.cached_fixed_step_trace, static_argnames=("num_steps",))
    final, states, velocities, features, times = run(jax.random.key(2), observation, num_steps=num_steps, noise=noise)
    assert features.shape == (num_steps, noise.shape[0], base.action_horizon, base.action_in_proj.out_features)
    trace = [(times[i], states[i], velocities[i], states[i + 1]) for i in range(num_steps)]
    return trace, final


@pytest.mark.parametrize("num_steps", [2, 4])
def test_cached_path_matches_fixed_step_intermediates(num_steps):
    config = Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=2,
        action_dim=4,
        max_token_len=8,
        dtype="float32",
    )
    base = config.create(jax.random.key(0))
    observation = config.fake_obs()
    noise = jax.random.normal(jax.random.key(1), (1, 2, 4), dtype=jnp.float32)
    fixed_trace, fixed_final = fixed_step_trace(base, observation, noise, num_steps)
    cached_trace, cached_final = cached_step_trace(base, observation, noise, num_steps)

    # Float32 CPU paths may reorder fused arithmetic; 1e-5 bounds that roundoff.
    for step, (fixed, cached) in enumerate(zip(fixed_trace, cached_trace, strict=True)):
        for field, left, right in zip(("time", "action", "velocity", "next_action"), fixed, cached, strict=True):
            np.testing.assert_allclose(
                np.asarray(left),
                np.asarray(right),
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"step={step} field={field}",
            )
    np.testing.assert_allclose(np.asarray(cached_final), np.asarray(fixed_final), rtol=1e-5, atol=1e-5)
    production = nnx_utils.module_jit(base.sample_actions, static_argnames=("num_steps",))(
        jax.random.key(2), observation, num_steps=num_steps, noise=noise
    )
    np.testing.assert_allclose(np.asarray(fixed_final), np.asarray(production), rtol=1e-5, atol=1e-5)
