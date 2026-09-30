"""Cached action-expert feature extraction preserves fixed-step π₀.₅ output."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models.pi0_config import Pi0Config
from openpi.shared import nnx_utils


@pytest.mark.parametrize("num_steps", [1, 2, 4, 10])
def test_cached_euler_matches_fixed_sampler(num_steps):
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
    noise = jnp.ones((1, 2, 4))
    sample = nnx_utils.module_jit(base.sample_actions, static_argnames=("num_steps",))
    fixed_before = sample(jax.random.key(1), observation, num_steps=num_steps, noise=noise)

    preprocessed = _model.preprocess_observation(None, observation, train=False)
    context = base.prepare_action_prefix(preprocessed)
    action = noise
    for step in range(num_steps):
        time = jnp.full((1,), 1 - step / num_steps, dtype=jnp.float32)
        velocity, features = base.cached_velocity_and_action_features(preprocessed, action, time, context)
        assert velocity.shape == noise.shape
        assert features.shape == (1, 2, base.action_in_proj.out_features)
        action = action - velocity / num_steps

    np.testing.assert_allclose(np.asarray(action), np.asarray(fixed_before), rtol=1e-5, atol=1e-5)
    fixed_after = sample(jax.random.key(1), observation, num_steps=num_steps, noise=noise)
    np.testing.assert_array_equal(np.asarray(fixed_before), np.asarray(fixed_after))
