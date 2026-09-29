"""Focused checks for the frozen-base velocity residual head."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from openpi.models.pi0_config import Pi0Config
from openpi.models.velocity_residual import FEATURE_WIDTH
from openpi.models.velocity_residual import gaussian_nll
from openpi.models.velocity_residual import init_head
from openpi.models.velocity_residual import load_head
from openpi.models.velocity_residual import predict_log_sigma
from openpi.models.velocity_residual import residual_energy
from openpi.models.velocity_residual import sample_flow
from openpi.models.velocity_residual import save_head
from openpi.models.velocity_residual import split_episodes
from openpi.models.velocity_residual import split_identity
from openpi.shared import nnx_utils


def test_feature_shape_and_head_dimensions():
    params = init_head(jax.random.key(0))
    assert params["layer_0"]["kernel"].shape == (1056, 256)
    assert params["layer_1"]["kernel"].shape == (256, 64)
    assert params["layer_2"]["kernel"].shape == (64, 1)
    prediction = predict_log_sigma(params, jnp.zeros((3, 10, FEATURE_WIDTH)), jnp.array([0.1, 0.5, 0.9]))
    assert prediction.shape == (3,)
    padded = jnp.zeros((1, 2, FEATURE_WIDTH)).at[:, 1, :].set(1000)
    baseline = predict_log_sigma(params, jnp.zeros((1, 2, FEATURE_WIDTH)), jnp.array([0.5]))
    masked = predict_log_sigma(params, padded, jnp.array([0.5]), jnp.array([[True, False]]))
    np.testing.assert_array_equal(np.asarray(masked), np.asarray(baseline))
    with pytest.raises(ValueError, match="Expected"):
        predict_log_sigma(params, jnp.zeros((3, 10, 64)), jnp.ones(3))


def test_flow_time_and_target():
    actions = jnp.ones((8192, 10, 32))
    x_t, time, target = sample_flow(jax.random.key(0), actions)
    assert float(time.min()) >= 0.001
    assert float(time.max()) <= 1.0
    assert 0.59 < float(time.mean()) < 0.62  # Beta(1.5,1) mean, with endpoint rescaling.
    noise = target + actions
    np.testing.assert_allclose(
        np.asarray(x_t), np.asarray(time[:, None, None] * noise + (1 - time[:, None, None]) * actions), atol=1e-6
    )


def test_residual_reduction_padding_and_nll_optimum():
    residual = jnp.array([[[2.0, 0.0], [100.0, 100.0]]])
    energy = residual_energy(residual, jnp.array([[True, False]]))
    np.testing.assert_allclose(np.asarray(energy), [2.0])
    with pytest.raises(ValueError, match="valid action"):
        residual_energy(residual, jnp.array([[False, False]]))
    optimum = 0.5 * jnp.log(energy)
    assert float(gaussian_nll(optimum, energy)) < float(gaussian_nll(optimum + 0.2, energy))
    assert float(gaussian_nll(optimum, energy)) < float(gaussian_nll(optimum - 0.2, energy))
    assert np.isfinite(float(gaussian_nll(jnp.array([1000.0]), jnp.array([1e6]))))


def test_episode_isolation_and_identity():
    ids = np.repeat(np.arange(20), 3)
    train, val, train_ids, val_ids = split_episodes(ids, seed=42)
    assert not np.intersect1d(ids[train], ids[val]).size
    assert len(train) + len(val) == len(ids)
    assert split_identity(train_ids, val_ids, "revision") != split_identity(train_ids, val_ids, "other")


def test_checkpoint_reload_and_base_freeze(tmp_path):
    head = init_head(jax.random.key(1))
    frozen_base = {"weight": jnp.ones((2, 2))}
    base_before = np.asarray(frozen_base["weight"]).copy()
    energy = jnp.array([3.0])
    features = jnp.ones((1, 10, FEATURE_WIDTH))
    time = jnp.array([0.3])
    optimizer = optax.adam(1e-3)
    state = optimizer.init(head)

    def loss_fn(p):
        return gaussian_nll(predict_log_sigma(p, features, time), energy)

    grads = jax.grad(loss_fn)(head)
    updates, _ = optimizer.update(grads, state, head)
    changed = optax.apply_updates(head, updates)
    np.testing.assert_array_equal(np.asarray(frozen_base["weight"]), base_before)
    assert any(
        not np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(jax.tree.leaves(head), jax.tree.leaves(changed), strict=True)
    )
    path = tmp_path / "head.npz"
    save_head(path, changed, {"base_checkpoint_identity": "base-id", "dataset_revision": "training"})
    reloaded, metadata = load_head(path, "base-id")
    assert metadata["dataset_revision"] == "training"
    for a, b in zip(jax.tree.leaves(changed), jax.tree.leaves(reloaded), strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    with pytest.raises(ValueError, match="Wrong frozen checkpoint"):
        load_head(path, "wrong-base")


def test_disabled_head_preserves_pi05_inference():
    """A separate head update cannot mutate the base model or its action sampler."""
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
    sample = nnx_utils.module_jit(base.sample_actions, static_argnames=("num_steps",))
    observation = config.fake_obs()
    noise = jnp.ones((1, 2, 4))
    before = sample(jax.random.key(1), observation, num_steps=1, noise=noise)
    head = init_head(jax.random.key(2))
    features = jnp.ones((1, 2, FEATURE_WIDTH))

    def loss_fn(p):
        return gaussian_nll(predict_log_sigma(p, features, jnp.array([0.5])), jnp.array([2.0]))

    gradients = jax.grad(loss_fn)(head)
    head = jax.tree.map(lambda p, g: p - 1e-3 * g, head, gradients)
    assert head["layer_2"]["kernel"].shape == (64, 1)
    after = sample(jax.random.key(1), observation, num_steps=1, noise=noise)
    np.testing.assert_array_equal(np.asarray(before), np.asarray(after))
