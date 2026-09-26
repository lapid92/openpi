from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import pi0_fast
from openpi.shared import download
from openpi.shared import nnx_utils


def test_pi0_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


@pytest.mark.parametrize("mode", ["eval", "train"])
def test_pi05_zero_tvm_alpha_matches_baseline_loss_exactly(mode: str):
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy")
    model = config.create(key)
    teacher = config.create(jax.random.key(1))
    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)
    train = mode == "train"

    losses = model.compute_tvm_loss(
        key,
        obs,
        act,
        teacher=teacher,
        alpha=0.0,
        fm_loss_weight=1.0,
        train=train,
    )
    baseline_loss = model.compute_loss(key, obs, act, train=train)

    assert set(losses) == {"loss", "fm_loss", "tvm_loss"}
    for value in losses.values():
        assert value.shape == (batch_size, config.action_horizon)
        assert jax.numpy.all(jax.numpy.isfinite(value))
    assert jax.numpy.array_equal(losses["loss"], baseline_loss)
    assert jax.numpy.array_equal(losses["fm_loss"], baseline_loss)
    assert jax.numpy.count_nonzero(losses["tvm_loss"]) == 0


def test_pi05_two_time_zero_initialized_path_preserves_one_time_output():
    key = jax.random.key(7)
    config = pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy")
    model = config.create(key)
    observation = _model.preprocess_observation(key, config.fake_obs(2), train=False)
    actions = config.fake_act(2)
    prefix = model.embed_prefix(observation)

    for target in (0.05, 0.4, 0.9):
        target_time = jnp.full((2,), target)
        baseline = model._predict_velocity_from_prefix(  # noqa: SLF001
            observation, actions, None, target_time, *prefix
        )
        for source in (0.1, 0.6, 1.0):
            source_time = jnp.full((2,), source)
            actual = model._predict_velocity_from_prefix(  # noqa: SLF001
                observation, actions, source_time, target_time, *prefix
            )
            np.testing.assert_allclose(actual, baseline, rtol=1e-5, atol=1e-5)


def test_pi05_target_time_jvp_holds_actions_and_source_time_fixed():
    key = jax.random.key(8)
    config = pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy")
    model = config.create(key)
    observation = _model.preprocess_observation(key, config.fake_obs(2), train=False)
    actions = config.fake_act(2)
    prefix = model.embed_prefix(observation)
    source_time = jnp.array([0.8, 0.9])
    target_time = jnp.array([0.2, 0.4])

    def at_target(s):
        return model._predict_velocity_from_prefix(  # noqa: SLF001
            observation, actions, source_time, s, *prefix
        )

    _, derivative = jax.jvp(at_target, (target_time,), (jnp.ones_like(target_time),))
    epsilon = 1e-3
    finite_difference = (at_target(target_time + epsilon) - at_target(target_time - epsilon)) / (2 * epsilon)
    assert jnp.all(jnp.isfinite(derivative))
    np.testing.assert_allclose(derivative, finite_difference, rtol=3e-2, atol=3e-2)


def test_pi05_diagonal_conditioning_stays_exact_after_source_path_update():
    config = pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy")
    model = config.create(jax.random.key(9))
    assert set(nnx.state(model).to_pure_dict()["source_time_proj"]) == {"kernel"}
    width = model.action_in_proj.out_features
    model.source_time_proj.kernel.value = jnp.eye(width, dtype=model.source_time_proj.kernel.value.dtype) * 0.1
    obs, actions = config.fake_obs(2), config.fake_act(2)
    target_time = jnp.array([0.2, 0.4])
    source_time = jnp.array([0.8, 0.9])

    baseline = model.embed_suffix(obs, actions, target_time)[-1]
    diagonal = model.embed_suffix(obs, actions, target_time, target_time)[-1]
    off_diagonal = model.embed_suffix(obs, actions, target_time, source_time)[-1]
    np.testing.assert_array_equal(diagonal, baseline)
    assert jnp.any(off_diagonal != baseline)


@pytest.mark.parametrize("alpha", [0.125, 0.25])
def test_pi05_positive_tvm_loss_is_finite_and_shape_consistent(alpha: float):
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy")
    model = config.create(key)
    teacher = config.create(jax.random.key(1))
    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    losses = model.compute_tvm_loss(
        key,
        obs,
        act,
        teacher=teacher,
        alpha=alpha,
        fm_loss_weight=1.0,
    )

    assert set(losses) == {"loss", "fm_loss", "tvm_loss"}
    for value in losses.values():
        assert value.shape == (batch_size, config.action_horizon)
        assert jax.numpy.all(jax.numpy.isfinite(value))


def test_pi05_tvm_loss_has_finite_student_gradients_and_does_not_mutate_teacher():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy")
    model = config.create(key)
    teacher = config.create(jax.random.key(1))
    obs, act = config.fake_obs(batch_size=2), config.fake_act(batch_size=2)
    teacher_before = nnx.state(teacher).to_pure_dict()

    def loss_fn(differentiable_model, teacher_model):
        losses = differentiable_model.compute_tvm_loss(
            key,
            obs,
            act,
            teacher=teacher_model,
            alpha=0.25,
            fm_loss_weight=1.0,
        )
        return jax.numpy.mean(losses["loss"])

    loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param))(model, teacher)

    assert jax.numpy.isfinite(loss)
    grad_leaves = jax.tree.leaves(grads)
    assert grad_leaves
    assert all(jax.numpy.all(jax.numpy.isfinite(grad)) for grad in grad_leaves)
    assert any(jax.numpy.any(grad != 0) for grad in grad_leaves)
    source_grad_leaves = jax.tree.leaves(grads["source_time_proj"])
    assert source_grad_leaves
    assert all(jnp.all(jnp.isfinite(grad)) for grad in source_grad_leaves)
    # The dummy Gemma's adaRMS modulation starts at zero, so this path receives no gradient
    # until a released checkpoint supplies trained modulation weights.

    teacher_after = nnx.state(teacher).to_pure_dict()
    teacher_equal = jax.tree.map(jax.numpy.array_equal, teacher_before, teacher_after)
    assert all(jax.tree.leaves(teacher_equal))


def test_pi0_lora_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_fast_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)


def test_pi0_fast_lora_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)

    lora_filter = nnx_utils.PathRegex(".*lora.*")
    model_state = nnx.state(model)

    lora_state_elems = list(model_state.filter(lora_filter))
    assert len(lora_state_elems) > 0


@pytest.mark.manual
def test_model_restore():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    model = config.load(
        _model.restore_params(download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params"))
    )

    loss = model.compute_loss(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = model.sample_actions(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)
