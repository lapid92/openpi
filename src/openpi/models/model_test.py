from flax import nnx
import jax
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


@pytest.mark.parametrize("alpha", [0.0, 0.125, 0.25])
def test_pi05_tvm_loss_is_finite_and_shape_consistent(alpha: float):
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
    if alpha == 0.0:
        assert jax.numpy.array_equal(losses["loss"], losses["fm_loss"])
        assert jax.numpy.count_nonzero(losses["tvm_loss"]) == 0
        # TVM deliberately retains its uniform diagonal-time FM sampler at alpha=0;
        # it is not expected to reproduce compute_loss's beta-distributed sample.
        baseline_loss = model.compute_loss(key, obs, act)
        assert not jax.numpy.allclose(losses["fm_loss"], baseline_loss)


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
