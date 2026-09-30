"""Focused CPU-only tests for bounded adaptive Euler integration."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import adaptive_euler
from openpi.models.adaptive_euler import AdaptiveEulerSampler
from openpi.models.adaptive_euler import AdaptiveStepConfig
from openpi.models.pi0_config import Pi0Config
from openpi.models.velocity_residual import init_head
from openpi.models.velocity_residual import load_head
from openpi.models.velocity_residual import save_head
from openpi.shared import nnx_utils


@pytest.mark.parametrize(
    "changes",
    [
        {"max_velocity_evaluations": 0},
        {"max_velocity_evaluations": 1.5},
        {"min_step": 0},
        {"min_step": -0.1},
        {"max_step": float("inf")},
        {"reference_step": float("nan")},
        {"reference_sigma": 0},
        {"score_power": -1},
        {"min_step": 0.6, "max_step": 0.5},
        {"max_step": 1.1},
        {"max_velocity_evaluations": 2, "max_step": 0.4},
    ],
)
def test_config_rejects_invalid_bounds(changes):
    kwargs = {
        "max_velocity_evaluations": 3,
        "min_step": 0.1,
        "max_step": 0.5,
        "reference_step": 0.2,
        "reference_sigma": 1.0,
    }
    kwargs.update(changes)
    with pytest.raises(ValueError, match=r"positive|finite|Require|cover"):
        AdaptiveStepConfig(**kwargs)


def test_step_mapping_clips_and_rejects_invalid_state():
    config = AdaptiveStepConfig(3, 0.1, 0.5, 0.2, 1.0)
    assert config.proposed_step(100) == pytest.approx(0.1)
    assert config.proposed_step(0.01) == pytest.approx(0.5)
    assert config.bounded_step(1.0, 1.0, 0) == pytest.approx(1 / 3)
    for sigma in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="Predicted sigma"):
            config.proposed_step(sigma)
    for time, used in ((0, 0), (1.1, 0), (0.5, 3), (0.5, -1)):
        with pytest.raises(ValueError, match="Invalid integration state"):
            config.bounded_step(time, 1.0, used)


class FakeFrozenBase:
    pi05 = True
    action_horizon = 2
    action_dim = 4

    def __init__(self):
        self.action_out_proj = SimpleNamespace(kernel=SimpleNamespace(value=np.zeros((1024, 4))))
        self.prefix_calls = 0
        self.velocity_calls = []
        self.sample_actions = lambda noise: noise - 0.25

    def prepare_action_prefix(self, observation):
        self.prefix_calls += 1
        return object()

    def cached_velocity_and_action_features(self, observation, actions, time, context):
        self.velocity_calls.append(float(np.asarray(time)[0]))
        feature_value = len(self.velocity_calls)
        features = jnp.full((actions.shape[0], self.action_horizon, 1024), feature_value, dtype=jnp.float32)
        return jnp.ones_like(actions), features


def make_fake_sampler(monkeypatch, config, sigma=1.0):
    base = FakeFrozenBase()
    sigma_features = []

    def predict(head, features, time):
        sigma_features.append((float(np.asarray(features)[0, 0, 0]), float(np.asarray(time)[0])))
        return jnp.array([np.log(sigma)], dtype=jnp.float32)

    monkeypatch.setattr(adaptive_euler, "predict_log_sigma", predict)

    def fake_compile(frozen_base, step_config):
        frozen_state = object()

        def run(state, head, observation, noise):
            assert state is frozen_state
            context = frozen_base.prepare_action_prefix(observation)
            action = noise
            time = 1.0
            times = np.zeros(step_config.max_velocity_evaluations, dtype=np.float32)
            sigmas = np.zeros_like(times)
            steps = np.zeros_like(times)
            next_times = np.zeros_like(times)
            count = 0
            while time > 0 and count < step_config.max_velocity_evaluations:
                velocity, features = frozen_base.cached_velocity_and_action_features(
                    observation, action, jnp.array([time], dtype=jnp.float32), context
                )
                predicted_sigma = float(np.exp(np.asarray(predict(head, features, jnp.array([time]))[0])))
                step = step_config.bounded_step(time, predicted_sigma, count)
                following = max(0.0, time - step)
                times[count], sigmas[count], steps[count], next_times[count] = time, predicted_sigma, step, following
                action = action + (-step) * velocity
                time = following
                count += 1
            return action, jnp.asarray(time), jnp.asarray(count), times, sigmas, steps, next_times

        return frozen_state, run

    monkeypatch.setattr(adaptive_euler, "_compile_adaptive_sample", fake_compile)
    head = init_head(jax.random.key(0))
    metadata = {"base_checkpoint_identity": "checkpoint-abc", "model_config": "pi05_libero", "feature_width": 1024}
    sampler = AdaptiveEulerSampler(
        base, head, config, head_metadata=metadata, base_checkpoint_identity="checkpoint-abc"
    )
    return base, sampler, sigma_features


def test_backward_time_exact_termination_and_same_pass_features(monkeypatch):
    config = AdaptiveStepConfig(3, 0.1, 0.5, 0.2, 1.0)
    base, sampler, sigma_features = make_fake_sampler(monkeypatch, config)
    noise = jnp.zeros((1, 2, 4), dtype=jnp.float32)
    observation = SimpleNamespace(state=np.zeros((1, 4)))
    fixed_before = base.sample_actions(noise)
    result = sampler.sample_actions(jax.random.key(0), observation, noise=noise)

    assert result.velocity_evaluations == 3
    assert base.prefix_calls == 1
    assert len(base.velocity_calls) == len(sigma_features) == result.velocity_evaluations
    assert [entry["time"] for entry in result.trace] == pytest.approx(base.velocity_calls)
    assert [entry["time"] for entry in result.trace] == pytest.approx([t for _, t in sigma_features])
    assert [feature for feature, _ in sigma_features] == [1.0, 2.0, 3.0]
    assert all(entry["next_time"] < entry["time"] for entry in result.trace)
    assert all(entry["step"] <= config.max_step for entry in result.trace)
    assert result.trace[-1]["next_time"] == 0.0
    np.testing.assert_allclose(np.asarray(result.actions), -np.ones((1, 2, 4)), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(base.sample_actions(noise)), np.asarray(fixed_before))


def test_batch_one_and_noise_shape_enforced(monkeypatch):
    config = AdaptiveStepConfig(2, 0.1, 0.5, 0.5, 1.0)
    base, sampler, _ = make_fake_sampler(monkeypatch, config)
    with pytest.raises(ValueError, match="batch size one"):
        sampler.sample_actions(jax.random.key(0), SimpleNamespace(state=np.zeros((2, 4))))
    with pytest.raises(ValueError, match="noise shape"):
        sampler.sample_actions(jax.random.key(0), SimpleNamespace(state=np.zeros((1, 4))), noise=np.zeros((1, 3, 4)))
    assert base.prefix_calls == 0


def test_head_reload_requires_matching_frozen_checkpoint(tmp_path):
    path = tmp_path / "head.npz"
    head = init_head(jax.random.key(3))
    save_head(path, head, {"base_checkpoint_identity": "checkpoint-abc"})
    restored, metadata = load_head(path, "checkpoint-abc")
    assert metadata["base_checkpoint_identity"] == "checkpoint-abc"
    for layer in head:
        for field in head[layer]:
            np.testing.assert_array_equal(np.asarray(restored[layer][field]), np.asarray(head[layer][field]))
    with pytest.raises(ValueError, match="Wrong frozen checkpoint"):
        load_head(path, "checkpoint-other")


def test_extreme_score_mapping_is_finite_and_clipped():
    config = AdaptiveStepConfig(2, 0.1, 0.5, 0.2, 1.0, score_power=1000.0)
    assert config.proposed_step(1e-300) == pytest.approx(0.5)
    assert config.proposed_step(1e300) == pytest.approx(0.1)


def test_constructor_rejects_checkpoint_and_shape_mismatch(monkeypatch):
    config = AdaptiveStepConfig(2, 0.1, 0.5, 0.5, 1.0)
    head = init_head(jax.random.key(0))
    metadata = {"base_checkpoint_identity": "checkpoint-abc", "model_config": "pi05_libero", "feature_width": 1024}
    with pytest.raises(ValueError, match="checkpoint identity"):
        AdaptiveEulerSampler(
            FakeFrozenBase(), head, config, head_metadata=metadata, base_checkpoint_identity="checkpoint-other"
        )
    bad_head = {**head, "layer_2": {**head["layer_2"], "kernel": jnp.zeros((63, 1))}}
    with pytest.raises(ValueError, match="parameter shapes"):
        AdaptiveEulerSampler(
            FakeFrozenBase(), bad_head, config, head_metadata=metadata, base_checkpoint_identity="checkpoint-abc"
        )


def test_from_paths_loads_matching_head_and_rejects_wrong_identity(tmp_path, monkeypatch):
    checkpoint = tmp_path / "base"
    checkpoint.mkdir()
    head_path = tmp_path / "head.npz"
    head = init_head(jax.random.key(0))
    metadata = {"base_checkpoint_identity": "checkpoint-abc", "model_config": "pi05_libero", "feature_width": 1024}
    save_head(head_path, head, metadata)
    config = AdaptiveStepConfig(2, 0.1, 0.5, 0.5, 1.0)
    monkeypatch.setattr(adaptive_euler, "checkpoint_identity", lambda path: "checkpoint-other")
    with pytest.raises(ValueError, match="Wrong frozen checkpoint"):
        AdaptiveEulerSampler.from_paths(checkpoint, head_path, config)

    fake_base = FakeFrozenBase()
    monkeypatch.setattr(adaptive_euler, "checkpoint_identity", lambda path: "checkpoint-abc")
    monkeypatch.setattr(
        adaptive_euler.config_lib,
        "get_config",
        lambda name: SimpleNamespace(model=SimpleNamespace(load=lambda params: fake_base)),
    )
    monkeypatch.setattr(adaptive_euler.model_lib, "restore_params", lambda path, dtype: object())
    monkeypatch.setattr(adaptive_euler, "_compile_adaptive_sample", lambda base, rule: (object(), lambda *args: None))
    sampler = AdaptiveEulerSampler.from_paths(checkpoint, head_path, config)
    assert isinstance(sampler, AdaptiveEulerSampler)


@pytest.mark.parametrize(("num_steps", "model_dtype"), [(2, "float32"), (10, "bfloat16")])
def test_compiled_dummy_adaptive_matches_fixed_output(monkeypatch, num_steps, model_dtype):
    config = Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=2,
        action_dim=4,
        max_token_len=8,
        dtype=model_dtype,
    )
    base = config.create(jax.random.key(0))
    head = init_head(jax.random.key(1))
    # Dummy expert features are 64 wide; use a JIT-safe constant score here.
    monkeypatch.setattr(adaptive_euler, "predict_log_sigma", lambda params, features, time: jnp.zeros_like(time))
    step_config = AdaptiveStepConfig(
        max_velocity_evaluations=num_steps,
        min_step=1 / num_steps,
        max_step=1 / num_steps,
        reference_step=1 / num_steps,
        reference_sigma=1.0,
    )
    frozen_state, compiled = adaptive_euler._compile_adaptive_sample(base, step_config, record_states=True)  # noqa: SLF001
    observation = config.fake_obs()
    noise = jax.random.normal(jax.random.key(2), (1, 2, 4), dtype=jnp.float32)
    fixed = nnx_utils.module_jit(base.sample_actions, static_argnames=("num_steps",))(
        jax.random.key(3), observation, num_steps=num_steps, noise=noise
    )
    actions, time, count, times, sigmas, steps, next_times, states, velocities = compiled(
        frozen_state, head, observation, noise
    )
    assert states.shape == (num_steps + 1, 1, 2, 4)
    assert velocities.shape == (num_steps, 1, 2, 4)
    assert int(count) == num_steps
    assert float(time) == 0.0
    expected_times = [1 - i / num_steps for i in range(num_steps)]
    expected_next_times = [max(0, 1 - (i + 1) / num_steps) for i in range(num_steps)]
    np.testing.assert_allclose(np.asarray(times), expected_times, atol=1e-6)
    np.testing.assert_allclose(np.asarray(steps), [1 / num_steps] * num_steps, atol=1e-6)
    np.testing.assert_allclose(np.asarray(next_times), expected_next_times, atol=1e-6)
    assert np.all(np.isfinite(np.asarray(sigmas)))
    np.testing.assert_allclose(np.asarray(actions), np.asarray(fixed), atol=1e-5, rtol=1e-5)
    if model_dtype == "bfloat16":
        normal_state, normal_compiled = adaptive_euler._compile_adaptive_sample(base, step_config)  # noqa: SLF001
        normal_actions, normal_time, normal_count, *_ = normal_compiled(normal_state, head, observation, noise)
        assert int(normal_count) == num_steps
        assert float(normal_time) == 0.0
        np.testing.assert_allclose(np.asarray(normal_actions), np.asarray(fixed), atol=1e-5, rtol=1e-5)
        np.testing.assert_array_equal(np.asarray(states[-1]), np.asarray(actions))
        assert np.all(np.isfinite(np.asarray(velocities)))
