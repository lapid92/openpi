"""Focused tests for the two-time sampler's time arguments and legacy aliases."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import pi0


class _FakeLLM:
    def __call__(self, tokens, *, mask, positions, kv_cache=None, adarms_cond=None):
        del mask, positions, adarms_cond
        if kv_cache is None:
            return (None, None), object()
        return (None, tokens[1]), None


def _fake_model():
    """The suffix encodes a known velocity, avoiding heavyweight model weights."""

    def embed_suffix(_observation, actions, target_time, source_time):
        velocity = actions + target_time[:, None, None] + 2 * source_time[:, None, None]
        batch = actions.shape[0]
        return velocity, jnp.ones((batch, 1), dtype=jnp.bool_), jnp.zeros((1,), dtype=jnp.bool_), None

    return SimpleNamespace(
        action_horizon=1,
        action_dim=1,
        PaliGemma=SimpleNamespace(llm=_FakeLLM()),
        action_out_proj=lambda value: value,
        embed_prefix=lambda observation: (
            jnp.zeros((observation.state.shape[0], 1, 1)),
            jnp.ones((observation.state.shape[0], 1), dtype=jnp.bool_),
            jnp.zeros((1,), dtype=jnp.bool_),
        ),
        embed_suffix=embed_suffix,
    )


@pytest.fixture(autouse=True)
def _identity_preprocess(monkeypatch):
    monkeypatch.setattr(pi0._model, "preprocess_observation", lambda _rng, observation, *, train: observation)  # noqa: SLF001


def _sample(sampler, steps, noise=0.0):
    model = _fake_model()
    observation = SimpleNamespace(state=jnp.zeros((1, 1)))
    return pi0.Pi0.sample_actions(
        model,
        jax.random.PRNGKey(0),
        observation,
        num_steps=steps,
        noise=jnp.array([[[noise]]]),
        sampler=sampler,
    )


@pytest.mark.parametrize("steps", [1, 2, 5])
def test_fm_only_matches_legacy_current_time_exactly(steps):
    # This is the no-TVM inference path: the model receives equal source/target times.
    np.testing.assert_array_equal(_sample("fm_only", steps), _sample("current_time", steps))


@pytest.mark.parametrize("steps", [1, 2, 5])
def test_jump_matches_legacy_target_time_exactly(steps):
    np.testing.assert_array_equal(_sample("jump", steps), _sample("target_time", steps))


def test_one_step_queries_the_expected_time_pair():
    np.testing.assert_allclose(_sample("fm_only", 1), [[[-3.0]]])
    np.testing.assert_allclose(_sample("jump", 1), [[[-2.0]]])


def test_two_step_jump_uses_next_time_at_each_euler_update():
    # From x=0: t=1 -> 0.5 gives v=2.5 and x=-1.25; the next v=-0.25.
    np.testing.assert_allclose(_sample("jump", 2), [[[-1.125]]])


def test_unknown_sampler_fails_before_inference():
    with pytest.raises(ValueError, match="Unknown sampler"):
        _sample("not_a_sampler", 1)
