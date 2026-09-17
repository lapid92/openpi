import jax
import numpy as np
import pytest

from openpi.training import tvm


@pytest.mark.parametrize(
    ("step", "expected_alpha"),
    [
        (0, 0.0),
        (9, 0.0),
        (10, 0.0),
        (20, 0.5),
        (30, 1.0),
        (40, 1.0),
    ],
)
def test_loss_weights_follow_warmup_and_linear_ramp(step: int, expected_alpha: float):
    config = tvm.TVMTrainingConfig(
        enabled=True,
        warmup_steps=10,
        ramp_steps=20,
        alpha_final=1.0,
        fm_loss_weight=0.75,
    )

    alpha, fm_loss_weight = tvm.loss_weights(config, step)

    np.testing.assert_allclose(alpha, expected_alpha)
    np.testing.assert_allclose(fm_loss_weight, 0.75)


def test_loss_weights_are_jittable():
    config = tvm.TVMTrainingConfig(enabled=True, warmup_steps=10, ramp_steps=20, alpha_final=0.25)
    alpha, fm_loss_weight = jax.jit(lambda step: tvm.loss_weights(config, step))(jax.numpy.asarray(20))

    np.testing.assert_allclose(alpha, 0.125)
    np.testing.assert_allclose(fm_loss_weight, 1.0)


def test_disabled_loss_weights_preserve_flow_matching():
    alpha, fm_loss_weight = tvm.loss_weights(tvm.TVMTrainingConfig(), step=1_000_000)

    np.testing.assert_allclose(alpha, 0.0)
    np.testing.assert_allclose(fm_loss_weight, 1.0)


def test_zero_length_ramp_steps_at_end_of_warmup():
    config = tvm.TVMTrainingConfig(enabled=True, warmup_steps=10, ramp_steps=0, alpha_final=0.25)

    np.testing.assert_allclose(tvm.loss_weights(config, 9)[0], 0.0)
    np.testing.assert_allclose(tvm.loss_weights(config, 10)[0], 0.25)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("warmup_steps", -1, "warmup_steps"),
        ("ramp_steps", -1, "ramp_steps"),
        ("alpha_final", -0.1, "alpha_final"),
        ("fm_loss_weight", -0.1, "fm_loss_weight"),
    ],
)
def test_config_rejects_negative_values(field: str, value: float, message: str):
    with pytest.raises(ValueError, match=message):
        tvm.TVMTrainingConfig(**{field: value})
