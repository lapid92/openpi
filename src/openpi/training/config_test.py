import dataclasses

import pytest

from openpi.models import pi0_config
from openpi.training import config as train_config
from openpi.training import tvm


def test_pi05_libero_tvm_config_defaults():
    config = train_config.get_config("pi05_libero_tvm")

    assert isinstance(config.model, pi0_config.Pi0Config)
    assert config.model.pi05
    assert config.model.action_horizon == 10
    assert config.batch_size == 256
    assert config.ema_decay == 0.999
    assert config.num_train_steps == 30_000
    assert config.tvm == tvm.TVMTrainingConfig(
        enabled=True,
        warmup_steps=10_000,
        ramp_steps=10_000,
        alpha_final=0.25,
        fm_loss_weight=1.0,
    )


def test_tvm_requires_pi05_model():
    config = train_config.get_config("debug")

    with pytest.raises(ValueError, match=r"requires a pi0\.5 model"):
        dataclasses.replace(config, tvm=tvm.TVMTrainingConfig(enabled=True, alpha_final=0.25))


def test_tvm_requires_ema_teacher():
    config = train_config.get_config("debug_pi05")

    with pytest.raises(ValueError, match="requires EMA parameters"):
        dataclasses.replace(
            config,
            tvm=tvm.TVMTrainingConfig(enabled=True, alpha_final=0.25),
            ema_decay=None,
        )
