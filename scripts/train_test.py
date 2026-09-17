import dataclasses
import os
import pathlib

import numpy as np
import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.training import config as _config
from openpi.training import tvm as _tvm

from . import train


def test_stack_microbatches():
    microbatches = [({"observation": np.full((2, 3), index)}, np.full((2, 4), index)) for index in range(3)]

    observations, actions = train._stack_microbatches(microbatches)  # noqa: SLF001

    np.testing.assert_array_equal(observations["observation"][:, 0, 0], np.arange(3))
    assert observations["observation"].shape == (3, 2, 3)
    assert actions.shape == (3, 2, 4)


@pytest.mark.parametrize("config_name", ["debug"])
def test_train(tmp_path: pathlib.Path, config_name: str):
    config = dataclasses.replace(
        _config._CONFIGS_DICT[config_name],  # noqa: SLF001
        batch_size=2,
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="test",
        overwrite=False,
        resume=False,
        num_train_steps=2,
        log_interval=1,
    )
    train.main(config)

    # test resuming
    config = dataclasses.replace(config, resume=True, num_train_steps=4)
    train.main(config)


def test_train_pi05_tvm_and_resume(tmp_path: pathlib.Path):
    config = dataclasses.replace(
        _config.get_config("debug_pi05"),
        tvm=_tvm.TVMTrainingConfig(
            enabled=True,
            warmup_steps=0,
            ramp_steps=1,
            alpha_final=0.25,
            fm_loss_weight=1.0,
        ),
        batch_size=2,
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="test_tvm",
        overwrite=False,
        resume=False,
        num_train_steps=2,
        log_interval=1,
        save_interval=1,
    )
    train.main(config)

    # Resuming must continue from the restored optimizer step, which also drives the TVM ramp.
    train.main(dataclasses.replace(config, resume=True, num_train_steps=4))


def test_train_pi05_tvm_with_gradient_accumulation(tmp_path: pathlib.Path):
    config = dataclasses.replace(
        _config.get_config("debug_pi05"),
        tvm=_tvm.TVMTrainingConfig(
            enabled=True,
            warmup_steps=0,
            ramp_steps=1,
            alpha_final=0.25,
            fm_loss_weight=1.0,
        ),
        batch_size=4,
        gradient_accumulation_steps=2,
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="test_tvm_accumulation",
        overwrite=False,
        resume=False,
        num_train_steps=2,
        log_interval=1,
        save_interval=1,
    )

    train.main(config)
    train.main(dataclasses.replace(config, resume=True, num_train_steps=4))
