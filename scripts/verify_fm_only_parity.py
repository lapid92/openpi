"""Verify exact fm_only/current_time action parity for a no-TVM Pi0.5 checkpoint."""

import argparse

from flax import nnx
import jax
import numpy as np

from openpi.training import config as training_config
from openpi.training import weight_loaders


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="No-TVM Pi0.5 checkpoint directory")
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 2, 5, 10])
    args = parser.parse_args()

    config = training_config.get_config("pi05_libero").model
    model = config.create(jax.random.key(0))
    graphdef, state = nnx.split(model)
    loaded = weight_loaders.CheckpointWeightLoader(f"{args.checkpoint}/params").load(state.to_pure_dict())
    state.replace_by_pure_dict(loaded)
    model = nnx.merge(graphdef, state)
    observation = config.fake_obs(1)
    noise = jax.random.normal(jax.random.key(23), (1, config.action_horizon, config.action_dim))
    for steps in args.steps:
        legacy = model.sample_actions(jax.random.key(42), observation, num_steps=steps, noise=noise, sampler="current_time")
        fm_only = model.sample_actions(jax.random.key(42), observation, num_steps=steps, noise=noise, sampler="fm_only")
        np.testing.assert_array_equal(fm_only, legacy)
        print(f"steps={steps}: exact action parity, shape={fm_only.shape}")


if __name__ == "__main__":
    main()
