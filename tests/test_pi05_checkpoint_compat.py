"""Older pi0.5 checkpoints remain usable for inference after TVM support is added."""

import flax.nnx as nnx
import jax
import numpy as np

from openpi.models.pi0_config import Pi0Config


def test_one_time_checkpoint_loads_with_zero_source_projection():
    cfg = Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy")
    model = cfg.create(jax.random.key(0))
    _, state = nnx.split(model)
    params = state.to_pure_dict()
    del params["source_time_proj"]
    restored = cfg.load(params)
    np.testing.assert_array_equal(restored.source_time_proj.kernel.value, 0)
