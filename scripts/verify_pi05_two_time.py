"""Check released-weight parity of the two-time Pi0.5 output."""

import argparse
import json

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import model as model_lib
from openpi.models.pi0 import make_attn_mask
from openpi.models.pi0 import posemb_sincos
from openpi.training import weight_loaders
from scripts import run_pi05_libero_tvm_early


def _legacy_velocity(model, actions, target_time, prefix):
    """The released one-time Pi0.5 suffix computation, independent of the new source path."""
    prefix_tokens, prefix_mask, prefix_ar_mask = prefix
    suffix_tokens = model.action_in_proj(actions)
    time_emb = posemb_sincos(target_time, model.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
    time_emb = model.time_mlp_in(time_emb)
    time_emb = nnx.swish(time_emb)
    time_emb = model.time_mlp_out(time_emb)
    adarms_cond = nnx.swish(time_emb)
    suffix_mask = jnp.ones(suffix_tokens.shape[:2], dtype=jnp.bool_)
    suffix_ar_mask = jnp.array([True] + [False] * (model.action_horizon - 1))
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attn_mask = make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1
    (_, suffix_out), _ = model.PaliGemma.llm(
        [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
    )
    return model.action_out_proj(suffix_out[:, -model.action_horizon :])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default=run_pi05_libero_tvm_early.BASE_PARAMS)
    args = parser.parse_args()

    config = run_pi05_libero_tvm_early.make_config().model
    model = config.create(jax.random.key(0))
    graphdef, state = nnx.split(model)
    loaded = weight_loaders.CheckpointWeightLoader(args.params).load(state.to_pure_dict())
    state.replace_by_pure_dict(loaded)
    model = nnx.merge(graphdef, state)

    observation = model_lib.preprocess_observation(jax.random.key(1), config.fake_obs(1), train=False)
    actions = config.fake_act(1)
    prefix = model.embed_prefix(observation)
    results = []
    for source, target in ((0.9, 0.8), (0.7, 0.3), (0.2, 0.1), (0.4, 0.4)):
        source_time = jnp.array([source], dtype=jnp.float32)
        target_time = jnp.array([target], dtype=jnp.float32)
        reference = _legacy_velocity(model, actions, target_time, prefix)
        actual = model._predict_velocity_from_prefix(  # noqa: SLF001
            observation, actions, source_time, target_time, *prefix
        )
        max_error = float(jnp.max(jnp.abs(reference - actual)))
        if not jnp.all(jnp.isfinite(actual)) or max_error > 1e-5:
            raise AssertionError(f"Output parity failed at ({source}, {target}): {max_error}")
        results.append({"source_time": source, "target_time": target, "max_abs_error": max_error})
    print(json.dumps({"released_weight_output_parity": results}, indent=2))


if __name__ == "__main__":
    main()
