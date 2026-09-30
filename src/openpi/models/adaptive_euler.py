"""Bounded, batch-one Euler sampling with a frozen π₀.₅ residual head.

The step rule is deliberately configurable and has not been calibrated for
LIBERO success. Existing Pi0.sample_actions remains the fixed-step entry point.
"""

from __future__ import annotations

import dataclasses
import math
import numbers
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.models.velocity_residual import checkpoint_identity
from openpi.models.velocity_residual import load_head
from openpi.models.velocity_residual import predict_log_sigma
from openpi.training import config as config_lib


@dataclasses.dataclass(frozen=True)
class AdaptiveStepConfig:
    max_velocity_evaluations: int
    min_step: float
    max_step: float
    reference_step: float
    reference_sigma: float
    score_power: float = 1.0

    def __post_init__(self):
        if (
            isinstance(self.max_velocity_evaluations, bool)
            or not isinstance(self.max_velocity_evaluations, numbers.Integral)
            or self.max_velocity_evaluations < 1
        ):
            raise ValueError("max_velocity_evaluations must be a positive integer")
        values = (self.min_step, self.max_step, self.reference_step, self.reference_sigma, self.score_power)
        if not all(not isinstance(value, bool) and math.isfinite(value) and value > 0 for value in values):
            raise ValueError("Step-rule parameters must be positive and finite")
        if self.min_step > self.max_step or self.max_step > 1:
            raise ValueError("Require 0 < min_step <= max_step <= 1")
        if self.max_velocity_evaluations * self.max_step < 1:
            raise ValueError("max_step and evaluation budget cannot cover t=1 to t=0")

    def proposed_step(self, sigma: float) -> float:
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("Predicted sigma must be positive and finite")
        log_raw = math.log(self.reference_step) + self.score_power * (math.log(self.reference_sigma) - math.log(sigma))
        if log_raw <= math.log(self.min_step):
            return self.min_step
        if log_raw >= math.log(self.max_step):
            return self.max_step
        return math.exp(log_raw)

    def bounded_step(self, time: float, sigma: float, evaluations_used: int) -> float:
        if not (0 < time <= 1) or not (0 <= evaluations_used < self.max_velocity_evaluations):
            raise ValueError("Invalid integration state")
        evaluations_left = self.max_velocity_evaluations - evaluations_used
        # The budget floor guarantees arrival at zero without exceeding max_step.
        needed = time / evaluations_left
        step = min(time, max(self.proposed_step(sigma), needed))
        if step > self.max_step + 1e-9:
            raise RuntimeError("Step budget became infeasible")
        return step


@dataclasses.dataclass(frozen=True)
class AdaptiveSample:
    actions: jax.Array
    velocity_evaluations: int
    trace: tuple[dict[str, float], ...]


def _compile_adaptive_sample(base, config: AdaptiveStepConfig, *, record_states: bool = False):
    """Freeze the base and compile one cached integration loop.

    Full action and velocity arrays are kept only for explicit diagnostics.
    """
    graphdef, state = nnx.split(base)
    max_evals = config.max_velocity_evaluations
    log_min = math.log(config.min_step)
    log_max = math.log(config.max_step)
    log_reference = math.log(config.reference_step)
    log_reference_sigma = math.log(config.reference_sigma)

    def run(frozen_state, head, raw_observation, noise):
        model = nnx.merge(graphdef, frozen_state)
        observation = model_lib.preprocess_observation(None, raw_observation, train=False)
        context = model.prepare_action_prefix(observation)
        times = jnp.zeros((max_evals,), dtype=jnp.float32)
        sigmas = jnp.zeros((max_evals,), dtype=jnp.float32)
        steps = jnp.zeros((max_evals,), dtype=jnp.float32)
        next_times = jnp.zeros((max_evals,), dtype=jnp.float32)

        def evaluate(actions, time, index):
            time_batch = jnp.broadcast_to(time, (1,))
            velocity, features = model.cached_velocity_and_action_features(observation, actions, time_batch, context)
            log_sigma = predict_log_sigma(head, features, time_batch)[0]
            sigma = jnp.exp(log_sigma)
            log_step = log_reference + config.score_power * (log_reference_sigma - log_sigma)
            proposed = jnp.minimum(config.max_step, jnp.exp(jnp.clip(log_step, log_min, log_max)))
            needed = time / (max_evals - index)
            step = jnp.minimum(time, jnp.maximum(proposed, needed))
            following = jnp.maximum(0.0, time - step)
            # Production uses a weak scalar dt, so its product with bfloat16
            # velocity has bfloat16 dtype. Preserve that update precision.
            update_step = (-step).astype(velocity.dtype)
            next_action = actions + update_step * velocity
            return next_action, following, sigma, step, velocity

        def cond(carry):
            return (carry[1] > 0) & (carry[2] < max_evals)

        if record_states:
            states = jnp.zeros((max_evals + 1, *noise.shape), dtype=noise.dtype).at[0].set(noise)
            velocities = jnp.zeros((max_evals, *noise.shape), dtype=noise.dtype)

            def body(carry):
                actions, time, index, times, sigmas, steps, next_times, states, velocities = carry
                next_action, following, sigma, step, velocity = evaluate(actions, time, index)
                return (
                    next_action,
                    following,
                    index + 1,
                    times.at[index].set(time),
                    sigmas.at[index].set(sigma),
                    steps.at[index].set(step),
                    next_times.at[index].set(following),
                    states.at[index + 1].set(next_action),
                    velocities.at[index].set(velocity),
                )

            return jax.lax.while_loop(cond, body, (noise, 1.0, 0, times, sigmas, steps, next_times, states, velocities))

        def body(carry):
            actions, time, index, times, sigmas, steps, next_times = carry
            next_action, following, sigma, step, _ = evaluate(actions, time, index)
            return (
                next_action,
                following,
                index + 1,
                times.at[index].set(time),
                sigmas.at[index].set(sigma),
                steps.at[index].set(step),
                next_times.at[index].set(following),
            )

        return jax.lax.while_loop(cond, body, (noise, 1.0, 0, times, sigmas, steps, next_times))

    return state, jax.jit(run)


class AdaptiveEulerSampler:
    def __init__(
        self, frozen_base, head, config: AdaptiveStepConfig, *, head_metadata: dict, base_checkpoint_identity: str
    ):
        if not frozen_base.pi05:
            raise ValueError("Adaptive residual sampling requires π₀.₅")
        if int(frozen_base.action_out_proj.kernel.value.shape[0]) != 1024:
            raise ValueError("Frozen base action-expert width does not match the residual head")
        if head_metadata.get("base_checkpoint_identity") != base_checkpoint_identity:
            raise ValueError("Residual head does not match the frozen checkpoint identity")
        if head_metadata.get("model_config") != "pi05_libero" or head_metadata.get("feature_width") != 1024:
            raise ValueError("Residual head metadata does not match standard π₀.₅ LIBERO")
        for index, shape in enumerate(((1056, 256), (256, 64), (64, 1))):
            layer = head[f"layer_{index}"]
            if layer["kernel"].shape != shape or layer["bias"].shape != (shape[1],):
                raise ValueError("Residual head parameter shapes are incompatible")
        self._base = frozen_base
        self._head = head
        self._config = config
        self._base_state, self._sample_compiled = _compile_adaptive_sample(frozen_base, config)

    @classmethod
    def from_paths(cls, frozen_checkpoint: str | Path, head_checkpoint: str | Path, config: AdaptiveStepConfig):
        """Load both frozen weights and matching head from saved checkpoints."""
        frozen_checkpoint = Path(frozen_checkpoint)
        identity = checkpoint_identity(frozen_checkpoint)
        head, metadata = load_head(head_checkpoint, identity)
        train_config = config_lib.get_config("pi05_libero")
        base = train_config.model.load(model_lib.restore_params(frozen_checkpoint / "params", dtype=jnp.bfloat16))
        return cls(base, head, config, head_metadata=metadata, base_checkpoint_identity=identity)

    def sample_actions(self, rng, observation: model_lib.Observation, *, noise=None) -> AdaptiveSample:
        if observation.state.shape[0] != 1:
            raise ValueError("Adaptive sampler currently supports batch size one")
        shape = (1, self._base.action_horizon, self._base.action_dim)
        if noise is None:
            noise = jax.random.normal(rng, shape, dtype=jnp.float32)
        else:
            noise = jnp.asarray(noise, dtype=jnp.float32)
            if noise.shape != shape:
                raise ValueError(f"Expected noise shape {shape}, got {noise.shape}")
        actions, time, count, times, sigmas, steps, next_times = self._sample_compiled(
            self._base_state, self._head, observation, noise
        )
        count = int(np.asarray(count))
        if float(np.asarray(time)) != 0.0:
            raise RuntimeError("Adaptive Euler did not terminate at t=0 within its evaluation budget")
        trace = tuple(
            {
                "time": float(times[i]),
                "sigma": float(sigmas[i]),
                "step": float(steps[i]),
                "next_time": float(next_times[i]),
            }
            for i in range(count)
        )
        return AdaptiveSample(actions, count, trace)
