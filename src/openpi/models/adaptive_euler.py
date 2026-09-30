"""Bounded, batch-one Euler sampling with a frozen π₀.₅ residual head.

The step rule is deliberately configurable and has not been calibrated for
LIBERO success. Existing Pi0.sample_actions remains the fixed-step entry point.
"""

from __future__ import annotations

import dataclasses
import math
import numbers
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.models.velocity_residual import checkpoint_identity
from openpi.models.velocity_residual import load_head
from openpi.models.velocity_residual import predict_log_sigma
from openpi.shared import nnx_utils
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
        self._prepare = nnx_utils.module_jit(frozen_base.prepare_action_prefix)
        self._velocity_and_features = nnx_utils.module_jit(frozen_base.cached_velocity_and_action_features)

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
        observation = model_lib.preprocess_observation(None, observation, train=False)
        if observation.state.shape[0] != 1:
            raise ValueError("Adaptive sampler currently supports batch size one")
        shape = (1, self._base.action_horizon, self._base.action_dim)
        if noise is None:
            noise = jax.random.normal(rng, shape, dtype=jnp.float32)
        else:
            noise = jnp.asarray(noise, dtype=jnp.float32)
            if noise.shape != shape:
                raise ValueError(f"Expected noise shape {shape}, got {noise.shape}")
        context = self._prepare(observation)
        actions = noise
        time = 1.0
        trace = []
        for evaluation in range(self._config.max_velocity_evaluations):
            velocity, features = self._velocity_and_features(
                observation, actions, jnp.array([time], dtype=jnp.float32), context
            )
            sigma = float(
                np.asarray(jnp.exp(predict_log_sigma(self._head, features, jnp.array([time], dtype=jnp.float32)))[0])
            )
            step = self._config.bounded_step(time, sigma, evaluation)
            actions = actions - step * velocity
            next_time = max(0.0, time - step)
            trace.append({"time": time, "sigma": sigma, "step": step, "next_time": next_time})
            time = next_time
            if time == 0.0:
                break
        if time != 0.0:
            raise RuntimeError("Adaptive Euler did not terminate at t=0 within its evaluation budget")
        return AdaptiveSample(actions, len(trace), tuple(trace))
