import dataclasses

import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class TVMTrainingConfig:
    """Configuration for the optional TVM-style fine-tuning objective."""

    enabled: bool = False
    warmup_steps: int = 0
    ramp_steps: int = 0
    alpha_final: float = 0.0
    fm_loss_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.warmup_steps < 0:
            raise ValueError("TVM warmup_steps must be non-negative")
        if self.ramp_steps < 0:
            raise ValueError("TVM ramp_steps must be non-negative")
        if self.alpha_final < 0:
            raise ValueError("TVM alpha_final must be non-negative")
        if self.fm_loss_weight < 0:
            raise ValueError("TVM fm_loss_weight must be non-negative")


def loss_weights(config: TVMTrainingConfig, step):
    """Returns the TVM and flow-matching weights for an optimizer step."""
    if not config.enabled:
        return jnp.asarray(0.0, dtype=jnp.float32), jnp.asarray(1.0, dtype=jnp.float32)

    step = jnp.asarray(step, dtype=jnp.float32)
    warmup_steps = float(config.warmup_steps)
    if config.ramp_steps == 0:
        ramp_fraction = jnp.where(step < warmup_steps, 0.0, 1.0)
    else:
        ramp_fraction = jnp.clip((step - warmup_steps) / float(config.ramp_steps), 0.0, 1.0)
    return (
        ramp_fraction * config.alpha_final,
        jnp.asarray(config.fm_loss_weight, dtype=jnp.float32),
    )
