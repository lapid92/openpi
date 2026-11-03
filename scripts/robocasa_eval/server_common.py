"""Request validation and response metadata shared by both policy adapters."""
from __future__ import annotations
import enum
import hashlib
import json
import pathlib
import time

def metadata_default(value):
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, enum.Enum):
        return value.value
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize metadata value of type {type(value)}")


def write_metadata(path, metadata):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=metadata_default))
    temp.replace(path)


def take_request(obs):
    """Never allow a missing seed to silently select a stateful/global RNG."""
    import numpy as np
    obs = dict(obs)
    seed = obs.pop("campaign_policy_seed")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("campaign_policy_seed must be an integer")
    seed = int(seed)
    if not 0 <= seed < 2**32:
        raise ValueError("campaign_policy_seed must fit uint32")
    if np.asarray(obs["observation/state"]).shape != (16,):
        raise ValueError("Expected unbatched 16D RoboCasa state")
    if not isinstance(obs.get("prompt"), str) or not obs["prompt"].strip():
        raise ValueError("An explicit nonempty prompt is required")
    for name in ("image", "right_image", "wrist_image"):
        value = np.asarray(obs["observation/" + name])
        if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8:
            raise ValueError(f"{name} must be uint8 HWC RGB")
    return obs, seed


def finish(actions, seed, count, elapsed, started, metadata):
    import numpy as np
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != 12 or not np.isfinite(actions).all():
        raise ValueError(f"Invalid policy action output: shape={actions.shape}")
    return {
        "actions": actions, "campaign_policy_seed": seed,
        "action_network_evaluations": count, "inference_seconds": elapsed,
        "total_inference_seconds": time.monotonic() - started,
        "sampler_parity": metadata.get("sampler_parity", "original_sampler_with_hook"),
    }
