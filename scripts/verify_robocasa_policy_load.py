"""Check released RoboCasa policy loading and the default jump inference path."""

import argparse
import json

import numpy as np

from openpi.training import config
from openpi.policies import policy_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Released RoboCasa Human300 checkpoint directory")
    args = parser.parse_args()
    cfg = config.get_config("pi05_robocasa_human300_tvm_released")
    policy = policy_config.create_trained_policy(cfg, args.checkpoint)
    obs = {
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/right_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/state": np.zeros(16, dtype=np.float32),
        "prompt": "adjust the toaster oven temperature",
    }
    actions = np.asarray(policy.infer(obs)["actions"])
    if actions.shape != (cfg.model.action_horizon, 12) or not np.isfinite(actions).all():
        raise AssertionError(f"Unexpected policy actions: {actions.shape}")
    print(json.dumps({"checkpoint": args.checkpoint, "actions_shape": list(actions.shape), "finite": True}))


if __name__ == "__main__":
    main()
