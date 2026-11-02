"""Focused checks of the released RoboCasa state, action, and camera contract."""

import json

import numpy as np
import pytest

from openpi.models.model import ModelType
from openpi.policies.robocasa_policy import RobocasaInputs, RobocasaOutputs


def test_three_cameras_and_twelve_actions():
    image = np.arange(3 * 4 * 5, dtype=np.uint8).reshape(3, 4, 5)
    data = {
        "observation/image": image,
        "observation/right_image": image + 1,
        "observation/wrist_image": image + 2,
        "observation/state": np.arange(16, dtype=np.float32),
        "actions": np.arange(24, dtype=np.float32).reshape(2, 12),
        "prompt": "adjust the toaster oven temperature",
    }
    result = RobocasaInputs(action_dim=32, model_type=ModelType.PI05)(data)
    assert list(result["image"]) == ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
    np.testing.assert_array_equal(result["image"]["base_0_rgb"], image.transpose(1, 2, 0))
    np.testing.assert_array_equal(result["image"]["left_wrist_0_rgb"], (image + 2).transpose(1, 2, 0))
    np.testing.assert_array_equal(result["image"]["right_wrist_0_rgb"], (image + 1).transpose(1, 2, 0))
    assert all(result["image_mask"].values())
    np.testing.assert_array_equal(result["state"][:16], data["observation/state"])
    np.testing.assert_array_equal(result["actions"][:, :12], data["actions"])
    assert result["state"].shape == (32,)
    assert result["actions"].shape == (2, 32)
    assert result["prompt"] == data["prompt"]
    np.testing.assert_array_equal(RobocasaOutputs()({"actions": result["actions"]})["actions"], data["actions"])


def test_pi05_requires_right_camera():
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="right_image"):
        RobocasaInputs(action_dim=32, model_type=ModelType.PI05)(
            {"observation/image": image, "observation/wrist_image": image, "observation/state": np.zeros(16)}
        )


def test_owner_normalization_order(tmp_path):
    from openpi.groot_utils.groot_openpi_dataset import _load_norm_stats_from_groot_dataset

    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "stats.json").write_text(
        json.dumps(
            {
                "observation.state": {"mean": list(range(16)), "std": [1] * 16},
                "action": {"mean": list(range(12)), "std": [1] * 12},
            }
        )
    )
    stats = _load_norm_stats_from_groot_dataset({"path": str(tmp_path)})
    np.testing.assert_array_equal(stats["state"].mean[:16], [7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 14, 15])
    np.testing.assert_array_equal(stats["actions"].mean[:12], [5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4])
    assert len(stats["state"].mean) == len(stats["actions"].mean) == 32


def test_owner_dataset_item_order(monkeypatch):
    from openpi.groot_utils.groot_openpi_dataset import GrootOpenpiSingleDataset, LeRobotSingleDataset

    def owner_item(_self, _index):
        def values(start, count, horizon=1):
            return np.tile(np.arange(start, start + count, dtype=np.float32), (horizon, 1))

        return {
            "state.end_effector_position_relative": values(7, 3),
            "state.end_effector_rotation_relative": values(10, 4),
            "state.base_position": values(0, 3),
            "state.base_rotation": values(3, 4),
            "state.gripper_qpos": values(14, 2),
            "action.end_effector_position": values(5, 3, 50),
            "action.end_effector_rotation": values(8, 3, 50),
            "action.gripper_close": values(11, 1, 50),
            "action.base_motion": values(0, 4, 50),
            "action.control_mode": values(4, 1, 50),
            "video.robot0_agentview_left": [np.zeros((224, 224, 3), dtype=np.uint8)],
            "video.robot0_agentview_right": [np.ones((224, 224, 3), dtype=np.uint8)],
            "video.robot0_eye_in_hand": [np.full((224, 224, 3), 2, dtype=np.uint8)],
            "annotation.human.task_description": ["adjust the toaster oven temperature"],
        }

    monkeypatch.setattr(LeRobotSingleDataset, "__getitem__", owner_item)
    dataset = object.__new__(GrootOpenpiSingleDataset)
    item = dataset[0]
    np.testing.assert_array_equal(item["observation/state"], [7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 14, 15])
    np.testing.assert_array_equal(item["actions"][0], [5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4])
    assert item["actions"].shape == (50, 12)
    assert item["prompt"] == "adjust the toaster oven temperature"
    assert item["observation/right_image"].mean() == 1
