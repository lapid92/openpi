"""RoboCasa Human300 Groot-LeRobot adapter.

Keeps the released owner's item ordering and mixture behavior from
robocasa-benchmark/openpi@ca4c6d710db75e276bc7c866a57bd7e4aee5b6e8.
"""

import json
import pathlib
from typing import SupportsIndex

import numpy as np

import openpi.shared.normalize as _normalize
import openpi.transforms as _transforms

from robocasa.utils.groot_utils.groot_dataset import (
    LE_ROBOT_MODALITY_FILENAME,
    LeRobotMixtureDataset,
    LeRobotSingleDataset,
    ModalityConfig,
)
from robocasa.utils.groot_utils.embodiment_tags import EmbodimentTag


def get_modality_keys(dataset_path: pathlib.Path) -> dict[str, list[str]]:
    """
    Get the modality keys from the dataset path.
    Returns a dictionary with modality types as keys and their corresponding modality keys as values,
    maintaining the order: video, state, action, annotation
    """
    modality_path = dataset_path / LE_ROBOT_MODALITY_FILENAME
    with open(modality_path, "r") as f:
        modality_meta = json.load(f)

    # Initialize dictionary with ordered keys
    modality_dict = {}
    for key in modality_meta.keys():
        modality_dict[key] = []
        for modality in modality_meta[key]:
            modality_dict[key].append(f"{key}.{modality}")
    return modality_dict


class GrootOpenpiSingleDataset(LeRobotSingleDataset):
    def __init__(
        self,
        dataset_meta: dict,
        action_horizon: int,
    ):
        dataset_path = dataset_meta["path"]
        dataset_path = pathlib.Path(dataset_path)
        filter_key = dataset_meta["filter_key"]
        delta_indices = list(range(0, action_horizon))
        delta_indices_obs = [0]
        modality_keys_dict = get_modality_keys(dataset_path)
        video_modality_keys = modality_keys_dict["video"]
        language_modality_keys = modality_keys_dict["annotation"]
        state_modality_keys = modality_keys_dict["state"]
        action_modality_keys = modality_keys_dict["action"]
        state_modality_keys = [key for key in state_modality_keys if key != "state.dummy_tensor"]
        modality_configs = {
            "video": ModalityConfig(
                delta_indices=delta_indices_obs,
                modality_keys=video_modality_keys,  # we will include all video modalities
            ),
            "state": ModalityConfig(
                delta_indices=delta_indices_obs,
                modality_keys=state_modality_keys,
            ),
            "action": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=action_modality_keys,
            ),
            "language": ModalityConfig(
                delta_indices=[0],
                modality_keys=language_modality_keys,
            ),
        }

        super().__init__(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            video_backend="opencv",
            video_backend_kwargs=None,
            transforms=None,
            filter_key=filter_key,
        )

    def __getitem__(self, index: SupportsIndex) -> dict:
        item = super().__getitem__(index)

        state = np.concatenate([
            item["state.end_effector_position_relative"],
            item["state.end_effector_rotation_relative"],
            item["state.base_position"],
            item["state.base_rotation"],
            item["state.gripper_qpos"],
        ], axis=1)
        actions = np.concatenate([
            item["action.end_effector_position"],
            item["action.end_effector_rotation"],
            item["action.gripper_close"],
            item["action.base_motion"],
            item["action.control_mode"],
        ], axis=1)

        new_item = {
            "observation/image": item["video.robot0_agentview_left"][0],
            "observation/wrist_image": item["video.robot0_eye_in_hand"][0],
            "observation/state": state[0],
            "actions": actions,
            "prompt": item["annotation.human.task_description"][0],
        }
        # Add right camera if available
        if "video.robot0_agentview_right" in item:
            new_item["observation/right_image"] = item["video.robot0_agentview_right"][0]
        return new_item


class GrootOpenpiMultiDataset(LeRobotMixtureDataset):
    def __init__(
            self,
            dataset_meta_list,
            action_horizon: int,
            dataset_weights=None,
            dataset_weights_alpha=0.4,
            metadata_config: dict | None = None,
        ):
        datasets = []
        for ds_meta in dataset_meta_list:
            ds_path = ds_meta["path"]
            ds_path = pathlib.Path(ds_path)
            filter_key = ds_meta["filter_key"]
            delta_indices = list(range(0, action_horizon))
            delta_indices_obs = [0]
            modality_keys_dict = get_modality_keys(ds_path)
            video_modality_keys = modality_keys_dict["video"]
            language_modality_keys = modality_keys_dict["annotation"]
            state_modality_keys = modality_keys_dict["state"]
            action_modality_keys = modality_keys_dict["action"]
            state_modality_keys = [key for key in state_modality_keys if key != "state.dummy_tensor"]
            modality_configs = {
                "video": ModalityConfig(
                    delta_indices=delta_indices_obs,
                    modality_keys=video_modality_keys,  # we will include all video modalities
                ),
                "state": ModalityConfig(
                    delta_indices=delta_indices_obs,
                    modality_keys=state_modality_keys,
                ),
                "action": ModalityConfig(
                    delta_indices=delta_indices,
                    modality_keys=action_modality_keys,
                ),
                "language": ModalityConfig(
                    delta_indices=[0],
                    modality_keys=language_modality_keys,
                ),
            }
            this_dataset = LeRobotSingleDataset(
                dataset_path=ds_path,
                modality_configs=modality_configs,
                embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
                video_backend="opencv",
                video_backend_kwargs=None,
                transforms=None,
                filter_key=filter_key,
            )
            datasets.append(this_dataset)

        if dataset_weights is None:
            ds_weights = np.array([len(dataset) ** dataset_weights_alpha for dataset in datasets], dtype=float)
        else:
            ds_weights = np.asarray(dataset_weights, dtype=float)
            if ds_weights.shape != (len(datasets),) or np.any(ds_weights <= 0):
                raise ValueError("dataset_weights must contain one positive weight per Human300 task")
        # The owner mixture expects the first dataset to have weight 1.
        ds_weights = ds_weights / ds_weights[0]
        dataset_mixture = list(zip(datasets, ds_weights))
        # set balance_dataset_weights to False, since we are calculating weights ourselves
        super().__init__(
            data_mixture=dataset_mixture,
            mode="train",
            balance_dataset_weights=False,
            balance_trajectory_weights=False,
            metadata_config=metadata_config or {"percentile_mixing_method": "weighted_average"},
        )

    def sample_step(self, index: int) -> tuple[LeRobotSingleDataset, int, int]:
        """Sample a task, trajectory, and step randomly, as in the owner trainer."""
        del index

        # Sample dataset
        dataset_index = np.random.choice(len(self.datasets), p=self.dataset_sampling_weights)
        dataset = self.datasets[dataset_index]

        # Sample trajectory
        trajectory_index = np.random.choice(
            len(dataset.trajectory_ids), p=self.trajectory_sampling_weights[dataset_index]
        )
        trajectory_id = dataset.trajectory_ids[trajectory_index]

        # Sample step
        base_index = np.random.choice(dataset.trajectory_lengths[trajectory_index])
        return dataset, trajectory_id, base_index

    def __getitem__(self, index: SupportsIndex) -> dict:
        item = super().__getitem__(index)

        state = np.concatenate([
            item["state.end_effector_position_relative"],
            item["state.end_effector_rotation_relative"],
            item["state.base_position"],
            item["state.base_rotation"],
            item["state.gripper_qpos"],
        ], axis=1)
        actions = np.concatenate([
            item["action.end_effector_position"],
            item["action.end_effector_rotation"],
            item["action.gripper_close"],
            item["action.base_motion"],
            item["action.control_mode"],
        ], axis=1)

        new_item = {
            "observation/image": item["video.robot0_agentview_left"][0],
            "observation/wrist_image": item["video.robot0_eye_in_hand"][0],
            "observation/state": state[0],
            "actions": actions,
            "prompt": item["annotation.human.task_description"][0],
        }
        # Add right camera if available
        if "video.robot0_agentview_right" in item:
            new_item["observation/right_image"] = item["video.robot0_agentview_right"][0]
        return new_item


def _load_norm_stats_from_groot_dataset(ds_meta: dict) -> dict[str, _transforms.NormStats] | None:
    def pad_zeros(input, targ_len):
        return np.concatenate([input, np.zeros(targ_len - len(input))])

    def pad_ones(input, targ_len):
        return np.concatenate([input, np.ones(targ_len - len(input))])

    dataset_path = ds_meta["path"]
    dataset_path = pathlib.Path(dataset_path)
    path = dataset_path / "meta" / "stats.json"
    data = json.loads(path.read_text())

    """
    the groot state ordering
    "state.base_position" 0, 1, 2
    "state.base_rotation" 3, 4, 5, 6
    "state.end_effector_position_relative" 7, 8, 9
    "state.end_effector_rotation_relative" 10, 11, 12, 13
    "state.gripper_qpos" 14, 15

    the desired state ordering
    "state.end_effector_position_relative" 7, 8, 9
    "state.end_effector_rotation_relative" 10, 11, 12, 13
    "state.base_position" 0, 1, 2
    "state.base_rotation" 3, 4, 5, 6
    "state.gripper_qpos" 14, 15
    """
    raw_states_stats = data["observation.state"]
    raw_states_mean = np.array(raw_states_stats["mean"])
    raw_states_std = np.array(raw_states_stats["std"])

    # Reorder statistics to match the owner's model state vector.
    states_indices = [7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 14, 15]
    states_mean = raw_states_mean[states_indices]
    states_std = raw_states_std[states_indices]

    states_stats = _normalize.NormStats(
        mean=pad_zeros(states_mean, targ_len=32),
        std=pad_ones(states_std, targ_len=32),
    )

    """
    the groot action ordering
    "action.base_motion" 0, 1, 2, 3
    "action.control_mode" 4
    "action.end_effector_position" 5, 6, 7
    "action.end_effector_rotation" 8, 9, 10
    "action.gripper_close" 11

    the desired action ordering
    "action.end_effector_position" 5, 6, 7
    "action.end_effector_rotation" 8, 9, 10
    "action.gripper_close" 11
    "action.base_motion" 0, 1, 2, 3
    "action.control_mode" 4
    """
    raw_actions_stats = data["action"]
    raw_actions_mean = np.array(raw_actions_stats["mean"])
    raw_actions_std = np.array(raw_actions_stats["std"])

    # Reorder statistics to match the owner's model action vector.
    actions_indices = [5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4]
    actions_mean = raw_actions_mean[actions_indices]
    actions_std = raw_actions_std[actions_indices]

    actions_stats = _normalize.NormStats(
        mean=pad_zeros(actions_mean, targ_len=32),
        std=pad_ones(actions_std, targ_len=32),
    )

    return {
        "state": states_stats,
        "actions": actions_stats,
    }

def compute_overall_statistics(
    per_task_stats: list[dict[str, dict[str, list[float] | np.ndarray]]],
    dataset_sampling_weights: list[float] | np.ndarray,
) -> dict[str, dict[str, list[float]]]:
    """
    Computes overall statistics from per-task statistics using dataset sample weights.

    Args:
        per_task_stats: List of per-task statistics.
        Example format of one element in the per-task statistics list:
            {
                "state.gripper": {
                    "min": [...],
                    "max": [...],
                    "mean": [...],
                    "std": [...],
                    "q01": [...],
                    "q99": [...],
                },
                ...
            }
        dataset_sampling_weights: List of sample weights for each task.

    Returns:
        A dict of overall statistics per modality.
    """
    # Normalize the sample weights to sum to 1
    dataset_sampling_weights = np.array(dataset_sampling_weights)
    normalized_weights = dataset_sampling_weights / dataset_sampling_weights.sum()

    # Initialize overall statistics dict
    overall_stats: dict[str, dict[str, list[float]]] = {}

    # Get the list of modality keys
    modality_keys = per_task_stats[0].keys()

    for modality in modality_keys:
        # Number of dimensions (assuming consistent across tasks)
        num_dims = len(per_task_stats[0][modality].mean)

        # Initialize accumulators for means and variances
        weighted_means = np.zeros(num_dims)
        weighted_squares = np.zeros(num_dims)

        for task_idx, task_stats in enumerate(per_task_stats):
            w_i = normalized_weights[task_idx]
            stats = task_stats[modality]
            means = np.array(stats.mean)
            stds = np.array(stats.std)

            # Update weighted sums for mean and variance
            weighted_means += w_i * means
            weighted_squares += w_i * (stds**2 + means**2)

        # Compute overall mean
        overall_mean = weighted_means.tolist()

        # Compute overall variance and std deviation
        overall_variance = weighted_squares - weighted_means**2
        overall_std = np.sqrt(overall_variance).tolist()

        # Store the overall statistics for the modality
        overall_stats[modality] = _normalize.NormStats(
            mean=overall_mean,
            std=overall_std,
        )

    return overall_stats


def _load_norm_stats_from_groot_mixture_dataset(dataset_meta_list) -> dict[str, _transforms.NormStats] | None:
    # Merge the dataset statistics
    per_dataset_norm_stats = []
    for ds_meta in dataset_meta_list:
        per_dataset_norm_stats.append(_load_norm_stats_from_groot_dataset(ds_meta))

    return compute_overall_statistics(
        per_dataset_norm_stats,
        dataset_sampling_weights=np.ones(len(dataset_meta_list)),
    )
