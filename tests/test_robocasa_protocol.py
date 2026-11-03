"""Offline regression checks for the evaluated RoboCasa protocol boundaries."""

import hashlib
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

from robocasa_eval import protocol, report, results
from robocasa_eval.manifest import make_manifest
if np is not None:
    from robocasa_eval import server_common, smol_interfaces


def manifest(model="smolvla", steps=4, seed=7, task_count=1):
    groups = protocol.GROUPS
    return dict(
        campaign_id="campaign", protocol_id="corrected-v2", checkpoint_revision="pinned",
        model=model, denoising_steps=steps, scene_seed=seed, policy_seed=0,
        split="target" if model == "smolvla" else "pretrain",
        replan_steps=50 if model == "smolvla" else 5,
        horizon_multiplier=1.5, episodes_per_task=50,
        owner_robocasa_adapter="/pinned/owner.py" if model == "smolvla" else None,
        owner_robocasa_adapter_sha256="a" * 64 if model == "smolvla" else None,
        tasks=[dict(task=f"Task{i}", group=groups[i % len(groups)]) for i in range(task_count)],
        reference=dict(group_rates_pct=dict.fromkeys(groups), overall_rate_pct=None),
    )


def episode(m, idx=0, task="Task0", success=False):
    model = m["model"]
    row = {key: m[key] for key in (
        "campaign_id", "protocol_id", "checkpoint_revision", "model", "split",
        "denoising_steps", "scene_seed", "replan_steps",
    )}
    row.update(
        manifest_sha256=results.manifest_digest(m), task=task,
        group=next(item["group"] for item in m["tasks"] if item["task"] == task),
        episode_index=idx, status="success" if success else "policy_failure",
        success=success, prompt=task if model == "smolvla" else None,
        instruction="Move the object", policy_queries=2,
        action_network_evaluations=2 * m["denoising_steps"],
        policy_seed=protocol.stable_seed(
            f"public_{model}_baseline", task, idx, m["policy_seed"]
        ),
        scene_reset_protocol=protocol.RESET_PROTOCOL,
        model_xml_sha256="a" * 64, initial_state_sha256="b" * 64,
    )
    return row


class ProtocolTests(unittest.TestCase):
    def test_evaluated_grid_and_model_specific_choices(self):
        self.assertEqual(protocol.STEPS, (1, 2, 4, 10, 15, 20))
        self.assertEqual(protocol.SEEDS, (7, 8, 9))
        for model in ("pi05", "smolvla"):
            for steps in protocol.STEPS:
                m = manifest(model, steps, task_count=50)
                protocol.validate_manifest(m)
                wrong = dict(m, split="pretrain" if model == "smolvla" else "target")
                with self.assertRaises(ValueError):
                    protocol.validate_manifest(wrong)
                wrong = dict(m, replan_steps=5 if model == "smolvla" else 50)
                with self.assertRaises(ValueError):
                    protocol.validate_manifest(wrong)

    def test_seed_is_stable_and_query_specific(self):
        seed = protocol.stable_seed("public_smolvla_baseline", "Task0", 0, 0)
        self.assertEqual(seed, protocol.stable_seed("public_smolvla_baseline", "Task0", 0, 0))
        self.assertNotEqual(seed, protocol.stable_seed("public_smolvla_baseline", "Task0", 1, 0))
        self.assertNotEqual(protocol.stable_seed(seed, 0), protocol.stable_seed(seed, 1))
        self.assertTrue(0 <= seed < 2**32)


@unittest.skipIf(np is None, "NumPy is not installed in the local Python")
class SmolInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.obs = {
            "state.base_position": np.array([0, 1, 2]),
            "state.base_rotation": np.array([3, 4, 5, 6]),
            "state.end_effector_position_relative": np.array([7, 8, 9]),
            "state.end_effector_rotation_relative": np.array([10, 11, 12, 13]),
            "state.gripper_qpos": np.array([14, 15]),
        }

    def test_corrected_state_and_action_order(self):
        state = smol_interfaces.convert_state(self.obs)
        np.testing.assert_array_equal(state, np.arange(16, dtype=np.float32))
        self.assertEqual(state.dtype, np.float32)
        action = smol_interfaces.convert_action(np.arange(12, dtype=np.float32))
        expected = {
            "action.base_motion": [0, 1, 2, 3], "action.control_mode": [4],
            "action.end_effector_position": [5, 6, 7],
            "action.end_effector_rotation": [8, 9, 10], "action.gripper_close": [11],
        }
        self.assertEqual({key: value.tolist() for key, value in action.items()}, expected)
        self.assertEqual(action["action.gripper_close"][0], 11)

    def test_pinned_owner_audit_checks_both_interfaces(self):
        owner = '''import numpy as np
def convert_action(action):
    return {"action.base_motion": action[:4], "action.control_mode": action[4:5],
            "action.end_effector_position": action[5:8],
            "action.end_effector_rotation": action[8:11], "action.gripper_close": action[11:12]}
def convert_observation(raw_obs):
    agent_pos = np.concatenate([raw_obs["state." + k] for k in
        ("base_position", "base_rotation", "end_effector_position_relative",
         "end_effector_rotation_relative", "gripper_qpos")]).astype(np.float32)
    return agent_pos
'''
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "owner.py"
            path.write_text(owner)
            audit = smol_interfaces.audit_interface(
                self.obs, [np.linspace(-1, 1, 12, dtype=np.float32)], path,
                hashlib.sha256(owner.encode()).hexdigest()
            )
            self.assertTrue(audit["passed"])
            self.assertEqual(audit["action_chunks_checked"], 1)
            self.assertEqual(audit["reference_sha256"], hashlib.sha256(owner.encode()).hexdigest())
            path.write_text(owner.replace('action[11:12]', 'action[10:11]'))
            with self.assertRaises(ValueError):
                smol_interfaces.audit_interface(
                    self.obs, [], path, hashlib.sha256(path.read_bytes()).hexdigest())
            path.write_text(owner.replace('"base_position", "base_rotation"',
                                          '"base_rotation", "base_position"'))
            with self.assertRaises(ValueError):
                smol_interfaces.audit_interface(
                    self.obs, [], path, hashlib.sha256(path.read_bytes()).hexdigest())


@unittest.skipIf(np is None, "NumPy is not installed in the local Python")
class ServerRequestTests(unittest.TestCase):
    def setUp(self):
        self.request = {
            "observation/state": np.arange(16, dtype=np.float32),
            "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/right_image": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
            "prompt": "Move the object", "campaign_policy_seed": 123,
        }

    def test_request_requires_explicit_uint32_seed_and_rgb(self):
        clean, seed = server_common.take_request(self.request)
        self.assertEqual(seed, 123)
        self.assertNotIn("campaign_policy_seed", clean)
        self.assertIn("campaign_policy_seed", self.request)
        for bad in (True, -1, 2**32, 1.5):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                server_common.take_request(dict(self.request, campaign_policy_seed=bad))
        with self.assertRaises(KeyError):
            server_common.take_request({k: v for k, v in self.request.items() if k != "campaign_policy_seed"})
        with self.assertRaises(ValueError):
            server_common.take_request(dict(self.request, **{"observation/image": np.zeros((8, 8, 3), dtype=np.float32)}))
        with self.assertRaises(ValueError):
            server_common.take_request(dict(self.request, prompt=""))

    def test_finish_rejects_wrong_action_shape(self):
        with self.assertRaises(ValueError):
            server_common.finish(np.zeros((50, 11)), 123, 4, 0.1, 0.0, {})
        self.assertEqual(server_common.finish(np.zeros((50, 12)), 123, 4, 0.1, 0.0, {})[
            "action_network_evaluations"], 4)


class EvidenceTests(unittest.TestCase):
    def test_episode_identity_nfe_and_conflicting_duplicate(self):
        m = manifest()
        row = episode(m)
        self.assertEqual(results.validate_episode(row, m), ("Task0", 0))
        for change in (
            {"action_network_evaluations": 7}, {"prompt": "different"},
            {"policy_seed": 123}, {"scene_reset_protocol": "per_episode_seed"},
            {"manifest_sha256": "bad"}, {"episode_index": 50},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                results.validate_episode(dict(row, **change), m)
        duplicate = dict(row, status="success", success=True)
        with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
            results.summarize([row, duplicate], m)

    def test_coverage_and_null_smol_reference(self):
        m = manifest()
        rows = [episode(m, idx=i, success=i % 2 == 0) for i in range(50)]
        _, _, groups = results.summarize(rows[:-1], m)
        self.assertFalse(groups[-1]["coverage_complete"])
        self.assertEqual(groups[-1]["comparison_status"], "incomplete_do_not_compare")
        _, _, groups = results.summarize(rows, m)
        self.assertTrue(groups[-1]["coverage_complete"])
        self.assertEqual(groups[-1]["success_rate_pct"], 50)
        self.assertIsNone(groups[-1]["public_reference_pct"])

    def test_report_cell_requires_2500_unique_completed_episodes(self):
        m = make_manifest("smolvla", 7, 4, "campaign", "/pinned/owner.py", "a" * 64)
        rows = [episode(m, idx=i, task=item["task"]) for item in m["tasks"] for i in range(50)]
        self.assertEqual(len(report.validate_cell(m, rows, "smolvla", 7, 4)), 2500)
        with self.assertRaisesRegex(ValueError, "Incomplete coverage"):
            report.validate_cell(m, rows[:-1], "smolvla", 7, 4)
        with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
            report.validate_cell(m, rows + [dict(rows[0], success=True, status="success")],
                                 "smolvla", 7, 4)



if __name__ == "__main__":
    unittest.main()
