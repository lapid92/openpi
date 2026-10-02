"""Synthetic tests for selective fixed-step accounting, score provenance and inference."""

# ruff: noqa: PT009, PT027
import copy
import hashlib
import importlib.util
import math
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("selective_analyze", Path(__file__).with_name("analyze.py"))
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)
SHA = hashlib.sha256(b"manifest").hexdigest()
STATE = hashlib.sha256(b"state").hexdigest()
OBS = hashlib.sha256(b"observation").hexdigest()


def fixture(*, smoke=False):
    categories = ["Robot Initial States", "Objects Layout", "Camera Viewpoints", "Camera Viewpoints"]
    conditions = [
        {
            "condition_id": f"c{i}",
            "suite": "libero_goal",
            "task_name": f"task{i}",
            "family": f"family{i // 2}",
            "category": category,
            "severity": 2,
            "cases": [{"seed": 2001, "init_index": 0, "initial_state_sha256": STATE}],
        }
        for i, category in enumerate(categories)
    ]
    m = {
        "flow_steps": [1, 10],
        "checkpoint": {"sha256": STATE},
        "head": {"sha256": OBS},
        "noise_policy": {"shape": [10, 32]},
        "gpu_uuids": ["GPU-test"],
        "metadata_rule": {"Robot Initial States": 10, "Objects Layout": 10, "Camera Viewpoints": 1},
        "settings": {
            "action_horizon": 10,
            "action_dim": 32,
            "replan_steps": 5,
            "max_policy_steps": {"libero_goal": 30},
        },
        "analysis": {
            "initial_log_sigma_pair_tolerance": 1e-6,
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 20261003,
            "sigma_min_rescues": 10,
            "sigma_min_regressions": 10,
            "sigma_min_conditions": 5,
        },
        "benchmarks": {
            "libero_plus": {
                "commit": "abc",
                "conditions": conditions,
                "smoke_cases": [
                    {"condition_id": c["condition_id"], "seed": 900101, "init_index": 0, "initial_state_sha256": STATE}
                    for c in conditions
                ],
            }
        },
    }
    scores = [0.8, 0.1, 0.2, 0.9]
    rows = []
    for i, c in enumerate(conditions):
        seed = 900101 if smoke else 2001
        for arm in a.FIXED:
            n = int(arm.split("_")[1])
            chunks = [
                {
                    "chunk_index": j,
                    "velocity_evaluations": n,
                    "noise_sha256": a.noise_digest("libero_plus", c["suite"], c["task_name"], seed, 0, j, (10, 32)),
                    "action_sha256": hashlib.sha256(f"action{n}-{j}".encode()).hexdigest(),
                    "observation_sha256": OBS,
                    "policy_ms": 10.0 if n == 1 else 50.0,
                    "request_ms": 12.0 if n == 1 else 52.0,
                    "head_ms": 0.5,
                    "head_evaluations": 1,
                    "sigma_time": 1.0,
                    "first_log_sigma": scores[i] + 0.1 * j,
                    "first_sigma": math.exp(scores[i] + 0.1 * j),
                    "reference_velocity_evaluations": 0,
                    "verification": [],
                }
                for j in range(2)
            ]
            rows.append(
                {
                    "benchmark": "libero_plus",
                    "benchmark_commit": "abc",
                    "condition_id": c["condition_id"],
                    "suite": c["suite"],
                    "task_name": c["task_name"],
                    "family": c["family"],
                    "category": c["category"],
                    "severity": 2,
                    "seed": seed,
                    "init_index": 0,
                    "flow_steps": n,
                    "policy_arm": arm,
                    "phase": "smoke" if smoke else "main",
                    "manifest_sha256": SHA,
                    "checkpoint_sha256": STATE,
                    "head_sha256": OBS,
                    "gpu_uuid": "GPU-test",
                    "initial_state_sha256": STATE,
                    "stabilized_state_sha256": STATE,
                    "status": "ok",
                    "success": (i in (1, 2)) if n == 1 else (i in (0, 3)),
                    "policy_steps": 7,
                    "episode_ms": 1000.0,
                    "total_velocity_evaluations": 2 * n,
                    "chunks": chunks,
                    "initial_sigma": chunks[0]["first_sigma"],
                    "initial_log_sigma": chunks[0]["first_log_sigma"],
                }
            )
        if smoke:
            selected = m["metadata_rule"][c["category"]]
            replay = copy.deepcopy(
                next(r for r in rows if r["condition_id"] == c["condition_id"] and r["flow_steps"] == selected)
            )
            replay["policy_arm"] = "metadata"
            rows.append(replay)
    return m, rows


class SelectiveTests(unittest.TestCase):
    def setUp(self):
        self.m, self.rows = fixture()

    def validate(self, *, smoke=False):
        return a.validate(self.m, self.rows, "libero_plus", SHA, smoke=smoke)

    def test_complete_fixed_pairs(self):
        found, errors = self.validate()
        self.assertEqual(len(found), 8)
        self.assertEqual(errors, [])

    def test_primary_counts_derived_rule_and_equal_case_costs(self):
        result = a.analyze(self.m, self.rows, "libero_plus", SHA)
        g = result["overall"]
        self.assertEqual([g["strategies"][x]["successes"] for x in a.STRATEGIES], [2, 2, 2])
        self.assertEqual(g["comparisons"]["fixed10_vs_fixed1"]["rescues"], 2)
        self.assertEqual(g["comparisons"]["fixed10_vs_fixed1"]["regressions"], 2)
        primary = g["comparisons"]["metadata_vs_fixed10"]
        self.assertEqual((primary["rescues"], primary["regressions"]), (1, 1))
        self.assertEqual(primary["net_success_difference"], 0)
        self.assertEqual(primary["cost_differences"]["mean_policy_call_ms"]["mean"], -20)
        self.assertFalse(g["conservative_primary_support"])
        self.assertEqual(result["sigma"]["overall"]["auc_higher_initial_sigma_predicts_rescue"], 1)
        self.assertFalse(result["sigma"]["overall"]["credible_count_threshold_met"])
        self.assertIsNone(
            result["sigma"]["by_type"]["Robot Initial States"]["auc_higher_initial_sigma_predicts_rescue"]
        )

    def test_missing_duplicate_and_unexpected_arm_rejected(self):
        self.rows.pop()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.validate()
        self.m, self.rows = fixture()
        self.rows.append(copy.deepcopy(self.rows[0]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.validate()
        self.rows.pop()
        self.rows[0]["policy_arm"] = "metadata"
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            self.validate()

    def test_head_and_policy_identity_rejected(self):
        for field in ("checkpoint_sha256", "head_sha256", "manifest_sha256", "benchmark_commit"):
            self.m, self.rows = fixture()
            self.rows[0][field] = "bad"
            with self.assertRaisesRegex(ValueError, "Provenance"):
                self.validate()

    def test_finite_boolean_and_integer_fields(self):
        for field, bad in (("success", 1), ("flow_steps", 1.0), ("episode_ms", float("nan"))):
            self.m, self.rows = fixture()
            self.rows[0][field] = bad
            with self.assertRaises(ValueError):
                self.validate()

    def test_same_pass_head_and_no_reference_calls(self):
        for field, bad in (
            ("head_evaluations", 2),
            ("sigma_time", 0.9),
            ("reference_velocity_evaluations", 1),
            ("head_ms", 999),
            ("first_sigma", -1),
            ("first_log_sigma", float("nan")),
        ):
            self.m, self.rows = fixture()
            self.rows[0]["chunks"][0][field] = bad
            with self.assertRaises(ValueError):
                self.validate()

    def test_each_chunk_noise(self):
        self.rows[0]["chunks"][1]["noise_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "noise"):
            self.validate()

    def test_initial_score_consistency_and_pair_agreement(self):
        self.rows[0]["initial_sigma"] += 0.1
        with self.assertRaisesRegex(ValueError, "Initial score"):
            self.validate()
        self.m, self.rows = fixture()
        row = self.rows[0]
        row["chunks"][0]["first_log_sigma"] += 0.1
        row["chunks"][0]["first_sigma"] = math.exp(row["chunks"][0]["first_log_sigma"])
        row["initial_log_sigma"] = row["chunks"][0]["first_log_sigma"]
        row["initial_sigma"] = row["chunks"][0]["first_sigma"]
        with self.assertRaisesRegex(ValueError, "Paired initial sigma"):
            self.validate()

    def test_pair_stabilized_state_and_first_observation(self):
        for field in ("stabilized_state_sha256",):
            self.m, self.rows = fixture()
            self.rows[0][field] = "0" * 64
            with self.assertRaisesRegex(ValueError, "Paired"):
                self.validate()
        self.m, self.rows = fixture()
        self.rows[0]["chunks"][0]["observation_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "initial observations"):
            self.validate()

    def test_smoke_wrapper_exact_replay(self):
        self.m, self.rows = fixture(smoke=True)
        self.assertEqual(len(self.validate(smoke=True)[0]), 12)
        meta = next(r for r in self.rows if r["policy_arm"] == "metadata")
        meta["chunks"][1]["action_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "trajectory digest"):
            self.validate(smoke=True)

    def test_separate_smoke_conditions_outside_main_registry(self):
        self.m, self.rows = fixture(smoke=True)
        spec = self.m["benchmarks"]["libero_plus"]
        spec["smoke_conditions"] = copy.deepcopy(spec["conditions"])
        for condition in spec["smoke_conditions"]:
            condition["condition_id"] = "smoke-" + condition["condition_id"]
            condition["cases"][0]["seed"] = 900101
        spec.pop("smoke_cases")
        for row in self.rows:
            row["condition_id"] = "smoke-" + row["condition_id"]
        self.assertEqual(len(self.validate(smoke=True)[0]), 12)
        with self.assertRaises(ValueError):
            self.validate()

    def test_smoke_wrapper_step_mapping(self):
        self.m, self.rows = fixture(smoke=True)
        meta = next(r for r in self.rows if r["policy_arm"] == "metadata")
        meta["flow_steps"] = 1
        with self.assertRaisesRegex(ValueError, "Provenance"):
            self.validate(smoke=True)

    def test_error_retained_not_outcome(self):
        error = dict(self.rows[0], status="error", error="synthetic interruption")
        self.rows.append(error)
        found, errors = self.validate()
        self.assertEqual((len(found), len(errors)), (8, 1))

    def test_auc_direction_ties_singleclass(self):
        self.assertEqual(a.auc([3, 4, 1, 2], [1, 1, 0, 0]), 1)
        self.assertEqual(a.auc([1, 2, 3, 4], [1, 1, 0, 0]), 0)
        self.assertEqual(a.auc([1, 1], [1, 0]), 0.5)
        self.assertIsNone(a.auc([1, 2], [1, 1]))

    def test_condition_and_family_cluster_sensitivity(self):
        cases = [{"condition_id": "a", "family": "same"}, {"condition_id": "b", "family": "same"}]
        self.assertEqual(a.cluster_ci(cases, [-1, 1], "condition_id", self.m["analysis"]), [-1, 1])
        self.assertIsNone(a.cluster_ci(cases, [-1, 1], "family", self.m["analysis"]))

    def test_zero_discordance_sigma_unmeasured(self):
        for row in self.rows:
            row["success"] = True
        result = a.analyze(self.m, self.rows, "libero_plus", SHA)
        self.assertIsNone(result["sigma"]["overall"]["auc_higher_initial_sigma_predicts_rescue"])
        self.assertEqual(result["sigma"]["overall"]["bootstrap"]["condition_id"]["valid_draws"], 0)

    def test_cost_estimator_weights_cases_not_chunk_counts(self):
        row = self.rows[0]
        self.assertEqual(a.row_cost(row)["mean_policy_call_ms"], 10)
        row["chunks"][1]["policy_ms"] = 30
        self.assertEqual(a.row_cost(row)["mean_policy_call_ms"], 20)


if __name__ == "__main__":
    unittest.main()
