"""Synthetic-only scientific accounting regression tests; no simulator or GPU."""

# ruff: noqa: PT009, PT027
# Keep stdlib unittest so the isolated simulator runtime need not install pytest.
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("study_analyze", Path(__file__).with_name("analyze.py"))
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)

SHA = hashlib.sha256(b"synthetic manifest").hexdigest()
CHECKPOINT = hashlib.sha256(b"frozen checkpoint").hexdigest()
DIGEST = hashlib.sha256(b"state").hexdigest()
UUID = "GPU-synthetic-test-only"


def fixture():
    conditions = [
        {
            "condition_id": "c" + str(i),
            "suite": "libero_goal",
            "task_name": "synthetic_task_" + str(i),
            "family": "family_" + str(i),
            "category": "synthetic",
            "cases": [{"seed": 100 + j, "init_index": j, "initial_state_sha256": DIGEST} for j in range(3)],
        }
        for i in range(2)
    ]
    manifest = {
        "flow_steps": [1, 2, 4, 10],
        "checkpoint": {"sha256": CHECKPOINT},
        "gpu_uuids": [UUID],
        "noise_policy": {"shape": [10, 32]},
        "settings": {
            "max_policy_steps": {"libero_goal": 30},
            "replan_steps": 5,
            "action_horizon": 10,
            "action_dim": 32,
        },
        "benchmarks": {
            "libero": {
                "conditions": conditions,
                "commit": "abc",
                "smoke_cases": [
                    {"condition_id": "c0", "seed": 900001, "init_index": 0, "initial_state_sha256": DIGEST}
                ],
            }
        },
    }
    rows = []
    for condition in conditions:
        for j, case in enumerate(condition["cases"]):
            # Across 6 cases: base 2/6, arm2 4/6 (4 rescues,2 regressions), arm4 6/6, arm10 0/6.
            outcomes = {1: j == 2, 2: j != 2, 4: True, 10: False}
            for arm in module.ARMS:
                chunks = [
                    {
                        "chunk_index": chunk,
                        "noise_sha256": module.noise_digest(
                            "libero",
                            condition["suite"],
                            condition["task_name"],
                            case["seed"],
                            case["init_index"],
                            chunk,
                        ),
                        "action_sha256": DIGEST,
                        "observation_sha256": DIGEST,
                        "velocity_evaluations": arm,
                        "policy_ms": 10.0 * arm,
                        "request_ms": 10.0 * arm + 1,
                    }
                    for chunk in range(2)
                ]
                rows.append(
                    {
                        "benchmark": "libero",
                        "benchmark_commit": "abc",
                        "phase": "main",
                        "condition_id": condition["condition_id"],
                        "suite": condition["suite"],
                        "task_name": condition["task_name"],
                        "family": condition["family"],
                        "category": condition["category"],
                        "seed": case["seed"],
                        "init_index": case["init_index"],
                        "flow_steps": arm,
                        "status": "ok",
                        "success": outcomes[arm],
                        "manifest_sha256": SHA,
                        "checkpoint_sha256": CHECKPOINT,
                        "gpu_uuid": UUID,
                        "initial_state_sha256": DIGEST,
                        "stabilized_state_sha256": DIGEST,
                        "episode_ms": 1000.0,
                        "policy_steps": 7,
                        "total_velocity_evaluations": 2 * arm,
                        "chunks": chunks,
                    }
                )
    return manifest, rows


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.manifest, self.rows = fixture()

    def validate(self):
        return module.validate(self.manifest, self.rows, "libero", SHA)

    def test_correct_paired_counts_denominators_and_costs(self):
        result = module.analyze(self.manifest, self.rows, "libero", SHA)
        arms = result["overall"]["arms"]
        self.assertEqual([arms[str(a)]["successes"] for a in module.ARMS], [2, 4, 6, 0])
        pair = arms["2"]["paired_vs_one"]
        self.assertEqual((pair["rescues"], pair["regressions"], pair["one_step_failures"]), (4, 2, 4))
        self.assertEqual(pair["fraction_one_step_failures_rescued"], 1)
        self.assertAlmostEqual(pair["net_success_difference"], 1 / 3)
        self.assertIn("Unresolved", pair["assessment"])
        self.assertEqual(arms["4"]["cost"]["total_velocity_evaluations"], 48)
        self.assertEqual(arms["4"]["cost"]["scored_policy_call_ms"]["mean"], 40)
        self.assertEqual(arms["4"]["cost"]["simulator_inclusive_episode_ms"]["mean"], 1000)
        self.assertIsNone(
            result["by_family"]["family_0"]["arms"]["2"]["paired_vs_one"]["net_difference_condition_cluster_ci95"]
        )
        self.assertEqual(result["uncertainty"]["replicates"], 10000)

    def test_cluster_interval_reproducible(self):
        cases = [
            {"condition_id": "a", "success": {"1": False, "2": True}},
            {"condition_id": "b", "success": {"1": True, "2": False}},
        ]
        ci = module.cluster_interval(cases, 2)
        self.assertEqual(ci, [-1.0, 1.0])
        self.assertEqual(ci, module.cluster_interval(cases, 2))

    def test_missing_and_duplicate_completed_rejected(self):
        self.rows.pop()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.validate()
        self.manifest, self.rows = fixture()
        self.rows.append(copy.deepcopy(self.rows[0]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.validate()

    def test_identity_mismatches_rejected(self):
        for field, bad in [
            ("benchmark", "libero_plus"),
            ("checkpoint_sha256", "0" * 64),
            ("manifest_sha256", "0" * 64),
            ("gpu_uuid", "GPU-other"),
            ("phase", "smoke"),
            ("task_name", "other"),
            ("family", "other"),
        ]:
            with self.subTest(field=field):
                self.manifest, self.rows = fixture()
                self.rows[0][field] = bad
                with self.assertRaises(ValueError):
                    self.validate()

    def test_noise_shape_is_part_of_protocol(self):
        self.manifest["noise_policy"]["shape"] = [50, 32]
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            self.validate()
        self.manifest["settings"]["action_horizon"] = 50
        with self.assertRaisesRegex(ValueError, "noise mismatch"):
            self.validate()

    def test_episode_keys_strictly_integer(self):
        for field in ("flow_steps", "seed", "init_index"):
            self.manifest, self.rows = fixture()
            self.rows[0][field] = float(self.rows[0][field])
            with self.assertRaisesRegex(ValueError, "integers"):
                self.validate()

    def test_bool_success_required(self):
        for value in [1, 0, None, "false"]:
            self.rows[0]["success"] = value
            with self.assertRaisesRegex(ValueError, "boolean"):
                self.validate()

    def test_each_chunk_noise_recomputed(self):
        self.rows[0]["chunks"][1]["noise_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "noise mismatch"):
            self.validate()

    def test_wrong_initial_or_stabilized_state_rejected(self):
        self.rows[0]["initial_state_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Initial state"):
            self.validate()
        self.manifest, self.rows = fixture()
        self.rows[0]["stabilized_state_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "stabilized"):
            self.validate()

    def test_first_observation_mismatch_rejected_but_later_divergence_allowed(self):
        self.rows[0]["chunks"][1]["observation_sha256"] = "0" * 64
        self.validate()
        self.rows[0]["chunks"][0]["observation_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "first observations"):
            self.validate()

    def test_cost_accounting_and_nan_rejected(self):
        for field, bad in [
            ("velocity_evaluations", 99),
            ("policy_ms", float("nan")),
            ("request_ms", -1),
            ("chunk_index", True),
        ]:
            self.manifest, self.rows = fixture()
            self.rows[0]["chunks"][1][field] = bad
            with self.assertRaises(ValueError):
                self.validate()
        self.manifest, self.rows = fixture()
        self.rows[0]["total_velocity_evaluations"] = 9
        with self.assertRaisesRegex(ValueError, "Total velocity"):
            self.validate()
        self.manifest, self.rows = fixture()
        self.rows[0]["policy_steps"] = 1
        with self.assertRaisesRegex(ValueError, "Chunk/action"):
            self.validate()

    def test_errors_separate_from_completed_outcomes(self):
        error = copy.deepcopy(self.rows[0])
        error.update(status="error", error="synthetic infrastructure interruption")
        error.pop("success")
        self.rows.append(error)
        found, errors = self.validate()
        self.assertEqual(len(found), 24)
        self.assertEqual(len(errors), 1)
        self.rows = [error, *self.rows[1:-1]]
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.validate()

    def test_zero_failure_rescue_fraction_is_null(self):
        for row in self.rows:
            row["success"] = True
        result = module.analyze(self.manifest, self.rows, "libero", SHA)
        self.assertIsNone(result["overall"]["arms"]["2"]["paired_vs_one"]["fraction_one_step_failures_rescued"])
        self.assertIn("Unresolved", result["assessment"])
        json.dumps(result, allow_nan=False)

    def test_smoke_separate_exact_plan(self):
        self.rows = self.rows[:4]
        for row in self.rows:
            row.update(phase="smoke", seed=900001)
            for chunk in row["chunks"]:
                chunk["noise_sha256"] = module.noise_digest(
                    "libero", row["suite"], row["task_name"], 900001, 0, chunk["chunk_index"]
                )
        found, _ = module.validate(self.manifest, self.rows, "libero", SHA, smoke=True)
        self.assertEqual(len(found), 4)
        with self.assertRaises(ValueError):
            self.validate()


class ClientPlanningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = str(Path(__file__).parent)
        sys.path.insert(0, directory)
        try:
            spec = importlib.util.spec_from_file_location("study_client", Path(directory) / "client.py")
            cls.client = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.client)
        finally:
            sys.path.remove(directory)

    def test_shards_partition_complete_pairs_and_rotate_arms(self):
        manifest, _ = fixture()
        all_keys = []
        for worker in range(4):
            plan = self.client.planned_episodes(manifest, "libero", worker_index=worker, workers=4)
            keys = [(c["condition_id"], x["seed"], x["init_index"], arm) for c, x, arm in plan]
            all_keys.extend(keys)
            for index in range(0, len(keys), 4):
                pair = keys[index : index + 4]
                self.assertEqual(len({key[:3] for key in pair}), 1)
                self.assertEqual({key[3] for key in pair}, {1, 2, 4, 10})
        self.assertEqual(len(all_keys), 24)
        self.assertEqual(len(set(all_keys)), 24)
        plan = self.client.planned_episodes(manifest, "libero")
        self.assertEqual([x[2] for x in plan[:8]], [1, 2, 4, 10, 2, 4, 10, 1])

    def test_resume_reaudits_pairs_and_preserves_error_attempt(self):
        manifest, rows = fixture()
        plan = self.client.planned_episodes(manifest, "libero")
        health = {"checkpoint_sha256": CHECKPOINT, "gpu_uuid": UUID}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            error = copy.deepcopy(rows[0])
            error.update(status="error", error="synthetic failure")
            for row in [error, *rows[:4]]:
                self.client.append_record(path, row)
            completed = self.client.validate_existing(path, SHA, plan, "main", health)
            self.assertEqual(len(completed), 4)
            self.assertEqual(len(path.read_text().splitlines()), 5)
            rows[1]["stabilized_state_sha256"] = "0" * 64
            path.write_text("\n".join(json.dumps(r) for r in rows[:4]) + "\n")
            with self.assertRaisesRegex(RuntimeError, "Pair mismatch"):
                self.client.validate_existing(path, SHA, plan, "main", health)

    def test_resume_rejects_unknown_status_and_nonbool_outcome(self):
        manifest, rows = fixture()
        plan = self.client.planned_episodes(manifest, "libero")
        health = {"checkpoint_sha256": CHECKPOINT, "gpu_uuid": UUID}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            for change in ({"status": "running"}, {"success": 1}):
                row = dict(rows[0], **change)
                path.write_text(json.dumps(row) + "\n")
                with self.assertRaises(RuntimeError):
                    self.client.validate_existing(path, SHA, plan, "main", health)


class SupervisorGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = str(Path(__file__).parent)
        sys.path.insert(0, directory)
        try:
            spec = importlib.util.spec_from_file_location("study_supervise", Path(directory) / "supervise.py")
            cls.supervisor = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.supervisor)
        finally:
            sys.path.remove(directory)

    def make_smoke(self, directory):
        manifest, rows = fixture()
        directory = Path(directory)
        manifest_path = directory / "protocol.json"
        manifest_path.write_text(json.dumps(manifest))
        mh = module.digest_file(manifest_path)
        rows = rows[:4]
        for row in rows:
            row.update(phase="smoke", seed=900001, manifest_sha256=mh)
            for chunk in row["chunks"]:
                chunk["noise_sha256"] = module.noise_digest(
                    "libero", row["suite"], row["task_name"], 900001, 0, chunk["chunk_index"]
                )
        path = directory / "libero-smoke.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        summary = module.analyze(manifest, rows, "libero", mh, smoke=True)
        summary["record_files"] = [{"path": str(path), "sha256": module.digest_file(path)}]
        summary_path = directory / "libero-smoke-summary.json"
        summary_path.write_text(json.dumps(summary))
        return manifest, manifest_path, path, summary_path, rows, summary

    def test_valid_smoke_gate_without_process_launches(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(sys.modules, {"analyze": module}):
            manifest, manifest_path, *_ = self.make_smoke(directory)
            with mock.patch.object(self.supervisor.subprocess, "Popen") as launch:
                self.supervisor.validate_smoke_gate(manifest, manifest_path, directory)
                launch.assert_not_called()

    def test_gate_reaudits_missing_records_and_detects_stale_hash(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(sys.modules, {"analyze": module}):
            manifest, mp, path, _, rows, _ = self.make_smoke(directory)
            path.write_text("\n".join(json.dumps(r) for r in rows[:3]) + "\n")
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                self.supervisor.validate_smoke_gate(manifest, mp, directory)
            rows[0]["episode_ms"] = 1001
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                self.supervisor.validate_smoke_gate(manifest, mp, directory)

    def test_gate_rejects_wrong_summary_identity(self):
        for field, value in (
            ("benchmark", "libero_plus"),
            ("phase", "main"),
            ("completed_episodes", 1),
            ("audit", "pending"),
        ):
            with tempfile.TemporaryDirectory() as directory, mock.patch.dict(sys.modules, {"analyze": module}):
                manifest, mp, _, summary_path, _, summary = self.make_smoke(directory)
                summary[field] = value
                summary_path.write_text(json.dumps(summary))
                with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                    self.supervisor.validate_smoke_gate(manifest, mp, directory)


class ReporterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("study_report", Path(__file__).with_name("report.py"))
        cls.reporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.reporter)

    def test_reporter_smoke_counts_latency_units_and_separation(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, mp, _, sp, _, summary = SupervisorGateTests().make_smoke(directory)
            summary["wandb_url"] = "https://example.test/synthetic-run"
            sp.write_text(json.dumps(summary))
            output = Path(directory) / "REPORT.md"
            argv = ["report.py", "--manifest", str(mp), "--results", directory, "--output", str(output), "--smoke"]
            with mock.patch.object(sys, "argv", argv):
                self.reporter.main()
            text = output.read_text()
            self.assertIn("Smoke validation only; excluded from main inference.", text)
            self.assertIn("| 2 | 1/1 (100.00%) | 1 / 0 |", text)
            self.assertIn("| 1 | 1 | 2 | 10.00 / 10.00 | 1.00 / 1.00 |", text)
            self.assertIn("Each benchmark is reported separately.", text)
            self.assertNotIn("Complete planned paired evaluations", text)

    def test_reporter_rejects_stale_summary_provenance(self):
        for field, bad in (
            ("manifest_sha256", "0" * 64),
            ("benchmark", "libero_plus"),
            ("phase", "main"),
            ("checkpoint_sha256", "0" * 64),
            ("benchmark_commit", "other"),
        ):
            with tempfile.TemporaryDirectory() as directory:
                _, mp, _, sp, _, summary = SupervisorGateTests().make_smoke(directory)
                summary.update(wandb_url="https://example.test/synthetic-run")
                summary[field] = bad
                sp.write_text(json.dumps(summary))
                argv = [
                    "report.py",
                    "--manifest",
                    str(mp),
                    "--results",
                    directory,
                    "--output",
                    str(Path(directory) / "REPORT.md"),
                    "--smoke",
                ]
                with mock.patch.object(sys, "argv", argv), self.assertRaises(ValueError):
                    self.reporter.main()

    def test_reporter_requires_validated_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            _, mp, _, sp, _, summary = SupervisorGateTests().make_smoke(directory)
            summary.update(audit="pending", wandb_url="https://example.test/synthetic-run")
            sp.write_text(json.dumps(summary))
            argv = [
                "report.py",
                "--manifest",
                str(mp),
                "--results",
                directory,
                "--output",
                str(Path(directory) / "REPORT.md"),
                "--smoke",
            ]
            with mock.patch.object(sys, "argv", argv), self.assertRaises(ValueError):
                self.reporter.main()


if __name__ == "__main__":
    unittest.main()
