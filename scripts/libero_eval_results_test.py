from __future__ import annotations

import copy
import csv
import json
import pathlib
import tempfile
import unittest

from scripts import libero_eval_results as results


class LiberoEvalResultsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def make_records(self) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        sequence = 0
        for suite_index, suite in enumerate(results.SUITES):
            for task_id in range(results.TASKS_PER_SUITE):
                for episode_idx in range(results.TRIALS_PER_TASK):
                    success = episode_idx < 10 + suite_index
                    records.append(
                        {
                            "schema_version": 1,
                            "eval_id": "eval-0001",
                            "artifact_id": "sha256:checkpoint-digest",
                            "artifact_name": "pi05-libero-tvm-30k",
                            "checkpoint_name": "29999",
                            "checkpoint_path": "/volt/checkpoints/pi05/29999",
                            "checkpoint_s3_uri": "s3://bucket/pi05/29999",
                            "model_class": "tvm",
                            "training_run_id": "pi05_libero_tvm_seed42",
                            "train_steps_completed": 30000,
                            "checkpoint_label": "30k",
                            "config": "pi05_libero_tvm",
                            "git_sha": "0123456789abcdef0123456789abcdef01234567",
                            "flow_steps": 2,
                            "eval_seed": 7,
                            "replan_steps": 5,
                            "trials_per_task": 50,
                            "status": "complete",
                            "suite": suite,
                            "task_id": task_id,
                            "episode_idx": episode_idx,
                            "success": success,
                            "exception": None,
                            "started_at_utc": f"2026-09-20T00:{sequence // 60:02d}:{sequence % 60:02d}Z",
                            "finished_at_utc": f"2026-09-20T00:{sequence // 60:02d}:{sequence % 60:02d}.500000Z",
                            "client_log_path": f"/volt/logs/{suite}.log",
                        }
                    )
                    sequence = (sequence + 1) % 3600
        return records

    def write_jsonl(self, records: list[dict[str, object]], name: str = "episodes.jsonl") -> pathlib.Path:
        path = self.root / name
        with path.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        return path

    def test_aggregate_complete_evaluation_and_write_rfc4180_csv(self) -> None:
        records = self.make_records()
        # Input order must not influence aggregate results.
        records.reverse()
        input_path = self.write_jsonl(records)
        output_path = self.root / "summary.v1.csv"

        results.aggregate_jsonl_to_csv([input_path], output_path)

        raw = output_path.read_bytes()
        self.assertIn(b"\r\n", raw)
        with output_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["schema_version"], "1")
        self.assertEqual(row["episodes_total"], "2000")
        self.assertEqual(row["successes_spatial"], "100")
        self.assertEqual(row["successes_object"], "110")
        self.assertEqual(row["successes_goal"], "120")
        self.assertEqual(row["successes_libero10"], "130")
        self.assertEqual(row["successes_total"], "460")
        self.assertEqual(row["failures_total"], "1540")
        self.assertEqual(row["success_rate"], "0.230000")
        self.assertEqual(row["eval_started_at_utc"], "2026-09-20T00:00:00Z")
        self.assertEqual(row["status"], "complete")
        self.assertEqual(tuple(row), results.SUMMARY_FIELDS)

    def test_duplicate_episode_is_rejected(self) -> None:
        records = self.make_records()
        records[-1] = copy.deepcopy(records[0])

        with self.assertRaisesRegex(results.ValidationError, "duplicate episode key"):
            results.aggregate_records(records)

    def test_incomplete_evaluation_is_rejected(self) -> None:
        with self.assertRaisesRegex(results.ValidationError, "expected 2000 episodes, got 1999"):
            results.aggregate_records(self.make_records()[:-1])

    def test_inconsistent_protocol_metadata_is_rejected(self) -> None:
        records = self.make_records()
        records[-1]["flow_steps"] = 3

        with self.assertRaisesRegex(results.ValidationError, "inconsistent evaluation metadata: flow_steps"):
            results.aggregate_records(records)

    def test_non_boolean_success_is_rejected(self) -> None:
        records = self.make_records()
        records[0]["success"] = 1

        with self.assertRaisesRegex(results.ValidationError, "success must be a JSON boolean"):
            results.aggregate_records(records)

    def test_complete_episode_with_exception_is_rejected(self) -> None:
        records = self.make_records()
        records[0]["exception"] = "environment crashed"

        with self.assertRaisesRegex(results.ValidationError, "complete episode contains exception"):
            results.aggregate_records(records)

    def test_inconsistent_suite_log_path_is_rejected(self) -> None:
        records = self.make_records()
        records[0]["client_log_path"] = "/volt/logs/other.log"

        with self.assertRaisesRegex(results.ValidationError, "inconsistent client_log_path"):
            results.aggregate_records(records)

    def test_existing_output_is_not_replaced_by_default(self) -> None:
        output_path = self.root / "summary.v1.csv"
        output_path.write_text("sentinel\n", encoding="utf-8")
        summary = results.aggregate_records(self.make_records())

        with self.assertRaises(FileExistsError):
            results.write_summary_csv(summary, output_path)
        self.assertEqual(output_path.read_text(encoding="utf-8"), "sentinel\n")

    def test_invalid_git_sha_and_naive_timestamp_are_rejected(self) -> None:
        records = self.make_records()
        records[0]["git_sha"] = "short"
        with self.assertRaisesRegex(results.ValidationError, "full 40-character"):
            results.aggregate_records(records)

        records = self.make_records()
        records[0]["started_at_utc"] = "2026-09-20T00:00:00"
        with self.assertRaisesRegex(results.ValidationError, "must include a UTC offset"):
            results.aggregate_records(records)

    def test_schema_version_and_exception_field_are_required(self) -> None:
        records = self.make_records()
        records[0]["schema_version"] = 2
        with self.assertRaisesRegex(results.ValidationError, "unsupported schema_version"):
            results.aggregate_records(records)

        records = self.make_records()
        del records[0]["exception"]
        with self.assertRaisesRegex(results.ValidationError, "missing required fields: exception"):
            results.aggregate_records(records)


if __name__ == "__main__":
    unittest.main()
