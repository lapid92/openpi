import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robocasa_eval import report
from robocasa_eval.manifest import make_manifest


def fixture(model="smolvla", seed=7, steps=2):
    manifest = make_manifest(model, seed, steps, "campaign",
                             owner_robocasa_adapter="/pinned/robocasa.py" if model == "smolvla" else None,
                             owner_sha256="a" * 64 if model == "smolvla" else None)
    rows = []
    for task in manifest["tasks"]:
        for idx in range(50):
            row = dict(task=task["task"], group=task["group"], episode_index=idx,
                       manifest_sha256=report.manifest_digest(manifest),
                       campaign_id=manifest["campaign_id"], protocol_id=manifest["protocol_id"],
                       checkpoint_revision=manifest["checkpoint_revision"], denoising_steps=steps,
                       scene_seed=seed, replan_steps=manifest["replan_steps"],
                       scene_reset_protocol=report.RESET_PROTOCOL,
                       policy_seed=report.stable_seed(f"public_{model}_baseline", task["task"], idx, 0),
                       policy_queries=1, action_network_evaluations=steps,
                       status="policy_failure", success=False,
                       model_xml_sha256="a" * 64, initial_state_sha256="b" * 64)
            if model == "smolvla":
                row.update(model=model, split="target", prompt=task["task"])
            else:
                row["instruction"] = f"Do {task['task']}"
            rows.append(row)
    return manifest, rows


class ValidateCell(unittest.TestCase):
    def test_complete_cell_and_identical_duplicate(self):
        manifest, rows = fixture()
        self.assertEqual(len(report.validate_cell(manifest, rows + [copy.deepcopy(rows[0])], "smolvla", 7, 2)), 2500)

    def test_missing_and_conflicting_episode(self):
        manifest, rows = fixture()
        with self.assertRaisesRegex(ValueError, "Incomplete coverage"):
            report.validate_cell(manifest, rows[:-1], "smolvla", 7, 2)
        conflict = dict(rows[0], status="success", success=True)
        with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
            report.validate_cell(manifest, rows + [conflict], "smolvla", 7, 2)

    def test_nfe_identity_and_hash_fail_closed(self):
        manifest, rows = fixture()
        for change, message in (({"action_network_evaluations": 3}, "NFE"),
                                ({"prompt": "wrong"}, "protocol"),
                                ({"model_xml_sha256": ""}, "model_xml")):
            altered = [dict(rows[0], **change), *rows[1:]]
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, message):
                report.validate_cell(manifest, altered, "smolvla", 7, 2)

    def test_pi05_instruction_protocol(self):
        manifest, rows = fixture("pi05")
        self.assertEqual(len(report.validate_cell(manifest, rows, "pi05", 7, 2)), 2500)
        rows[0]["instruction"] = ""
        with self.assertRaisesRegex(ValueError, "instruction"):
            report.validate_cell(manifest, rows, "pi05", 7, 2)

    def test_rejects_invalid_smol_v1_protocol(self):
        manifest, rows = fixture()
        manifest["protocol_id"] = "smolvla_target_bare_task_h50_sequential_reset_policy0_interfaces_v1"
        with self.assertRaisesRegex(ValueError, "pinned smolvla protocol: protocol_id"):
            report.validate_cell(manifest, rows, "smolvla", 7, 2)

    def test_rejects_conflicting_raw_and_csv_evidence(self):
        manifest, rows = fixture()
        with tempfile.TemporaryDirectory() as directory:
            cell = Path(directory)
            (cell / "manifest.json").write_text(json.dumps(manifest))
            (cell / "episodes").mkdir()
            (cell / "episodes" / "first.json").write_text(json.dumps(rows[0]))
            with (cell / "episodes.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerow(dict(rows[0], status="success", success=True))
            with self.assertRaisesRegex(ValueError, "Raw/CSV episode evidence conflicts"):
                report.load_cell(cell)
            with (cell / "episodes.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerow(dict(rows[0], prompt="changed but same outcome"))
            with self.assertRaisesRegex(ValueError, "Raw/CSV episode evidence conflicts"):
                report.load_cell(cell)
            with (cell / "episodes.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerow(rows[0])
            self.assertEqual(len(report.load_cell(cell)[1]), 1)


class ReportMath(unittest.TestCase):
    def test_three_seed_sample_sd_and_unpaired_hashes(self):
        def fake_load(path):
            return {"model": path.parent.parent.name}, []

        def fake_validate(_manifest, _rows, model, seed, steps):
            return {(group, 0): dict(group=group, success=seed != 7,
                                     model_xml_sha256=str(steps),
                                     initial_state_sha256=str(seed))
                    for group in report.GROUPS[:-1]}

        with patch.object(report, "load_cell", side_effect=fake_load), patch.object(
                report, "validate_cell", side_effect=fake_validate):
            output = report.build_report("unused")
        self.assertEqual(output["total_episodes"], 108)
        row = next(r for r in output["summary"] if r["model"] == "smolvla"
                   and r["steps"] == 2 and r["group"] == "overall")
        self.assertAlmostEqual(row["mean_pct"], 200 / 3)
        self.assertAlmostEqual(row["sample_sd_pct"], 100 / 3 ** 0.5)
        comparison = next(r for r in output["scene_hash_comparison"]
                          if r["model"] == "pi05" and r["seed"] == 7)
        self.assertEqual(comparison["model_xml_sha256_equal"], 0)
        self.assertEqual(comparison["initial_state_sha256_equal"], 3)


if __name__ == "__main__":
    unittest.main()
