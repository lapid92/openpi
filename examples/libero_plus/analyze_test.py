"""Fail-closed tests for the declared LIBERO-Plus pilot analyzer."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from examples.libero_plus import analyze


@pytest.fixture
def complete_records(tmp_path):
    source_manifest = Path(__file__).with_name("pilot_manifest.json")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(source_manifest.read_bytes())
    manifest = json.loads(manifest_path.read_text())
    manifest_hash = analyze.sha256_file(manifest_path)
    rows = []
    for task in manifest["tasks"]:
        for seed in task["seeds"]:
            case_digest = f"{task['id']:032x}{seed:032x}"
            for arm in manifest["flow_steps"]:
                times = (1 - np.arange(arm) / arm).tolist()
                row = {
                    "task_id": task["id"],
                    "task_name": task["name"],
                    "category": task["category"],
                    "seed": seed,
                    "flow_steps": arm,
                    "status": "ok",
                    "success": False,
                    "policy_steps": 1,
                    "episode_ms": 15.0,
                    "manifest_sha256": manifest_hash,
                    "benchmark_commit": manifest["benchmark_commit"],
                    "checkpoint_identity": manifest["base_checkpoint_identity"],
                    "head_checkpoint": manifest["head_sha256"],
                    "gpu_uuid": analyze.EXPECTED_UUID,
                    "initial_state_digest": case_digest,
                    "chunks": [
                        {
                            "chunk_index": 0,
                            "times": times,
                            "sigmas": [0.1] * arm,
                            "log_sigma": [float(np.log(0.1))] * arm,
                            "velocity_evaluations": arm,
                            "request_ms": 5.0,
                            "policy_ms": 4.0,
                            "noise_digest": case_digest,
                            "observation_digest": case_digest,
                            "action_digest": "a" * 64,
                        }
                    ],
                    "total_velocity_evaluations": arm,
                }
                rows.append(row)
    assert len(rows) == 48
    records_path = tmp_path / "records.jsonl"

    def write(items):
        records_path.write_text("".join(json.dumps(item) + "\n" for item in items))
        return records_path

    write(rows)
    return manifest_path, records_path, rows, write


def test_load_validated_requires_exact_complete_48_episode_set(complete_records):
    manifest_path, records_path, rows, write = complete_records
    manifest, _, records = analyze.load_validated(manifest_path, records_path)
    assert len(records) == 48
    assert len(manifest["tasks"]) == 6
    write(rows[:-1])
    with pytest.raises(ValueError, match="Expected 48 completed episodes"):
        analyze.load_validated(manifest_path, records_path)
    write([*rows[:-1], copy.deepcopy(rows[0])])
    with pytest.raises(ValueError, match="Duplicate episode"):
        analyze.load_validated(manifest_path, records_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "error", "Episode error"),
        ("total_velocity_evaluations", 99, "Total velocity count mismatch"),
        ("episode_ms", -1.0, "Invalid episode_ms"),
    ],
)
def test_episode_errors_counts_and_latency_fail_closed(complete_records, field, value, message):
    manifest_path, records_path, rows, write = complete_records
    changed = copy.deepcopy(rows)
    changed[0][field] = value
    write(changed)
    with pytest.raises(ValueError, match=message):
        analyze.load_validated(manifest_path, records_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("velocity_evaluations", 99, "Chunk index or velocity count mismatch"),
        ("times", [0.9], "Fixed Euler times changed"),
        ("request_ms", -1.0, "Invalid request_ms"),
        ("policy_ms", 6.0, "HTTP request time is shorter"),
    ],
)
def test_chunk_schedule_accounting_and_latency_fail_closed(complete_records, field, value, message):
    manifest_path, records_path, rows, write = complete_records
    changed = copy.deepcopy(rows)
    changed[0]["chunks"][0][field] = value
    write(changed)
    with pytest.raises(ValueError, match=message):
        analyze.load_validated(manifest_path, records_path)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("initial_state_digest", "Initial-state mismatch"),
        ("noise_digest", "First-chunk noise mismatch"),
        ("observation_digest", "First-chunk observation mismatch"),
    ],
)
def test_first_chunk_pairing_identity_fails_closed(complete_records, field, message):
    manifest_path, records_path, rows, write = complete_records
    changed = copy.deepcopy(rows)
    # Row 1 is the two-step arm for the same task instance and seed as row 0.
    if field == "initial_state_digest":
        changed[1][field] = "b" * 64
    else:
        changed[1]["chunks"][0][field] = "b" * 64
    write(changed)
    with pytest.raises(ValueError, match=message):
        analyze.load_validated(manifest_path, records_path)


def test_first_score_drift_uses_declared_tolerance_and_one_step_canonical(complete_records):
    manifest_path, records_path, rows, write = complete_records
    changed = copy.deepcopy(rows)
    canonical = changed[0]["chunks"][0]["log_sigma"][0]
    changed[1]["chunks"][0]["log_sigma"][0] = canonical + 0.009
    changed[1]["chunks"][0]["sigmas"][0] = float(np.exp(canonical + 0.009))
    write(changed)
    analyze.load_validated(manifest_path, records_path)
    changed[1]["chunks"][0]["log_sigma"][0] = canonical + 0.011
    changed[1]["chunks"][0]["sigmas"][0] = float(np.exp(canonical + 0.011))
    write(changed)
    with pytest.raises(ValueError, match="log sigma drift"):
        analyze.load_validated(manifest_path, records_path)


def test_paired_gate_and_sigma_auc_are_separate_from_action_agreement(complete_records, monkeypatch):
    manifest_path, records_path, rows, write = complete_records
    changed = copy.deepcopy(rows)
    # Four paired gains across Robot Initial States and Camera Viewpoints;
    # first-chunk sigma orders those gains above all nongains.
    gain_cases = {(259, 7), (259, 11), (609, 7), (609, 11)}
    for row in changed:
        gain = (row["task_id"], row["seed"]) in gain_cases
        sigma = 0.2 if gain else 0.1
        row["chunks"][0]["sigmas"] = [sigma] * row["flow_steps"]
        row["chunks"][0]["log_sigma"] = [float(np.log(sigma))] * row["flow_steps"]
        row["success"] = bool(gain and row["flow_steps"] == 2)
    write(changed)
    manifest, manifest_hash, records = analyze.load_validated(manifest_path, records_path)
    monkeypatch.setattr(analyze, "BOOTSTRAP_REPLICATES", 30)
    summary = analyze.summarize(manifest, manifest_hash, records)
    assert summary["pair_count"] == 12
    assert summary["episode_count"] == 48
    assert summary["by_arm"]["2"]["paired_vs_1"]["wins"] == 4
    assert summary["by_arm"]["2"]["paired_vs_1"]["losses"] == 0
    assert summary["decision_gate_pass"] is True
    assert summary["sigma_ranking"]["2"]["auc_sigma_t1_for_gain"] == 1.0
    assert summary["by_arm"]["1"]["velocity_evaluations_per_chunk"]["mean"] == 1
    assert summary["by_arm"]["10"]["velocity_evaluations_per_chunk"]["mean"] == 10
