"""Tests for the declared two-stage LIBERO-Plus evaluation protocol."""

import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import types

import pytest

from examples.libero_plus import two_stage_analyze

MANIFEST = Path(__file__).with_name("two_stage_manifest.json")


@pytest.fixture(scope="module")
def protocol():
    return json.loads(MANIFEST.read_text())


def test_screen_and_heldout_seeds_are_disjoint(protocol):
    screen = protocol["screening_seeds"]
    heldout = protocol["heldout_seeds"]
    smoke = protocol["smoke"]["seed"]
    assert len(screen) == len(set(screen)) == 3
    assert len(heldout) == len(set(heldout)) == 4
    assert set(screen).isdisjoint(heldout)
    assert smoke not in set(screen) | set(heldout)
    assert protocol["stage_1"]["flow_steps"] == [1]
    assert protocol["stage_2"]["flow_steps"] == [1, 2, 4, 10]


def test_declared_condition_coverage_and_screen_size(protocol):
    conditions = protocol["conditions"]
    assert len(conditions) == 12
    assert len({condition["condition_id"] for condition in conditions}) == len(conditions)
    assert len({condition["task_id"] for condition in conditions}) == len(conditions)
    assert {condition["suite"] for condition in conditions} == {"libero_goal", "libero_object", "libero_10"}
    assert {condition["perturbation_type"] for condition in conditions} == {"Robot Initial States", "Camera Viewpoints"}
    assert {condition["severity"] for condition in conditions} == {1, 3}
    assert len({condition["task_family"] for condition in conditions}) == 3
    assert protocol["stage_1"]["expected_episodes"] == len(conditions) * len(protocol["screening_seeds"])
    assert protocol["stage_2"]["expected_episodes_per_selected_condition"] == (
        len(protocol["heldout_seeds"]) * len(protocol["stage_2"]["flow_steps"])
    )
    assert all(condition["task_index"] == condition["task_id"] - 1 for condition in conditions)
    assert all(condition["init_state_index"] == 0 for condition in conditions)
    assert all(condition["max_policy_steps"] > 0 for condition in conditions)


def test_frozen_identities_noise_score_and_latency_are_declared(protocol):
    assert len(protocol["base_checkpoint_identity"]) == 64
    assert len(protocol["head_sha256"]) == 64
    assert len(protocol["benchmark_commit"]) == 40
    assert protocol["gpu_uuid"] == "GPU-4779cdde-a260-f8ec-da6a-6fa390bc7fd7"
    assert protocol["execution"]["replan_steps"] == 5
    assert protocol["execution"]["wait_steps"] == 10
    assert protocol["execution"]["first_score_log_sigma_tolerance"] == 0.01
    assert "same-pass" in protocol["execution"]["score"]
    assert "synchronized" in protocol["execution"]["latency"]
    assert "task-condition cluster" in protocol["stage_2"]["score_ranking_minimum"]


def _load_two_stage(monkeypatch):
    """Load protocol helpers without importing the isolated simulator runtime."""
    fake_libero = types.ModuleType("libero")
    fake_libero.__path__ = []
    fake_api = types.ModuleType("libero.libero")
    fake_api.__path__ = []
    fake_api.benchmark = object()
    fake_client = types.ModuleType("pilot_client")
    fake_client.run_episode = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "libero", fake_libero)
    monkeypatch.setitem(sys.modules, "libero.libero", fake_api)
    monkeypatch.setitem(sys.modules, "pilot_client", fake_client)
    spec = importlib.util.spec_from_file_location("isolated_two_stage", Path(__file__).with_name("two_stage.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def helpers(monkeypatch):
    return _load_two_stage(monkeypatch)


@pytest.fixture
def screen_records(protocol):
    return [
        _episode(protocol, condition, seed, 1, success=False)
        for condition in protocol["conditions"]
        for seed in protocol["screening_seeds"]
    ]


def _episode(protocol, condition, seed, arm, *, success=False):
    times = [1 - i / arm for i in range(arm)]
    digest = format(condition["task_id"], "032x") + format(seed, "032x")
    return {
        "condition_id": condition["condition_id"],
        "stage": "screen" if seed in protocol["screening_seeds"] else "compare",
        "task_family": condition["task_family"],
        "severity": condition["severity"],
        "task_id": condition["task_id"],
        "task_name": condition["task_name"],
        "category": condition["perturbation_type"],
        "suite": condition["suite"],
        "seed": seed,
        "flow_steps": arm,
        "checkpoint_identity": protocol["base_checkpoint_identity"],
        "head_checkpoint": protocol["head_sha256"],
        "benchmark_commit": protocol["benchmark_commit"],
        "gpu_uuid": protocol["gpu_uuid"],
        "status": "ok",
        "success": success,
        "chunks": [
            {
                "chunk_index": 0,
                "velocity_evaluations": arm,
                "times": times,
                "sigmas": [0.2] * arm,
                "log_sigma": [-1.6094379124341003] * arm,
                "noise_digest": digest,
                "observation_digest": digest,
                "action_digest": "a" * 64,
                "policy_ms": float(arm),
                "request_ms": float(arm + 1),
            }
        ],
        "total_velocity_evaluations": arm,
        "initial_state_digest": digest,
        "episode_ms": 100.0,
        "policy_steps": 1,
    }


def test_screen_selection_uses_only_declared_one_step_outcomes(protocol, helpers, screen_records):
    selected_by_suite = {}
    for suite in ("libero_goal", "libero_object", "libero_10"):
        candidates = [condition for condition in protocol["conditions"] if condition["suite"] == suite]
        chosen = next(
            condition
            for condition in candidates
            if condition["severity"] == 3 and condition["perturbation_type"] == "Robot Initial States"
        )
        selected_by_suite[suite] = chosen["condition_id"]
        for row in screen_records:
            if row["condition_id"] == chosen["condition_id"] and row["seed"] == protocol["screening_seeds"][0]:
                row["success"] = True
    assert helpers.select_conditions(protocol, screen_records) == list(selected_by_suite.values())
    assert len(helpers.planned_screen(protocol)) == 36
    assert all(arm == 1 and seed in protocol["screening_seeds"] for _, seed, arm in helpers.planned_screen(protocol))
    selection = {"selected_condition_ids": list(selected_by_suite.values())}
    assert len(helpers.planned_compare(protocol, selection)) == 48
    assert all(seed in protocol["heldout_seeds"] for _, seed, _ in helpers.planned_compare(protocol, selection))


def test_screen_rejects_missing_duplicate_or_heldout_seed(protocol, helpers, screen_records):
    assert len(helpers.validate_screen_records(protocol, screen_records)) == 36
    with pytest.raises(ValueError, match="Missing episodes"):
        helpers.validate_screen_records(protocol, screen_records[:-1])
    with pytest.raises(ValueError, match="Unexpected or duplicate"):
        helpers.validate_screen_records(protocol, [*screen_records, screen_records[0]])
    changed = [dict(row) for row in screen_records]
    changed[0]["seed"] = protocol["heldout_seeds"][0]
    with pytest.raises(ValueError, match="Unexpected or duplicate"):
        helpers.validate_screen_records(protocol, changed)


def test_comparison_requires_complete_pairing_and_identity(protocol, helpers):
    condition = protocol["conditions"][0]
    selection = {"selected_condition_ids": [condition["condition_id"]]}
    rows = [_episode(protocol, condition, seed, arm) for seed in protocol["heldout_seeds"] for arm in (1, 2, 4, 10)]
    assert len(helpers.validate_comparison_records(protocol, selection, rows)) == 16
    with pytest.raises(ValueError, match="Missing episodes"):
        helpers.validate_comparison_records(protocol, selection, rows[:-1])
    for field, message in (
        ("initial_state_digest", "Initial states differ"),
        ("noise_digest", "First noise_digest differs"),
        ("observation_digest", "First observation_digest differs"),
    ):
        changed = copy.deepcopy(rows)
        target = changed[1]
        if field == "initial_state_digest":
            target[field] = "b" * 64
        else:
            target["chunks"][0][field] = "b" * 64
        with pytest.raises(ValueError, match=message):
            helpers.validate_comparison_records(protocol, selection, changed)
    changed = copy.deepcopy(rows)
    changed[1]["chunks"][0]["log_sigma"][0] += 0.02
    changed[1]["chunks"][0]["sigmas"][0] = 0.2 * math.exp(0.02)
    with pytest.raises(ValueError, match="First score differs"):
        helpers.validate_comparison_records(protocol, selection, changed)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("checkpoint_identity", "b" * 64, "Episode identity mismatch"),
        ("head_checkpoint", "b" * 64, "Episode identity mismatch"),
        ("gpu_uuid", "GPU-wrong", "Episode identity mismatch"),
        ("total_velocity_evaluations", 2, "Total velocity count mismatch"),
    ],
)
def test_screen_rejects_identity_and_count_errors(protocol, helpers, screen_records, field, value, message):
    changed = copy.deepcopy(screen_records)
    changed[0][field] = value
    with pytest.raises(ValueError, match=message):
        helpers.validate_screen_records(protocol, changed)


def test_comparison_rejects_non_reference_time_schedule(protocol, helpers):
    condition = protocol["conditions"][0]
    selection = {"selected_condition_ids": [condition["condition_id"]]}
    rows = [_episode(protocol, condition, seed, arm) for seed in protocol["heldout_seeds"] for arm in (1, 2, 4, 10)]
    changed = copy.deepcopy(rows)
    changed[1]["chunks"][0]["times"] = [1.0, 0.75]  # two-step Euler must use [1, 0.5]
    with pytest.raises(ValueError, match=r"time|schedule|accounting"):
        helpers.validate_comparison_records(protocol, selection, changed)


def test_selection_cap_uses_declared_round_robin_and_severity_order(protocol, helpers, screen_records):
    first_seed = protocol["screening_seeds"][0]
    for row in screen_records:
        row["success"] = row["seed"] == first_seed
    selected = helpers.select_conditions(protocol, screen_records)
    assert selected == [
        "libero_goal-robot_initial_states-s3",
        "libero_object-robot_initial_states-s3",
        "libero_10-robot_initial_states-s3",
        "libero_goal-camera_viewpoints-s3",
    ]
    for row in screen_records:
        row["success"] = False
    assert helpers.select_conditions(protocol, screen_records) == []
    for row in screen_records:
        row["success"] = True
    assert helpers.select_conditions(protocol, screen_records) == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sigmas", [float("nan")], "Sigma values invalid"),
        ("sigmas", [0.0], "Sigma values invalid"),
        ("policy_ms", float("inf"), "Invalid latency"),
        ("request_ms", 0.0, "Invalid latency"),
        ("velocity_evaluations", 2, "Velocity accounting mismatch"),
    ],
)
def test_screen_rejects_invalid_score_latency_or_chunk_accounting(
    protocol, helpers, screen_records, field, value, message
):
    changed = copy.deepcopy(screen_records)
    changed[0]["chunks"][0][field] = value
    with pytest.raises(ValueError, match=message):
        helpers.validate_screen_records(protocol, changed)


def test_analyzer_rejects_tampered_selection_even_with_valid_screen_hash(protocol, screen_records, tmp_path):

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(protocol))
    manifest_hash = two_stage_analyze.sha256_file(manifest_path)
    for row in screen_records:
        row["manifest_sha256"] = manifest_hash
        row["success"] = row["seed"] == protocol["screening_seeds"][0] and row["condition_id"] == (
            "libero_goal-robot_initial_states-s3"
        )
    found = two_stage_analyze.validate(protocol, manifest_path, screen_records, "screen")
    assert two_stage_analyze.prespecified_selection(protocol, found) == ["libero_goal-robot_initial_states-s3"]
    forged = {
        "manifest_sha256": manifest_hash,
        "selected_condition_ids": ["libero_goal-robot_initial_states-s1"],
    }
    with pytest.raises(ValueError, match="Selection cannot be verified"):
        two_stage_analyze.validate(protocol, manifest_path, [], "compare", forged, screen_found=found)


def test_selection_hash_applies_to_comparison_rows_only(protocol, screen_records, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(protocol))
    manifest_hash = two_stage_analyze.sha256_file(manifest_path)
    selected = "libero_goal-robot_initial_states-s3"
    for row in screen_records:
        row["manifest_sha256"] = manifest_hash
        row["success"] = row["condition_id"] == selected and row["seed"] == protocol["screening_seeds"][0]
    selection = {"manifest_sha256": manifest_hash, "selected_condition_ids": [selected]}
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection))
    selection_hash = two_stage_analyze.sha256_file(selection_path)

    # Screening predates selection and has no selection SHA, even during comparison analysis.
    screen_found = two_stage_analyze.validate(
        protocol, manifest_path, screen_records, "screen", selection, selection_path
    )
    condition = next(item for item in protocol["conditions"] if item["condition_id"] == selected)
    compare_rows = [
        _episode(protocol, condition, seed, arm) for seed in protocol["heldout_seeds"] for arm in (1, 2, 4, 10)
    ]
    for row in compare_rows:
        row["manifest_sha256"] = manifest_hash
        row["selection_sha256"] = selection_hash
    assert (
        len(
            two_stage_analyze.validate(
                protocol, manifest_path, compare_rows, "compare", selection, selection_path, screen_found
            )
        )
        == 16
    )

    for invalid in (None, "b" * 64):
        changed = copy.deepcopy(compare_rows)
        if invalid is None:
            changed[0].pop("selection_sha256")
        else:
            changed[0]["selection_sha256"] = invalid
        with pytest.raises(ValueError, match="Record selection SHA mismatch"):
            two_stage_analyze.validate(
                protocol, manifest_path, changed, "compare", selection, selection_path, screen_found
            )
