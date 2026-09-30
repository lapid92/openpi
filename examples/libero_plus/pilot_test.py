# ruff: noqa: SLF001
"""Contract tests for the predeclared LIBERO-Plus pilot protocol."""

import json
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
from examples.libero_plus import policy_server

MANIFEST = Path(__file__).with_name("pilot_manifest.json")


@pytest.fixture(scope="module")
def protocol():
    return json.loads(MANIFEST.read_text())


def test_manifest_pins_benchmark_and_frozen_models(protocol):
    assert protocol["benchmark_commit"] == "4976dc30028e805ff8094b55501d532c48fec182"
    assert len(protocol["benchmark_assets_sha256"]) == 64
    assert len(protocol["classification_sha256"]) == 64
    assert "frozen" in protocol["fixed_model"].lower()
    assert protocol["flow_steps"] == [1, 2, 4, 10]
    assert protocol["suite"] == "libero_spatial"


def test_declared_instances_span_geometry_and_visual_conditions(protocol):
    tasks = protocol["tasks"]
    assert len(tasks) == 6
    assert [task["id"] for task in tasks] == [259, 291, 609, 618, 2124, 2284]
    assert {task["category"] for task in tasks} == {
        "Robot Initial States",
        "Camera Viewpoints",
        "Light Conditions",
    }
    for category in {task["category"] for task in tasks}:
        assert sum(task["category"] == category for task in tasks) == 2
    assert len({task["name"] for task in tasks}) == len(tasks)
    for task in tasks:
        assert task["task_index"] == task["id"] - 1
        assert task["difficulty_level"] == 1
        assert task["init_state_index"] == 0
        assert task["seeds"] == [7, 11]


def test_all_flow_arms_have_exactly_the_same_declared_instance_seed_pairs(protocol):
    pairs = {(task["id"], seed) for task in protocol["tasks"] for seed in task["seeds"]}
    assert len(pairs) == 12
    arm_pairs = {
        steps: {(task["id"], seed) for task in protocol["tasks"] for seed in task["seeds"]}
        for steps in protocol["flow_steps"]
    }
    assert all(value == pairs for value in arm_pairs.values())
    assert protocol["smoke"]["task_id"] not in {task["id"] for task in protocol["tasks"]}
    assert protocol["smoke"]["seed"] not in {seed for _, seed in pairs}


def test_protocol_fixes_execution_and_scoring_boundaries(protocol):
    assert protocol["render_size"] == 256
    assert protocol["image_size"] == 224
    assert protocol["replan_steps"] == 5
    assert protocol["wait_steps"] == 10
    assert protocol["max_policy_steps"] == 220
    assert "first Euler velocity evaluation" in protocol["score_summary"]
    assert "exact" in protocol["primary_pairing"]
    assert protocol["decision_gate"]


def test_deterministic_noise_is_arm_independent_and_chunk_specific(monkeypatch):
    one, digest_one = policy_server.deterministic_noise("libero_spatial", "task_a", 7, 0, (50, 32))
    again, digest_again = policy_server.deterministic_noise("libero_spatial", "task_a", 7, 0, (50, 32))
    next_chunk, next_digest = policy_server.deterministic_noise("libero_spatial", "task_a", 7, 1, (50, 32))
    assert one.dtype == "float32"
    assert one.shape == (50, 32)
    assert digest_one == digest_again
    assert (one == again).all()
    assert digest_one != next_digest
    assert not (one == next_chunk).all()
    with pytest.raises(ValueError, match="Invalid episode/noise key"):
        policy_server.deterministic_noise("libero_spatial", "task_a", 7, -1)


def test_wrong_checkpoint_or_untrained_head_rejected_before_model_load(monkeypatch, tmp_path):
    monkeypatch.setattr(policy_server, "checked_gpu_uuid", lambda: policy_server.EXPECTED_UUID)
    monkeypatch.setattr(policy_server, "checkpoint_identity", lambda _: "base-identity")

    def wrong_head(_path, identity):
        assert identity == "base-identity"
        raise ValueError("Wrong frozen checkpoint for predictor")

    monkeypatch.setattr(policy_server, "load_head", wrong_head)
    with pytest.raises(ValueError, match="Wrong frozen checkpoint"):
        policy_server.FrozenScoredPolicy(tmp_path, tmp_path / "head.npz")

    monkeypatch.setattr(policy_server, "load_head", lambda _path, identity: ({}, {"steps": 2999}))
    with pytest.raises(ValueError, match="3,000-step"):
        policy_server.FrozenScoredPolicy(tmp_path, tmp_path / "head.npz")


@pytest.mark.parametrize("steps", [1, 2, 4, 10])
def test_sigma_logging_reuses_fixed_trace_and_does_not_change_actions(monkeypatch, steps):
    monkeypatch.setattr(policy_server.model_lib.Observation, "from_dict", lambda _: object())
    calls = {"trace": 0, "score": 0}
    policy = policy_server.FrozenScoredPolicy.__new__(policy_server.FrozenScoredPolicy)
    policy.action_horizon = 50
    policy.action_dim = 32
    fake_policy = type("FakePolicy", (), {})()
    fake_policy._input_transform = lambda raw: {"state": raw["observation/state"]}
    fake_policy._output_transform = lambda outputs: {"actions": outputs["actions"][..., :7]}
    policy.policy = fake_policy

    def trace(_key, _obs, *, num_steps, noise):
        calls["trace"] += 1
        assert num_steps == steps
        features = jnp.zeros((steps, 1, 50, 1024), dtype=jnp.float32)
        times = jnp.linspace(1.0, 1.0 / steps, steps)
        return noise + 2, None, None, features, times

    def score(features, times):
        calls["score"] += 1
        assert features.shape == (steps, 1, 50, 1024)
        assert times.shape == (steps,)
        return jnp.full((steps, 1), -1.0)

    policy.trace = trace
    policy.score = score
    request = {
        "flow_steps": steps,
        "suite": "libero_spatial",
        "task_name": "task_a",
        "episode_seed": 7,
        "chunk_index": 0,
        "observation": {
            "image": np.zeros((224, 224, 3), dtype=np.uint8).tolist(),
            "wrist_image": np.zeros((224, 224, 3), dtype=np.uint8).tolist(),
            "state": [0.0] * 8,
            "prompt": "pick object",
        },
    }
    result = policy.infer(request)
    noise, digest = policy_server.deterministic_noise("libero_spatial", "task_a", 7, 0)
    np.testing.assert_allclose(result["actions"], (noise + 2)[:, :7], atol=0, rtol=0)
    assert result["noise_sha256"] == digest
    assert result["velocity_evaluations"] == steps
    assert len(result["sigma"]) == len(result["flow_times"]) == steps
    np.testing.assert_allclose(result["sigma"], np.exp(-1), atol=1e-7)
    assert calls == {"trace": 1, "score": 1}
