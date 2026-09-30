"""CPU-only contract tests for the offline residual/refinement diagnostic."""
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np
import pytest

SCRIPT = Path(__file__).with_name("study_velocity_residual_refinement.py")
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("study_velocity_residual_refinement", SCRIPT)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_diagnostic_split_is_deterministic_disjoint_complete_and_stratified():
    ids = list(range(16))
    tasks = {i: f"task_{i // 10}" if i % 10 == 0 else f"dummy_{i}" for i in range(40)}
    episodes = {episode: {"tasks": [f"task_{episode // 4}"], "length": 8} for episode in ids}
    metadata = SimpleNamespace(tasks=tasks, episodes=episodes)
    diagnostic, calibration, split_hash, suites = study.diagnostic_split(ids, metadata)
    assert study.diagnostic_split(ids, metadata) == (diagnostic, calibration, split_hash, suites)
    assert len(split_hash) == 64
    assert set(diagnostic).isdisjoint(calibration)
    assert set(diagnostic) | set(calibration) == set(ids)
    assert [sum(suites[e] == s for e in diagnostic) for s in range(4)] == [2] * 4
    assert [sum(suites[e] == s for e in calibration) for s in range(4)] == [2] * 4


def test_grouped_calibration_compares_variances_and_includes_t_one():
    logs = {}
    variance = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    times = np.array([0.0, 0.19, 0.20, 0.40, 0.80, 1.0])
    suites = np.array([0, 1, 2, 3, 0, 1])
    study.grouped_calibration(logs, "cal", variance, 2 * variance, times, suites)
    assert logs["cal/overall/count"] == 6
    assert logs["cal/overall/mean_sigma_squared"] == pytest.approx(3.5)
    assert logs["cal/overall/mean_residual_squared"] == pytest.approx(7.0)
    assert logs["cal/time_4/count"] == 2
    assert logs["cal/time_4/variance_ratio"] == pytest.approx(0.5)
    assert sum(logs[f"cal/score_{i}/count"] for i in range(5) if f"cal/score_{i}/count" in logs) == 6


def test_masked_mse_ignores_padded_steps():
    left = np.array([[[1, 1], [100, 100]], [[2, 2], [2, 2]]], dtype=np.float32)
    valid = np.array([[True, False], [True, True]])
    np.testing.assert_allclose(study.masked_mse(left, np.zeros_like(left), valid), [1.0, 4.0])


def test_masked_mse_rejects_all_padding():
    left = np.ones((1, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="valid"):
        study.masked_mse(left, np.zeros_like(left), np.zeros((1, 2), dtype=bool))


def test_ranking_gate_positive_signal():
    score = np.arange(16, dtype=np.float32)
    result = study.ranking_metrics(score, score.copy(), np.arange(16), np.arange(16) % 4, seed=1, bootstrap=200)
    assert result["gate_pass"] is True
    assert result["spearman"] == pytest.approx(1.0)
    assert result["auc_positive_benefit"] == pytest.approx(1.0)
    assert result["spearman_ci95"][0] > 0
    assert result["high_minus_low_ci95"][0] > 0


def test_ranking_constant_score_fails_gate_without_bootstrap_crash():
    score = np.ones(16, dtype=np.float32)
    result = study.ranking_metrics(score, np.arange(16), np.arange(16), np.arange(16) % 4, seed=1, bootstrap=20)
    assert result["gate_pass"] is False
    assert result["spearman_ci95"] is None
    assert result["high_minus_low_ci95"] is None


def test_auc_one_class_returns_none():
    score = np.array([0.1, 0.2, 0.3])
    assert study.auc(score, np.ones(3, dtype=bool)) is None
    assert study.auc(score, np.zeros(3, dtype=bool)) is None
