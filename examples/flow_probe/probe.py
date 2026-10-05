# ruff: noqa: C408
"""Prespecified, training-free midpoint velocity-change statistic.

Model coordinates are normalized, before unnormalization and absolute-action
transforms. The first 10 horizon entries and first 7 dimensions are executed by
the LIBERO adapter. All score arithmetic is float32; nonfinite values fail closed.
"""

import hashlib

import numpy as np

SIGNAL_ID = "midpoint-relative-velocity-rms-v1"
EPSILON = np.float32(1e-6)


def array_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def velocity_change_score(first, midpoint):
    first = np.asarray(first, dtype=np.float32)
    midpoint = np.asarray(midpoint, dtype=np.float32)
    if first.shape != (10, 32) or midpoint.shape != (10, 32):
        raise ValueError("Expected unbatched 10x32 velocity tensors")
    if not np.isfinite(first).all() or not np.isfinite(midpoint).all():
        raise ValueError("Nonfinite velocity")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        v = first[:, :7]
        delta = midpoint[:, :7] - v
        denominator_rms = np.sqrt(np.mean(v * v, dtype=np.float32))
        numerator_rms = np.sqrt(np.mean(delta * delta, dtype=np.float32))
        denominator = np.maximum(denominator_rms, EPSILON)
        score = np.float32(numerator_rms / denominator)
    if not np.isfinite(score):
        raise ValueError("Nonfinite score")
    return dict(
        signal_id=SIGNAL_ID,
        score=float(score),
        numerator_rms=float(numerator_rms),
        denominator_rms=float(denominator_rms),
        safeguarded_denominator=float(denominator),
        denominator_floor=float(EPSILON),
        dtype="float32",
        times=[1.0, 0.5],
        horizon_slice=[0, 10],
        dimension_slice=[0, 7],
        coordinate_system="normalized_model_actions",
        ranking_direction="descending",
    )


def validate_trace(states, velocities, times):
    states = np.asarray(states, dtype=np.float32)
    velocities = np.asarray(velocities, dtype=np.float32)
    times = np.asarray(times, dtype=np.float32)
    if states.shape != (3, 1, 10, 32) or velocities.shape != (2, 1, 10, 32):
        raise ValueError("Unexpected two-evaluation trace dimensions")
    if not np.array_equal(times, np.array([1.0, 0.5], dtype=np.float32)):
        raise ValueError("Unexpected probe time points")
    if not np.isfinite(states).all() or not np.isfinite(velocities).all():
        raise ValueError("Nonfinite trace")
    # XLA may fuse multiply-add; float32 rounding tolerance does not change score.
    expected = states[0] - np.float32(0.5) * velocities[0]
    difference = float(np.max(np.abs(states[1] - expected)))
    if not np.allclose(states[1], expected, rtol=1e-6, atol=1e-6):
        raise ValueError("Probe midpoint does not follow declared Euler state")
    return difference
