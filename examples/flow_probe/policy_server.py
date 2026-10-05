# ruff: noqa: SLF001, FBT002, C408
"""Side-channel initial-observation probe; fixed actions use the unchanged server."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from probe import array_sha256
from probe import validate_trace
from probe import velocity_change_score

# Load the audited implementation under a unique name, never shadow this module.
_RESCUE = Path(__file__).resolve().parents[1] / "rescue_characterization"
sys.path.insert(0, str(_RESCUE))
_spec = importlib.util.spec_from_file_location("rescue_fixed_server", _RESCUE / "policy_server.py")
fixed = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixed)


class ProbePolicy(fixed.FrozenPolicy):
    def __init__(self, manifest, gpu_uuid, manifest_sha256):
        super().__init__(manifest, gpu_uuid, manifest_sha256)
        import jax

        from openpi.models.velocity_residual import checkpoint_identity
        from openpi.models.velocity_residual import load_head
        from openpi.models.velocity_residual import predict_log_sigma
        from openpi.shared import nnx_utils

        spec = manifest["head"]
        if fixed.file_hash(spec["path"]) != spec["sha256"]:
            raise RuntimeError("Frozen head hash mismatch")
        self.head, self.head_metadata = load_head(spec["path"], checkpoint_identity(manifest["checkpoint"]["path"]))
        if self.head_metadata != spec["metadata"]:
            raise RuntimeError("Frozen head metadata mismatch")
        self.head_sha256 = spec["sha256"]
        self.probe_trace = nnx_utils.module_jit(
            self.policy._model.cached_fixed_step_trace, static_argnames=("num_steps",)
        )
        self.head_score = jax.jit(lambda features, times: predict_log_sigma(self.head, features, times))
        self.probe_dir = Path(manifest["probe_artifact_dir"])
        self.probe_dir.mkdir(parents=True, exist_ok=True)

    def health(self):
        value = super().health()
        value.update(
            head_sha256=self.head_sha256,
            residual_head=self.head_metadata,
            probe="midpoint-relative-velocity-rms-v1",
            probe_chunk=0,
            actual_probe_velocity_evaluations=2,
            theoretical_shared_extra_velocity_evaluations=1,
            probe_changes_actions=False,
            policy_latency_definition="unchanged production infer only, excludes separate probe",
        )
        return value

    def _probe(self, request, persist=False):
        import jax
        import jax.numpy as jnp
        import numpy as np

        from openpi.models import model as model_lib

        noise, noise_hash = fixed.deterministic_noise(
            request["benchmark"],
            request["suite"],
            request["task_name"],
            request["episode_seed"],
            request["init_index"],
            request["chunk_index"],
            self.noise_shape,
        )
        if request["chunk_index"] != 0:
            raise ValueError("Only initial observation probes are permitted")
        rng_before = np.asarray(jax.random.key_data(self.policy._rng)).copy()
        raw = fixed.checked_observation(request)
        input_hashes = {
            key: array_sha256(value) if isinstance(value, np.ndarray) else hashlib.sha256(value.encode()).hexdigest()
            for key, value in raw.items()
        }
        start = time.perf_counter()
        inputs = self.policy._input_transform(dict(raw))
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
        observation = model_lib.Observation.from_dict(inputs)
        _, states, velocities, features, times = self.probe_trace(
            jax.random.key(0), observation, num_steps=2, noise=jnp.asarray(noise)[None, ...]
        )
        # Host materialization synchronizes every trace output and cost timer.
        arrays = {
            "states": np.asarray(states, dtype=np.float32),
            "velocities": np.asarray(velocities, dtype=np.float32),
            "first_action_features": np.asarray(features[0], dtype=np.float32),
            "times": np.asarray(times, dtype=np.float32),
            "noise": noise,
        }
        trace_ms = 1000 * (time.perf_counter() - start)
        midpoint_difference = validate_trace(arrays["states"], arrays["velocities"], arrays["times"])
        if not np.array_equal(arrays["states"][0, 0], noise):
            raise RuntimeError("Probe initial noise changed")
        statistics = velocity_change_score(arrays["velocities"][0, 0], arrays["velocities"][1, 0])
        head_start = time.perf_counter()
        log_sigma = float(np.asarray(self.head_score(features[0], jnp.ones((1,), dtype=jnp.float32)))[0])
        head_ms = 1000 * (time.perf_counter() - head_start)
        sigma = float(np.exp(np.float32(log_sigma)))
        if not np.isfinite(log_sigma) or not np.isfinite(sigma):
            raise RuntimeError("Nonfinite frozen sigma")
        rng_unchanged = np.array_equal(rng_before, np.asarray(jax.random.key_data(self.policy._rng)))
        if not rng_unchanged:
            raise RuntimeError("Probe changed production RNG")
        result = dict(
            **statistics,
            first_log_sigma=log_sigma,
            first_sigma=sigma,
            sigma_time=1.0,
            noise_sha256=noise_hash,
            input_sha256=input_hashes,
            array_sha256={key: array_sha256(value) for key, value in arrays.items()},
            head_sha256=self.head_sha256,
            checkpoint_sha256=self.checkpoint_sha256,
            gpu_uuid=self.gpu_uuid,
            manifest_sha256=self.manifest_sha256,
            velocity_evaluations=2,
            prefix_evaluations=1,
            head_evaluations=1,
            duplicated_first_velocity_evaluations=1,
            theoretical_shared_extra_velocity_evaluations=1,
            trace_ms=trace_ms,
            head_ms=head_ms,
            rng_unchanged=rng_unchanged,
            midpoint_max_abs_difference=midpoint_difference,
            chunk_index=0,
            role="scored" if persist else "verification_or_warmup",
        )
        if persist:
            key = [
                request.get("phase", "unspecified"),
                request.get("condition_id"),
                request["benchmark"],
                request["suite"],
                request["task_name"],
                request["episode_seed"],
                request["init_index"],
                request["flow_steps"],
            ]
            directory = self.probe_dir / hashlib.sha256(json.dumps(key).encode()).hexdigest()
            directory.mkdir(exist_ok=True)
            path = directory / (uuid.uuid4().hex + ".npz")
            with path.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            result.update(raw_path=str(path), raw_sha256=fixed.file_hash(path))
            path.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        return result

    def infer(self, request, allow_cold=False, record_probe=True):
        # No probe state is fed to the public sampler, and no sampler RNG is rewound.
        result = super().infer(request, allow_cold=allow_cold)
        if request["chunk_index"] == 0:
            result["initial_probe"] = self._probe(request, persist=record_probe)
        result["probe_velocity_evaluations"] = 2 if request["chunk_index"] == 0 else 0
        result["total_velocity_evaluations"] = result["velocity_evaluations"] + result["probe_velocity_evaluations"]
        return result

    def warmup(self, request):
        timings = {}
        for steps in fixed.ALLOWED_STEPS:
            req = dict(request, flow_steps=steps, chunk_index=0)
            self.infer(req, allow_cold=True, record_probe=False)
            timings[str(steps)] = self.infer(req, allow_cold=True, record_probe=False)["policy_ms"]
            self.warmed.add((request["benchmark"], steps))
        return {
            "pass": True,
            "second_call_ms": timings,
            "velocity_evaluations": 2 * sum(fixed.ALLOWED_STEPS) + 16,
            "probe_velocity_evaluations": 16,
            "head_evaluations": 8,
            "prefix_evaluations": 16,
        }

    def verify(self, request):
        import jax
        import numpy as np

        steps = request["flow_steps"]
        before = super().infer(request)
        rng_before = self.policy._rng
        expected_rng, _ = jax.random.split(rng_before)
        probed = self.infer(request, record_probe=False)
        rng_ok = np.array_equal(
            np.asarray(jax.random.key_data(expected_rng)), np.asarray(jax.random.key_data(self.policy._rng))
        )
        after = super().infer(request)
        action_equal = before["action_sha256"] == probed["action_sha256"] == after["action_sha256"]
        probe_repeat_equal = True
        probe = probed.get("initial_probe")
        if request["chunk_index"] == 0:
            repeat = self._probe(request)
            probe_repeat_equal = (
                probe["array_sha256"] == repeat["array_sha256"]
                and probe["score"] == repeat["score"]
                and probe["first_log_sigma"] == repeat["first_log_sigma"]
            )
        # Separate existing cached/public tolerance gate, on the same raw input/noise.
        import jax.numpy as jnp

        from openpi.models import model as model_lib
        from openpi.shared import nnx_utils

        raw = fixed.checked_observation(request)
        inputs = self.policy._input_transform(dict(raw))
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
        noise, _ = fixed.deterministic_noise(
            request["benchmark"],
            request["suite"],
            request["task_name"],
            request["episode_seed"],
            request["init_index"],
            request["chunk_index"],
            self.noise_shape,
        )
        if not hasattr(self, "parity_trace"):
            self.parity_trace = nnx_utils.module_jit(
                self.policy._model.cached_fixed_step_trace, static_argnames=("num_steps",)
            )
        final, *_ = self.parity_trace(
            jax.random.key(0),
            model_lib.Observation.from_dict(inputs),
            num_steps=steps,
            noise=jnp.asarray(noise)[None, ...],
        )
        traced = self.policy._output_transform(
            {"state": np.asarray(inputs["state"])[0], "actions": np.asarray(final)[0]}
        )["actions"]
        trace_difference = float(np.max(np.abs(np.asarray(before["actions"]) - np.asarray(traced))))
        passed = bool(action_equal and rng_ok and probe_repeat_equal and trace_difference <= 1e-5)
        if passed:
            self.verified.add((request["benchmark"], steps))
        return {
            "pass": passed,
            "fixed_action_hash_bitwise_equal": action_equal,
            "cached_public_max_abs_difference": trace_difference,
            "cached_public_tolerance": 1e-5,
            "action_sha256": before["action_sha256"],
            "rng_progression_unchanged": bool(rng_ok),
            "probe_repeat_bitwise_equal": probe_repeat_equal,
            "initial_probe": probe,
            "flow_steps": steps,
            "chunk_index": request["chunk_index"],
            "velocity_evaluations_for_verification": 4 * steps + (4 if probe else 0),
            "probe_velocity_evaluations": 4 if probe else 0,
            "head_evaluations": 2 if probe else 0,
            "prefix_evaluations": 6 if probe else 4,
            "checkpoint_sha256": self.checkpoint_sha256,
            "head_sha256": self.head_sha256,
            "gpu_uuid": self.gpu_uuid,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).splitlines()
    if args.gpu_uuid not in inventory or args.gpu_uuid not in manifest["gpu_uuids"]:
        raise RuntimeError("GPU UUID mismatch")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
    fixed.Handler.policy = ProbePolicy(manifest, args.gpu_uuid, fixed.file_hash(args.manifest))
    print(json.dumps(fixed.Handler.policy.health()), flush=True)
    fixed.HTTPServer(("127.0.0.1", args.port), fixed.Handler).serve_forever()


if __name__ == "__main__":
    main()
