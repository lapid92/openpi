# ruff: noqa: SLF001, FBT002, C408
"""Frozen residual logging on existing fixed-step trace; metadata allocation is outcome independent."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frozen_flow"))
import policy_server as fixed
from protocol import file_hash

ARMS = (1, 10)


class SelectivePolicy(fixed.FrozenPolicy):
    def __init__(self, manifest, gpu_uuid, manifest_sha256):
        super().__init__(manifest, gpu_uuid, manifest_sha256)
        import jax

        from openpi.models.velocity_residual import checkpoint_identity
        from openpi.models.velocity_residual import load_head
        from openpi.models.velocity_residual import predict_log_sigma
        from openpi.shared import nnx_utils

        spec = manifest["head"]
        if file_hash(spec["path"]) != spec["sha256"]:
            raise RuntimeError("Frozen residual head hash mismatch")
        self.head, self.head_metadata = load_head(spec["path"], checkpoint_identity(manifest["checkpoint"]["path"]))
        if self.head_metadata != spec["metadata"]:
            raise RuntimeError("Frozen head metadata mismatch")
        self.head_sha256 = spec["sha256"]
        self.trace = nnx_utils.module_jit(self.policy._model.cached_fixed_step_trace, static_argnames=("num_steps",))
        self.score = jax.jit(lambda features, times: predict_log_sigma(self.head, features, times))
        self.conditions = {
            c["condition_id"]: c
            for b in manifest["benchmarks"].values()
            for c in (b["conditions"] + b.get("smoke_conditions", []))
        }

    def health(self):
        value = super().health()
        value.update(
            allowed_steps=list(ARMS),
            head_sha256=self.head_sha256,
            residual_head=self.head_metadata,
            policy_latency_definition="synchronized fixed trace, one initial-feature head evaluation and transforms; no reference policy calls",
            head_latency_definition="synchronized initial-feature head evaluation only, after trace synchronization",
        )
        return value

    def selected_steps(self, request):
        condition = self.conditions[request["condition_id"]]
        if condition["task_name"] != request["task_name"] or condition["suite"] != request["suite"]:
            raise ValueError("Condition identity mismatch")
        arm = request.get("policy_arm", "fixed_" + str(request["flow_steps"]))
        if arm == "metadata":
            steps = self.manifest["metadata_rule"][condition["category"]]
        elif arm in ("fixed_1", "fixed_10"):
            steps = int(arm.split("_")[1])
        else:
            raise ValueError("Unknown policy arm")
        if request["flow_steps"] != steps:
            raise ValueError("Declared step count differs from arm")
        return steps, arm

    def infer(self, request, allow_cold=False):
        import jax
        import jax.numpy as jnp
        import numpy as np

        from openpi.models import model as model_lib

        steps, arm = self.selected_steps(request)
        if steps not in ARMS or request["benchmark"] not in self.manifest["benchmarks"]:
            raise ValueError("Unsupported benchmark or fixed arm")
        if not allow_cold and (request["benchmark"], steps) not in self.warmed:
            raise RuntimeError("Arm not warmed")
        noise, noise_hash = fixed.deterministic_noise(
            request["benchmark"],
            request["suite"],
            request["task_name"],
            request["episode_seed"],
            request["init_index"],
            request["chunk_index"],
            self.noise_shape,
        )
        raw = fixed.checked_observation(request)
        start = time.perf_counter()
        inputs = self.policy._input_transform(dict(raw))
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
        observation = model_lib.Observation.from_dict(inputs)
        final, _, _, features, times = self.trace(
            jax.random.key(0), observation, num_steps=steps, noise=jnp.asarray(noise)[None, ...]
        )
        # Synchronize trace before separately measuring the frozen tiny head.
        final_np = np.asarray(final)
        first_features = features[0]
        jax.block_until_ready(first_features)
        head_start = time.perf_counter()
        log_sigma = float(
            np.asarray(self.score(first_features, jnp.ones((first_features.shape[0],), dtype=jnp.float32)))[0]
        )
        head_ms = 1000 * (time.perf_counter() - head_start)
        actions = np.asarray(
            self.policy._output_transform({"state": np.asarray(inputs["state"])[0], "actions": final_np[0]})["actions"]
        )
        policy_ms = 1000 * (time.perf_counter() - start)
        if actions.shape != (self.noise_shape[0], 7) or not np.all(np.isfinite(actions)) or not np.isfinite(log_sigma):
            raise RuntimeError("Nonfinite/invalid inference output")
        if float(np.asarray(times)[0]) != 1.0 or features.shape[0] != steps:
            raise RuntimeError("Unexpected trace time/count")
        return dict(
            actions=actions.tolist(),
            noise_sha256=noise_hash,
            action_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),
            velocity_evaluations=steps,
            policy_ms=policy_ms,
            head_ms=head_ms,
            head_evaluations=1,
            first_log_sigma=log_sigma,
            first_sigma=float(np.exp(log_sigma)),
            sigma_time=1.0,
            policy_arm=arm,
            selected_steps=steps,
            head_sha256=self.head_sha256,
            reference_velocity_evaluations=0,
        )

    def warmup(self, request):
        timings = {}
        for steps in ARMS:
            req = dict(request, flow_steps=steps, policy_arm="fixed_" + str(steps))
            self.infer(req, allow_cold=True)
            timings[str(steps)] = self.infer(req, allow_cold=True)["policy_ms"]
            self.warmed.add((request["benchmark"], steps))
        return {"pass": True, "second_call_ms": timings, "velocity_evaluations": 2 * sum(ARMS)}

    def verify(self, request):
        import numpy as np

        steps, arm = self.selected_steps(request)
        result = self.infer(request)
        noise, _ = fixed.deterministic_noise(
            request["benchmark"],
            request["suite"],
            request["task_name"],
            request["episode_seed"],
            request["init_index"],
            request["chunk_index"],
            self.noise_shape,
        )
        self.policy._sample_kwargs = {"num_steps": steps}
        reference = np.asarray(self.policy.infer(fixed.checked_observation(request), noise=noise)["actions"])
        difference = float(np.max(np.abs(np.asarray(result["actions"]) - reference)))
        passed = difference <= 1e-5
        if passed:
            self.verified.add((request["benchmark"], steps))
        return {
            "pass": passed,
            "max_abs_action_difference": difference,
            "tolerance": 1e-5,
            "flow_steps": steps,
            "policy_arm": arm,
            "first_sigma": result["first_sigma"],
            "first_log_sigma": result["first_log_sigma"],
            "noise_sha256": result["noise_sha256"],
            "velocity_evaluations_for_verification": 2 * steps,
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
    print(json.dumps({"verified_gpu_uuid": args.gpu_uuid, "inventory": inventory}), flush=True)
    fixed.Handler.policy = SelectivePolicy(manifest, args.gpu_uuid, file_hash(args.manifest))
    print(json.dumps(fixed.Handler.policy.health()), flush=True)
    fixed.HTTPServer(("127.0.0.1", args.port), fixed.Handler).serve_forever()


if __name__ == "__main__":
    main()
