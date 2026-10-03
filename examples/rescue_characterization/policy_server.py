# ruff: noqa: SLF001, FBT002, C408, N802
"""Frozen production sampler server. No residual head, training, or adaptive policy."""

import argparse
import dataclasses
import hashlib
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
import json
import os
from pathlib import Path
import subprocess
import time

from protocol import file_hash
from protocol import tree_hash

ALLOWED_STEPS = (1, 2, 4, 10)


def deterministic_noise(benchmark, suite, task_name, seed, init_index, chunk_index, shape=(10, 32)):
    import numpy as np

    key = [benchmark, suite, task_name, int(seed), int(init_index), int(chunk_index)]
    encoded = json.dumps(key, ensure_ascii=True, separators=(",", ":")).encode()
    noise = (
        np.random.default_rng(int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little"))
        .standard_normal(shape)
        .astype(np.float32)
    )
    return noise, hashlib.sha256(noise.tobytes()).hexdigest()


def checked_observation(payload):
    import numpy as np

    obs = payload["observation"]
    result = {}
    for name in ("image", "wrist_image"):
        x = np.asarray(obs[name])
        if x.shape != (224, 224, 3) or not np.issubdtype(x.dtype, np.integer) or np.any(x < 0) or np.any(x > 255):
            raise ValueError("Invalid image")
        result["observation/" + name] = x.astype(np.uint8)
    state = np.asarray(obs["state"], dtype=np.float32)
    if state.shape != (8,) or not np.all(np.isfinite(state)):
        raise ValueError("Invalid state")
    result["observation/state"] = state
    if not isinstance(obs["prompt"], str) or not obs["prompt"]:
        raise ValueError("Invalid prompt")
    result["prompt"] = obs["prompt"]
    return result


class FrozenPolicy:
    def __init__(self, manifest, gpu_uuid, manifest_sha256):
        # Caller verifies/masks UUID before any JAX import.
        import jax

        from openpi.policies import policy_config
        from openpi.training import config as config_lib

        self.manifest = manifest
        self.manifest_sha256 = manifest_sha256
        self.gpu_uuid = gpu_uuid
        if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
            raise RuntimeError("Expected exactly one masked GPU")
        checkpoint = Path(manifest["checkpoint"]["path"])
        digest, _ = tree_hash(checkpoint)
        if digest != manifest["checkpoint"]["sha256"]:
            raise RuntimeError("Checkpoint content changed")
        self.checkpoint_sha256 = digest
        for path, digest in manifest["source_sha256"].items():
            if file_hash(path) != digest:
                raise RuntimeError("Source identity mismatch: " + path)
        config = config_lib.get_config("pi05_libero")
        config = dataclasses.replace(
            config,
            data=dataclasses.replace(
                config.data, assets=config_lib.AssetsConfig(assets_dir=str(checkpoint / "assets"))
            ),
        )
        self.policy = policy_config.create_trained_policy(config, checkpoint)
        if self.policy._is_pytorch_model:
            raise RuntimeError("JAX checkpoint required")
        self.noise_shape = tuple(manifest["noise_policy"]["shape"])
        if (self.policy._model.action_horizon, self.policy._model.action_dim) != self.noise_shape:
            raise RuntimeError("Unexpected policy dimensions")
        self.warmed = set()
        self.verified = set()

    def health(self):
        return dict(
            status="ok",
            gpu_uuid=self.gpu_uuid,
            checkpoint_sha256=self.checkpoint_sha256,
            allowed_steps=list(ALLOWED_STEPS),
            manifest_sha256=self.manifest_sha256,
            warmed_steps={b: sorted(n for bb, n in self.warmed if bb == b) for b in self.manifest["benchmarks"]},
            verified_steps={b: sorted(n for bb, n in self.verified if bb == b) for b in self.manifest["benchmarks"]},
            residual_head=None,
            policy_latency_definition="synchronized public policy.infer including input/output transforms; excludes JSON/network",
        )

    def infer(self, request, allow_cold=False):
        import numpy as np

        steps = request["flow_steps"]
        if type(steps) is not int or steps not in ALLOWED_STEPS:
            raise ValueError("Invalid flow step arm")
        if not allow_cold and (request["benchmark"], steps) not in self.warmed:
            raise RuntimeError("Arm has not been warmed")
        if request["benchmark"] not in self.manifest["benchmarks"]:
            raise ValueError("Unknown benchmark")
        noise, digest = deterministic_noise(
            request["benchmark"],
            request["suite"],
            request["task_name"],
            request["episode_seed"],
            request["init_index"],
            request["chunk_index"],
            self.noise_shape,
        )
        obs = checked_observation(request)
        self.policy._sample_kwargs = {"num_steps": steps}
        start = time.perf_counter()
        result = self.policy.infer(obs, noise=noise)
        actions = np.asarray(result["actions"])
        elapsed = 1000 * (time.perf_counter() - start)
        if actions.shape != (self.noise_shape[0], 7) or not np.all(np.isfinite(actions)):
            raise RuntimeError("Invalid production actions")
        return dict(
            actions=actions.tolist(),
            noise_sha256=digest,
            action_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),
            velocity_evaluations=steps,
            policy_ms=elapsed,
        )

    def warmup(self, request):
        timings = {}
        for steps in ALLOWED_STEPS:
            req = dict(request, flow_steps=steps)
            self.infer(req, allow_cold=True)
            timings[str(steps)] = self.infer(req, allow_cold=True)["policy_ms"]
            self.warmed.add((request["benchmark"], steps))
        return dict(pass_=True, warmed_steps=sorted(self.warmed), second_call_ms=timings)

    def verify(self, request):
        # Compare production public sampler against existing cached trace on real weights.
        # Trace is verification-only and is never used during scored episodes.
        import jax
        import jax.numpy as jnp
        import numpy as np

        from openpi.models import model as model_lib
        from openpi.shared import nnx_utils

        steps = request["flow_steps"]
        result = self.infer(request)
        raw = checked_observation(request)
        inputs = self.policy._input_transform(dict(raw))
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
        observation = model_lib.Observation.from_dict(inputs)
        noise, _ = deterministic_noise(
            request["benchmark"],
            request["suite"],
            request["task_name"],
            request["episode_seed"],
            request["init_index"],
            request["chunk_index"],
            self.noise_shape,
        )
        if not hasattr(self, "trace"):
            self.trace = nnx_utils.module_jit(
                self.policy._model.cached_fixed_step_trace, static_argnames=("num_steps",)
            )
        final, *_ = self.trace(jax.random.key(0), observation, num_steps=steps, noise=jnp.asarray(noise)[None, ...])
        actions = self.policy._output_transform(
            {"state": np.asarray(inputs["state"])[0], "actions": np.asarray(final)[0]}
        )["actions"]
        difference = float(np.max(np.abs(np.asarray(result["actions"]) - np.asarray(actions))))
        passed = difference <= 1e-5
        if passed:
            self.verified.add((request["benchmark"], steps))
        return {
            "pass": passed,
            "max_abs_action_difference": difference,
            "tolerance": 1e-5,
            "flow_steps": steps,
            "velocity_evaluations_for_verification": 2 * steps,
            "checkpoint_sha256": self.checkpoint_sha256,
            "gpu_uuid": self.gpu_uuid,
        }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, status, payload):
        data = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.send(200, self.policy.health()) if self.path == "/health" else self.send(404, {"error": "not found"})

    def do_POST(self):
        try:
            size = int(self.headers["Content-Length"])
            if not 0 < size < 3000000:
                raise ValueError("Invalid request size")
            req = json.loads(self.rfile.read(size))
            method = {"/infer": self.policy.infer, "/warmup": self.policy.warmup, "/verify": self.policy.verify}.get(
                self.path
            )
            if method is None:
                self.send(404, {"error": "not found"})
                return
            self.send(200, method(req))
        except Exception as exc:
            self.send(500, {"error": repr(exc)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--gpu-uuid", required=True)
    p.add_argument("--port", type=int, required=True)
    a = p.parse_args()
    manifest = json.loads(Path(a.manifest).read_text())
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).splitlines()
    if a.gpu_uuid not in inventory or a.gpu_uuid not in manifest["gpu_uuids"]:
        raise RuntimeError("GPU UUID mismatch")
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu_uuid
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
    print(json.dumps(dict(verified_gpu_uuid=a.gpu_uuid, inventory=inventory)), flush=True)
    Handler.policy = FrozenPolicy(manifest, a.gpu_uuid, file_hash(a.manifest))
    print(json.dumps(Handler.policy.health()), flush=True)
    HTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
