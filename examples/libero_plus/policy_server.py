# ruff: noqa: SLF001
"""Frozen π₀.₅ LIBERO-Plus fixed-step policy with same-pass residual scores.

Run this process with CUDA_VISIBLE_DEVICES=3. The LIBERO simulator runs in a
separate CPU-only Python environment and sends one action-chunk request at a time.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
import json
import os
from pathlib import Path
import subprocess
import time

if os.environ.get("CUDA_VISIBLE_DEVICES") != "3":
    raise RuntimeError("Set CUDA_VISIBLE_DEVICES=3; no other GPU is permitted")

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.models.velocity_residual import checkpoint_identity
from openpi.models.velocity_residual import load_head
from openpi.models.velocity_residual import predict_log_sigma
from openpi.policies import policy_config
from openpi.shared import nnx_utils
from openpi.training import config as config_lib

EXPECTED_UUID = "GPU-4779cdde-a260-f8ec-da6a-6fa390bc7fd7"
ALLOWED_STEPS = (1, 2, 4, 10)


def checked_gpu_uuid() -> str:
    uuid = subprocess.check_output(
        ["nvidia-smi", "-i", "3", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip()
    if uuid != EXPECTED_UUID:
        raise RuntimeError(f"GPU 3 identity mismatch: {uuid}")
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu" or devices[0].id != 0:
        raise RuntimeError(f"Expected only masked cuda:0, got {devices}")
    return uuid


def deterministic_noise(suite: str, task_name: str, episode_seed: int, chunk_index: int, shape=(50, 32)):
    """Stable, arm-independent Gaussian chunk noise and its reproducible digest."""
    if not suite or not task_name or episode_seed < 0 or chunk_index < 0:
        raise ValueError("Invalid episode/noise key")
    payload = json.dumps(
        [suite, task_name, int(episode_seed), int(chunk_index)], ensure_ascii=True, separators=(",", ":")
    ).encode()
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    noise = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
    return noise, hashlib.sha256(noise.tobytes()).hexdigest()


def _checked_observation(payload: dict) -> dict:
    observation = payload["observation"]
    result = {}
    for name in ("image", "wrist_image"):
        image = np.asarray(observation[name])
        if image.shape != (224, 224, 3) or not np.issubdtype(image.dtype, np.integer):
            raise ValueError(f"{name} must be a [224,224,3] uint8 image")
        if np.any(image < 0) or np.any(image > 255):
            raise ValueError(f"{name} pixel values must be in [0,255]")
        result[f"observation/{name}"] = image.astype(np.uint8, copy=False)
    state = np.asarray(observation["state"], dtype=np.float32)
    if state.shape != (8,) or not np.all(np.isfinite(state)):
        raise ValueError("state must be eight finite float values")
    result["observation/state"] = state
    prompt = observation["prompt"]
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be nonempty text")
    result["prompt"] = prompt
    return result


class FrozenScoredPolicy:
    def __init__(self, checkpoint: Path, head_path: Path):
        self.gpu_uuid = checked_gpu_uuid()
        self.checkpoint = checkpoint.resolve()
        self.head_path = head_path.resolve()
        self.base_identity = checkpoint_identity(self.checkpoint)
        self.head, self.head_metadata = load_head(self.head_path, self.base_identity)
        if int(self.head_metadata.get("steps", -1)) != 3000:
            raise ValueError("This pilot requires the trained 3,000-step residual head")
        self.head_sha256 = hashlib.sha256(self.head_path.read_bytes()).hexdigest()
        config = config_lib.get_config("pi05_libero")
        config = dataclasses.replace(
            config,
            data=dataclasses.replace(
                config.data, assets=config_lib.AssetsConfig(assets_dir=str(self.checkpoint / "assets"))
            ),
        )
        self.policy = policy_config.create_trained_policy(config, self.checkpoint)
        if self.policy._is_pytorch_model:
            raise RuntimeError("Expected the frozen JAX π₀.₅ checkpoint")
        self.trace = nnx_utils.module_jit(self.policy._model.cached_fixed_step_trace, static_argnames=("num_steps",))

        def score_features(features, times):
            return jax.vmap(lambda f, t: predict_log_sigma(self.head, f, jnp.broadcast_to(t, (f.shape[0],))))(
                features, times
            )

        self.score = jax.jit(score_features)
        self.action_horizon = self.policy._model.action_horizon
        self.action_dim = self.policy._model.action_dim

    def health(self) -> dict:
        return {
            "status": "ok",
            "gpu_uuid": self.gpu_uuid,
            "visible_device": "cuda:0",
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "checkpoint": str(self.checkpoint),
            "base_checkpoint_identity": self.base_identity,
            "head": str(self.head_path),
            "head_sha256": self.head_sha256,
            "head_step": int(self.head_metadata["steps"]),
            "allowed_steps": list(ALLOWED_STEPS),
        }

    def infer(self, request: dict) -> dict:
        steps = request["flow_steps"]
        if type(steps) is not int or steps not in ALLOWED_STEPS:
            raise ValueError("flow_steps must be one of 1, 2, 4, 10")
        suite = request["suite"]
        task_name = request["task_name"]
        episode_seed = request["episode_seed"]
        chunk_index = request["chunk_index"]
        if type(episode_seed) is not int or type(chunk_index) is not int:
            raise ValueError("episode_seed and chunk_index must be integers")
        raw = _checked_observation(request)
        start = time.perf_counter()
        inputs = self.policy._input_transform(dict(raw))
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
        observation = model_lib.Observation.from_dict(inputs)
        noise, noise_sha256 = deterministic_noise(
            suite, task_name, episode_seed, chunk_index, (self.action_horizon, self.action_dim)
        )
        noise_batch = jnp.asarray(noise)[None, ...]
        final, _, _, features, times = self.trace(jax.random.key(0), observation, num_steps=steps, noise=noise_batch)
        log_sigma = self.score(features, times)
        # Converting all results to NumPy synchronizes the device before latency is recorded.
        final_np, log_sigma_np, times_np = map(np.asarray, (final, log_sigma, times))
        outputs = self.policy._output_transform({"state": np.asarray(inputs["state"])[0], "actions": final_np[0]})
        actions = np.asarray(outputs["actions"])
        if actions.shape != (self.action_horizon, 7) or not np.all(np.isfinite(actions)):
            raise RuntimeError(f"Unexpected transformed action shape or values: {actions.shape}")
        elapsed_ms = 1000 * (time.perf_counter() - start)
        log_sigma_values = log_sigma_np[:, 0].astype(float).tolist()
        return {
            "actions": actions.tolist(),
            "flow_times": times_np.astype(float).tolist(),
            "log_sigma": log_sigma_values,
            "sigma": np.exp(np.asarray(log_sigma_values)).tolist(),
            "first_step_sigma": float(np.exp(log_sigma_values[0])),
            "velocity_evaluations": steps,
            "noise_sha256": noise_sha256,
            "policy_ms": elapsed_ms,
            "action_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
        }

    def verify_against_public_policy(self, request: dict) -> dict:
        """Smoke-only extra evaluation. Never called by the pilot inference endpoint."""
        result = self.infer(request)
        noise, _ = deterministic_noise(
            request["suite"],
            request["task_name"],
            request["episode_seed"],
            request["chunk_index"],
            (self.action_horizon, self.action_dim),
        )
        original_kwargs = self.policy._sample_kwargs
        self.policy._sample_kwargs = {"num_steps": request["flow_steps"]}
        try:
            reference = self.policy.infer(_checked_observation(request), noise=noise)["actions"]
        finally:
            self.policy._sample_kwargs = original_kwargs
        max_abs = float(np.max(np.abs(np.asarray(result["actions"]) - np.asarray(reference))))
        return {
            "max_abs_action_difference": max_abs,
            "tolerance": 1e-5,
            "pass": max_abs <= 1e-5,
            "velocity_evaluations_for_verification": request["flow_steps"] * 2,
        }


class Handler(BaseHTTPRequestHandler):
    policy: FrozenScoredPolicy

    def _send(self, status: int, payload: dict):
        body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(HTTPStatus.OK, self.policy.health())
        else:
            self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/infer", "/verify"):
            self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            length = int(self.headers["Content-Length"])
            if length < 1 or length > 2_000_000:
                raise ValueError("request body size out of range")
            request = json.loads(self.rfile.read(length))
            response = (
                self.policy.infer(request)
                if self.path == "/infer"
                else self.policy.verify_against_public_policy(request)
            )
        except (KeyError, TypeError, ValueError) as exc:
            self._send(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except Exception as exc:
            self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(exc).__name__}: {exc}"})
            return
        self._send(HTTPStatus.OK, response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("/root/.cache/openpi/openpi-assets/checkpoints/pi05_libero")
    )
    parser.add_argument(
        "--head", type=Path, default=Path("/volt/data/openpi_velocity_runs/long_3000_20260930/head.npz")
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.host not in ("127.0.0.1", "localhost"):
        parser.error("Bind only to localhost; this endpoint carries raw observations")
    Handler.policy = FrozenScoredPolicy(args.checkpoint, args.head)
    server = HTTPServer((args.host, args.port), Handler)
    print(json.dumps({"listening": f"http://{args.host}:{args.port}", **Handler.policy.health()}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
