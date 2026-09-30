"""Closed-loop, paired LIBERO-Plus pilot client for the isolated Python 3.8 simulator."""

import argparse
import collections
import hashlib
import json
import math
import pathlib
import random
import subprocess
import time
import urllib.request

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools

DUMMY_ACTION = [0.0] * 6 + [-1.0]


def sha256_file(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def post_json(url, payload):
    data = json.dumps(payload, separators=(",", ":")).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def quat2axisangle(quat):
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(denominator, 0.0):
        return np.zeros(3)
    return quat[:3] * 2.0 * math.acos(quat[3]) / denominator


def observation_payload(obs, prompt, resize_size):
    image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    image = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, resize_size, resize_size))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, resize_size, resize_size))
    state = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
    return {"image": image.tolist(), "wrist_image": wrist.tolist(), "state": state.tolist(), "prompt": str(prompt)}


def validate_manifest(manifest, classification):
    entries = classification[manifest["suite"]]
    assert manifest["task_order_index"] == 0
    assert manifest["flow_steps"] == [1, 2, 4, 10]
    assert len(manifest["tasks"]) == 6
    assert {task["category"] for task in manifest["tasks"]} == {
        "Robot Initial States",
        "Camera Viewpoints",
        "Light Conditions",
    }
    for task in manifest["tasks"]:
        assert entries[task["task_index"]]["id"] == task["id"]
        assert entries[task["task_index"]]["name"] == task["name"]
        assert entries[task["task_index"]]["category"] == task["category"]
        assert entries[task["task_index"]]["difficulty_level"] == task["difficulty_level"] == 1
        assert task["seeds"] == [7, 11]


def episode_key(task, seed, flow_steps):
    return (task["name"], int(seed), int(flow_steps))


def planned_episodes(manifest, *, smoke=False):
    if smoke:
        smoke_case = manifest["smoke"]
        return [(smoke_case["task_id"], smoke_case["seed"], n) for n in smoke_case["steps"]]
    return [
        (task["id"], seed, n) for task in manifest["tasks"] for seed in task["seeds"] for n in manifest["flow_steps"]
    ]


def run_episode(suite, task, seed, flow_steps, manifest, server):
    np.random.seed(seed)
    random.seed(seed)
    from_libero_path = pathlib.Path(get_libero_path("bddl_files")) / task["problem_folder"] / task["bddl_file"]
    env = OffScreenRenderEnv(
        bddl_file_name=str(from_libero_path),
        camera_heights=manifest["render_size"],
        camera_widths=manifest["render_size"],
    )
    env.seed(seed)
    begun = time.perf_counter()
    try:
        env.reset()
        initial_states = suite.get_task_init_states(task["index"])
        init_index = task["init_state_index"]
        if not 0 <= init_index < len(initial_states):
            raise RuntimeError("Declared initial-state index is out of range")
        obs = env.set_init_state(initial_states[init_index])
        init_digest = hashlib.sha256(np.asarray(initial_states[init_index]).tobytes()).hexdigest()
        queue = collections.deque()
        chunks = []
        success = False
        done = False
        chunk_index = 0
        for step in range(manifest["wait_steps"] + manifest["max_policy_steps"]):
            if step < manifest["wait_steps"]:
                obs, _, done, _ = env.step(DUMMY_ACTION)
                if done:
                    raise RuntimeError("Environment finished during dummy stabilization")
                continue
            if not queue:
                observation = observation_payload(obs, task["language"], manifest["image_size"])
                observation_digest = hashlib.sha256(
                    json.dumps(observation, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                payload = {
                    "flow_steps": flow_steps,
                    "suite": manifest["suite"],
                    "task_name": task["name"],
                    "episode_seed": seed,
                    "chunk_index": chunk_index,
                    "observation": observation,
                }
                called = time.perf_counter()
                response = post_json(server + "/infer", payload)
                request_ms = (time.perf_counter() - called) * 1000.0
                if response["velocity_evaluations"] != flow_steps:
                    raise RuntimeError("Unaccounted velocity evaluations")
                if len(response["actions"]) < manifest["replan_steps"]:
                    raise RuntimeError("Too few actions returned")
                if len(response["sigma"]) != flow_steps:
                    raise RuntimeError("Missing per-step sigma")
                queue.extend(response["actions"][: manifest["replan_steps"]])
                chunks.append(
                    {
                        "chunk_index": chunk_index,
                        "sigmas": response["sigma"],
                        "times": response["flow_times"],
                        "velocity_evaluations": response["velocity_evaluations"],
                        "request_ms": request_ms,
                        "policy_ms": response["policy_ms"],
                        "noise_digest": response["noise_sha256"],
                        "action_digest": response["action_sha256"],
                        "observation_digest": observation_digest,
                        "log_sigma": response["log_sigma"],
                    }
                )
                chunk_index += 1
            action = queue.popleft()
            obs, _, done, _ = env.step(action)
            if done:
                success = True
                break
        return {
            "task_name": task["name"],
            "task_id": task["id"],
            "category": task["category"],
            "seed": seed,
            "flow_steps": flow_steps,
            "success": success,
            "policy_steps": max(0, step + 1 - manifest["wait_steps"]),
            "episode_ms": (time.perf_counter() - begun) * 1000.0,
            "initial_state_digest": init_digest,
            "chunks": chunks,
            "total_velocity_evaluations": sum(chunk["velocity_evaluations"] for chunk in chunks),
            "status": "ok",
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(pathlib.Path(__file__).with_name("pilot_manifest.json")))
    parser.add_argument("--classification", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8765")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    manifest_path = pathlib.Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    if sha256_file(args.classification) != manifest["classification_sha256"]:
        raise RuntimeError("Classification identity changed")
    validate_manifest(manifest, json.loads(pathlib.Path(args.classification).read_text()))
    benchmark_root = pathlib.Path(args.classification).resolve().parents[3]
    imported = pathlib.Path(benchmark.__file__).resolve()
    if benchmark_root not in imported.parents:
        raise RuntimeError("Imported LIBERO package is not the declared benchmark checkout")
    benchmark_commit = subprocess.check_output(
        ["git", "-C", str(benchmark_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if benchmark_commit != manifest["benchmark_commit"]:
        raise RuntimeError("Benchmark Git commit changed")
    asset_root = benchmark_root.parent
    setup = json.loads((asset_root / "setup_status.json").read_text())
    if setup["assets_sha256"] != manifest["benchmark_assets_sha256"]:
        raise RuntimeError("Installed benchmark asset identity changed")
    if sha256_file(asset_root / "assets.zip") != manifest["benchmark_assets_sha256"]:
        raise RuntimeError("Benchmark asset archive changed")
    with urllib.request.urlopen(args.server + "/health", timeout=180) as response:
        health = json.load(response)
    if health["gpu_uuid"] != "GPU-4779cdde-a260-f8ec-da6a-6fa390bc7fd7":
        raise RuntimeError("Wrong GPU in policy server")
    if (
        health["base_checkpoint_identity"] != manifest["base_checkpoint_identity"]
        or health["head_sha256"] != manifest["head_sha256"]
    ):
        raise RuntimeError("Frozen model checkpoint identity changed")
    suite = benchmark.get_benchmark_dict()[manifest["suite"]](task_order_index=0)
    tasks = {}
    for item in manifest["tasks"]:
        task = suite.get_task(item["task_index"])
        if task.name != item["name"]:
            raise RuntimeError("Benchmark task order changed")
        tasks[item["id"]] = task._asdict()
        tasks[item["id"]]["id"] = item["id"]
        tasks[item["id"]]["category"] = item["category"]
        tasks[item["id"]]["index"] = item["task_index"]
        tasks[item["id"]]["init_state_index"] = item["init_state_index"]
    if args.smoke:
        item = manifest["smoke"]["task_id"]
        task = suite.get_task(item - 1)
        tasks[item] = task._asdict()
        tasks[item]["id"] = item
        tasks[item]["category"] = "Light Conditions"
        tasks[item]["index"] = item - 1
        tasks[item]["init_state_index"] = 0
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_hash = sha256_file(manifest_path)
    completed = set()
    if output.exists():
        for line in output.read_text().splitlines():
            record = json.loads(line)
            if record["manifest_sha256"] != manifest_hash:
                raise RuntimeError("Cannot resume under a different protocol")
            if record["status"] == "ok":
                completed.add((record["task_id"], record["seed"], record["flow_steps"]))
    for task_id, seed, flow_steps in planned_episodes(manifest, smoke=args.smoke):
        if (task_id, seed, flow_steps) in completed:
            continue
        task = tasks[task_id]
        try:
            record = run_episode(suite, task, seed, flow_steps, manifest, args.server)
        except Exception as error:
            record = {
                "task_id": task_id,
                "task_name": task["name"],
                "category": task["category"],
                "seed": seed,
                "flow_steps": flow_steps,
                "status": "error",
                "error": repr(error),
            }
        record["manifest_sha256"] = manifest_hash
        record["benchmark_commit"] = manifest["benchmark_commit"]
        record["checkpoint_identity"] = health["base_checkpoint_identity"]
        record["head_checkpoint"] = health["head_sha256"]
        record["gpu_uuid"] = health["gpu_uuid"]
        with output.open("a") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            handle.flush()
        print(
            json.dumps(
                {
                    key: record.get(key)
                    for key in (
                        "task_id",
                        "seed",
                        "flow_steps",
                        "status",
                        "success",
                        "policy_steps",
                        "episode_ms",
                        "error",
                    )
                }
            ),
            flush=True,
        )
        if record["status"] != "ok":
            raise RuntimeError("Episode failed; stop for inspection")


if __name__ == "__main__":
    main()
