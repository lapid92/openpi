# ruff: noqa: FBT002, PERF401, B905, C408, B007
"""Paired simulator runner; all execution and durable JSONL stay on the pod."""

import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
import urllib.request

from protocol import file_hash
from protocol import tree_hash

from recording import Recorder, rng_hash
from recording_audit import audit_record, audit_records

DUMMY_ACTION = [0.0] * 6 + [-1.0]


def post_json(url, payload):
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as response:
        return json.load(response)


def digest_array(x):
    import numpy as np

    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def observation_payload(obs, prompt, size):
    import numpy as np
    from openpi_client import image_tools

    q = np.asarray(obs["robot0_eef_quat"], dtype=np.float64).copy()
    q[3] = np.clip(q[3], -1.0, 1.0)
    denom = np.sqrt(1 - q[3] * q[3])
    aa = np.zeros(3) if math.isclose(denom, 0.0) else q[:3] * 2 * math.acos(q[3]) / denom
    result = {
        "state": np.concatenate((obs["robot0_eef_pos"], aa, obs["robot0_gripper_qpos"])).tolist(),
        "prompt": str(prompt),
    }
    for dest, src in (("image", "agentview_image"), ("wrist_image", "robot0_eye_in_hand_image")):
        im = np.ascontiguousarray(obs[src][::-1, ::-1])
        result[dest] = image_tools.convert_to_uint8(image_tools.resize_with_pad(im, size, size)).tolist()
    return result


def planned_episodes(manifest, benchmark, smoke=False, worker_index=0, workers=1):
    spec = manifest["benchmarks"][benchmark]
    conditions = {c["condition_id"]: c for c in spec["conditions"]}
    cases = (
        spec["smoke_cases"]
        if smoke
        else [dict(case, condition_id=c["condition_id"]) for c in spec["conditions"] for case in c["cases"]]
    )
    result = []
    condition_indices = {c["condition_id"]: i for i, c in enumerate(spec["conditions"])}
    for ordinal, case in enumerate(cases):
        if condition_indices[case["condition_id"]] % workers != worker_index:
            continue
        arms = manifest["flow_steps"]
        order = arms[ordinal % 4 :] + arms[: ordinal % 4]
        for arm in order:
            result.append((conditions[case["condition_id"]], case, arm))
    return result


def record_key(record):
    return (record["condition_id"], record["seed"], record["init_index"], record["flow_steps"])


def append_record(path, record):
    with path.open("a") as f:
        f.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def validate_existing(path, manifest_hash, plan, phase, health):
    completed = {}
    expected = {(c["condition_id"], s["seed"], s["init_index"], arm) for c, s, arm in plan}
    if not path.exists():
        return completed
    for line in path.read_text().splitlines():
        row = json.loads(line)
        key = record_key(row)
        if row["manifest_sha256"] != manifest_hash or row["phase"] != phase or key not in expected:
            raise RuntimeError("Resume protocol/plan mismatch")
        if row["checkpoint_sha256"] != health["checkpoint_sha256"] or row["gpu_uuid"] != health["gpu_uuid"]:
            raise RuntimeError("Resume checkpoint/GPU mismatch")
        if row["status"] not in ("ok", "error"):
            raise RuntimeError("Unknown resumed status")
        if row["status"] == "ok":
            if key in completed or not row.get("chunks") or "stabilized_state_sha256" not in row:
                raise RuntimeError("Duplicate/incomplete resumed record")
            if row["total_velocity_evaluations"] != sum(c["velocity_evaluations"] for c in row["chunks"]):
                raise RuntimeError("Invalid resumed accounting")
            if (
                type(row.get("success")) is not bool
                or not math.isfinite(row.get("episode_ms", -1))
                or row["episode_ms"] <= 0
            ):
                raise RuntimeError("Invalid resumed success/timing")
            for i, chunk in enumerate(row["chunks"]):
                if chunk["chunk_index"] != i or chunk["velocity_evaluations"] != row["flow_steps"]:
                    raise RuntimeError("Invalid resumed chunk accounting")
                if any(not math.isfinite(chunk.get(k, -1)) or chunk[k] <= 0 for k in ("policy_ms", "request_ms")):
                    raise RuntimeError("Invalid resumed chunk timing")
                for k in ("noise_sha256", "action_sha256", "observation_sha256"):
                    if len(chunk.get(k, "")) != 64 or any(x not in "0123456789abcdef" for x in chunk[k]):
                        raise RuntimeError("Invalid resumed chunk hash")
            assert_pair(row, completed)
            completed[key] = row
    return completed


def assert_pair(row, completed):
    for other in completed.values():
        if record_key(row)[:3] != record_key(other)[:3]:
            continue
        for field in ("initial_state_sha256", "stabilized_state_sha256", "checkpoint_sha256", "gpu_uuid"):
            if row[field] != other[field]:
                raise RuntimeError("Pair mismatch: " + field)
        if row["chunks"][0]["observation_sha256"] != other["chunks"][0]["observation_sha256"]:
            raise RuntimeError("Initial observation differs between arms")
        for x, y in zip(row["chunks"], other["chunks"]):
            if x.get("simulator_rng_sha256") != y.get("simulator_rng_sha256"):
                raise RuntimeError("Arm-dependent simulator RNG stream")
            if x["noise_sha256"] != y["noise_sha256"]:
                raise RuntimeError("Arm-dependent noise stream")


def run_episode(suite, condition, case, arm, manifest, benchmark, server, smoke=False, prepare_only=False):
    from libero.libero.envs import OffScreenRenderEnv
    import numpy as np

    settings = manifest["settings"]
    seed = case["seed"]
    random.seed(seed)
    np.random.seed(seed)
    begun = time.perf_counter()
    env = OffScreenRenderEnv(
        bddl_file_name=condition["bddl_path"],
        camera_heights=settings["render_size"],
        camera_widths=settings["render_size"],
    )
    chunks = []
    recorder = None
    try:
        env.seed(seed)
        env.reset()
        states = suite.get_task_init_states(condition["task_index"])
        state = states[case["init_index"]]
        initial_hash = digest_array(state)
        if initial_hash != case["initial_state_sha256"]:
            raise RuntimeError("Declared initial state content changed")
        obs = env.set_init_state(state)
        for _ in range(settings["wait_steps"]):
            obs, _, done, _ = env.step(DUMMY_ACTION)
            if done or env.check_success():
                raise RuntimeError("Success/termination during stabilization")
        stable = digest_array(env.get_sim_state())
        if prepare_only:
            observation = observation_payload(obs, condition["prompt"], settings["image_size"])
            payload = dict(
                benchmark=benchmark,
                suite=condition["suite"],
                task_name=condition["task_name"],
                episode_seed=seed,
                init_index=case["init_index"],
                chunk_index=0,
                flow_steps=1,
                observation=observation,
            )
            warmup = post_json(server + "/warmup", payload)
            parity = [post_json(server + "/verify", dict(payload, flow_steps=n)) for n in manifest["flow_steps"]]
            if not all(v["pass"] for v in parity):
                raise RuntimeError("Real checkpoint sampler parity failed")
            return dict(
                status="verified",
                warmup=warmup,
                verification=parity,
                initial_state_sha256=initial_hash,
                stabilized_state_sha256=stable,
                observation_sha256=hashlib.sha256(
                    json.dumps(observation, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            )
        recorder = Recorder(manifest["recording"]["root"], benchmark, condition, case, arm)
        recorder.observe(obs)
        queue = collections.deque()
        success = False
        max_steps = settings["max_policy_steps"][condition["suite"]]
        for step in range(max_steps):
            if not queue:
                observation = observation_payload(obs, condition["prompt"], settings["image_size"])
                payload = dict(
                    benchmark=benchmark,
                    suite=condition["suite"],
                    task_name=condition["task_name"],
                    episode_seed=seed,
                    init_index=case["init_index"],
                    chunk_index=len(chunks),
                    flow_steps=arm,
                    observation=observation,
                )
                verification = []
                called = time.perf_counter()
                response = post_json(server + "/infer", payload)
                request_ms = 1000 * (time.perf_counter() - called)
                if response["velocity_evaluations"] != arm or len(response["actions"]) < settings["replan_steps"]:
                    raise RuntimeError("Sampler accounting mismatch")
                actions = np.asarray(response["actions"])
                if actions.shape != (manifest["noise_policy"]["shape"][0], 7) or not np.all(np.isfinite(actions)):
                    raise RuntimeError("Invalid actions")
                queue.extend(response["actions"][: settings["replan_steps"]])
                chunks.append(
                    dict(
                        chunk_index=len(chunks),
                        simulator_rng_sha256=rng_hash(),
                        actions=response["actions"],
                        noise_sha256=response["noise_sha256"],
                        action_sha256=response["action_sha256"],
                        observation_sha256=hashlib.sha256(
                            json.dumps(observation, sort_keys=True, separators=(",", ":")).encode()
                        ).hexdigest(),
                        velocity_evaluations=response["velocity_evaluations"],
                        policy_ms=response["policy_ms"],
                        request_ms=request_ms,
                        verification=verification,
                    )
                )
            executed_action = queue.popleft()
            recorder.action(obs, executed_action)
            obs, _, done, _ = env.step(executed_action)
            recorder.observe(obs)
            success = bool(env.check_success())
            recorder.success.append(success)
            if done and not success:
                raise RuntimeError("Simulator terminated without task completion")
            if success:
                break
        return dict(
            recording=recorder.finish(),
            success=success,
            policy_steps=step + 1,
            episode_ms=1000 * (time.perf_counter() - begun),
            initial_state_sha256=initial_hash,
            stabilized_state_sha256=stable,
            chunks=chunks,
            total_velocity_evaluations=sum(c["velocity_evaluations"] for c in chunks),
            status="ok",
        )
    finally:
        if recorder is not None:
            recorder.close()
        env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--benchmark", choices=["libero", "libero_plus"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--server", required=True)
    p.add_argument("--worker-index", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    a = p.parse_args()
    if not 0 <= a.worker_index < a.workers:
        raise ValueError("Invalid worker shard")
    manifest = json.loads(Path(a.manifest).read_text())
    spec = manifest["benchmarks"][a.benchmark]
    from libero.libero import benchmark

    root = Path(spec["root"]).resolve()
    if root not in Path(benchmark.__file__).resolve().parents:
        raise RuntimeError("Wrong simulator import root")
    if subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() != spec["commit"]:
        raise RuntimeError("Benchmark commit changed")
    for path, digest in manifest["source_sha256"].items():
        if file_hash(path) != digest:
            raise RuntimeError("Preprocessing/source content changed: " + path)
    if spec.get("classification_path") and file_hash(spec["classification_path"]) != spec["classification_sha256"]:
        raise RuntimeError("Classification changed")
    if tree_hash(spec["installed_assets_path"])[0] != spec["installed_assets_sha256"]:
        raise RuntimeError("Installed simulator assets changed")
    for c in spec["conditions"]:
        if (
            file_hash(c["bddl_asset_path"]) != c["bddl_sha256"]
            or file_hash(c["init_file_path"]) != c["init_file_sha256"]
        ):
            raise RuntimeError("Task asset changed")
    with urllib.request.urlopen(a.server + "/health", timeout=900) as response:
        health = json.load(response)
    if (
        health["gpu_uuid"] not in manifest["gpu_uuids"]
        or health["checkpoint_sha256"] != manifest["checkpoint"]["sha256"]
        or health["manifest_sha256"] != file_hash(a.manifest)
    ):
        raise RuntimeError("Server identity mismatch")
    if not a.prepare_only and (
        health["warmed_steps"].get(a.benchmark) != manifest["flow_steps"]
        or health["verified_steps"].get(a.benchmark) != manifest["flow_steps"]
    ):
        raise RuntimeError("Full run requires warmup and real-checkpoint parity at all arms")
    plan = planned_episodes(manifest, a.benchmark, a.smoke, a.worker_index, a.workers)
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if a.prepare_only:
        if not plan:
            raise ValueError("Preparation shard is empty; omit --smoke when preparing all workers")
        condition, case, arm = plan[0]
        # Separate seed ensures validation never consumes a scored case.
        case = dict(condition["cases"][0], seed=900001)
        suite = benchmark.get_benchmark_dict()[condition["suite"]](task_order_index=0)
        record = run_episode(suite, condition, case, arm, manifest, a.benchmark, a.server, prepare_only=True)
        record.update(
            benchmark=a.benchmark,
            condition_id=condition["condition_id"],
            gpu_uuid=health["gpu_uuid"],
            checkpoint_sha256=health["checkpoint_sha256"],
            manifest_sha256=file_hash(a.manifest),
            phase="preparation",
        )
        append_record(output, record)
        print(json.dumps(record), flush=True)
        return
    phase = "smoke" if a.smoke else "main"
    manifest_hash = file_hash(a.manifest)
    completed = validate_existing(output, manifest_hash, plan, phase, health)
    if completed:
        audit_records(list(completed.values()), manifest, inspect_video=False, allow_partial=True)
    suites = {}
    for condition, case, arm in plan:
        row = dict(
            benchmark=a.benchmark,
            condition_id=condition["condition_id"],
            suite=condition["suite"],
            task_name=condition["task_name"],
            family=condition["family"],
            category=condition["category"],
            severity=condition["severity"],
            prompt=condition["prompt"],
            seed=case["seed"],
            init_index=case["init_index"],
            flow_steps=arm,
            phase=phase,
            manifest_sha256=manifest_hash,
            checkpoint_sha256=health["checkpoint_sha256"],
            gpu_uuid=health["gpu_uuid"],
            benchmark_commit=spec["commit"],
        )
        if record_key(row) in completed:
            continue
        try:
            if condition["suite"] not in suites:
                suites[condition["suite"]] = benchmark.get_benchmark_dict()[condition["suite"]](task_order_index=0)
            suite = suites[condition["suite"]]
            if suite.get_task(condition["task_index"]).name != condition["task_name"]:
                raise RuntimeError("Task order changed")
            row.update(run_episode(suite, condition, case, arm, manifest, a.benchmark, a.server, a.smoke))
            assert_pair(row, completed)
            audit_record(row, manifest, inspect_video=False)
        except Exception as exc:
            row.update(status="error", error=repr(exc))
        append_record(output, row)
        print(json.dumps({k: v for k, v in row.items() if k != "chunks"}), flush=True)
        if row["status"] != "ok":
            raise RuntimeError("Evaluation halted; preserved error record")
        completed[record_key(row)] = row


if __name__ == "__main__":
    main()
