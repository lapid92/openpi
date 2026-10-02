# ruff: noqa: B905, FBT002, C408
"""Independent paired fixed arms with frozen residual logging and metadata replay smoke."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frozen_flow"))
import client as fixed
from protocol import file_hash
from protocol import tree_hash


def planned(manifest, benchmark, smoke, worker, workers):
    spec = manifest["benchmarks"][benchmark]
    selected_conditions = spec["smoke_conditions"] if smoke else spec["conditions"]
    conditions = {c["condition_id"]: c for c in selected_conditions}
    indices = {c["condition_id"]: i for i, c in enumerate(selected_conditions)}
    cases = [dict(s, condition_id=c["condition_id"]) for c in selected_conditions for s in c["cases"]]
    result = []
    for ordinal, case in enumerate(cases):
        if indices[case["condition_id"]] % workers != worker:
            continue
        arms = ["fixed_1", "fixed_10"] if ordinal % 2 == 0 else ["fixed_10", "fixed_1"]
        if smoke:
            arms.append("metadata")
        result.extend((conditions[case["condition_id"]], case, arm) for arm in arms)
    return result


def selected_steps(manifest, condition, arm):
    return manifest["metadata_rule"][condition["category"]] if arm == "metadata" else int(arm.split("_")[1])


def key(row):
    return row["condition_id"], row["seed"], row["init_index"], row["policy_arm"]


def pair_check(row, completed, tolerance):
    fixed.assert_pair(row, completed)
    for other in completed.values():
        if key(other)[:3] != key(row)[:3]:
            continue
        if abs(row["initial_log_sigma"] - other["initial_log_sigma"]) > tolerance:
            raise RuntimeError("Paired initial log sigma differs")
        if "metadata" in (row["policy_arm"], other["policy_arm"]) and row["flow_steps"] == other["flow_steps"]:
            for field in ("success", "policy_steps", "total_velocity_evaluations"):
                if row[field] != other[field]:
                    raise RuntimeError("Metadata replay differs: " + field)
            if len(row["chunks"]) != len(other["chunks"]):
                raise RuntimeError("Metadata replay chunk count differs")
            for x, y in zip(row["chunks"], other["chunks"]):
                for field in (
                    "action_sha256",
                    "noise_sha256",
                    "observation_sha256",
                    "velocity_evaluations",
                    "first_log_sigma",
                ):
                    if x[field] != y[field]:
                        raise RuntimeError("Metadata replay differs: " + field)


def run_episode(suite, condition, case, arm, manifest, benchmark, server, prepare_only=False):
    responses = []
    original = fixed.post_json
    steps = selected_steps(manifest, condition, arm)

    def transport(url, payload):
        target_arm = "fixed_" + str(payload["flow_steps"]) if prepare_only else arm
        request = dict(payload, condition_id=condition["condition_id"], policy_arm=target_arm)
        result = original(url, request)
        if url.endswith("/infer"):
            if result["head_sha256"] != manifest["head"]["sha256"] or result["reference_velocity_evaluations"] != 0:
                raise RuntimeError("Head identity or forbidden extra velocity calls")
            responses.append(result)
        return result

    fixed.post_json = transport
    try:
        result = fixed.run_episode(
            suite, condition, case, steps, manifest, benchmark, server, prepare_only=prepare_only
        )
    finally:
        fixed.post_json = original
    if prepare_only:
        verification = result["verification"]
        if len({v["noise_sha256"] for v in verification}) != 1:
            raise RuntimeError("Parity observations use different initial noise")
        if (
            max(v["first_log_sigma"] for v in verification) - min(v["first_log_sigma"] for v in verification)
            > manifest["analysis"]["initial_log_sigma_pair_tolerance"]
        ):
            raise RuntimeError("Initial same-observation log sigma differs between arms")
        return result
    if len(responses) != len(result["chunks"]):
        raise RuntimeError("Missing chunk scores")
    for chunk, response in zip(result["chunks"], responses):
        for field in (
            "first_sigma",
            "first_log_sigma",
            "sigma_time",
            "head_ms",
            "head_evaluations",
            "reference_velocity_evaluations",
        ):
            chunk[field] = response[field]
    result.update(
        initial_sigma=result["chunks"][0]["first_sigma"], initial_log_sigma=result["chunks"][0]["first_log_sigma"]
    )
    return result


def validate_row(row, manifest, health, phase, expected):
    if key(row) not in expected or row["phase"] != phase or row["manifest_sha256"] != health["manifest_sha256"]:
        raise RuntimeError("Resume protocol mismatch")
    for field in ("head_sha256", "checkpoint_sha256", "gpu_uuid"):
        if row[field] != health[field]:
            raise RuntimeError("Resume identity mismatch")
    condition = next(
        c
        for c in (
            manifest["benchmarks"][row["benchmark"]]["conditions"]
            + manifest["benchmarks"][row["benchmark"]].get("smoke_conditions", [])
        )
        if c["condition_id"] == row["condition_id"]
    )
    if row["flow_steps"] != selected_steps(manifest, condition, row["policy_arm"]):
        raise RuntimeError("Record arm/step mapping mismatch")
    for field in ("suite", "task_name", "family", "category", "severity"):
        if row[field] != condition[field]:
            raise RuntimeError("Record condition metadata mismatch")
    if row["status"] not in ("ok", "error"):
        raise RuntimeError("Invalid record status")
    if row["status"] == "error":
        return
    if type(row["success"]) is not bool or not row["chunks"]:
        raise RuntimeError("Incomplete episode")
    if row["total_velocity_evaluations"] != sum(c["velocity_evaluations"] for c in row["chunks"]):
        raise RuntimeError("Incomplete velocity accounting")
    if len(row["chunks"]) != math.ceil(row["policy_steps"] / manifest["settings"]["replan_steps"]):
        raise RuntimeError("Missing chunks")
    if not math.isfinite(row["episode_ms"]) or row["episode_ms"] <= 0:
        raise RuntimeError("Invalid episode timing")
    for index, chunk in enumerate(row["chunks"]):
        if chunk["chunk_index"] != index:
            raise RuntimeError("Noncontiguous chunk index")
        for field in ("action_sha256", "noise_sha256", "observation_sha256"):
            if len(chunk[field]) != 64 or any(c not in "0123456789abcdef" for c in chunk[field]):
                raise RuntimeError("Invalid chunk hash")
        if any(chunk[field] < 0 for field in ("head_ms", "policy_ms", "request_ms")):
            raise RuntimeError("Negative latency")
        if (
            chunk["velocity_evaluations"] != row["flow_steps"]
            or chunk["head_evaluations"] != 1
            or chunk["sigma_time"] != 1.0
            or chunk["reference_velocity_evaluations"] != 0
        ):
            raise RuntimeError("Invalid score/velocity accounting")
        if not all(
            math.isfinite(chunk[f]) for f in ("first_log_sigma", "first_sigma", "head_ms", "policy_ms", "request_ms")
        ):
            raise RuntimeError("Nonfinite record")
        if not math.isclose(math.exp(chunk["first_log_sigma"]), chunk["first_sigma"], rel_tol=1e-6):
            raise RuntimeError("Inconsistent sigma")
    if (
        row["initial_log_sigma"] != row["chunks"][0]["first_log_sigma"]
        or row["initial_sigma"] != row["chunks"][0]["first_sigma"]
    ):
        raise RuntimeError("Initial score not first chunk")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--benchmark", default="libero_plus", choices=["libero_plus"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.worker_index < args.workers:
        raise ValueError("Invalid shard")
    manifest = json.loads(Path(args.manifest).read_text())
    mh = file_hash(args.manifest)
    spec = manifest["benchmarks"][args.benchmark]
    from libero.libero import benchmark

    root = Path(spec["root"]).resolve()
    if (
        root not in Path(benchmark.__file__).resolve().parents
        or subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() != spec["commit"]
    ):
        raise RuntimeError("Benchmark import/commit mismatch")
    for path, digest in manifest["source_sha256"].items():
        if file_hash(path) != digest:
            raise RuntimeError("Source identity changed: " + path)
    if (
        file_hash(spec["classification_path"]) != spec["classification_sha256"]
        or tree_hash(spec["installed_assets_path"])[0] != spec["installed_assets_sha256"]
    ):
        raise RuntimeError("Installed benchmark identity changed")
    for condition in spec["conditions"] + spec.get("smoke_conditions", []):
        if (
            file_hash(condition["bddl_asset_path"]) != condition["bddl_sha256"]
            or file_hash(condition["init_file_path"]) != condition["init_file_sha256"]
        ):
            raise RuntimeError("Condition asset changed")
    with urllib.request.urlopen(args.server + "/health", timeout=900) as response:
        health = json.load(response)
    if (
        health["manifest_sha256"] != mh
        or health["head_sha256"] != manifest["head"]["sha256"]
        or health["checkpoint_sha256"] != manifest["checkpoint"]["sha256"]
        or health["gpu_uuid"] not in manifest["gpu_uuids"]
    ):
        raise RuntimeError("Policy server identity mismatch")
    if not args.prepare_only and (
        health["warmed_steps"].get(args.benchmark) != [1, 10] or health["verified_steps"].get(args.benchmark) != [1, 10]
    ):
        raise RuntimeError("Warmup and production parity required")
    plan = planned(manifest, args.benchmark, args.smoke, args.worker_index, args.workers)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.prepare_only:
        if not plan:
            raise ValueError("Empty preparation shard")
        condition, _, arm = plan[0]
        case = dict(condition["cases"][0], seed=manifest["preparation_seed"])
        suite = benchmark.get_benchmark_dict()[condition["suite"]](task_order_index=0)
        row = run_episode(suite, condition, case, arm, manifest, args.benchmark, args.server, prepare_only=True)
        row.update(
            benchmark=args.benchmark,
            condition_id=condition["condition_id"],
            seed=case["seed"],
            init_index=case["init_index"],
            phase="preparation",
            manifest_sha256=mh,
            checkpoint_sha256=health["checkpoint_sha256"],
            head_sha256=health["head_sha256"],
            gpu_uuid=health["gpu_uuid"],
        )
        fixed.append_record(output, row)
        print(json.dumps(row), flush=True)
        return
    phase = "smoke" if args.smoke else "main"
    expected = {(c["condition_id"], s["seed"], s["init_index"], a) for c, s, a in plan}
    tolerance = manifest["analysis"]["initial_log_sigma_pair_tolerance"]
    completed = {}
    if output.exists():
        for text in output.read_text().splitlines():
            row = json.loads(text)
            validate_row(row, manifest, health, phase, expected)
            if row["status"] == "ok":
                if key(row) in completed:
                    raise RuntimeError("Duplicate completed record")
                pair_check(row, completed, tolerance)
                completed[key(row)] = row
    suites = {}
    for condition, case, arm in plan:
        row = dict(
            benchmark=args.benchmark,
            condition_id=condition["condition_id"],
            suite=condition["suite"],
            task_name=condition["task_name"],
            family=condition["family"],
            category=condition["category"],
            severity=condition["severity"],
            seed=case["seed"],
            init_index=case["init_index"],
            policy_arm=arm,
            flow_steps=selected_steps(manifest, condition, arm),
            phase=phase,
            manifest_sha256=mh,
            head_sha256=health["head_sha256"],
            checkpoint_sha256=health["checkpoint_sha256"],
            gpu_uuid=health["gpu_uuid"],
            benchmark_commit=spec["commit"],
        )
        if key(row) in completed:
            continue
        try:
            if condition["suite"] not in suites:
                suites[condition["suite"]] = benchmark.get_benchmark_dict()[condition["suite"]](task_order_index=0)
            suite = suites[condition["suite"]]
            if suite.get_task(condition["task_index"]).name != condition["task_name"]:
                raise RuntimeError("Task order changed")
            row.update(run_episode(suite, condition, case, arm, manifest, args.benchmark, args.server))
            validate_row(row, manifest, health, phase, expected)
            pair_check(row, completed, tolerance)
        except Exception as error:
            row.update(status="error", error=repr(error))
        fixed.append_record(output, row)
        print(json.dumps({k: v for k, v in row.items() if k != "chunks"}), flush=True)
        if row["status"] != "ok":
            raise RuntimeError("Fail closed; original error retained")
        completed[key(row)] = row


if __name__ == "__main__":
    main()
