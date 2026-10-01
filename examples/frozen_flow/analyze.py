"""Strict audits and paired fixed-flow summaries. No benchmark pooling."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

ARMS = (1, 2, 4, 10)
BOOTSTRAP_SEED = 20261001
BOOTSTRAP_REPS = 10000


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def noise_digest(benchmark, suite, task_name, seed, init_index, chunk_index, shape=(10, 32)):
    key = [benchmark, suite, task_name, seed, init_index, chunk_index]
    payload = json.dumps(key, ensure_ascii=True, separators=(",", ":")).encode()
    seed_value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    noise = np.random.default_rng(seed_value).standard_normal(shape).astype(np.float32)
    return hashlib.sha256(noise.tobytes()).hexdigest()


def planned_cases(manifest, benchmark, *, smoke=False):
    config = manifest["benchmarks"][benchmark]
    conditions = config["conditions"]
    if len({c["condition_id"] for c in conditions}) != len(conditions):
        raise ValueError("Duplicate condition declaration")
    by_id = {c["condition_id"]: c for c in conditions}
    if smoke:
        cases = [(by_id[c["condition_id"]], c) for c in config["smoke_cases"]]
    else:
        cases = [(condition, case) for condition in conditions for case in condition["cases"]]
    keys = [(c["condition_id"], p["seed"], p["init_index"]) for c, p in cases]
    if not cases or len(set(keys)) != len(keys):
        raise ValueError("Empty or duplicate planned cases")
    return cases


def _sha(value, field):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError("Invalid SHA256: " + field)


def _number(value, field, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < minimum:
        raise ValueError("Invalid finite number: " + field)


def validate(manifest, rows, benchmark, manifest_sha256, *, smoke=False):
    if tuple(manifest["flow_steps"]) != ARMS:
        raise ValueError("Invalid flow arms")
    _sha(manifest_sha256, "manifest")
    shape = manifest["noise_policy"]["shape"]
    if not isinstance(shape, list) or len(shape) != 2 or any(type(x) is not int or x <= 0 for x in shape):
        raise ValueError("Invalid noise shape")
    if shape != [manifest["settings"]["action_horizon"], manifest["settings"]["action_dim"]]:
        raise ValueError("Noise and model shape mismatch")
    cases = planned_cases(manifest, benchmark, smoke=smoke)
    expected = {}
    for condition, case in cases:
        for arm in ARMS:
            expected[(condition["condition_id"], case["seed"], case["init_index"], arm)] = (condition, case)
    found, errors = {}, []
    for row in rows:
        if any(type(row.get(field)) is not int for field in ("seed", "init_index", "flow_steps")):
            raise ValueError("Episode key fields must be integers")
        key = (row.get("condition_id"), row.get("seed"), row.get("init_index"), row.get("flow_steps"))
        if key not in expected:
            raise ValueError("Unexpected episode: " + repr(key))
        condition, case = expected[key]
        identity = {
            "benchmark": benchmark,
            "benchmark_commit": manifest["benchmarks"][benchmark]["commit"],
            "suite": condition["suite"],
            "task_name": condition["task_name"],
            "family": condition["family"],
            "category": condition["category"],
            "manifest_sha256": manifest_sha256,
            "checkpoint_sha256": manifest["checkpoint"]["sha256"],
            "phase": "smoke" if smoke else "main",
        }
        for field, value in identity.items():
            if row.get(field) != value:
                raise ValueError("Provenance mismatch: " + field)
        if row.get("gpu_uuid") not in manifest["gpu_uuids"]:
            raise ValueError("Undeclared GPU UUID")
        if row.get("status") == "error":
            if not isinstance(row.get("error"), str) or not row["error"]:
                raise ValueError("Missing error description")
            errors.append(row)
            continue
        if row.get("status") != "ok" or type(row.get("success")) is not bool:
            raise ValueError("Outcome is not a complete boolean result")
        if key in found:
            raise ValueError("Duplicate completed episode")
        if row.get("initial_state_sha256") != case["initial_state_sha256"]:
            raise ValueError("Initial state differs from protocol")
        for field in ("initial_state_sha256", "stabilized_state_sha256"):
            _sha(row.get(field), field)
        horizon = manifest["settings"]["max_policy_steps"][condition["suite"]]
        policy_steps = row.get("policy_steps")
        if type(policy_steps) is not int or not 1 <= policy_steps <= horizon:
            raise ValueError("Invalid policy step count")
        chunks = row.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("Missing chunk records")
        replan_steps = manifest["settings"]["replan_steps"]
        if len(chunks) != math.ceil(policy_steps / replan_steps):
            raise ValueError("Chunk/action execution accounting mismatch")
        for i, chunk in enumerate(chunks):
            if type(chunk.get("chunk_index")) is not int or chunk["chunk_index"] != i:
                raise ValueError("Noncontiguous chunk indices")
            if type(chunk.get("velocity_evaluations")) is not int or chunk["velocity_evaluations"] != key[3]:
                raise ValueError("Incorrect velocity accounting")
            for field in ("noise_sha256", "action_sha256", "observation_sha256"):
                _sha(chunk.get(field), field)
            expected_noise = noise_digest(
                benchmark, condition["suite"], condition["task_name"], key[1], key[2], i, shape=tuple(shape)
            )
            if chunk["noise_sha256"] != expected_noise:
                raise ValueError("Deterministic chunk noise mismatch")
            for field in ("policy_ms", "request_ms"):
                _number(chunk.get(field), field)
            if chunk["request_ms"] + 0.01 < chunk["policy_ms"]:
                raise ValueError("Request latency less than policy latency")
        if (
            type(row.get("total_velocity_evaluations")) is not int
            or row["total_velocity_evaluations"] != len(chunks) * key[3]
        ):
            raise ValueError("Total velocity accounting mismatch")
        _number(row.get("episode_ms"), "episode_ms")
        if sum(c["request_ms"] for c in chunks) > row["episode_ms"] + 1:
            raise ValueError("Episode latency below its request sum")
        found[key] = row
    if set(found) != set(expected):
        raise ValueError("Incomplete planned evaluation: " + repr(sorted(set(expected) - set(found))))
    for condition, case in cases:
        pair = [found[(condition["condition_id"], case["seed"], case["init_index"], arm)] for arm in ARMS]
        for field in ("initial_state_sha256", "stabilized_state_sha256", "gpu_uuid"):
            if len({row[field] for row in pair}) != 1:
                raise ValueError("Paired " + field + " differs")
        if len({row["chunks"][0]["observation_sha256"] for row in pair}) != 1:
            raise ValueError("Paired first observations differ")
        for index in range(min(len(row["chunks"]) for row in pair)):
            if len({row["chunks"][index]["noise_sha256"] for row in pair}) != 1:
                raise ValueError("Paired chunk noise differs")
    return found, errors


def stats(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"count": 0, "mean": None, "p50": None, "p95": None, "sum": 0}
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "p50": float(np.quantile(values, 0.5)),
        "p95": float(np.quantile(values, 0.95)),
        "sum": float(np.sum(values)),
    }


def cluster_interval(cases, arm):
    groups = {}
    for case in cases:
        groups.setdefault(case["condition_id"], []).append(int(case["success"][str(arm)]) - int(case["success"]["1"]))
    if len(groups) < 2:
        return None
    sums = np.asarray([sum(g) for g in groups.values()])
    counts = np.asarray([len(g) for g in groups.values()])
    rng = np.random.default_rng(BOOTSTRAP_SEED + arm)
    indices = rng.integers(0, len(groups), (BOOTSTRAP_REPS, len(groups)))
    differences = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    return np.quantile(differences, [0.025, 0.975]).astype(float).tolist()


def group_summary(cases, found, minimum_failures=20, minimum_rescues=5):
    arms = {}
    failures = sum(not case["success"]["1"] for case in cases)
    for arm in ARMS:
        rows = [found[(c["condition_id"], c["seed"], c["init_index"], arm)] for c in cases]
        chunks = [chunk for row in rows for chunk in row["chunks"]]
        successes = sum(row["success"] for row in rows)
        result = {
            "episodes": len(rows),
            "successes": successes,
            "success_rate": successes / len(rows),
            "cost": {
                "velocity_evaluations_per_chunk": stats([c["velocity_evaluations"] for c in chunks]),
                "velocity_evaluations_per_episode": stats([r["total_velocity_evaluations"] for r in rows]),
                "total_velocity_evaluations": sum(r["total_velocity_evaluations"] for r in rows),
                "scored_policy_call_ms": stats([c["policy_ms"] for c in chunks]),
                "http_request_ms": stats([c["request_ms"] for c in chunks]),
                "simulator_inclusive_episode_ms": stats([r["episode_ms"] for r in rows]),
            },
        }
        if arm != 1:
            rescues = sum(not c["success"]["1"] and c["success"][str(arm)] for c in cases)
            regressions = sum(c["success"]["1"] and not c["success"][str(arm)] for c in cases)
            result["paired_vs_one"] = {
                "rescues": rescues,
                "regressions": regressions,
                "ties": len(cases) - rescues - regressions,
                "one_step_failures": failures,
                "fraction_one_step_failures_rescued": rescues / failures if failures else None,
                "net_success_difference": (rescues - regressions) / len(cases),
                "net_difference_condition_cluster_ci95": cluster_interval(cases, arm),
                "rescue_condition_count": len(
                    {c["condition_id"] for c in cases if not c["success"]["1"] and c["success"][str(arm)]}
                ),
                "assessment": "Unresolved: too few one-step failures or rescues."
                if failures < minimum_failures or rescues < minimum_rescues
                else "Paired rescue cases observed; interpret the net difference and clustered uncertainty.",
            }
        arms[str(arm)] = result
    return {"paired_cases": len(cases), "conditions": len({c["condition_id"] for c in cases}), "arms": arms}


def analyze(manifest, rows, benchmark, manifest_sha256, *, smoke=False):
    found, errors = validate(manifest, rows, benchmark, manifest_sha256, smoke=smoke)
    cases = []
    for condition, case in planned_cases(manifest, benchmark, smoke=smoke):
        cases.append(
            {
                "condition_id": condition["condition_id"],
                "family": condition["family"],
                "suite": condition["suite"],
                "task_name": condition["task_name"],
                "seed": case["seed"],
                "init_index": case["init_index"],
                "success": {
                    str(arm): found[(condition["condition_id"], case["seed"], case["init_index"], arm)]["success"]
                    for arm in ARMS
                },
            }
        )
    minimum_failures = manifest.get("analysis", {}).get("minimum_one_step_failures", 20)
    minimum_rescues = manifest.get("analysis", {}).get("minimum_rescues", 5)
    result = {
        "benchmark": benchmark,
        "phase": "smoke" if smoke else "main",
        "manifest_sha256": manifest_sha256,
        "checkpoint_sha256": manifest["checkpoint"]["sha256"],
        "benchmark_commit": manifest["benchmarks"][benchmark]["commit"],
        "audit": "passed",
        "completed_episodes": len(found),
        "error_attempts": errors,
        "error_attempt_count": len(errors),
        "overall": group_summary(cases, found, minimum_failures, minimum_rescues),
        "by_family": {
            family: group_summary([c for c in cases if c["family"] == family], found, minimum_failures, minimum_rescues)
            for family in sorted({c["family"] for c in cases})
        },
        "by_condition": {
            cid: group_summary([c for c in cases if c["condition_id"] == cid], found, minimum_failures, minimum_rescues)
            for cid in sorted({c["condition_id"] for c in cases})
        },
        "paired_cases": cases,
        "uncertainty": {
            "method": "condition-cluster percentile bootstrap, complete paired cases retained within clusters",
            "replicates": BOOTSTRAP_REPS,
            "seed": BOOTSTRAP_SEED,
            "seed_by_arm": {str(arm): BOOTSTRAP_SEED + arm for arm in ARMS[1:]},
            "note": "Null interval means fewer than two conditions. Few clusters or no observed discordances can yield uninformative or degenerate intervals; these do not establish population equivalence. Costs are measured on potentially different later trajectories and episode lengths.",
        },
    }
    failures = result["overall"]["arms"]["2"]["paired_vs_one"]["one_step_failures"]
    rescue_max = max(result["overall"]["arms"][str(a)]["paired_vs_one"]["rescues"] for a in ARMS[1:])
    threshold = manifest.get("analysis", {}).get("minimum_one_step_failures", 20)
    minimum_rescues = manifest.get("analysis", {}).get("minimum_rescues", 5)
    result["assessment"] = (
        "Smoke only; excluded from study inference."
        if smoke
        else "Unresolved: too few one-step failures or rescues under the declared budget."
        if failures < threshold or rescue_max < minimum_rescues
        else "Paired rescue and regression cases observed; interpret net differences with clustered uncertainty."
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-name")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    rows = [json.loads(line) for path in args.records for line in path.read_text().splitlines() if line.strip()]
    summary = analyze(manifest, rows, args.benchmark, digest_file(args.manifest), smoke=args.smoke)
    summary["record_files"] = [{"path": str(p), "sha256": digest_file(p)} for p in args.records]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if args.wandb_project:
        import wandb  # Optional reporting dependency; audits run without W&B.

        with wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            job_type="frozen-flow-analysis",
            config={
                "protocol_sha256": summary["manifest_sha256"],
                "benchmark": args.benchmark,
                "manifest_sha256": summary["manifest_sha256"],
            },
        ) as run:
            metrics = {"episodes": summary["completed_episodes"], "error_attempts": summary["error_attempt_count"]}
            for arm, values in summary["overall"]["arms"].items():
                prefix = args.benchmark + "/fixed_" + arm + "/"
                metrics[prefix + "success_rate"] = values["success_rate"]
                metrics[prefix + "policy_ms_mean"] = values["cost"]["scored_policy_call_ms"]["mean"]
                metrics[prefix + "episode_ms_mean"] = values["cost"]["simulator_inclusive_episode_ms"]["mean"]
                if "paired_vs_one" in values:
                    for field in (
                        "rescues",
                        "regressions",
                        "net_success_difference",
                        "fraction_one_step_failures_rescued",
                    ):
                        if values["paired_vs_one"][field] is not None:
                            metrics[prefix + field] = values["paired_vs_one"][field]
            run.log(metrics)
            summary["wandb_url"] = run.url
            args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
            artifact = wandb.Artifact(
                "frozen-flow-" + args.benchmark + "-" + summary["phase"] + "-" + summary["manifest_sha256"][:12],
                type="evaluation",
            )
            for path in [args.manifest, *args.records, args.output]:
                artifact.add_file(str(path))
            run.log_artifact(artifact)
    print(json.dumps({"output": str(args.output), "audit": summary["audit"], "wandb_url": summary.get("wandb_url")}))


if __name__ == "__main__":
    main()
