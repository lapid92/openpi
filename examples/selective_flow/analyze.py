"""Audit paired fixed arms and the predeclared metadata rule; no fitted selector."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

FIXED = ("fixed_1", "fixed_10")
STRATEGIES = (*FIXED, "metadata")


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def noise_digest(benchmark, suite, task, seed, initial, chunk, shape):
    key = json.dumps([benchmark, suite, task, seed, initial, chunk], separators=(",", ":"), ensure_ascii=True).encode()
    noise = (
        np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "little"))
        .standard_normal(shape)
        .astype(np.float32)
    )
    return hashlib.sha256(noise.tobytes()).hexdigest()


def planned_cases(manifest, benchmark, *, smoke=False):
    spec = manifest["benchmarks"][benchmark]
    conditions = spec["smoke_conditions"] if smoke and "smoke_conditions" in spec else spec["conditions"]
    by_id = {c["condition_id"]: c for c in conditions}
    if len(by_id) != len(conditions):
        raise ValueError("Duplicate condition declaration")
    if smoke and "smoke_conditions" not in spec:
        result = [(by_id[x["condition_id"]], x) for x in spec["smoke_cases"]]
    else:
        result = [(c, x) for c in conditions for x in c["cases"]]
    keys = [(c["condition_id"], x["seed"], x["init_index"]) for c, x in result]
    if not result or len(set(keys)) != len(keys):
        raise ValueError("Empty or duplicate planned cases")
    return result


def number(value, field, *, positive=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        raise ValueError("Invalid finite number: " + field)


def sha(value, field):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError("Invalid SHA256: " + field)


def validate(manifest, rows, benchmark, manifest_sha256, *, smoke=False):
    if manifest["flow_steps"] != [1, 10]:
        raise ValueError("Expected fixed 1/10 arms")
    shape = manifest["noise_policy"]["shape"]
    if shape != [manifest["settings"]["action_horizon"], manifest["settings"]["action_dim"]]:
        raise ValueError("Noise/model shape mismatch")
    cases = planned_cases(manifest, benchmark, smoke=smoke)
    expected = {
        (c["condition_id"], x["seed"], x["init_index"], arm): (c, x)
        for c, x in cases
        for arm in (STRATEGIES if smoke else FIXED)
    }
    found, errors = {}, []
    tolerance = manifest["analysis"]["initial_log_sigma_pair_tolerance"]
    for row in rows:
        if any(type(row.get(k)) is not int for k in ("seed", "init_index", "flow_steps")):
            raise ValueError("Episode identity must use integers")
        key = (row.get("condition_id"), row["seed"], row["init_index"], row.get("policy_arm"))
        if key not in expected:
            raise ValueError("Unexpected episode")
        condition, case = expected[key]
        effective = (
            manifest["metadata_rule"][condition["category"]] if key[3] == "metadata" else int(key[3].split("_")[1])
        )
        identity = {
            "benchmark": benchmark,
            "benchmark_commit": manifest["benchmarks"][benchmark]["commit"],
            "suite": condition["suite"],
            "task_name": condition["task_name"],
            "family": condition["family"],
            "category": condition["category"],
            "severity": condition["severity"],
            "phase": "smoke" if smoke else "main",
            "manifest_sha256": manifest_sha256,
            "checkpoint_sha256": manifest["checkpoint"]["sha256"],
            "head_sha256": manifest["head"]["sha256"],
            "flow_steps": effective,
        }
        for field, value in identity.items():
            if row.get(field) != value:
                raise ValueError("Provenance mismatch: " + field)
        if row.get("gpu_uuid") not in manifest["gpu_uuids"]:
            raise ValueError("Undeclared GPU")
        if row.get("status") == "error":
            if not isinstance(row.get("error"), str) or not row["error"]:
                raise ValueError("Missing error description")
            errors.append(row)
            continue
        if row.get("status") != "ok" or type(row.get("success")) is not bool:
            raise ValueError("Incomplete boolean outcome")
        if key in found:
            raise ValueError("Duplicate completed episode")
        if row.get("initial_state_sha256") != case["initial_state_sha256"]:
            raise ValueError("Initial state mismatch")
        for field in ("initial_state_sha256", "stabilized_state_sha256"):
            sha(row.get(field), field)
        steps = row.get("policy_steps")
        if type(steps) is not int or not 1 <= steps <= manifest["settings"]["max_policy_steps"][condition["suite"]]:
            raise ValueError("Invalid policy horizon")
        chunks = row.get("chunks")
        if not isinstance(chunks, list) or len(chunks) != math.ceil(steps / manifest["settings"]["replan_steps"]):
            raise ValueError("Chunk/action accounting mismatch")
        for index, chunk in enumerate(chunks):
            if type(chunk.get("chunk_index")) is not int or chunk["chunk_index"] != index:
                raise ValueError("Invalid chunk index")
            if type(chunk.get("velocity_evaluations")) is not int or chunk["velocity_evaluations"] != effective:
                raise ValueError("Velocity accounting mismatch")
            if (
                type(chunk.get("head_evaluations")) is not int
                or chunk["head_evaluations"] != 1
                or chunk.get("sigma_time") != 1.0
            ):
                raise ValueError("Expected one same-pass t=1 head evaluation")
            for field in ("noise_sha256", "action_sha256", "observation_sha256"):
                sha(chunk.get(field), field)
            if chunk["noise_sha256"] != noise_digest(
                benchmark, condition["suite"], condition["task_name"], key[1], key[2], index, tuple(shape)
            ):
                raise ValueError("Chunk noise mismatch")
            for field in ("policy_ms", "request_ms", "head_ms"):
                number(chunk.get(field), field)
            if chunk["request_ms"] + 0.01 < chunk["policy_ms"] or chunk["head_ms"] > chunk["policy_ms"] + 0.01:
                raise ValueError("Latency boundary mismatch")
            log_score = chunk.get("first_log_sigma")
            if isinstance(log_score, bool) or not isinstance(log_score, int | float) or not math.isfinite(log_score):
                raise ValueError("Invalid log sigma")
            number(chunk.get("first_sigma"), "first_sigma", positive=True)
            if abs(math.log(chunk["first_sigma"]) - log_score) > 1e-5:
                raise ValueError("Sigma/log consistency mismatch")
            if (
                type(chunk.get("reference_velocity_evaluations")) is not int
                or chunk["reference_velocity_evaluations"] != 0
            ):
                raise ValueError("Forbidden reference velocity evaluations")
            if chunk.get("verification", []):
                raise ValueError("Reference evaluations in scored chunk")
        if (
            row.get("initial_sigma") != chunks[0]["first_sigma"]
            or row.get("initial_log_sigma") != chunks[0]["first_log_sigma"]
        ):
            raise ValueError("Initial score differs from first chunk")
        if (
            type(row.get("total_velocity_evaluations")) is not int
            or row["total_velocity_evaluations"] != len(chunks) * effective
        ):
            raise ValueError("Total velocity accounting mismatch")
        number(row.get("episode_ms"), "episode_ms")
        if row["episode_ms"] + 1 < sum(c["request_ms"] for c in chunks):
            raise ValueError("Episode timing below requests")
        found[key] = row
    if set(found) != set(expected):
        raise ValueError("Incomplete planned records")
    for condition, case in cases:
        prefix = (condition["condition_id"], case["seed"], case["init_index"])
        paired = [found[(*prefix, arm)] for arm in (STRATEGIES if smoke else FIXED)]
        for field in ("initial_state_sha256", "stabilized_state_sha256", "gpu_uuid"):
            if len({r[field] for r in paired}) != 1:
                raise ValueError("Paired identity mismatch: " + field)
        if len({r["chunks"][0]["observation_sha256"] for r in paired}) != 1:
            raise ValueError("Paired initial observations differ")
        if max(r["initial_log_sigma"] for r in paired) - min(r["initial_log_sigma"] for r in paired) > tolerance:
            raise ValueError("Paired initial sigma disagreement")
        for index in range(min(len(r["chunks"]) for r in paired)):
            if len({r["chunks"][index]["noise_sha256"] for r in paired}) != 1:
                raise ValueError("Paired noise differs")
        if smoke:
            mapped = found[(*prefix, "fixed_" + str(manifest["metadata_rule"][condition["category"]]))]
            metadata = found[(*prefix, "metadata")]
            for field in ("success", "policy_steps", "total_velocity_evaluations"):
                if mapped[field] != metadata[field]:
                    raise ValueError("Metadata smoke differs from mapped fixed trajectory")
            if len(mapped["chunks"]) != len(metadata["chunks"]):
                raise ValueError("Metadata smoke chunk count differs")
            for first, second in zip(mapped["chunks"], metadata["chunks"], strict=True):
                for field in ("action_sha256", "observation_sha256", "noise_sha256", "first_log_sigma"):
                    if first[field] != second[field]:
                        raise ValueError("Metadata smoke trajectory digest differs")
    return found, errors


def stats(values):
    x = np.asarray(values, dtype=float)
    return {
        "count": len(x),
        "mean": float(x.mean()),
        "p50": float(np.quantile(x, 0.5)),
        "p95": float(np.quantile(x, 0.95)),
        "sum": float(x.sum()),
    }


def cluster_ci(cases, values, field, analysis):
    groups = {}
    for case, value in zip(cases, values, strict=True):
        groups.setdefault(case[field], []).append(value)
    if len(groups) < 2:
        return None
    sums = np.asarray([sum(x) for x in groups.values()])
    counts = np.asarray([len(x) for x in groups.values()])
    seed = analysis["bootstrap_seed"] + (1000 if field == "family" else 0)
    indices = np.random.default_rng(seed).integers(0, len(groups), (analysis["bootstrap_replicates"], len(groups)))
    return np.quantile(sums[indices].sum(axis=1) / counts[indices].sum(axis=1), [0.025, 0.975]).tolist()


def row_cost(row):
    return {
        "mean_policy_call_ms": float(np.mean([x["policy_ms"] for x in row["chunks"]])),
        "total_policy_ms": sum(x["policy_ms"] for x in row["chunks"]),
        "episode_ms": row["episode_ms"],
        "velocity_evaluations_per_chunk": row["flow_steps"],
        "total_velocity_evaluations": row["total_velocity_evaluations"],
    }


def comparison(cases, candidate, reference, analysis):
    deltas = [int(c["records"][candidate]["success"]) - int(c["records"][reference]["success"]) for c in cases]
    result = {
        "rescues": sum(d == 1 for d in deltas),
        "regressions": sum(d == -1 for d in deltas),
        "net_success_difference": float(np.mean(deltas)),
        "condition_cluster_ci95": cluster_ci(cases, deltas, "condition_id", analysis),
        "family_cluster_ci95": cluster_ci(cases, deltas, "family", analysis),
        "cost_differences": {},
    }
    failures = sum(not c["records"][reference]["success"] for c in cases)
    result["reference_failures"] = failures
    result["fraction_reference_failures_rescued"] = result["rescues"] / failures if failures else None
    for metric in row_cost(cases[0]["records"][candidate]):
        values = [row_cost(c["records"][candidate])[metric] - row_cost(c["records"][reference])[metric] for c in cases]
        result["cost_differences"][metric] = {
            "mean": float(np.mean(values)),
            "condition_cluster_ci95": cluster_ci(cases, values, "condition_id", analysis),
            "family_cluster_ci95": cluster_ci(cases, values, "family", analysis),
        }
    return result


def group_summary(cases, analysis):
    result = {
        "pairs": len(cases),
        "conditions": len({c["condition_id"] for c in cases}),
        "families": len({c["family"] for c in cases}),
        "strategies": {},
    }
    for arm in STRATEGIES:
        rows = [c["records"][arm] for c in cases]
        chunks = [x for row in rows for x in row["chunks"]]
        result["strategies"][arm] = {
            "successes": sum(r["success"] for r in rows),
            "episodes": len(rows),
            "success_rate": float(np.mean([r["success"] for r in rows])),
            "cost": {
                "policy_call_ms": stats([x["policy_ms"] for x in chunks]),
                "head_ms": stats([x["head_ms"] for x in chunks]),
                "request_ms": stats([x["request_ms"] for x in chunks]),
                "episode_ms": stats([r["episode_ms"] for r in rows]),
                "per_case_mean_policy_call_ms": stats([row_cost(r)["mean_policy_call_ms"] for r in rows]),
                "per_case_total_policy_ms": stats([row_cost(r)["total_policy_ms"] for r in rows]),
                "per_case_velocity_evaluations_per_chunk": stats([r["flow_steps"] for r in rows]),
                "total_velocity_evaluations": sum(r["total_velocity_evaluations"] for r in rows),
                "total_head_evaluations": len(chunks),
            },
        }
    result["comparisons"] = {
        "metadata_vs_fixed10": comparison(cases, "metadata", "fixed_10", analysis),
        "metadata_vs_fixed1": comparison(cases, "metadata", "fixed_1", analysis),
        "fixed10_vs_fixed1": comparison(cases, "fixed_10", "fixed_1", analysis),
    }
    primary = result["comparisons"]["metadata_vs_fixed10"]
    success_ci = primary["condition_cluster_ci95"]
    cost_ci = primary["cost_differences"]["mean_policy_call_ms"]["condition_cluster_ci95"]
    result["conservative_primary_support"] = bool(
        success_ci is not None and success_ci[0] >= 0 and cost_ci is not None and cost_ci[1] < 0
    )
    result["decision"] = (
        "Conservative support criterion met; inspect family sensitivity and limitations."
        if result["conservative_primary_support"]
        else (
            "Evidence of lower success than fixed 10: success interval is wholly negative."
            if success_ci is not None and success_ci[1] < 0
            else "Inconclusive under the predeclared joint success/cost support criterion."
        )
    )
    return result


def auc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    ends = np.r_[starts[1:], len(scores)]
    ranks = np.empty(len(scores), dtype=float)
    for start, end in zip(starts, ends, strict=True):
        ranks[order[start:end]] = (start + 1 + end) / 2
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def sigma_summary(cases, analysis):
    discordant = [c for c in cases if c["records"]["fixed_1"]["success"] != c["records"]["fixed_10"]["success"]]
    labels = [not c["records"]["fixed_1"]["success"] for c in discordant]
    scores = [c["records"]["fixed_1"]["initial_log_sigma"] for c in discordant]
    rescues = sum(labels)
    regressions = len(labels) - rescues
    condition_count = len({c["condition_id"] for c in discordant})
    result = {
        "cases": len(cases),
        "discordant_cases": len(discordant),
        "rescues": rescues,
        "regressions": regressions,
        "discordant_conditions": condition_count,
        "auc_higher_initial_sigma_predicts_rescue": auc(scores, labels),
        "credible_count_threshold_met": rescues >= analysis.get("sigma_min_rescues", 10)
        and regressions >= analysis.get("sigma_min_regressions", 10)
        and condition_count >= analysis.get("sigma_min_conditions", 5),
        "bootstrap": {},
        "scope": "Retrospective rescue-versus-regression ranking among discordant fixed-arm cases, not an inference-time gate or a fitted predictor.",
    }
    for field in ("condition_id", "family"):
        groups = {}
        for i, case in enumerate(discordant):
            groups.setdefault(case[field], []).append(i)
        draws = []
        if len(groups) >= 2 and rescues and regressions:
            blocks = list(groups.values())
            rng = np.random.default_rng(analysis["bootstrap_seed"] + 2000 + (1000 if field == "family" else 0))
            for _ in range(analysis["bootstrap_replicates"]):
                indices = [i for block in rng.integers(0, len(blocks), len(blocks)) for i in blocks[block]]
                value = auc([scores[i] for i in indices], [labels[i] for i in indices])
                if value is not None:
                    draws.append(value)
        result["bootstrap"][field] = {
            "clusters": len(groups),
            "valid_draws": len(draws),
            "requested_draws": analysis["bootstrap_replicates"],
            "ci95": np.quantile(draws, [0.025, 0.975]).tolist() if draws else None,
            "note": "Single-class resamples excluded and counted; interval conditional on valid draws.",
        }
    result["assessment"] = (
        "Count threshold met; interpret uncertainty and type/family sensitivity."
        if result["credible_count_threshold_met"]
        else "Descriptive only: too few rescues, regressions or conditions."
    )
    return result


def analyze(manifest, rows, benchmark, manifest_sha256, *, smoke=False):
    found, errors = validate(manifest, rows, benchmark, manifest_sha256, smoke=smoke)
    cases = []
    for condition, case in planned_cases(manifest, benchmark, smoke=smoke):
        prefix = (condition["condition_id"], case["seed"], case["init_index"])
        records = {arm: found[(*prefix, arm)] for arm in FIXED}
        selected = manifest["metadata_rule"][condition["category"]]
        records["metadata"] = records["fixed_" + str(selected)]
        cases.append(
            {
                "condition_id": condition["condition_id"],
                "family": condition["family"],
                "category": condition["category"],
                "seed": case["seed"],
                "init_index": case["init_index"],
                "selected_flow_steps": selected,
                "records": records,
            }
        )
    analysis = manifest["analysis"]
    return {
        "benchmark": benchmark,
        "benchmark_commit": manifest["benchmarks"][benchmark]["commit"],
        "phase": "smoke" if smoke else "main",
        "manifest_sha256": manifest_sha256,
        "checkpoint_sha256": manifest["checkpoint"]["sha256"],
        "head_sha256": manifest["head"]["sha256"],
        "audit": "passed",
        "completed_episodes": len(found),
        "error_attempts": errors,
        "error_attempt_count": len(errors),
        "overall": group_summary(cases, analysis),
        "by_type": {
            value: group_summary([c for c in cases if c["category"] == value], analysis)
            for value in sorted({c["category"] for c in cases})
        },
        "by_family": {
            value: group_summary([c for c in cases if c["family"] == value], analysis)
            for value in sorted({c["family"] for c in cases})
        },
        "sigma": {
            "overall": sigma_summary(cases, analysis),
            "by_type": {
                value: sigma_summary([c for c in cases if c["category"] == value], analysis)
                for value in sorted({c["category"] for c in cases})
            },
        },
        "paired_cases": [
            {k: v for k, v in c.items() if k != "records"}
            | {
                "success": {a: c["records"][a]["success"] for a in STRATEGIES},
                "initial_sigma": c["records"]["fixed_1"]["initial_sigma"],
                "initial_log_sigma": c["records"]["fixed_1"]["initial_log_sigma"],
            }
            for c in cases
        ],
        "limitations": [
            "Metadata results reuse matched fixed-arm records; smoke separately checks actual wrapper equivalence.",
            "Primary costs use equal-case mean scored-policy-call latency; total policy time and simulator time are separate.",
            "The support criterion is not a conventional noninferiority margin, equivalence test or joint 95% confidence statement.",
            "Condition-cluster inference has family-cluster sensitivity; repeated states and few families limit generalization.",
            "Sigma ranking conditions on observed discordance and cannot by itself justify an episode-level selector.",
            "No trained selector, modified predictor, adaptive speedup or pooled benchmark rate is inferred.",
            "Smoke results are excluded from primary scientific interpretation."
            if smoke
            else "All declared main cases retained.",
        ],
        "uncertainty": {
            "bootstrap_replicates": analysis["bootstrap_replicates"],
            "effect_condition_seed": analysis["bootstrap_seed"],
            "effect_family_seed": analysis["bootstrap_seed"] + 1000,
            "sigma_condition_seed": analysis["bootstrap_seed"] + 2000,
            "sigma_family_seed": analysis["bootstrap_seed"] + 3000,
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--records", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--wandb-project")
    p.add_argument("--wandb-name")
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    rows = [json.loads(line) for path in args.records for line in path.read_text().splitlines() if line.strip()]
    result = analyze(manifest, rows, args.benchmark, digest_file(args.manifest), smoke=args.smoke)
    result["record_files"] = [{"path": str(path), "sha256": digest_file(path)} for path in args.records]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if args.wandb_project:
        import wandb

        with wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            job_type="selective-fixed-step-analysis",
            config={
                k: result[k]
                for k in (
                    "benchmark",
                    "benchmark_commit",
                    "phase",
                    "manifest_sha256",
                    "checkpoint_sha256",
                    "head_sha256",
                )
            },
        ) as run:
            metrics = {}
            for arm, item in result["overall"]["strategies"].items():
                metrics[arm + "/success_rate"] = item["success_rate"]
                metrics[arm + "/mean_policy_call_ms_equal_case"] = item["cost"]["per_case_mean_policy_call_ms"]["mean"]
            metrics["primary_support"] = int(result["overall"]["conservative_primary_support"])
            run.log(metrics)
            result["wandb_url"] = run.url
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            artifact = wandb.Artifact(
                "selective-fixed-" + args.benchmark + "-" + result["phase"] + "-" + result["manifest_sha256"][:12],
                type="evaluation",
            )
            for path in [args.manifest, *args.records, args.output]:
                artifact.add_file(str(path))
            run.log_artifact(artifact)
    print(json.dumps({"audit": result["audit"], "output": str(args.output), "wandb_url": result.get("wandb_url")}))


if __name__ == "__main__":
    main()
