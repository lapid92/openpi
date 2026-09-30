"""Fail-closed paired analysis of the predeclared no-training LIBERO-Plus pilot."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta
import wandb

ARMS = (1, 2, 4, 10)
EXPECTED_UUID = "GPU-4779cdde-a260-f8ec-da6a-6fa390bc7fd7"
BOOTSTRAP_SEED = 20261003
BOOTSTRAP_REPLICATES = 2000


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_binomial_interval(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    lower = 0.0 if successes == 0 else float(beta.ppf(0.025, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta.ppf(0.975, successes + 1, total - successes))
    return [lower, upper]


def auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return None
    return float(np.mean((positive[:, None] > negative[None, :]) + 0.5 * (positive[:, None] == negative[None, :])))


def cluster_bootstrap_interval(items: list[dict], measure, *, seed=BOOTSTRAP_SEED) -> list[float] | None:
    by_task = defaultdict(list)
    for item in items:
        by_task[item["task_id"]].append(item)
    task_ids = sorted(by_task)
    if len(task_ids) < 2:
        return None
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(task_ids, size=len(task_ids), replace=True)
        value = measure([row for task_id in sampled for row in by_task[task_id]])
        if value is not None and np.isfinite(value):
            estimates.append(value)
    if not estimates:
        return None
    return np.quantile(estimates, [0.025, 0.975]).astype(float).tolist()


def _finite_nonnegative(value, field):
    if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
        raise ValueError(f"Invalid {field}: {value}")
    return float(value)


def load_validated(manifest_path: Path, records_path: Path):
    manifest = json.loads(manifest_path.read_text())
    manifest_sha256 = sha256_file(manifest_path)
    required = {"base_checkpoint_identity", "head_sha256", "benchmark_commit", "flow_steps", "tasks"}
    if required - manifest.keys():
        raise ValueError(f"Protocol lacks pinned identities: {sorted(required - manifest.keys())}")
    if tuple(manifest["flow_steps"]) != ARMS:
        raise ValueError("Flow arms changed from the declared protocol")
    log_sigma_tolerance = manifest.get("score_pairing_log_sigma_tolerance")
    if log_sigma_tolerance != 0.01:
        raise ValueError("Declared first-score numerical tolerance changed")
    expected = {(task["id"], seed, arm): task for task in manifest["tasks"] for seed in task["seeds"] for arm in ARMS}
    lines = records_path.read_text().splitlines()
    if len(lines) != len(expected):
        raise ValueError(f"Expected {len(expected)} completed episodes, found {len(lines)} lines")
    records = {}
    for line_number, line in enumerate(lines, start=1):
        row = json.loads(line)
        key = (row.get("task_id"), row.get("seed"), row.get("flow_steps"))
        if key not in expected:
            raise ValueError(f"Unexpected episode on line {line_number}: {key}")
        if key in records:
            raise ValueError(f"Duplicate episode: {key}")
        if row.get("status") != "ok":
            error_message = row.get("error")
            raise ValueError(f"Episode error on line {line_number}: {error_message}")
        task = expected[key]
        for field, value in (
            ("manifest_sha256", manifest_sha256),
            ("benchmark_commit", manifest["benchmark_commit"]),
            ("checkpoint_identity", manifest["base_checkpoint_identity"]),
            ("head_checkpoint", manifest["head_sha256"]),
            ("gpu_uuid", EXPECTED_UUID),
            ("task_name", task["name"]),
            ("category", task["category"]),
        ):
            if row.get(field) != value:
                raise ValueError(f"{field} mismatch for {key}")
        if type(row.get("success")) is not bool:
            raise ValueError(f"Success must be Boolean for {key}")
        chunks = row.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            raise ValueError(f"No action chunks for {key}")
        for chunk_index, chunk in enumerate(chunks):
            if chunk.get("chunk_index") != chunk_index or chunk.get("velocity_evaluations") != key[2]:
                raise ValueError(f"Chunk index or velocity count mismatch for {key}")
            expected_times = 1 - np.arange(key[2]) / key[2]
            if not np.allclose(chunk.get("times"), expected_times, rtol=0, atol=1e-5):
                raise ValueError(f"Fixed Euler times changed for {key}")
            sigmas = np.asarray(chunk.get("sigmas"), dtype=float)
            if sigmas.shape != (key[2],) or not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0):
                raise ValueError(f"Missing or invalid sigma for {key}")
            log_sigma = np.asarray(chunk.get("log_sigma"), dtype=float)
            if log_sigma.shape != (key[2],) or not np.all(np.isfinite(log_sigma)):
                raise ValueError(f"Missing or invalid log sigma for {key}")
            if not np.allclose(np.exp(log_sigma), sigmas, rtol=1e-6, atol=1e-8):
                raise ValueError(f"Sigma and log sigma disagree for {key}")
            for field in ("request_ms", "policy_ms"):
                _finite_nonnegative(chunk.get(field), field)
            if chunk["request_ms"] + 1e-3 < chunk["policy_ms"]:
                raise ValueError(f"HTTP request time is shorter than scored policy time for {key}")
            if any(
                len(chunk.get(field, "")) != 64 for field in ("noise_digest", "action_digest", "observation_digest")
            ):
                raise ValueError(f"Missing chunk identity digests for {key}")
        if row.get("total_velocity_evaluations") != key[2] * len(chunks):
            raise ValueError(f"Total velocity count mismatch for {key}")
        if len(row.get("initial_state_digest", "")) != 64:
            raise ValueError(f"Missing initial-state digest for {key}")
        if not 1 <= row.get("policy_steps", 0) <= manifest["max_policy_steps"]:
            raise ValueError(f"Policy-step count out of range for {key}")
        _finite_nonnegative(row.get("episode_ms"), "episode_ms")
        records[key] = row
    if set(records) != set(expected):
        raise ValueError("Pilot has missing declared episodes")
    for task in manifest["tasks"]:
        for seed in task["seeds"]:
            task_id = task["id"]
            rows = [records[(task_id, seed, arm)] for arm in ARMS]
            if len({row["initial_state_digest"] for row in rows}) != 1:
                raise ValueError(f"Initial-state mismatch for task {task['id']}, seed {seed}")
            if len({row["chunks"][0]["noise_digest"] for row in rows}) != 1:
                raise ValueError(f"First-chunk noise mismatch for task {task['id']}, seed {seed}")
            if len({row["chunks"][0]["observation_digest"] for row in rows}) != 1:
                raise ValueError(f"First-chunk observation mismatch for task {task['id']}, seed {seed}")
            canonical_log_sigma = rows[0]["chunks"][0]["log_sigma"][0]
            if max(abs(row["chunks"][0]["log_sigma"][0] - canonical_log_sigma) for row in rows) > log_sigma_tolerance:
                raise ValueError(f"t=1 log sigma drift exceeds declared tolerance for task {task['id']}, seed {seed}")
    return manifest, manifest_sha256, records


def _latency(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.5)),
        "p95": float(np.quantile(values, 0.95)),
    }


def summarize(manifest: dict, manifest_sha256: str, records: dict) -> dict:
    case_rows = []
    for task in manifest["tasks"]:
        for seed in task["seeds"]:
            rows = {arm: records[(task["id"], seed, arm)] for arm in ARMS}
            case_rows.append(
                {
                    "task_id": task["id"],
                    "task_name": task["name"],
                    "category": task["category"],
                    "seed": seed,
                    "sigma_t1": float(rows[1]["chunks"][0]["sigmas"][0]),
                    "max_first_log_sigma_drift": max(
                        abs(rows[arm]["chunks"][0]["log_sigma"][0] - rows[1]["chunks"][0]["log_sigma"][0])
                        for arm in ARMS
                    ),
                    "success": {str(arm): rows[arm]["success"] for arm in ARMS},
                }
            )
    by_category = {}
    for category in sorted({task["category"] for task in manifest["tasks"]}):
        pairs = [case for case in case_rows if case["category"] == category]
        by_category[category] = {}
        for arm in ARMS:
            successes = sum(case["success"][str(arm)] for case in pairs)
            by_category[category][str(arm)] = {
                "successes": successes,
                "episodes": len(pairs),
                "success_rate": successes / len(pairs),
                "episode_exact_ci95_exploratory": exact_binomial_interval(successes, len(pairs)),
            }
    arm_results = {}
    ranking = {}
    gate = False
    for arm in ARMS:
        rows = [records[(case["task_id"], case["seed"], arm)] for case in case_rows]
        all_chunks = [chunk for row in rows for chunk in row["chunks"]]
        successes = sum(row["success"] for row in rows)
        arm_results[str(arm)] = {
            "successes": successes,
            "episodes": len(rows),
            "success_rate": successes / len(rows),
            "episode_exact_ci95_exploratory": exact_binomial_interval(successes, len(rows)),
            "velocity_evaluations_per_chunk": _latency([chunk["velocity_evaluations"] for chunk in all_chunks]),
            "velocity_evaluations_per_episode": _latency([row["total_velocity_evaluations"] for row in rows]),
            "scored_policy_chunk_ms": _latency([chunk["policy_ms"] for chunk in all_chunks]),
            "http_request_chunk_ms": _latency([chunk["request_ms"] for chunk in all_chunks]),
            "episode_ms": _latency([row["episode_ms"] for row in rows]),
            "chunks": len(all_chunks),
        }
        if arm == 1:
            continue
        paired = []
        for case in case_rows:
            low = int(case["success"]["1"])
            high = int(case["success"][str(arm)])
            paired.append({**case, "gain": int(high > low), "loss": int(high < low), "delta": high - low})
        wins = sum(pair["gain"] for pair in paired)
        losses = sum(pair["loss"] for pair in paired)
        discordant = wins + losses
        gain_categories = sorted({pair["category"] for pair in paired if pair["gain"]})
        arm_results[str(arm)]["paired_vs_1"] = {
            "wins": wins,
            "losses": losses,
            "ties": len(paired) - discordant,
            "net_success_rate_difference": (wins - losses) / len(paired),
            "net_difference_task_cluster_bootstrap_ci95": cluster_bootstrap_interval(
                paired, lambda sample: float(np.mean([item["delta"] for item in sample]))
            ),
            "discordant_win_fraction": wins / discordant if discordant else None,
            "discordant_win_fraction_exact_ci95": exact_binomial_interval(wins, discordant),
            "gain_categories": gain_categories,
        }
        gate = gate or (wins >= 4 and losses <= 1 and len(gain_categories) >= 2)
        scores = np.asarray([pair["sigma_t1"] for pair in paired])
        labels = np.asarray([pair["gain"] for pair in paired])
        ranking[str(arm)] = {
            "positive_gains": wins,
            "nongains": len(paired) - wins,
            "auc_sigma_t1_for_gain": auc(scores, labels),
            "auc_task_cluster_bootstrap_ci95": cluster_bootstrap_interval(
                paired,
                lambda sample: auc(
                    np.asarray([item["sigma_t1"] for item in sample]),
                    np.asarray([item["gain"] for item in sample]),
                ),
                seed=BOOTSTRAP_SEED + arm,
            ),
            "mean_sigma_gain": float(scores[labels == 1].mean()) if wins else None,
            "mean_sigma_nongain": float(scores[labels == 0].mean()) if wins < len(paired) else None,
        }
    return {
        "protocol_name": manifest["name"],
        "manifest_sha256": manifest_sha256,
        "benchmark_commit": manifest["benchmark_commit"],
        "base_checkpoint_identity": manifest["base_checkpoint_identity"],
        "head_sha256": manifest["head_sha256"],
        "gpu_uuid": EXPECTED_UUID,
        "max_first_log_sigma_drift": max(case["max_first_log_sigma_drift"] for case in case_rows),
        "pair_count": len(case_rows),
        "episode_count": len(records),
        "by_category": by_category,
        "by_arm": arm_results,
        "sigma_ranking": ranking,
        "first_chunk_cases": case_rows,
        "decision_gate_pass": gate,
        "decision": "extra-step benefit clears declared pilot gate"
        if gate
        else "stop; no meaningful selective extra-step benefit under declared pilot gate",
        "uncertainty_note": "Exact binomial intervals treat episodes as independent and are exploratory. Task-cluster bootstrap resamples six benchmark task instances, retaining both seeds; pilot intervals are coarse.",
        "separate_measures": "No demonstration-action MSE or fine-step numerical reference is measured by this closed-loop pilot.",
    }


def _add_interval(metrics: dict, prefix: str, interval: list[float] | None) -> None:
    if interval is not None:
        metrics[f"{prefix}_ci95_low"] = interval[0]
        metrics[f"{prefix}_ci95_high"] = interval[1]


def log_wandb(
    manifest_path: Path, records_path: Path, summary_path: Path, manifest: dict, summary: dict, project: str, name: str
):
    with wandb.init(project=project, name=name, job_type="libero-plus-no-training-pilot", config=manifest) as run:
        metrics = {}
        for arm, result in summary["by_arm"].items():
            prefix = f"fixed_{arm}"
            metrics.update(
                {
                    f"{prefix}/success_rate": result["success_rate"],
                    f"{prefix}/successes": result["successes"],
                    f"{prefix}/policy_chunk_ms_mean": result["scored_policy_chunk_ms"]["mean"],
                    f"{prefix}/policy_chunk_ms_p50": result["scored_policy_chunk_ms"]["p50"],
                    f"{prefix}/policy_chunk_ms_p95": result["scored_policy_chunk_ms"]["p95"],
                    f"{prefix}/http_request_chunk_ms_p95": result["http_request_chunk_ms"]["p95"],
                    f"{prefix}/episode_ms_mean": result["episode_ms"]["mean"],
                    f"{prefix}/episode_ms_p95": result["episode_ms"]["p95"],
                    f"{prefix}/velocity_evaluations_per_chunk_mean": result["velocity_evaluations_per_chunk"]["mean"],
                    f"{prefix}/velocity_evaluations_per_chunk_p95": result["velocity_evaluations_per_chunk"]["p95"],
                    f"{prefix}/velocity_evaluations_per_episode_mean": result["velocity_evaluations_per_episode"][
                        "mean"
                    ],
                    f"{prefix}/velocity_evaluations_per_episode_p95": result["velocity_evaluations_per_episode"]["p95"],
                }
            )
            _add_interval(metrics, f"{prefix}/success_rate", result["episode_exact_ci95_exploratory"])
            if "paired_vs_1" in result:
                pair = result["paired_vs_1"]
                metrics.update(
                    {
                        f"{prefix}/paired_wins_vs_1": pair["wins"],
                        f"{prefix}/paired_losses_vs_1": pair["losses"],
                        f"{prefix}/paired_net_success_difference": pair["net_success_rate_difference"],
                    }
                )
                _add_interval(
                    metrics,
                    f"{prefix}/paired_net_success_difference",
                    pair["net_difference_task_cluster_bootstrap_ci95"],
                )
                if pair["discordant_win_fraction"] is not None:
                    metrics[f"{prefix}/discordant_win_fraction"] = pair["discordant_win_fraction"]
                _add_interval(metrics, f"{prefix}/discordant_win_fraction", pair["discordant_win_fraction_exact_ci95"])
                ranking = summary["sigma_ranking"][arm]
                if ranking["auc_sigma_t1_for_gain"] is not None:
                    metrics[f"{prefix}/sigma_t1_gain_auc"] = ranking["auc_sigma_t1_for_gain"]
                _add_interval(metrics, f"{prefix}/sigma_t1_gain_auc", ranking["auc_task_cluster_bootstrap_ci95"])
        for category, arms in summary["by_category"].items():
            slug = category.lower().replace(" ", "_")
            for arm, result in arms.items():
                prefix = f"category/{slug}/fixed_{arm}"
                metrics[f"{prefix}/success_rate"] = result["success_rate"]
                metrics[f"{prefix}/successes"] = result["successes"]
                _add_interval(metrics, f"{prefix}/success_rate", result["episode_exact_ci95_exploratory"])
        metrics["max_first_log_sigma_drift"] = summary["max_first_log_sigma_drift"]
        run.log(metrics)
        table = wandb.Table(
            columns=["task_id", "category", "seed", "sigma_t1", "success_1", "success_2", "success_4", "success_10"]
        )
        for row in summary["first_chunk_cases"]:
            table.add_data(
                row["task_id"],
                row["category"],
                row["seed"],
                row["sigma_t1"],
                *[int(row["success"][str(arm)]) for arm in ARMS],
            )
        run.log({"paired_cases": table})
        artifact = wandb.Artifact("libero-plus-no-training-pilot", type="evaluation")
        for path in (manifest_path, records_path, summary_path):
            artifact.add_file(str(path))
        run.log_artifact(artifact)
        run.summary["decision_gate_pass"] = summary["decision_gate_pass"]
        run.summary["manifest_sha256"] = summary["manifest_sha256"]
        run.summary["paired_case_count"] = summary["pair_count"]
        run.summary["completed_episode_count"] = summary["episode_count"]
        return run.url


def analyze_smoke(manifest_path: Path, records_path: Path, output_path: Path, project: str, name: str) -> dict:
    manifest = json.loads(manifest_path.read_text())
    lines = records_path.read_text().splitlines()
    if len(lines) != 1:
        raise ValueError("Smoke run must contain exactly one episode")
    row = json.loads(lines[0])
    expected = manifest["smoke"]
    if (row.get("task_id"), row.get("seed"), row.get("flow_steps")) != (
        expected["task_id"],
        expected["seed"],
        expected["steps"][0],
    ):
        raise ValueError("Smoke task, seed, or fixed-step arm changed")
    if row.get("status") != "ok" or type(row.get("success")) is not bool:
        error_message = row.get("error")
        raise ValueError(f"Smoke episode failed: {error_message}")
    for field, value in (
        ("manifest_sha256", sha256_file(manifest_path)),
        ("benchmark_commit", manifest["benchmark_commit"]),
        ("checkpoint_identity", manifest["base_checkpoint_identity"]),
        ("head_checkpoint", manifest["head_sha256"]),
        ("gpu_uuid", EXPECTED_UUID),
    ):
        if row.get(field) != value:
            raise ValueError(f"Smoke {field} mismatch")
    chunks = row.get("chunks")
    if not chunks or row.get("total_velocity_evaluations") != len(chunks) * row["flow_steps"]:
        raise ValueError("Smoke velocity accounting mismatch")
    for i, chunk in enumerate(chunks):
        if chunk["chunk_index"] != i or chunk["velocity_evaluations"] != row["flow_steps"]:
            raise ValueError("Smoke chunk velocity accounting mismatch")
        if abs(chunk["times"][0] - 1.0) > 1e-6 or not np.isfinite(chunk["sigmas"][0]):
            raise ValueError("Smoke first-pass score invalid")
    summary = {
        "mode": "smoke",
        "protocol_name": manifest["name"],
        "manifest_sha256": sha256_file(manifest_path),
        "task_id": row["task_id"],
        "seed": row["seed"],
        "flow_steps": row["flow_steps"],
        "success": row["success"],
        "policy_steps": row["policy_steps"],
        "action_chunks": len(chunks),
        "velocity_evaluations": row["total_velocity_evaluations"],
        "sigma_t1": float(chunks[0]["sigmas"][0]),
        "scored_policy_chunk_ms": _latency([chunk["policy_ms"] for chunk in chunks]),
        "http_request_chunk_ms": _latency([chunk["request_ms"] for chunk in chunks]),
        "episode_ms": float(row["episode_ms"]),
        "benchmark_commit": manifest["benchmark_commit"],
        "base_checkpoint_identity": manifest["base_checkpoint_identity"],
        "head_sha256": manifest["head_sha256"],
        "gpu_uuid": EXPECTED_UUID,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    with wandb.init(project=project, name=name, job_type="libero-plus-smoke", config=manifest) as run:
        run.log(
            {
                "smoke/success": int(summary["success"]),
                "smoke/policy_steps": summary["policy_steps"],
                "smoke/velocity_evaluations": summary["velocity_evaluations"],
                "smoke/sigma_t1": summary["sigma_t1"],
                "smoke/scored_policy_chunk_ms_mean": summary["scored_policy_chunk_ms"]["mean"],
                "smoke/episode_ms": summary["episode_ms"],
            }
        )
        artifact = wandb.Artifact("libero-plus-no-training-smoke", type="evaluation")
        for path in (manifest_path, records_path, output_path):
            artifact.add_file(str(path))
        run.log_artifact(artifact)
        summary["wandb_url"] = run.url
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("pilot_manifest.json"))
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wandb-project", default="pi05-libero-velocity-residual")
    parser.add_argument("--wandb-name", default="libero-plus-no-training-pilot")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        summary = analyze_smoke(args.manifest, args.records, args.output, args.wandb_project, args.wandb_name)
        print(
            json.dumps(
                {"summary": str(args.output), "wandb_url": summary["wandb_url"], "smoke_success": summary["success"]}
            ),
            flush=True,
        )
        return
    manifest, manifest_sha256, records = load_validated(args.manifest, args.records)
    summary = summarize(manifest, manifest_sha256, records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    url = log_wandb(args.manifest, args.records, args.output, manifest, summary, args.wandb_project, args.wandb_name)
    summary["wandb_url"] = url
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {"summary": str(args.output), "wandb_url": url, "decision_gate_pass": summary["decision_gate_pass"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
