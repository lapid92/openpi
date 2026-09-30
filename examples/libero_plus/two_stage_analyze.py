"""Audit and report two-stage frozen pi0.5 LIBERO-Plus episodes; log W&B artifacts."""

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
BOOTSTRAP_SEED = 20261004
BOOTSTRAP_REPS = 2000


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def interval(successes: int, count: int):
    if not count:
        return None
    low = 0.0 if successes == 0 else float(beta.ppf(0.025, successes, count - successes + 1))
    high = 1.0 if successes == count else float(beta.ppf(0.975, successes + 1, count - successes))
    return [low, high]


def stats(values):
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "p50": float(np.quantile(a, 0.5)), "p95": float(np.quantile(a, 0.95))}


def auc(scores, labels):
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return None
    return float(np.mean((positive[:, None] > negative[None, :]) + 0.5 * (positive[:, None] == negative[None, :])))


def cluster_interval(items, value_fn, *, seed=BOOTSTRAP_SEED):
    groups = defaultdict(list)
    for item in items:
        groups[item["condition_id"]].append(item)
    keys = sorted(groups)
    if len(keys) < 2:
        return None
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(BOOTSTRAP_REPS):
        chosen = rng.choice(keys, len(keys), replace=True)
        value = value_fn([item for key in chosen for item in groups[key]])
        if value is not None and np.isfinite(value):
            estimates.append(value)
    if not estimates:
        return None
    return np.quantile(estimates, [0.025, 0.975]).astype(float).tolist()


def validate(manifest, manifest_path, rows, stage, selection=None, selection_path=None, screen_found=None):
    manifest_hash = sha256_file(manifest_path)
    conditions = {c["condition_id"]: c for c in manifest["conditions"]}
    if stage == "smoke":
        expected = {(manifest["smoke"]["condition_id"], manifest["smoke"]["seed"], 1)}
    elif stage == "screen":
        expected = {
            (c["condition_id"], seed, 1) for c in manifest["conditions"] for seed in manifest["screening_seeds"]
        }
    else:
        if selection is None or not selection["selected_condition_ids"]:
            raise ValueError("Frozen nonempty selection required")
        ids = selection["selected_condition_ids"]
        if len(ids) != len(set(ids)) or len(ids) > manifest["stage_1"]["max_selected_conditions"]:
            raise ValueError("Selection has duplicate or too many conditions")
        if screen_found is None or ids != prespecified_selection(manifest, screen_found):
            raise ValueError("Selection cannot be verified against complete screen")
        expected = {
            (cid, seed, arm)
            for cid in selection["selected_condition_ids"]
            for seed in manifest["heldout_seeds"]
            for arm in ARMS
        }
        if selection["manifest_sha256"] != manifest_hash:
            raise ValueError("Selection protocol hash mismatch")
    found = {}
    for row in rows:
        key = (row.get("condition_id"), row.get("seed"), row.get("flow_steps"))
        if key not in expected or key in found or row.get("stage") != stage or row.get("status") != "ok":
            raise ValueError("Unexpected, duplicate, or failed episode: " + repr(key))
        c = conditions[key[0]]
        for field, value in (
            ("task_id", c["task_id"]),
            ("task_name", c["task_name"]),
            ("category", c["perturbation_type"]),
            ("suite", c["suite"]),
            ("task_family", c["task_family"]),
            ("severity", c["severity"]),
            ("checkpoint_identity", manifest["base_checkpoint_identity"]),
            ("head_checkpoint", manifest["head_sha256"]),
            ("benchmark_commit", manifest["benchmark_commit"]),
            ("gpu_uuid", manifest["gpu_uuid"]),
            ("manifest_sha256", manifest_hash),
        ):
            if row.get(field) != value:
                raise ValueError("Record identity mismatch: " + field)
        if stage == "compare" and selection_path and row.get("selection_sha256") != sha256_file(selection_path):
            raise ValueError("Record selection SHA mismatch")
        if type(row.get("success")) is not bool or not 1 <= row.get("policy_steps", 0) <= c["max_policy_steps"]:
            raise ValueError("Invalid outcome/horizon")
        if not np.isfinite(row.get("episode_ms", np.nan)) or row["episode_ms"] < 0:
            raise ValueError("Invalid episode latency")
        chunks = row.get("chunks")
        if not chunks or row.get("total_velocity_evaluations") != len(chunks) * key[2]:
            raise ValueError("Velocity count mismatch")
        for i, chunk in enumerate(chunks):
            if chunk.get("chunk_index") != i or chunk.get("velocity_evaluations") != key[2]:
                raise ValueError("Chunk count mismatch")
            if not np.allclose(chunk.get("times", []), 1 - np.arange(key[2]) / key[2], atol=1e-5, rtol=0):
                raise ValueError("Euler time mismatch")
            sigma = np.asarray(chunk.get("sigmas", []), dtype=float)
            logs = np.asarray(chunk.get("log_sigma", []), dtype=float)
            if (
                sigma.shape != (key[2],)
                or logs.shape != (key[2],)
                or not np.all(np.isfinite(sigma))
                or not np.all(np.isfinite(logs))
                or np.any(sigma <= 0)
                or not np.allclose(np.exp(logs), sigma, rtol=1e-6, atol=1e-8)
            ):
                raise ValueError("Invalid same-pass score")
            times = np.asarray([chunk.get("policy_ms"), chunk.get("request_ms")], dtype=float)
            if not np.all(np.isfinite(times)) or np.any(times < 0) or times[1] + 1e-3 < times[0]:
                raise ValueError("Chunk latency mismatch")
            if any(
                len(chunk.get(field, "")) != 64 for field in ("noise_digest", "action_digest", "observation_digest")
            ):
                raise ValueError("Chunk digest missing")
        if len(row.get("initial_state_digest", "")) != 64:
            raise ValueError("Initial state digest missing")
        found[key] = row
    if set(found) != expected:
        raise ValueError("Missing declared episodes: " + repr(sorted(expected - set(found))))
    if stage == "compare":
        tol = manifest["execution"]["first_score_log_sigma_tolerance"]
        for cid in selection["selected_condition_ids"]:
            for seed in manifest["heldout_seeds"]:
                paired = [found[(cid, seed, arm)] for arm in ARMS]
                if len({r["initial_state_digest"] for r in paired}) != 1:
                    raise ValueError("First initial state mismatch")
                for field in ("noise_digest", "observation_digest"):
                    if len({r["chunks"][0][field] for r in paired}) != 1:
                        raise ValueError("First chunk " + field + " mismatch")
                scores = [r["chunks"][0]["log_sigma"][0] for r in paired]
                if max(abs(s - scores[0]) for s in scores) > tol:
                    raise ValueError("Initial sigma mismatch")
    return found


def prespecified_selection(manifest, found):
    qualified = []
    for condition in manifest["conditions"]:
        successes = sum(found[(condition["condition_id"], seed, 1)]["success"] for seed in manifest["screening_seeds"])
        if successes in (1, 2):
            qualified.append(condition)
    suites = ("libero_goal", "libero_object", "libero_10")
    ranked = {
        suite: sorted(
            (c for c in qualified if c["suite"] == suite),
            key=lambda c: (-c["severity"], 0 if c["perturbation_type"] == "Robot Initial States" else 1, c["task_id"]),
        )
        for suite in suites
    }
    selected = []
    while len(selected) < manifest["stage_1"]["max_selected_conditions"]:
        added = False
        for suite in suites:
            if ranked[suite] and len(selected) < manifest["stage_1"]["max_selected_conditions"]:
                selected.append(ranked[suite].pop(0)["condition_id"])
                added = True
        if not added:
            break
    return selected


def _screen_summary(manifest, found):
    conditions = {}
    for c in manifest["conditions"]:
        rows = [found[(c["condition_id"], seed, 1)] for seed in manifest["screening_seeds"]]
        wins = sum(row["success"] for row in rows)
        conditions[c["condition_id"]] = {
            "suite": c["suite"],
            "task_family": c["task_family"],
            "type": c["perturbation_type"],
            "severity": c["severity"],
            "successes": wins,
            "episodes": len(rows),
            "success_rate": wins / len(rows),
            "exact_ci95_exploratory": interval(wins, len(rows)),
            "qualifies": wins in (1, 2),
        }
    return {
        "stage": "screen",
        "episodes": len(found),
        "conditions": conditions,
        "qualifying_conditions": [cid for cid, value in conditions.items() if value["qualifies"]],
    }


def _comparison_summary(manifest, found, selection):
    cases = []
    for cid in selection["selected_condition_ids"]:
        c = next(item for item in manifest["conditions"] if item["condition_id"] == cid)
        for seed in manifest["heldout_seeds"]:
            rows = {arm: found[(cid, seed, arm)] for arm in ARMS}
            cases.append(
                {
                    "condition_id": cid,
                    "suite": c["suite"],
                    "task_family": c["task_family"],
                    "type": c["perturbation_type"],
                    "severity": c["severity"],
                    "seed": seed,
                    "initial_sigma": rows[1]["chunks"][0]["sigmas"][0],
                    "success": {str(arm): rows[arm]["success"] for arm in ARMS},
                }
            )
    by_condition, by_family, by_type, by_arm, ranking = {}, {}, {}, {}, {}
    for field, destination in (("condition_id", by_condition), ("task_family", by_family), ("type", by_type)):
        for value in sorted({case[field] for case in cases}):
            subset = [case for case in cases if case[field] == value]
            destination[value] = {}
            for arm in ARMS:
                wins = sum(case["success"][str(arm)] for case in subset)
                destination[value][str(arm)] = {
                    "successes": wins,
                    "episodes": len(subset),
                    "rate": wins / len(subset),
                    "exact_ci95_exploratory": interval(wins, len(subset)),
                }
                if arm != 1:
                    pair_wins = sum(case["success"][str(arm)] and not case["success"]["1"] for case in subset)
                    pair_losses = sum(case["success"]["1"] and not case["success"][str(arm)] for case in subset)
                    destination[value][str(arm)]["paired_vs_1"] = {
                        "wins": pair_wins,
                        "losses": pair_losses,
                        "ties": len(subset) - pair_wins - pair_losses,
                    }
    gate = False
    for arm in ARMS:
        rows = [found[(case["condition_id"], case["seed"], arm)] for case in cases]
        chunks = [chunk for row in rows for chunk in row["chunks"]]
        wins = sum(row["success"] for row in rows)
        by_arm[str(arm)] = {
            "successes": wins,
            "episodes": len(rows),
            "rate": wins / len(rows),
            "exact_ci95_exploratory": interval(wins, len(rows)),
            "velocity_evaluations_per_chunk": stats([chunk["velocity_evaluations"] for chunk in chunks]),
            "velocity_evaluations_per_episode": stats([row["total_velocity_evaluations"] for row in rows]),
            "scored_policy_chunk_ms": stats([chunk["policy_ms"] for chunk in chunks]),
            "http_request_chunk_ms": stats([chunk["request_ms"] for chunk in chunks]),
            "episode_ms": stats([row["episode_ms"] for row in rows]),
            "chunks": len(chunks),
        }
        if arm == 1:
            continue
        paired = [
            {
                **case,
                "win": int(case["success"][str(arm)] and not case["success"]["1"]),
                "loss": int(case["success"]["1"] and not case["success"][str(arm)]),
                "delta": int(case["success"][str(arm)]) - int(case["success"]["1"]),
            }
            for case in cases
        ]
        gain = sum(p["win"] for p in paired)
        loss = sum(p["loss"] for p in paired)
        win_conditions = sorted({p["condition_id"] for p in paired if p["win"]})
        by_arm[str(arm)]["paired_vs_1"] = {
            "wins": gain,
            "losses": loss,
            "ties": len(paired) - gain - loss,
            "net_success_rate_difference": (gain - loss) / len(paired),
            "net_difference_condition_cluster_ci95": cluster_interval(
                paired, lambda sample: float(np.mean([p["delta"] for p in sample])), seed=BOOTSTRAP_SEED + arm
            ),
            "discordant_win_fraction_exact_ci95": interval(gain, gain + loss),
            "win_conditions": win_conditions,
        }
        gate = gate or (gain >= 3 and loss <= 1 and len(win_conditions) >= 2)
        failures = [p for p in paired if not p["success"]["1"]]
        rescues = sum(p["win"] for p in failures)
        unrescued = len(failures) - rescues
        measurable = rescues >= 4 and unrescued >= 4 and len({p["condition_id"] for p in failures}) >= 2
        result = {
            "one_step_failures": len(failures),
            "rescued": rescues,
            "unrescued": unrescued,
            "estimable": measurable,
            "auroc_initial_sigma": None,
            "auroc_condition_cluster_ci95": None,
            "ranking_gate_pass": False,
        }
        if measurable:

            def fn(sample):
                return auc(np.asarray([p["initial_sigma"] for p in sample]), np.asarray([p["win"] for p in sample]))

            result["auroc_initial_sigma"] = fn(failures)
            result["auroc_condition_cluster_ci95"] = cluster_interval(failures, fn, seed=BOOTSTRAP_SEED + 100 + arm)
            ci = result["auroc_condition_cluster_ci95"]
            result["ranking_gate_pass"] = bool(result["auroc_initial_sigma"] > 0.65 and ci and ci[0] > 0.5)
        ranking[str(arm)] = result
    return {
        "stage": "compare",
        "episodes": len(found),
        "paired_cases": len(cases),
        "by_condition": by_condition,
        "by_task_family": by_family,
        "by_perturbation_type": by_type,
        "by_arm": by_arm,
        "sigma_ranking_on_one_step_failures": ranking,
        "first_chunk_cases": cases,
        "success_benefit_gate_pass": gate,
        "decision": "selective fixed-step benefit clears declared gate"
        if gate
        else "stop; no meaningful selective extra-step benefit under declared gate",
        "uncertainty_note": "Exact episode intervals are exploratory; paired differences and score AUROC use condition-cluster bootstrap. Few selected conditions yield coarse intervals.",
        "separate_measures": "No demonstration-action error or fine-step numerical reference was measured.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("smoke", "screen", "compare"))
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("two_stage_manifest.json"))
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--screen-records", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wandb-project", default="pi05-libero-velocity-residual")
    parser.add_argument("--wandb-name", required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    selection = json.loads(args.selection.read_text()) if args.selection else None
    screen_found = None
    if args.stage == "compare":
        if not args.screen_records or selection["screen_sha256"] != sha256_file(args.screen_records):
            raise ValueError("Comparison selection does not match frozen full screen")
        screen_found = validate(manifest, args.manifest, load_rows(args.screen_records), "screen")
        if selection["selected_condition_ids"] != prespecified_selection(manifest, screen_found):
            raise ValueError("Frozen selection differs from prespecified rule")
    found = validate(
        manifest, args.manifest, load_rows(args.records), args.stage, selection, args.selection, screen_found
    )
    if (
        args.stage == "screen"
        and selection
        and (
            selection["manifest_sha256"] != sha256_file(args.manifest)
            or selection["screen_sha256"] != sha256_file(args.records)
            or selection["selected_condition_ids"] != prespecified_selection(manifest, found)
        )
    ):
        raise ValueError("Screen selection file does not match prespecified rule and complete records")
    if args.stage == "screen":
        summary = _screen_summary(manifest, found)
    elif args.stage == "compare":
        summary = _comparison_summary(manifest, found, selection)
    else:
        row = next(iter(found.values()))
        summary = {
            "stage": "smoke",
            "success": row["success"],
            "episodes": 1,
            "velocity_evaluations_per_chunk": stats([c["velocity_evaluations"] for c in row["chunks"]]),
            "scored_policy_chunk_ms": stats([c["policy_ms"] for c in row["chunks"]]),
        }
    summary.update(
        manifest_sha256=sha256_file(args.manifest),
        records_sha256=sha256_file(args.records),
        benchmark_commit=manifest["benchmark_commit"],
        base_checkpoint_identity=manifest["base_checkpoint_identity"],
        head_sha256=manifest["head_sha256"],
        gpu_uuid=manifest["gpu_uuid"],
    )
    if selection:
        summary["selection_sha256"] = sha256_file(args.selection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        job_type="libero-plus-two-stage-" + args.stage,
        config={"protocol": manifest, "stage": args.stage, "manifest_sha256": summary["manifest_sha256"]},
    ) as run:
        metrics = {"episodes": summary["episodes"]}
        if args.stage == "screen":
            for cid, item in summary["conditions"].items():
                metrics["condition/" + cid + "/success_rate"] = item["success_rate"]
                metrics["condition/" + cid + "/qualifies"] = int(item["qualifies"])
        elif args.stage == "compare":
            metrics["success_benefit_gate_pass"] = int(summary["success_benefit_gate_pass"])
            for arm, item in summary["by_arm"].items():
                prefix = "fixed_" + arm + "/"
                metrics.update(
                    {
                        prefix + "success_rate": item["rate"],
                        prefix + "velocity_evals_per_chunk_mean": item["velocity_evaluations_per_chunk"]["mean"],
                        prefix + "velocity_evals_per_chunk_p95": item["velocity_evaluations_per_chunk"]["p95"],
                        prefix + "policy_chunk_ms_mean": item["scored_policy_chunk_ms"]["mean"],
                        prefix + "policy_chunk_ms_p95": item["scored_policy_chunk_ms"]["p95"],
                        prefix + "episode_ms_mean": item["episode_ms"]["mean"],
                        prefix + "episode_ms_p95": item["episode_ms"]["p95"],
                    }
                )
                if "paired_vs_1" in item:
                    metrics[prefix + "paired_wins_vs_1"] = item["paired_vs_1"]["wins"]
                    metrics[prefix + "paired_losses_vs_1"] = item["paired_vs_1"]["losses"]
            for arm, item in summary["sigma_ranking_on_one_step_failures"].items():
                if item["estimable"]:
                    metrics["sigma_ranking/" + arm + "/auroc"] = item["auroc_initial_sigma"]
        else:
            metrics["smoke_success"] = int(summary["success"])
        run.log(metrics)
        artifact = wandb.Artifact(
            "libero-plus-two-stage-" + args.stage + "-" + summary["records_sha256"][:12],
            type="evaluation",
            metadata={
                "manifest_sha256": summary["manifest_sha256"],
                "records_sha256": summary["records_sha256"],
                "checkpoint_identity": summary["base_checkpoint_identity"],
                "gpu_uuid": summary["gpu_uuid"],
            },
        )
        for path in (args.manifest, args.records, args.output, args.selection, args.screen_records):
            if path:
                artifact.add_file(str(path))
        run.log_artifact(artifact)
        summary["wandb_url"] = run.url
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "wandb_url": summary["wandb_url"]}, sort_keys=True))


if __name__ == "__main__":
    main()
