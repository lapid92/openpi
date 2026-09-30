"""Two-stage frozen pi0.5 LIBERO-Plus evaluation (simulator Python 3.8)."""

import argparse
import hashlib
import json
import math
import pathlib
import subprocess
import urllib.request

from libero.libero import benchmark
from pilot_client import run_episode

ARMS = (1, 2, 4, 10)
GPU_UUID = "GPU-4779cdde-a260-f8ec-da6a-6fa390bc7fd7"


def sha256_file(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def validate_manifest(manifest, classification):
    if manifest["stage_1"]["flow_steps"] != [1] or tuple(manifest["stage_2"]["flow_steps"]) != ARMS:
        raise ValueError("Flow schedules changed")
    if len(manifest["conditions"]) != 12 or manifest["stage_1"]["expected_episodes"] != 36:
        raise ValueError("Screen size changed")
    screen, heldout = set(manifest["screening_seeds"]), set(manifest["heldout_seeds"])
    if len(screen) != 3 or len(heldout) != 4 or screen & heldout or manifest["smoke"]["seed"] in screen | heldout:
        raise ValueError("Seeds overlap")
    ids = set()
    for c in manifest["conditions"]:
        if c["condition_id"] in ids:
            raise ValueError("Duplicate condition")
        ids.add(c["condition_id"])
        entry = classification[c["suite"]][c["task_index"]]
        if (entry["id"], entry["name"], entry["category"], entry["difficulty_level"]) != (
            c["task_id"],
            c["task_name"],
            c["perturbation_type"],
            c["severity"],
        ):
            raise ValueError("Classification mismatch: " + c["condition_id"])
        if c["task_index"] != c["task_id"] - 1 or c["init_state_index"] != 0:
            raise ValueError("Task or initial state index mismatch")
    if manifest["smoke"]["condition_id"] not in ids:
        raise ValueError("Smoke condition absent")
    return True


def planned_screen(manifest):
    return [(c["condition_id"], seed, 1) for c in manifest["conditions"] for seed in manifest["screening_seeds"]]


def planned_compare(manifest, selection):
    return [
        (cid, seed, arm)
        for cid in selection["selected_condition_ids"]
        for seed in manifest["heldout_seeds"]
        for arm in ARMS
    ]


def _validate_records(manifest, records, planned, manifest_hash=None):
    expected = set(planned)
    found = {}
    conditions = {c["condition_id"]: c for c in manifest["conditions"]}
    for row in records:
        key = (row.get("condition_id"), row.get("seed"), row.get("flow_steps"))
        if key not in expected or key in found:
            raise ValueError("Unexpected or duplicate episode: " + repr(key))
        c = conditions[key[0]]
        for field, value in (
            ("task_id", c["task_id"]),
            ("task_name", c["task_name"]),
            ("category", c["perturbation_type"]),
            ("suite", c["suite"]),
            ("checkpoint_identity", manifest["base_checkpoint_identity"]),
            ("head_checkpoint", manifest["head_sha256"]),
            ("benchmark_commit", manifest["benchmark_commit"]),
            ("gpu_uuid", GPU_UUID),
        ):
            if row.get(field) != value:
                raise ValueError("Episode identity mismatch: " + field)
        if manifest_hash and row.get("manifest_sha256") != manifest_hash:
            raise ValueError("Protocol hash mismatch")
        if row.get("status") != "ok" or type(row.get("success")) is not bool:
            raise ValueError("Incomplete episode")
        if row.get("stage") != (
            "smoke"
            if key[1] == manifest["smoke"]["seed"]
            else "screen"
            if key[2] == 1 and key[1] in manifest["screening_seeds"]
            else "compare"
        ):
            raise ValueError("Stage identity mismatch")
        if row.get("task_family") != c["task_family"] or row.get("severity") != c["severity"]:
            raise ValueError("Condition metadata mismatch")
        chunks = row.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("Missing chunks")
        for i, chunk in enumerate(chunks):
            if chunk.get("chunk_index") != i or chunk.get("velocity_evaluations") != key[2]:
                raise ValueError("Velocity accounting mismatch")
            if len(chunk.get("sigmas", [])) != key[2] or len(chunk.get("times", [])) != key[2]:
                raise ValueError("Score or time accounting mismatch")
            expected_times = [1.0 - j / key[2] for j in range(key[2])]
            if any(
                not math.isfinite(t) or abs(t - target) > 1e-5
                for j, t in enumerate(chunk["times"])
                for target in (expected_times[j],)
            ):
                raise ValueError("Euler times mismatch")
            sigmas, logs = chunk["sigmas"], chunk.get("log_sigma", [])
            if (
                len(logs) != key[2]
                or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in logs)
                or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in sigmas)
                or any(
                    abs(math.exp(log) - sigma) > 1e-6 * max(1.0, sigma)
                    for j, log in enumerate(logs)
                    for sigma in (sigmas[j],)
                )
            ):
                raise ValueError("Sigma values invalid")
            for field in ("noise_digest", "action_digest", "observation_digest"):
                if len(chunk.get(field, "")) != 64:
                    raise ValueError("Missing digest: " + field)
            policy_ms, request_ms = chunk.get("policy_ms"), chunk.get("request_ms")
            if (
                any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in (policy_ms, request_ms))
                or request_ms + 1e-3 < policy_ms
            ):
                raise ValueError("Invalid latency")
        if row.get("total_velocity_evaluations") != len(chunks) * key[2]:
            raise ValueError("Total velocity count mismatch")
        if len(row.get("initial_state_digest", "")) != 64:
            raise ValueError("Missing initial state digest")
        if (
            not isinstance(row.get("episode_ms"), (int, float))
            or not math.isfinite(row["episode_ms"])
            or row["episode_ms"] < 0
            or not 1 <= row.get("policy_steps", 0) <= c["max_policy_steps"]
        ):
            raise ValueError("Invalid episode timing or horizon")
        found[key] = row
    if set(found) != expected:
        raise ValueError("Missing episodes: " + repr(sorted(expected - set(found))))
    return found


def validate_screen_records(manifest, records, manifest_hash=None):
    return _validate_records(manifest, records, planned_screen(manifest), manifest_hash)


def select_conditions(manifest, screen_records):
    records = validate_screen_records(manifest, screen_records)
    qualified = []
    for c in manifest["conditions"]:
        successes = sum(records[(c["condition_id"], seed, 1)]["success"] for seed in manifest["screening_seeds"])
        if successes in (1, 2):
            qualified.append(c)
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


def validate_comparison_records(manifest, selection, records, manifest_hash=None, selection_hash=None):
    selected = selection["selected_condition_ids"]
    if not selected:
        raise ValueError("No selected conditions; comparison must stop")
    if len(set(selected)) != len(selected) or len(selected) > manifest["stage_1"]["max_selected_conditions"]:
        raise ValueError("Invalid selection")
    found = _validate_records(manifest, records, planned_compare(manifest, selection), manifest_hash)
    if selection_hash and any(row.get("selection_sha256") != selection_hash for row in found.values()):
        raise ValueError("Selection hash mismatch")
    tolerance = manifest["execution"]["first_score_log_sigma_tolerance"]
    for cid in selected:
        for seed in manifest["heldout_seeds"]:
            rows = [found[(cid, seed, arm)] for arm in ARMS]
            if len({r["initial_state_digest"] for r in rows}) != 1:
                raise ValueError("Initial states differ across arms")
            for field in ("noise_digest", "observation_digest"):
                if len({r["chunks"][0][field] for r in rows}) != 1:
                    raise ValueError("First " + field + " differs across arms")
            reference = rows[0]["chunks"][0]["log_sigma"][0]
            if max(abs(r["chunks"][0]["log_sigma"][0] - reference) for r in rows) > tolerance:
                raise ValueError("First score differs across arms")
    return found


def load_records(path):
    return [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line.strip()]


def _task(suite, condition):
    task = suite.get_task(condition["task_index"])
    if task.name != condition["task_name"]:
        raise ValueError("Task order changed")
    result = task._asdict()
    result.update(
        id=condition["task_id"],
        category=condition["perturbation_type"],
        index=condition["task_index"],
        init_state_index=condition["init_state_index"],
    )
    return result


def _check_runtime(manifest, classification_path, server):
    if sha256_file(classification_path) != manifest["classification_sha256"]:
        raise ValueError("Classification hash changed")
    validate_manifest(manifest, json.loads(pathlib.Path(classification_path).read_text()))
    repo = pathlib.Path(classification_path).resolve().parents[3]
    if repo not in pathlib.Path(benchmark.__file__).resolve().parents:
        raise ValueError("Wrong benchmark import")
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != manifest["benchmark_commit"]:
        raise ValueError("Benchmark commit changed")
    if sha256_file(repo.parent / "assets.zip") != manifest["benchmark_assets_sha256"]:
        raise ValueError("Benchmark assets changed")
    with urllib.request.urlopen(server + "/health", timeout=180) as response:
        health = json.load(response)
    if (health["gpu_uuid"], health["base_checkpoint_identity"], health["head_sha256"], health["head_step"]) != (
        GPU_UUID,
        manifest["base_checkpoint_identity"],
        manifest["head_sha256"],
        3000,
    ):
        raise ValueError("GPU or frozen checkpoint identity changed")


def _load_selection(manifest, manifest_hash, screen_path, selection_path):
    records = load_records(screen_path)
    validate_screen_records(manifest, records, manifest_hash)
    selection = json.loads(pathlib.Path(selection_path).read_text())
    if selection["manifest_sha256"] != manifest_hash or selection["screen_sha256"] != sha256_file(screen_path):
        raise ValueError("Selection is not tied to complete screen")
    if selection["selected_condition_ids"] != select_conditions(manifest, records):
        raise ValueError("Selection differs from prespecified rule")
    return selection


def _append_episode(path, record):
    with path.open("a") as handle:
        handle.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
        handle.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("smoke", "screen", "select", "compare"))
    parser.add_argument(
        "--manifest", type=pathlib.Path, default=pathlib.Path(__file__).with_name("two_stage_manifest.json")
    )
    parser.add_argument("--classification", type=pathlib.Path)
    parser.add_argument("--server", default="http://127.0.0.1:8765")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--screen-records", type=pathlib.Path)
    parser.add_argument("--selection", type=pathlib.Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    manifest_hash = sha256_file(args.manifest)
    if args.stage == "select":
        if not args.screen_records:
            parser.error("--screen-records required for select")
        records = load_records(args.screen_records)
        validate_screen_records(manifest, records, manifest_hash)
        selection = {
            "manifest_sha256": manifest_hash,
            "screen_sha256": sha256_file(args.screen_records),
            "selected_condition_ids": select_conditions(manifest, records),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists() and json.loads(args.output.read_text()) != selection:
            raise ValueError("Existing selection differs")
        args.output.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
        print(json.dumps(selection, sort_keys=True), flush=True)
        return
    if not args.classification:
        parser.error("--classification required")
    if args.stage == "compare" and (not args.screen_records or not args.selection):
        parser.error("--screen-records and --selection required")
    _check_runtime(manifest, args.classification, args.server)
    selection = (
        _load_selection(manifest, manifest_hash, args.screen_records, args.selection)
        if args.stage == "compare"
        else None
    )
    by_id = {c["condition_id"]: c for c in manifest["conditions"]}
    suites = {}
    if args.stage == "smoke":
        planned = [(manifest["smoke"]["condition_id"], manifest["smoke"]["seed"], manifest["smoke"]["flow_steps"])]
    elif args.stage == "screen":
        planned = planned_screen(manifest)
    else:
        planned = planned_compare(manifest, selection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prior = load_records(args.output) if args.output.exists() else []
    completed = set()
    if prior:
        _validate_records(
            manifest, prior, [(r["condition_id"], r["seed"], r["flow_steps"]) for r in prior], manifest_hash
        )
        if selection and any(r.get("selection_sha256") != sha256_file(args.selection) for r in prior):
            raise ValueError("Resume selection hash mismatch")
    for row in prior:
        if row.get("manifest_sha256") != manifest_hash or row.get("stage") != args.stage:
            raise ValueError("Cannot resume incompatible episode file")
        key = (row["condition_id"], row["seed"], row["flow_steps"])
        if key not in planned or key in completed or row["status"] != "ok":
            raise ValueError("Cannot resume duplicate, failed, or unexpected episode")
        completed.add(key)
    for cid, seed, arm in planned:
        if (cid, seed, arm) in completed:
            continue
        c = by_id[cid]
        if c["suite"] not in suites:
            suites[c["suite"]] = benchmark.get_benchmark_dict()[c["suite"]](task_order_index=0)
        suite = suites[c["suite"]]
        task = _task(suite, c)
        settings = {
            "render_size": manifest["execution"]["render_size"],
            "image_size": manifest["execution"]["image_size"],
            "wait_steps": manifest["execution"]["wait_steps"],
            "replan_steps": manifest["execution"]["replan_steps"],
            "max_policy_steps": c["max_policy_steps"],
            "suite": c["suite"],
        }
        try:
            record = run_episode(suite, task, seed, arm, settings, args.server)
        except Exception as error:
            record = {
                "task_id": c["task_id"],
                "task_name": c["task_name"],
                "category": c["perturbation_type"],
                "seed": seed,
                "flow_steps": arm,
                "status": "error",
                "error": repr(error),
            }
        record.update(
            stage=args.stage,
            condition_id=cid,
            suite=c["suite"],
            task_family=c["task_family"],
            severity=c["severity"],
            manifest_sha256=manifest_hash,
            benchmark_commit=manifest["benchmark_commit"],
            checkpoint_identity=manifest["base_checkpoint_identity"],
            head_checkpoint=manifest["head_sha256"],
            gpu_uuid=GPU_UUID,
        )
        if selection:
            record["selection_sha256"] = sha256_file(args.selection)
        _append_episode(args.output, record)
        print(
            json.dumps(
                {
                    k: record.get(k)
                    for k in ("stage", "condition_id", "seed", "flow_steps", "success", "status", "error")
                }
            ),
            flush=True,
        )
        if record["status"] != "ok":
            raise RuntimeError("Episode failed; stopped for inspection")


if __name__ == "__main__":
    main()
