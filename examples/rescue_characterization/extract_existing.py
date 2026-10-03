"""Extract historical fixed arms without rerunning or changing prior studies."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from analysis import pair_records, summarize

ROOT = Path("/volt/code/frozen-flow-study")
SOURCES = (
    ("frozen_flow", Path("/volt/artifacts/frozen-flow-study/runs")),
    ("selective_flow", Path("/volt/artifacts/selective-flow-study/runs")),
)
EXECUTION_FIELDS = ("action_dim", "action_horizon", "batch_size", "image_size", "max_policy_steps",
                    "render_size", "replan_steps", "success", "wait_steps")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(root=ROOT, sources=SOURCES):
    rows, inventory, excluded = [], [], Counter()
    expected = set()
    for study, run_dir in sources:
        manifest_path = root / "examples" / study / "protocol.json"
        manifest = json.loads(manifest_path.read_text())
        manifest_hash = file_hash(manifest_path)
        for benchmark_name, spec in manifest["benchmarks"].items():
            for condition in spec["conditions"]:
                for case in condition["cases"]:
                    for arm in manifest["flow_steps"]:
                        expected.add((study, "main", condition["condition_id"], case["seed"], case["init_index"], arm))
            smoke_cases = ([(c["condition_id"], x) for c in spec["smoke_conditions"] for x in c["cases"]]
                           if "smoke_conditions" in spec else [(x["condition_id"], x) for x in spec["smoke_cases"]])
            for condition_id, case in smoke_cases:
                for arm in manifest["flow_steps"]:
                    expected.add((study, "smoke", condition_id, case["seed"], case["init_index"], arm))
        noise = {"policy": manifest["noise_policy"], "algorithm": manifest["settings"]["noise"]}
        execution = {k: manifest["settings"][k] for k in EXECUTION_FIELDS}
        inventory.append({"study": study, "manifest_path": str(manifest_path),
                          "manifest_sha256": manifest_hash, "checkpoint_sha256": manifest["checkpoint"]["sha256"],
                          "noise": noise, "execution": execution,
                          "benchmark_commits": {k: b["commit"] for k, b in manifest["benchmarks"].items()}})
        for path in sorted(run_dir.glob("*.jsonl")):
            if "-main-worker-" not in path.name and "-smoke.jsonl" not in path.name:
                continue
            source_hash = file_hash(path)
            for lineno, raw in enumerate(path.read_bytes().splitlines(), 1):
                row = json.loads(raw)
                if row.get("policy_arm") not in (None, "fixed_1", "fixed_2", "fixed_4", "fixed_10"):
                    excluded["non_fixed_policy"] += 1
                    continue
                if row.get("phase") not in ("main", "smoke"):
                    raise ValueError("Unexpected episode phase")
                if row.get("status") != "ok":
                    raise ValueError(f"Execution error preserved at {path}:{lineno}; resolve before analysis")
                benchmark = manifest["benchmarks"][row["benchmark"]]
                conditions = benchmark.get("smoke_conditions", benchmark["conditions"]) if row["phase"] == "smoke" else benchmark["conditions"]
                condition = next(c for c in conditions if c["condition_id"] == row["condition_id"])
                plan = condition["cases"] if row["phase"] == "main" or "smoke_conditions" in benchmark else [
                    c for c in benchmark["smoke_cases"] if c["condition_id"] == row["condition_id"]]
                declared = [c for c in plan if (c["seed"], c["init_index"]) == (row["seed"], row["init_index"])]
                if len(declared) != 1:
                    raise ValueError("Episode absent from unique declared case")
                for field, expected_value in (
                    ("manifest_sha256", manifest_hash), ("checkpoint_sha256", manifest["checkpoint"]["sha256"]),
                    ("benchmark_commit", benchmark["commit"]), ("initial_state_sha256", declared[0]["initial_state_sha256"]),
                    ("task_name", condition["task_name"]), ("suite", condition["suite"]),
                    ("family", condition["family"]), ("category", condition["category"]),
                ):
                    if row.get(field) != expected_value:
                        raise ValueError(f"Identity mismatch {field} at {path}:{lineno}")
                if row["gpu_uuid"] not in manifest["gpu_uuids"]:
                    raise ValueError("Undeclared GPU")
                row["severity"] = condition["severity"]
                row["study"] = study
                row["noise_policy_sha256"] = digest(noise)
                row["execution_sha256"] = digest(execution)
                row["bddl_sha256"] = condition["bddl_sha256"]
                row["prompt"] = condition["prompt"]
                # Deliberately no study/phase/manifest identifier: detect cross-study overlap.
                physical = {k: row[k] for k in ("benchmark", "benchmark_commit", "checkpoint_sha256", "suite",
                            "task_name", "seed", "init_index", "initial_state_sha256",
                            "bddl_sha256", "prompt", "noise_policy_sha256", "execution_sha256")}
                row["physical_case_id"] = digest(physical)
                row["source_refs"] = [{"path": str(path), "line": lineno,
                                       "file_sha256": source_hash, "record_sha256": hashlib.sha256(raw).hexdigest()}]
                # Historical rows retain only hashes, not action arrays or videos.
                for key in ("head_sha256", "initial_sigma", "initial_log_sigma", "policy_arm"):
                    row.pop(key, None)
                rows.append(row)
    observed = {(r["study"], r["phase"], r["condition_id"], r["seed"], r["init_index"], r["flow_steps"]) for r in rows}
    if observed != expected:
        raise ValueError(f"Incomplete or unexpected source coverage: missing={len(expected-observed)}, extra={len(observed-expected)}")
    cases = pair_records(rows)
    for case in cases:
        wanted = {1, 10} if case["study"] == "selective_flow" else {1, 2, 4, 10}
        if set(map(int, case["arms"])) != wanted:
            raise ValueError("Incomplete physical pairing despite complete logical coverage")
    audit = {"source_inventories": inventory, "episode_records": len(rows), "case_records": len(cases),
             "unique_cases": sum(c["include_unique_population"] for c in cases),
             "duplicate_cases": [{"case_id": c["case_id"], "duplicate_of": c["duplicate_of"]} for c in cases if c["duplicate_of"]],
             "excluded_records": dict(excluded),
             "notes": ["All original main and smoke fixed-arm JSONL files are enumerated.",
                       "Main/smoke and study strata remain separate. No cross-study population pooling.",
                       "Historical action hashes are not numerical action traces; videos must be regenerated as audited replays.",
                       "Independent study fixed arms used head instrumentation; head outputs are not predictors in this analysis.",
                       "Manifest identities verified against raw rows; hardware/checkpoint disk validation is a separate audit."]}
    return cases, audit


def replay_selection(cases):
    population = [c for c in cases if c["phase"] == "main" and c["include_unique_population"]]
    chosen = []
    for case in population:
        if case["study"] == "frozen_flow" and (case["success_vector"][0] is False or case["regression_steps"]):
            chosen.append(case)
        elif case["study"] == "selective_flow" and (case["rescue_steps"] or case["regression_steps"]):
            chosen.append(case)
    never = [c for c in population if c["study"] == "selective_flow"
             and c["success_vector"][0] is False and not c["rescue_steps"]]
    # Exactly 40 available; deterministic condition/seed order if the cohort grows.
    chosen.extend(sorted(never, key=lambda c: (c["condition_id"], c["seed"], c["init_index"]))[:40])
    entries = []
    for case in sorted(chosen, key=lambda c: (c["study"], c["benchmark"], c["condition_id"], c["seed"])):
        entry = dict(case)
        entry["replay_steps"] = sorted(int(k) for k in case["arms"])
        entry["selection_reason"] = "rescue" if case["rescue_steps"] else "regression" if case["regression_steps"] else "never_rescued"
        entries.append(entry)
    return {"purpose": "Outcome-selected historical replay for failure review only, excluded from new population rates",
            "selection": "All old four-arm one-step failures and any one-step regression; independent all 10-step rescues and regressions plus up to 40 never-rescued in deterministic condition/seed order",
            "unknown_first_success_preserved": True, "maximum_replay_episodes": sum(len(c["replay_steps"]) for c in entries),
            "case_count": len(entries), "cases": entries}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/volt/artifacts/rescue-characterization/existing"))
    parser.add_argument("--repo-output", type=Path, default=ROOT / "examples/rescue_characterization/results/existing")
    parser.add_argument("--replicates", type=int, default=10000)
    args = parser.parse_args()
    cases, audit = extract()
    summary = summarize(cases, args.replicates)
    for output in (args.output, args.repo_output):
        output.mkdir(parents=True, exist_ok=True)
        for filename, data in (("audit.json", audit), ("summary.json", summary)):
            (output / filename).write_text(json.dumps(data, indent=2) + "\n")
        # Compact episode provenance plus full vectors, never duplicate chunk records.
        (output / "cases.jsonl").write_text("".join(json.dumps(c, separators=(",", ":")) + "\n" for c in cases))
        selected = [c for c in cases if c["success_vector"][0] is False or c["regression_steps"]]
        (output / "failures-and-regressions.jsonl").write_text("".join(json.dumps(c, separators=(",", ":")) + "\n" for c in selected))
    print(json.dumps({key: {k: v for k, v in item.items() if k not in ("distributions", "uncertainty", "family_sensitivity")}
                      for key, item in summary.items()}, indent=2))


if __name__ == "__main__":
    main()

