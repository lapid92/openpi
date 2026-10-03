"""Retrospective fixed-arm analysis. No adaptive policy or predictor is fitted."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

STEPS = (1, 2, 4, 10)


def classify(success_by_step):
    outcomes = {int(k): v for k, v in success_by_step.items()}
    if any(k not in STEPS or type(v) is not bool for k, v in outcomes.items()):
        raise ValueError("Expected Boolean outcomes for fixed 1/2/4/10 arms")
    vector = [outcomes.get(k) for k in STEPS]
    complete = all(v is not None for v in vector)
    successes = [k for k in STEPS if outcomes.get(k) is True]
    # Missing lower arms forbid assigning the first successful step.
    first = "unknown"
    if successes and all(k in outcomes for k in STEPS if k <= successes[0]):
        first = str(successes[0])
    elif complete:
        first = "never"
    tested = [outcomes[k] for k in STEPS if k in outcomes]
    nonmonotonic = any(tested[i] and not tested[j] for i in range(len(tested)) for j in range(i + 1, len(tested)))
    return {
        "success_vector": vector,
        "pattern": "/".join("unknown" if v is None else "succeed" if v else "fail" for v in vector),
        "complete_four_arm": complete, "first_success": first,
        "first_tested_success": str(successes[0]) if successes else ("never" if complete else "unknown"),
        "persists_at_larger_tested": all(outcomes[k] for k in outcomes if k > successes[0]) if successes else None,
        "nonmonotonic": nonmonotonic,
        "rescue_steps": [k for k in STEPS[1:] if outcomes.get(1) is False and outcomes.get(k) is True],
        "regression_steps": [k for k in STEPS[1:] if outcomes.get(1) is True and outcomes.get(k) is False],
    }


def pair_records(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["study"], row["phase"], row["physical_case_id"])].append(row)
    cases = []
    for (study, phase, physical_id), members in sorted(groups.items()):
        arms = {}
        for row in members:
            if row.get("status") != "ok":
                raise ValueError("Execution error must not be scored as failure")
            if type(row.get("success")) is not bool:
                raise ValueError("Non-Boolean success")
            arm = row["flow_steps"]
            if arm not in STEPS:
                raise ValueError("Non-fixed arm")
            if arm in arms:
                # Exact source duplicates are retained as references, never scored twice.
                comparable = ("success", "initial_state_sha256", "stabilized_state_sha256", "chunks",
                              "checkpoint_sha256", "benchmark_commit", "noise_policy_sha256", "execution_sha256")
                if any(row.get(k) != arms[arm].get(k) for k in comparable):
                    raise ValueError("Conflicting duplicate arm")
                arms[arm]["source_refs"].extend(row["source_refs"])
            else:
                arms[arm] = dict(row, source_refs=list(row.get("source_refs", [])))
        reference = next(iter(arms.values()))
        for row in arms.values():
            for key in ("checkpoint_sha256", "benchmark_commit", "initial_state_sha256", "stabilized_state_sha256", "noise_policy_sha256", "execution_sha256"):
                if row.get(key) != reference.get(key):
                    raise ValueError("Pair mismatch: " + key)
            if not row.get("chunks") or not reference.get("chunks"):
                raise ValueError("Missing chunk evidence")
            if row["chunks"][0]["observation_sha256"] != reference["chunks"][0]["observation_sha256"]:
                raise ValueError("Initial observation mismatch")
            for a, b in zip(row["chunks"], reference["chunks"]):
                if a["noise_sha256"] != b["noise_sha256"]:
                    raise ValueError("Noise pairing mismatch")
            if any(c["velocity_evaluations"] != row["flow_steps"] for c in row["chunks"]):
                raise ValueError("Fixed sampler accounting mismatch")
        omit = {"chunks", "success", "flow_steps", "source_refs", "episode_ms", "policy_steps", "total_velocity_evaluations", "gpu_uuid", "status"}
        case = {k: v for k, v in reference.items() if k not in omit}
        case.update(classify({k: r["success"] for k, r in arms.items()}))
        case["case_id"] = hashlib.sha256(json.dumps([study, phase, physical_id]).encode()).hexdigest()
        case["arms"] = {str(k): {"success": r["success"], "source_refs": r.get("source_refs", []),
                                "policy_steps": r.get("policy_steps"), "chunks_count": len(r["chunks"]),
                                "video_path": r.get("recording", {}).get("video_path", r.get("video_path")),
                                "trace_path": r.get("recording", {}).get("trace_path", r.get("trace_path")),
                                "recording": r.get("recording")} for k, r in sorted(arms.items())}
        cases.append(case)
    # Main has precedence over a repeat smoke physical case. Cohorts remain separate.
    seen = {}
    for case in sorted(cases, key=lambda c: (c["phase"] != "main", c["study"], c["case_id"])):
        physical_id = case["physical_case_id"]
        case["duplicate_of"] = seen.get(physical_id)
        case["include_unique_population"] = physical_id not in seen
        seen.setdefault(physical_id, case["case_id"])
    return cases


def counts(cases):
    failures = [c for c in cases if c["success_vector"][0] is False]
    successes = [c for c in cases if c["success_vector"][0] is True]
    first = Counter(c["first_success"] for c in failures)
    result = {"cases": len(cases), "one_step_failures": len(failures), "one_step_successes": len(successes),
              "complete_four_arm": sum(c["complete_four_arm"] for c in cases),
              "patterns": dict(sorted(Counter(c["pattern"] for c in cases).items())),
              "first_success_among_one_step_failures": dict(sorted(first.items())),
              "first_success_rates_among_one_step_failures": {k: v / len(failures) for k, v in sorted(first.items())},
              "first_success_rates_among_all_cases": {k: v / len(cases) for k, v in sorted(first.items())},
              "nonmonotonic": sum(c["nonmonotonic"] for c in cases)}
    result["steps"] = {}
    for i, step in enumerate(STEPS[1:], 1):
        paired = [c for c in cases if c["success_vector"][0] is not None and c["success_vector"][i] is not None]
        eligible_failures = sum(c["success_vector"][0] is False for c in paired)
        eligible_successes = sum(c["success_vector"][0] is True for c in paired)
        rescues = sum(step in c["rescue_steps"] for c in paired)
        regressions = sum(step in c["regression_steps"] for c in paired)
        result["steps"][str(step)] = {
            "paired_cases": len(paired), "rescues": rescues, "regressions": regressions,
            "eligible_failures": eligible_failures, "eligible_successes": eligible_successes,
            "rescue_rate_among_failures": rescues / eligible_failures if eligible_failures else None,
            "regression_rate_among_successes": regressions / eligible_successes if eligible_successes else None,
            "net_success_difference": (rescues - regressions) / len(paired) if paired else None}
    return result


def cluster_intervals(cases, cluster="condition_id", replicates=2000, seed=20261003):
    groups = defaultdict(list)
    for case in cases:
        groups[(case.get("suite"), case.get(cluster))].append(case)
    if len(groups) < 2:
        return {"cluster": cluster, "clusters": len(groups), "intervals": None,
                "reason": "Fewer than two clusters; no interval"}
    rng = random.Random(seed)
    clusters = list(groups.values())
    values = defaultdict(list)
    for _ in range(replicates):
        sample = [case for _ in clusters for case in rng.choice(clusters)]
        result = counts(sample)
        for k in ("2", "4", "10", "never"):
            nfail = result["one_step_failures"]
            if result["complete_four_arm"] == result["cases"]:
                n = result["first_success_among_one_step_failures"].get(k, 0)
                if nfail:
                    values["first_" + k + "_among_failures"].append(n / nfail)
                values["first_" + k + "_among_all"].append(n / result["cases"])
        for step, metrics in result["steps"].items():
            for metric in ("rescue_rate_among_failures", "regression_rate_among_successes", "net_success_difference"):
                if metrics[metric] is not None:
                    values[step + "_" + metric].append(metrics[metric])
    intervals = {}
    for key, vals in values.items():
        vals.sort()
        intervals[key] = {"lower": vals[int(.025 * (len(vals) - 1))], "upper": vals[int(.975 * (len(vals) - 1))],
                          "valid_replicates": len(vals)}
    return {"cluster": cluster, "clusters": len(groups), "replicates": replicates, "seed": seed,
            "method": "Paired percentile cluster bootstrap; undefined-denominator replicates omitted; exploratory, no multiplicity correction",
            "intervals": intervals}


def summarize(cases, replicates=2000, seed=20261003):
    strata = defaultdict(list)
    for case in cases:
        if case.get("include_unique_population", True):
            strata[(case["study"], case["phase"], case["benchmark"])].append(case)
    result = {}
    for key, cohort in sorted(strata.items()):
        item = counts(cohort)
        item["uncertainty"] = cluster_intervals(cohort, "condition_id", replicates, seed)
        item["family_sensitivity"] = cluster_intervals(cohort, "family", replicates, seed + 1)
        item["primary_uncertainty"] = "family_sensitivity" if key[2] == "libero_plus" else "uncertainty"
        item["distributions"] = {}
        for field in ("suite", "family", "category", "severity", "condition_id"):
            groups = defaultdict(list)
            for case in cohort:
                groups[str(case.get(field, "unknown"))].append(case)
            item["distributions"][field] = {label: counts(group) for label, group in sorted(groups.items())}
        result["/".join(key)] = item
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--benchmark", choices=("libero", "libero_plus"))
    parser.add_argument("--records", type=Path, nargs="+")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10000)
    args = parser.parse_args()
    if args.cases:
        cases = [json.loads(line) for line in args.cases.read_text().splitlines()]
    else:
        if not args.manifest or not args.benchmark or not args.records:
            parser.error("Supply --cases or --manifest --benchmark --records")
        import analyze
        from extract_existing import EXECUTION_FIELDS, digest
        manifest = json.loads(args.manifest.read_text())
        manifest_hash = analyze.digest_file(args.manifest)
        rows = []
        for path in args.records:
            file_hash = analyze.digest_file(path)
            for lineno, raw in enumerate(path.read_bytes().splitlines(), 1):
                row = json.loads(raw)
                row["source_refs"] = [{"path": str(path), "line": lineno, "file_sha256": file_hash,
                                       "record_sha256": hashlib.sha256(raw).hexdigest()}]
                rows.append(row)
        # Reuse independent strict completeness/noise/accounting/parity audit.
        validated, errors = analyze.validate(manifest, rows, args.benchmark, manifest_hash, smoke=args.smoke)
        if errors:
            args.output.with_suffix(".error-attempts.json").write_text(json.dumps(errors, indent=2) + "\n")
        rows = list(validated.values())
        conditions = {c["condition_id"]: c for c in manifest["benchmarks"][args.benchmark]["conditions"]}
        for row in rows:
            condition = conditions[row["condition_id"]]
            row["study"] = manifest["study"]
            if row.get("severity", condition["severity"]) != condition["severity"]:
                raise ValueError("Severity mismatch")
            if row.get("prompt", condition["prompt"]) != condition["prompt"]:
                raise ValueError("Prompt mismatch")
            row["severity"] = condition["severity"]
            row["noise_policy_sha256"] = digest({"policy": manifest["noise_policy"], "algorithm": manifest["settings"]["noise"]})
            row["execution_sha256"] = digest({k: manifest["settings"][k] for k in EXECUTION_FIELDS})
            row["bddl_sha256"] = condition["bddl_sha256"]
            row["prompt"] = condition["prompt"]
            row["physical_case_id"] = digest({k: row[k] for k in ("benchmark", "benchmark_commit", "checkpoint_sha256",
                "suite", "task_name", "seed", "init_index", "initial_state_sha256", "bddl_sha256", "prompt",
                "noise_policy_sha256", "execution_sha256")})
        cases = pair_records(rows)
        vector_path = args.output.with_suffix(".cases.jsonl")
        vector_path.write_text("".join(json.dumps(c, separators=(",", ":")) + "\n" for c in cases))
    args.output.write_text(json.dumps(summarize(cases, args.replicates), indent=2) + "\n")


if __name__ == "__main__":
    main()
