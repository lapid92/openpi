# ruff: noqa: C408, PERF401, PLW2901
"""Post-hoc grouping of complete audited outcomes; no new rollouts or trained policy."""

import argparse
from collections import Counter
from collections import defaultdict
import csv
import json
from pathlib import Path
import subprocess
import sys

from analyze import ARMS
from analyze import analyze
from analyze import cluster_interval
from analyze import digest_file
from analyze import validate
import numpy as np

INPUT_COMMIT = "8d4aa1bd16b21f24f273a785ff2ea34be2a5a4be"
DIMS = {
    "family": "family",
    "task_identity": "task_name",
    "perturbation_type": "category",
    "severity": "severity",
    "condition": "condition_id",
}


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interval(cases, differences):
    groups = defaultdict(list)
    for case, delta in zip(cases, differences, strict=True):
        groups[case["condition_id"]].append(delta)
    if len(groups) < 2:
        return None
    sums = np.array([sum(v) for v in groups.values()])
    counts = np.array([len(v) for v in groups.values()])
    idx = np.random.default_rng(20261002).integers(0, len(groups), (10000, len(groups)))
    return np.quantile(sums[idx].sum(axis=1) / counts[idx].sum(axis=1), [0.025, 0.975]).tolist()


def has_sigma(value):
    if isinstance(value, dict):
        return any("sigma" in k.lower() or "residual" in k.lower() or has_sigma(v) for k, v in value.items())
    return isinstance(value, list) and any(has_sigma(v) for v in value)


def grouping(cases, dimension):
    result = defaultdict(list)
    for case in cases:
        result[str(case[DIMS[dimension]])].append(case)
    return result


def group_rows(benchmark, cases):
    rows = []
    for dimension in DIMS:
        for group, selected in sorted(grouping(cases, dimension).items()):
            n = len(selected)
            failures = sum(not c["success"]["1"] for c in selected)
            for arm in ARMS:
                successes = sum(c["success"][str(arm)] for c in selected)
                rescues = sum(not c["success"]["1"] and c["success"][str(arm)] for c in selected)
                regressions = sum(c["success"]["1"] and not c["success"][str(arm)] for c in selected)
                ci = cluster_interval(selected, arm) if arm != 1 else None
                rows.append(
                    dict(
                        benchmark=benchmark,
                        dimension=dimension,
                        group=group,
                        suites=";".join(sorted({c["suite"] for c in selected})),
                        families=";".join(sorted({c["family"] for c in selected})),
                        task_identities=";".join(sorted({c["task_name"] for c in selected})),
                        perturbation_types=";".join(sorted({c["category"] for c in selected})),
                        severities=";".join(sorted({str(c["severity"]) for c in selected})),
                        condition_ids=";".join(sorted({c["condition_id"] for c in selected})),
                        pairs=n,
                        conditions=len({c["condition_id"] for c in selected}),
                        arm=arm,
                        successes=successes,
                        success_rate=successes / n,
                        one_step_failures=failures,
                        rescues=rescues,
                        regressions=regressions,
                        net_success_gain=(rescues - regressions) / n,
                        rescue_fraction=rescues / failures if failures else None,
                        ci95_low=ci[0] if ci else None,
                        ci95_high=ci[1] if ci else None,
                    )
                )
    return rows


def loco(benchmark, cases, dimension):
    heldout = []
    for cid in sorted({c["condition_id"] for c in cases}):
        test = [c for c in cases if c["condition_id"] == cid]
        value = test[0][DIMS[dimension]]
        train = [c for c in cases if c["condition_id"] != cid and c[DIMS[dimension]] == value]
        gains = {a: sum(int(c["success"][str(a)]) - int(c["success"]["1"]) for c in train) for a in ARMS[1:]}
        best = max(ARMS[1:], key=lambda a: (gains[a], -a))
        chosen = best if train and gains[best] > 0 else 1
        assert all(c["condition_id"] != cid for c in train)
        for c in test:
            heldout.append(
                dict(
                    benchmark=benchmark,
                    dimension=dimension,
                    condition_id=cid,
                    seed=c["seed"],
                    init_index=c["init_index"],
                    group=str(value),
                    training_condition_ids=";".join(sorted({x["condition_id"] for x in train})),
                    training_pairs=len(train),
                    training_conditions=len({x["condition_id"] for x in train}),
                    training_gain_2=gains[2],
                    training_gain_4=gains[4],
                    training_gain_10=gains[10],
                    chosen_arm=chosen,
                    success=int(c["success"][str(chosen)]),
                    fixed1=int(c["success"]["1"]),
                    fixed10=int(c["success"]["10"]),
                )
            )
    ordered = {(c["condition_id"], c["seed"], c["init_index"]): c for c in cases}
    aligned = [ordered[(r["condition_id"], r["seed"], r["init_index"])] for r in heldout]
    summary = dict(
        benchmark=benchmark,
        dimension=dimension,
        pairs=len(heldout),
        conditions=len({r["condition_id"] for r in heldout}),
        heldout_pairs_with_training_support=sum(r["training_conditions"] > 0 for r in heldout),
        heldout_conditions_with_training_support=len(
            {r["condition_id"] for r in heldout if r["training_conditions"] > 0}
        ),
        singleton_condition_groups=len({r["condition_id"] for r in heldout if r["training_conditions"] == 0}),
        chosen_arm_counts=dict(sorted(Counter(str(r["chosen_arm"]) for r in heldout).items())),
        successes=sum(r["success"] for r in heldout),
        success_rate=sum(r["success"] for r in heldout) / len(heldout),
    )
    for baseline in ("fixed1", "fixed10"):
        deltas = [r["success"] - r[baseline] for r in heldout]
        summary[baseline] = dict(
            successes=sum(r[baseline] for r in heldout),
            successes_gained=sum(deltas),
            net_difference=float(np.mean(deltas)),
            condition_cluster_ci95=interval(aligned, deltas),
            rescues=sum(r["success"] and not r[baseline] for r in heldout),
            regressions=sum(not r["success"] and r[baseline] for r in heldout),
        )
    return summary, heldout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.manifest = args.manifest.resolve()
    args.results = args.results.resolve()
    args.output = args.output.resolve()
    root = Path(__file__).resolve().parents[2]
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    execution_commit = head
    head = INPUT_COMMIT
    manifest = json.loads(args.manifest.read_text())
    mh = digest_file(args.manifest)
    import hashlib

    def verify_git_input(path):
        original = subprocess.check_output(
            ["git", "-C", str(root), "show", INPUT_COMMIT + ":" + str(path.relative_to(root))]
        )
        assert hashlib.sha256(original).hexdigest() == digest_file(path), path

    verify_git_input(args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = dict(
        input_commit=head,
        execution_commit=execution_commit,
        manifest_sha256=mh,
        checkpoint_sha256=manifest["checkpoint"]["sha256"],
        exploratory=True,
        command=" ".join([str(Path(sys.executable)), *sys.argv]),
        source_sha256=digest_file(__file__),
        input_files={},
        audits={},
        sigma_availability={},
        method={
            "grouping": "all cases; no selected evaluation subset",
            "allocation": "leave one entire condition out; same-group training net gains versus one step; choose maximum positive gain from2,4,10, ties smaller arm; otherwise1",
            "uncertainty": "10000 condition-cluster percentile draws; descriptive conditional on frozen cross-fitted assignments, overlapping training sets not re-fit; not independent validation",
            "leakage": "no held-condition outcomes used for arm assignment; same base family and seeds may recur across conditions, so generalization only across held conditions",
            "limits": "post-hoc taxonomy/rule; no trained model, no deployed adaptive policy, no adaptive speedup; condition identity and singleton families cannot be supported out of condition",
        },
    )
    all_groups = []
    all_cases = []
    loco_summaries = []
    loco_cases = []
    for benchmark in ("libero", "libero_plus"):
        published_path = args.results / (benchmark + "-summary.json")
        verify_git_input(published_path)
        published = json.loads(published_path.read_text())
        files = sorted(args.results.glob(benchmark + "-main-worker-*.jsonl"))
        expected_hashes = {Path(x["path"]).name: x["sha256"] for x in published["record_files"]}
        rows = []
        locations = {}
        for file in files:
            verify_git_input(file)
            digest = digest_file(file)
            assert digest == expected_hashes[file.name], file
            provenance["input_files"][str(file.relative_to(root))] = digest
            for line, text in enumerate(file.read_text().splitlines(), 1):
                row = json.loads(text)
                rows.append(row)
                if row["status"] == "ok":
                    key = (row["condition_id"], row["seed"], row["init_index"], row["flow_steps"])
                    locations[key] = (str(file.relative_to(root)), line)
        reproduced = analyze(manifest, rows, benchmark, mh)
        assert reproduced == {k: v for k, v in published.items() if k not in ("record_files", "wandb_url")}
        found, _ = validate(manifest, rows, benchmark, mh)
        provenance["input_files"][str(published_path.relative_to(root))] = digest_file(published_path)
        provenance["audits"][benchmark] = "strict raw audit and exact published summary reproduction passed"
        sigma_rows = sum(has_sigma(r) for r in rows)
        provenance["sigma_availability"][benchmark] = dict(
            rows=len(rows),
            rows_with_sigma_or_residual_key=sigma_rows,
            conclusion="Absent; residual-head predictive value cannot be tested"
            if sigma_rows == 0
            else "Inspect located fields",
        )
        conditions = {c["condition_id"]: c for c in manifest["benchmarks"][benchmark]["conditions"]}
        cases = []
        for c in reproduced["paired_cases"]:
            meta = conditions[c["condition_id"]]
            c = dict(c, category=meta["category"], severity=meta["severity"])
            cases.append(c)
            row = {
                k: c[k]
                for k in ("condition_id", "family", "suite", "task_name", "category", "severity", "seed", "init_index")
            }
            row = dict(
                benchmark=benchmark,
                **row,
                unique_condition_initial_states=meta["distinct_declared_initial_state_hashes"],
                sigma_available=False,
            )
            for arm in ARMS:
                record = found[(c["condition_id"], c["seed"], c["init_index"], arm)]
                path, line = locations[(c["condition_id"], c["seed"], c["init_index"], arm)]
                row.update(
                    {
                        f"success_{arm}": int(c["success"][str(arm)]),
                        f"raw_path_{arm}": path,
                        f"raw_line_{arm}": line,
                        f"raw_url_{arm}": f"https://github.com/lapid92/openpi/blob/{head}/{path}#L{line}",
                        f"velocity_evaluations_{arm}": record["total_velocity_evaluations"],
                        f"policy_ms_sum_{arm}": sum(x["policy_ms"] for x in record["chunks"]),
                        f"episode_ms_{arm}": record["episode_ms"],
                    }
                )
                if arm != 1:
                    row[f"rescue_{arm}"] = int(not c["success"]["1"] and c["success"][str(arm)])
                    row[f"regression_{arm}"] = int(c["success"]["1"] and not c["success"][str(arm)])
            all_cases.append(row)
        all_groups.extend(group_rows(benchmark, cases))
        for dimension in ("family", "perturbation_type", "severity"):
            summary, held = loco(benchmark, cases, dimension)
            loco_summaries.append(summary)
            loco_cases.extend(held)
    write_csv(args.output / "grouped_results.csv", all_groups)
    write_csv(args.output / "all_paired_cases.csv", all_cases)
    write_csv(args.output / "heldout_allocations.csv", loco_cases)
    (args.output / "heldout_summary.json").write_text(json.dumps(loco_summaries, indent=2, allow_nan=False) + "\n")
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    print(json.dumps(loco_summaries, indent=2))


if __name__ == "__main__":
    main()
