"""Render audited selective study summaries with explicit inference limits."""

import argparse
import hashlib
import json
from pathlib import Path


def fmt(value):
    if value is None:
        return "unavailable"
    if isinstance(value, list):
        return "[" + ", ".join(fmt(x) for x in value) + "]"
    return f"{value:.5f}"


def render(manifest, summary, manifest_sha256, benchmark, *, smoke=False):
    expected = {
        "manifest_sha256": manifest_sha256,
        "benchmark": benchmark,
        "benchmark_commit": manifest["benchmarks"][benchmark]["commit"],
        "phase": "smoke" if smoke else "main",
        "checkpoint_sha256": manifest["checkpoint"]["sha256"],
        "head_sha256": manifest["head"]["sha256"],
        "audit": "passed",
    }
    if any(summary.get(k) != v for k, v in expected.items()):
        raise ValueError("Summary provenance or audit differs from protocol")
    lines = [
        "# Independent selective fixed-step study",
        "",
        "Smoke validation only; excluded from scientific inference." if smoke else summary["overall"]["decision"],
        "",
        f"Benchmark: {benchmark}; revision: {expected['benchmark_commit']}.",
        f"Manifest SHA256: {manifest_sha256}.",
        f"Checkpoint SHA256: {expected['checkpoint_sha256']}.",
        f"Frozen residual head SHA256: {expected['head_sha256']}.",
        f"Validated episode records: {summary['completed_episodes']}; error attempts: {summary['error_attempt_count']}.",
        f"W&B: {summary.get('wandb_url', 'not attached')}.",
        "",
        "Rule: 10 steps for Robot Initial States and Objects Layout; 1 for Camera Viewpoints.",
        "Rule results select matched fixed-arm records; smoke checks the actual rule wrapper.",
        "",
    ]
    for name, group in [("Overall", summary["overall"]), *summary["by_type"].items()]:
        lines += [
            f"## {name}",
            "",
            f"{group['pairs']} pairs; {group['conditions']} conditions; {group['families']} families.",
            "",
            "| Strategy | Success | Equal-case mean policy call ms | Policy call mean/p50/p95 ms | Head mean ms | Episode mean/p50/p95 ms | Velocity evaluations |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for arm, row in group["strategies"].items():
            cost = row["cost"]
            pol = cost["policy_call_ms"]
            ep = cost["episode_ms"]
            lines.append(
                f"| {arm} | {row['successes']}/{row['episodes']} | {fmt(cost['per_case_mean_policy_call_ms']['mean'])} | "
                + "/".join(fmt(pol[k]) for k in ("mean", "p50", "p95"))
                + " | "
                + fmt(cost["head_ms"]["mean"])
                + " | "
                + "/".join(fmt(ep[k]) for k in ("mean", "p50", "p95"))
                + f" | {cost['total_velocity_evaluations']} |"
            )
        lines += [
            "",
            "Differences are candidate minus reference; success units are fractions. Intervals are separate 95% cluster bootstrap intervals.",
            "",
        ]
        for contrast, row in group["comparisons"].items():
            cost = row["cost_differences"]["mean_policy_call_ms"]
            lines += [
                f"**{contrast}**: rescues/regressions {row['rescues']}/{row['regressions']}; "
                f"reference failures rescued {row['rescues']}/{row['reference_failures']} ({fmt(row['fraction_reference_failures_rescued'])}).",
                f"Success difference {fmt(row['net_success_difference'])}; condition CI {fmt(row['condition_cluster_ci95'])}; family sensitivity CI {fmt(row['family_cluster_ci95'])}.",
                f"Equal-case mean policy-call difference {fmt(cost['mean'])} ms; condition CI {fmt(cost['condition_cluster_ci95'])}; family sensitivity CI {fmt(cost['family_cluster_ci95'])}.",
                "",
            ]
    lines += [
        "## Initial sigma ranking",
        "",
        "Higher first-chunk initial sigma predicts rescue. AUC uses only discordant fixed-1/fixed-10 pairs; this retrospective conditioning cannot validate a prospective selector.",
        "",
    ]
    for name, row in [("Overall", summary["sigma"]["overall"]), *summary["sigma"]["by_type"].items()]:
        lines += [
            f"**{name}**: {row['rescues']} rescues, {row['regressions']} regressions, {row['discordant_conditions']} discordant conditions; AUC {fmt(row['auc_higher_initial_sigma_predicts_rescue'])}. {row['assessment']}"
        ]
        for field, bootstrap in row["bootstrap"].items():
            lines.append(
                f"{field} CI {fmt(bootstrap['ci95'])}; valid draws {bootstrap['valid_draws']}/{bootstrap['requested_draws']}. Single-class draws excluded."
            )
        lines.append("")
    lines += ["## Interpretation limits", ""]
    lines.extend("- " + item for item in summary["limitations"])
    lines += [
        "",
        "Primary support requires success CI lower >= 0 and policy-call cost CI upper < 0. This is neither equivalence nor a joint 95% guarantee.",
        "Full per-family results, paired-case outcomes, all cost distributions and bootstrap seeds are in the adjacent validated summary JSON.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    documents = []
    for benchmark in manifest["benchmarks"]:
        path = args.results / (benchmark + ("-smoke-summary.json" if args.smoke else "-summary.json"))
        documents.append(render(manifest, json.loads(path.read_text()), digest, benchmark, smoke=args.smoke))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n".join(documents))


if __name__ == "__main__":
    main()
