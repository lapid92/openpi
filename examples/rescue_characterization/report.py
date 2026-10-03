"""Render validated benchmark summaries without pooling policies."""

import argparse
import hashlib
import json
from pathlib import Path


def percent(value):
    return "undefined" if value is None else f"{100 * value:.2f}%"


def table(group):
    lines = [
        "| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |",
        "|---:|---:|---:|---:|---:|",
    ]
    for arm, row in group["arms"].items():
        if arm == "1":
            detail = ["—", "—", "—"]
        else:
            pair = row["paired_vs_one"]
            ci = pair["net_difference_condition_cluster_ci95"]
            interval = "unavailable (<2 clusters)" if ci is None else f"[{percent(ci[0])}, {percent(ci[1])}]"
            detail = [
                f"{pair['rescues']} / {pair['regressions']}",
                percent(pair["net_success_difference"]) + " " + interval,
                f"{pair['rescues']}/{pair['one_step_failures']} ({percent(pair['fraction_one_step_failures_rescued'])})",
            ]
        lines.append(
            f"| {arm} | {row['successes']}/{row['episodes']} ({percent(row['success_rate'])}) | "
            + " | ".join(detail)
            + " |"
        )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    manifest = json.loads(Path(a.manifest).read_text())
    results = Path(a.results)
    text = [
        "# Frozen π₀.₅ rescue characterization",
        "",
        "Smoke validation only; excluded from main inference."
        if a.smoke
        else "Complete planned paired evaluations; no training or adaptive step policy.",
        "",
        "Each benchmark is reported separately. LIBERO and LIBERO-Plus use the same frozen LIBERO checkpoint; no RoboCasa episodes were evaluated.",
        "",
        "Checkpoint full content SHA256: " + manifest["checkpoint"]["sha256"] + ".",
        "Historical metadata identity: dad4e2fbe79cceca79b83f3e53bb59c180e55ebb6815c67bfcdcf81768d7cee8.",
        "",
        "## Benchmark results",
        "",
    ]
    for bench, spec in manifest["benchmarks"].items():
        path = results / (bench + ("-smoke-summary.json" if a.smoke else "-summary.json"))
        summary = json.loads(path.read_text())
        identity = {
            "manifest_sha256": hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),
            "benchmark": bench,
            "phase": "smoke" if a.smoke else "main",
            "checkpoint_sha256": manifest["checkpoint"]["sha256"],
            "benchmark_commit": spec["commit"],
        }
        if any(summary.get(k) != v for k, v in identity.items()):
            raise ValueError("Summary provenance differs from protocol")
        if summary["audit"] != "passed":
            raise ValueError("Report requires validated records")
        text += [
            "### " + bench,
            "",
            summary["assessment"],
            "",
            f"Benchmark revision: {spec['commit']}. {summary['overall']['paired_cases']} paired cases, {summary['overall']['conditions']} task/condition clusters.",
            f"[W&B]({summary['wandb_url']})",
            "",
            table(summary["overall"]),
            "",
            "| Steps | Velocity evaluations/chunk | Total velocity evaluations | Policy call mean / p95 (ms) | Episode mean / p95 (s) |",
            "|---:|---:|---:|---:|---:|",
        ]
        for arm, row in summary["overall"]["arms"].items():
            c = row["cost"]
            pol = c["scored_policy_call_ms"]
            ep = c["simulator_inclusive_episode_ms"]
            text.append(
                f"| {arm} | {c['velocity_evaluations_per_chunk']['mean']:.0f} | {c['total_velocity_evaluations']} | {pol['mean']:.2f} / {pol['p95']:.2f} | {ep['mean'] / 1000:.2f} / {ep['p95'] / 1000:.2f} |"
            )
        patterns_path = results / (bench + ("-smoke-patterns.json" if a.smoke else "-patterns.json"))
        patterns = json.loads(patterns_path.read_text())
        for cohort in patterns.values():
            text += ["", "#### First-success and non-monotonic patterns", "",
                     "Full success-pattern counts: " + json.dumps(cohort.get("patterns", {}), sort_keys=True),
                     "First success among one-step failures: " + json.dumps(cohort["first_success_among_one_step_failures"], sort_keys=True),
                     "One-step failures: " + str(cohort["one_step_failures"]) + ".",
                     "Primary uncertainty uses whole base families for Plus and tasks for LIBERO. Full intervals and distributions are in " + patterns_path.name + ".",
                     "Visual failure labels remain pending independent review; numerical rescues alone do not establish a failure mechanism."]
        text += ["", "#### Task-family results", ""]
        for family, group in summary["by_family"].items():
            text += ["**" + family + "**", "", table(group), ""]
        text += [
            "Per-family costs, condition outcomes, and every paired case are in the JSON summary. Raw JSONL preserves chunk records and error attempts.",
            "",
        ]
    text += [
        "## Cost and uncertainty boundaries",
        "",
        "Policy latency includes synchronized production inference and input/output transforms; it excludes Gaussian noise generation, JSON and networking. Episode time includes simulator construction, reset, stabilization, policy requests and simulation; it excludes teardown. Compilation and parity are separate preparation records.",
        "Arm tables use condition-cluster intervals. The pattern JSON additionally gives the PRIMARY Plus family-cluster intervals. Use these for scientific conclusions; condition clustering is sensitivity. Zero discordance does not prove equivalence.",
        "Objects Layout supplies one state per condition repeated at ten seeds. Plus uses severity3/4 across7 axes with10 seeds; standard LIBERO uses20 seeds. Conditions are a predeclared subset, not the official full benchmark.",
        "Fewer than50 one-step failures fails the predeclared adequacy target. A descriptive recurring class requires at least10 rescues across3 families and5 conditions in the new population, with counterexamples and uncertainty. No adaptive speedup is claimed.",
        "",
        "## Reproduction",
        "",
        "Execution directory: /volt/code/frozen-flow-study on Volt pod uz2ptakxucbe. The manifest pins full checkpoint files, source files, benchmark assets, tasks, states, seeds and budgets. Status JSON records exact child argument lists and W&B links.",
        "~~~bash",
        "setsid -f .venv/bin/python examples/rescue_characterization/supervise.py --manifest examples/rescue_characterization/protocol.json --output /volt/artifacts/rescue-characterization/runs --phase smoke > /volt/artifacts/rescue-characterization/smoke-supervisor.log 2>&1",
        "setsid -f .venv/bin/python examples/rescue_characterization/supervise.py --manifest examples/rescue_characterization/protocol.json --output /volt/artifacts/rescue-characterization/runs --phase full > /volt/artifacts/rescue-characterization/full-supervisor.log 2>&1",
        "~~~",
        "",
        "Errors halt execution and remain on the pod; publication occurs only after complete audits. Publication failures are recorded in supervisor status and require publication retry, not episode replacement.",
    ]
    Path(a.output).write_text("\n".join(text) + "\n")


if __name__ == "__main__":
    main()
