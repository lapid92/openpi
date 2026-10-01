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
        "# Frozen π₀.₅ fixed flow-step study",
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
        "Intervals use 10,000 condition-cluster bootstrap draws retaining all states/arms. Plus conditions may share a base family, so condition clustering can understate family dependence. Single-condition family intervals are unavailable; zero discordance does not prove equivalence.",
        "Objects Layout supplies one state per condition repeated at ten seeds. Other selected conditions use ten state indices. Severity-2/3 Plus conditions are an exploratory subset, not the official full benchmark.",
        "Fewer than 20 one-step failures or five rescues leaves rescue benefit unresolved. No adaptive speedup is claimed.",
        "",
        "## RoboCasa blocker",
        "",
        "Public candidate changyeon/pi05_robocasa_as50_jax at revision 165da7e92fdbd140c4fe3e2a4bad6f0cabcda47e contains a completed Orbax save. Its configuration, camera/state/action adapter, and simulator provenance could not be verified. It was not evaluated with the LIBERO interface. See setup/ROBOCASA_DISCOVERY.md.",
        "",
        "## Reproduction",
        "",
        "Execution directory: /volt/code/frozen-flow-study on Volt pod uz2ptakxucbe. The manifest pins full checkpoint files, source files, benchmark assets, tasks, states, seeds and budgets. Status JSON records exact child argument lists and W&B links.",
        "~~~bash",
        "setsid -f .venv/bin/python examples/frozen_flow/supervise.py --manifest examples/frozen_flow/protocol.json --output /volt/artifacts/frozen-flow-study/runs --phase smoke > /volt/artifacts/frozen-flow-study/smoke-supervisor.log 2>&1",
        "setsid -f .venv/bin/python examples/frozen_flow/supervise.py --manifest examples/frozen_flow/protocol.json --output /volt/artifacts/frozen-flow-study/runs --phase full > /volt/artifacts/frozen-flow-study/full-supervisor.log 2>&1",
        "~~~",
        "",
        "Errors halt execution and remain on the pod; publication occurs only after complete audits. Publication failures are recorded in supervisor status and require publication retry, not episode replacement.",
    ]
    Path(a.output).write_text("\n".join(text) + "\n")


if __name__ == "__main__":
    main()
