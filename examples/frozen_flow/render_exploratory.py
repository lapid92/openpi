"""Render exploratory tables from audited CSVs; no policy or simulator execution."""

import csv
import json
from pathlib import Path

R = Path(__file__).parent / "results/d16dba54a815/exploratory"
rows = list(csv.DictReader((R / "grouped_results.csv").open()))
cases = list(csv.DictReader((R / "all_paired_cases.csv").open()))
held = json.loads((R / "heldout_summary.json").read_text())


def table(header, data):
    return (
        "\n".join(
            ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
            + ["| " + " | ".join(map(str, r)) + " |" for r in data]
        )
        + "\n"
    )


def groups(benchmark, dimension):
    out = {}
    for x in rows:
        if x["benchmark"] == benchmark and x["dimension"] == dimension:
            out.setdefault(x["group"], {})[x["arm"]] = x
    return out


def count_table(benchmark, dimension):
    data = []
    for name, a in groups(benchmark, dimension).items():
        b = a["1"]
        data.append(
            [name, b["pairs"], b["conditions"], b["one_step_failures"]]
            + [a[n]["rescues"] + "/" + a[n]["regressions"] for n in ["2", "4", "10"]]
        )
    return table(["Group", "Pairs", "Conditions", "1-step failures", "2 R/L", "4 R/L", "10 R/L"], data)


text = """# Exploratory fixed-step allocation analysis

**Decision: the records reveal a testable perturbation-type signal, but a new independent evaluation is needed before adopting an allocation rule.** Layout and robot-state conditions look more favorable to extra steps than camera conditions. Task-family labels transfer weakly in the retrospective check. No model, predictor, policy, protocol, or rollout was changed.

## Evidence and aggregate reproduction

Inputs are the audited main records at commit **8d4aa1bd16b21f24f273a785ff2ea34be2a5a4be**, not smoke runs. The analysis reruns the strict original record audit, reproduces all published summaries exactly, and checks hashes against the independent final audit. [Provenance](provenance.json) records input hashes and commands; [independent audit](../independent-final-review.json) and [original report](../full-REPORT.md) are the reference. Benchmarks are never pooled.

| Benchmark | Pairs / conditions | Successes at 1 / 2 / 4 / 10 | One-step failures | Rescues at 2 / 4 / 10 | Regressions at 2 / 4 / 10 |
|---|---|---|---|---|---|
| LIBERO | 400 / 40 | 386 / 390 / 390 / 386 | 14 | 8 / 11 / 11 | 4 / 7 / 11 |
| LIBERO-Plus | 480 / 48 | 454 / 456 / 459 / 463 | 26 | 15 / 16 / 16 | 13 / 11 / 7 |

Published overall net-difference 95% condition-cluster intervals (percentage points) are LIBERO: +1 [-0.5,2.5], +1 [-1,3], 0 [-2,2]; Plus: +0.417 [-2.083,3.125], +1.042 [-1.25,3.333], +1.875 [-0.417,4.375]. All include zero.

## LIBERO-Plus group counts

R/L means failures rescued / one-step successes lost. Denominators include every case, including one-step successes; conditioning only on the 26 failures would hide harm. Counts at different step arms overlap and must not be added.

"""
text += count_table("libero_plus", "perturbation_type") + "\n" + count_table("libero_plus", "severity")
text += "\nNet differences at ten steps, with exploratory 95% condition-cluster intervals:\n\n"
data = []
for dim in ["perturbation_type", "severity"]:
    for name, a in groups("libero_plus", dim).items():
        x = a["10"]
        data.append(
            [
                dim + ": " + name,
                f"{100 * float(x['net_success_gain']):.2f}",
                f"[{100 * float(x['ci95_low']):.2f}, {100 * float(x['ci95_high']):.2f}]",
            ]
        )
text += table(["Group", "Net pp", "95% interval"], data)
text += """
These labels were not randomized across task families: type and family are partly confounded. Severity is not a monotonic difficulty or benefit score: severity 2 has 14 failures and severity 3 has 12. All groups were examined after outcomes existed; intervals are unadjusted for multiple comparisons.

### Task families

"""
text += count_table("libero_plus", "family")
text += """
The moka-pot/stove family has five one-step failures across two layout conditions; ten steps rescues all five without a loss. Both conditions have only **one distinct initial state**, repeated with ten noise seeds each. This is a two-condition pattern, not twenty independent scene configurations. Conversely, soup-and-tomato basket cases contain both a robot-state condition with rescues and a camera condition with losses: a family label alone misses the interaction.

### Conditions and task identities

[All conditions](CONDITIONS.md) includes every tested condition, with full task identity, type, severity, denominator and R/L. “Promising” means observed rescues in at least one extra arm and no losses in any extra arm; “harmful-only” means losses and no rescues; “mixed” means both. These are retrospective descriptions, not selection criteria or validated recommendations. All neutral conditions are retained.

[Individual outcomes](CASES.md) lists every one-step failure and every case with any regression, separately by benchmark, with links to each arm's exact raw line. [All 880 paired cases](all_paired_cases.csv) also includes all-success ties. [Grouped results](grouped_results.csv) supplies every family, task identity, perturbation type, severity, condition grouping, with joint metadata, including per-arm net intervals and rescue fractions. Task identity includes the variation suffix; it is effectively a condition identifier here and does not itself provide independent-condition replication.

## Does a grouping transfer to held-out conditions?

For each condition, use only other conditions with the same group label to tabulate paired net gains vs one step. Choose the largest positive-gain arm, break ties toward fewer steps, and otherwise choose one step. Apply that arm to **all ten held-out cases**, not just known failures. This is a retrospective lookup calculation on existing fixed-arm records, not a trained predictor or executed adaptive policy. No extra trajectories were generated.

"""
data = []
for x in held:
    if x["benchmark"] == "libero_plus":
        ci = x["fixed1"]["condition_cluster_ci95"]
        data.append(
            [
                x["dimension"],
                x["heldout_conditions_with_training_support"],
                x["successes"],
                f"{x['fixed1']['rescues']}/{x['fixed1']['regressions']}",
                f"{100 * x['fixed1']['net_difference']:.2f} [{100 * ci[0]:.2f},{100 * ci[1]:.2f}]",
                x["fixed10"]["successes_gained"],
            ]
        )
text += table(
    [
        "Plus lookup",
        "Supported conditions /48",
        "Success /480",
        "R/L vs1",
        "Net pp vs1 [descriptive CI]",
        "Net successes vs fixed10",
    ],
    data,
)
text += """
The type lookup yields 462/480 versus 454/480 at one step and 463/480 at ten. It chooses one step for 160 cases, ten for 160, two for 140 and four for 20. This is a candidate signal, not proof of an adaptive speedup. Family lookup achieves only 456/480; three singleton families have no other-condition support and fall back to one step. Exact held-out choices and supporting condition IDs are in [held-out allocations](heldout_allocations.csv).

The positive type interval is **conditional on the already-selected lookup assignments**: bootstrap resampling does not refit the lookup. Training folds overlap, conditions share families, and the grouping was chosen after examining data. Therefore this interval is not selection-corrected evidence of generalization or a confirmatory significance test. No family-held-out or independent benchmark validation was performed.

## LIBERO, separately

LIBERO has only 14 one-step failures, below the declared adequacy threshold. Four steps rescues 11 but loses seven successes; ten rescues 11 and loses 11. The wine-bottle-on-rack task has two failures rescued at four and ten steps; the both-moka-pots-on-stove task has two rescues but four losses at those arms. These are each one condition with ten states, so no across-condition family uncertainty can be estimated.

Every LIBERO family appears in just one condition: family-grouped condition-held-out lookup has zero supported cases and defaults to one step. Type/severity are constant, so they are merely global arm-choice baselines, not informative features; their leave-one-condition-out lookup gives 386/400, eight rescues and eight regressions. Full separate condition and case tables are linked above.

## Predictor and uncertainty limits

**Initial residual-predictor sigma was not logged.** Recursive inspection of all 3,520 records and chunk keys found no sigma fields; this frozen study omitted the residual head. Ranking rescues against regressions by sigma is unmeasured, and no surrogate sigma was reconstructed.

Group intervals use 10,000 whole-condition bootstrap draws. A one-condition interval is unavailable, not zero uncertainty; zero-discordance intervals do not prove equivalence. Rescue fractions have only 26 possible Plus failures overall and often one or two per subgroup. Report raw numerators and denominators rather than treating large percentages as stable. Repeated layout seeds and cross-condition family dependence reduce effective diversity. Findings do not diagnose a physical failure mechanism from success labels alone.

## Reproduction and decision

Run only the following analysis commands on the pod; these do not invoke a policy or simulator:

```bash
cd /volt/code/frozen-flow-study
.venv/bin/python examples/frozen_flow/explore_groups.py --manifest examples/frozen_flow/protocol.json --results examples/frozen_flow/results/d16dba54a815 --output examples/frozen_flow/results/d16dba54a815/exploratory
.venv/bin/python examples/frozen_flow/render_exploratory.py
```

Validation and review are in [REVIEW.md](REVIEW.md). Raw input URLs are pinned to the audited input commit in each case row.

**Test, do not deploy:** retain perturbation type (especially layout versus camera) as an exploratory hypothesis for an independent, predeclared comparison. Existing records cannot validate an episode-level sigma selector or show that a deployable allocation policy generalizes. A new independent evaluation with diverse conditions and states is needed; none is launched or authorized by this analysis.
"""
(R / "ANALYSIS.md").write_text(text)
out = "# All condition descriptions\n\nR/L = rescues/regressions vs one step. All conditions are retained; classifications are descriptive after observing outcomes. Net CIs for singleton conditions are unavailable.\n"
for b in ["libero_plus", "libero"]:
    out += "\n## " + b + "\n\n"
    data = []
    for key, a in groups(b, "condition").items():
        x = a["1"]
        rs = [int(a[n]["rescues"]) for n in ["2", "4", "10"]]
        ls = [int(a[n]["regressions"]) for n in ["2", "4", "10"]]
        label = "neutral"
        if any(rs) and any(ls):
            label = "mixed"
        elif any(rs):
            label = "promising observed"
        elif any(ls):
            label = "harmful-only observed"
        data.append(
            [key, x["task_identities"], x["perturbation_types"], x["severities"], x["pairs"], x["one_step_failures"]]
            + [f"{rescues}/{losses}" for rescues, losses in zip(rs, ls, strict=True)]
            + [label]
        )
    out += table(
        [
            "Condition",
            "Task identity",
            "Type",
            "Severity",
            "N",
            "1 failures",
            "2 R/L",
            "4 R/L",
            "10 R/L",
            "Description",
        ],
        data,
    )
(R / "CONDITIONS.md").write_text(out)
out = "# Individual failure and regression cases\n\nS=success, F=failure. Each outcome links to the corresponding raw record at the audited input commit. All one-step failures and all one-step successes lost by any arm are shown. Unlisted all-success cases remain in all_paired_cases.csv. Condition metadata are in CONDITIONS.md.\n"
for b in ["libero_plus", "libero"]:
    selected = [
        x
        for x in cases
        if x["benchmark"] == b and (x["success_1"] == "0" or any(x["regression_" + n] == "1" for n in ["2", "4", "10"]))
    ]
    out += "\n## " + b + "\n\n"
    out += table(
        ["Condition", "Seed", "State index", "1", "2", "4", "10"],
        [
            [x["condition_id"], x["seed"], x["init_index"]]
            + [
                "[" + ("S" if x["success_" + n] == "1" else "F") + "](" + x["raw_url_" + n] + ")"
                for n in ["1", "2", "4", "10"]
            ]
            for x in selected
        ],
    )
(R / "CASES.md").write_text(out)
print("Rendered ANALYSIS.md CONDITIONS.md CASES.md")
