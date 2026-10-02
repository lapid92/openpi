# Independent exploratory analysis review

Date: 2026-10-02. Execution and review occurred on Volt pod `uz2ptakxucbe`, in `/volt/code/frozen-flow-study`.

**Verdict: approved as exploratory descriptive analysis.** This approval does not establish a deployable allocation policy, independent generalization, residual-predictor utility, or adaptive speedup. No new rollouts, model updates, or frozen protocol/source changes were made.

## Reviewer and Test Writer cycle

The Reviewer inspected the actual `explore_groups.py` and `render_exploratory.py` implementations and their generated CSV, JSON, and Markdown artifacts. The Test Writer independently recomputed counts and lookup assignments from raw episode records, checked all raw-line references and rendered coverage, recomputed bootstrap intervals, and exercised synthetic leakage/tie/fallback cases.

Resolved findings:

1. The initial exact-HEAD guard prevented rerunning the analysis after its source was committed. The final implementation records execution revision separately and verifies manifest, raw records, and original summaries byte-for-byte against immutable input commit `8d4aa1bd16b21f24f273a785ff2ea34be2a5a4be`. Raw links remain pinned to that input revision.
2. The initial prose definition of “promising” overlapped with mixed conditions. It now agrees with the table: any rescue and no loss at any extra arm; mixed takes precedence when any rescue and any regression occur.
3. The narrative no longer claims joint statistical groupings that were not computed. Five grouping dimensions are reported, with joint metadata columns.
4. The LIBERO lookup is correctly described as family-grouped, condition-held-out; no family-held-out validation was performed.

## Independent checks and coverage

- **880 paired case rows** correspond exactly to **3,520 raw episode links**. Every linked file, line, immutable URL, outcome, seed, initial-state index, task metadata, velocity total, summed policy time and episode time was checked against the raw record.
- **952 group rows** were independently recomputed. For each benchmark, dimension and arm, disjoint group totals reproduce the original benchmark denominator and success count. Rescue, regression, one-step failure and rescue-fraction denominators agree. No benchmarks were pooled.
- **57 non-null group bootstrap intervals** and **895 null group intervals** were independently checked; all **12 lookup-versus-baseline intervals** were recomputed using their declared deterministic seeds and 10,000 condition-cluster draws.
- **2,640 held-out allocation rows** were independently recomputed. Every training set excludes the entire held condition, uses only matching-group conditions, and applies its selected arm to all held cases, including original one-step successes. Training IDs/counts/gains, smaller-arm tie-breaking, nonpositive-gain fallback and final outcomes all agree.
- Synthetic cases verified that flipping all outcomes in a held condition cannot change that condition's chosen arm; positive-gain ties choose two steps; no-support and negative-gain folds choose one step.
- **88 condition rows** are present, covering all 40 LIBERO and 48 LIBERO-Plus conditions. All descriptive labels agree with mixed-first precedence.
- **79 individual case rows** cover exactly all one-step failures and all cases with any regression: 30 LIBERO and 49 LIBERO-Plus cases. Every rendered S/F entry points to its exact raw line. Neutral cases remain in the complete CSV.
- Recursive inspection of all raw record/chunk trees found **zero sigma/residual fields**; the frozen manifest specifies no residual head. Sigma predictive value remains unmeasured.

Independent lookup successes reproduce the artifacts:

| Benchmark | Family lookup | Perturbation-type lookup | Severity lookup |
|---|---:|---:|---:|
| LIBERO | 386/400 | 386/400 | 386/400 |
| LIBERO-Plus | 456/480 | 462/480 | 459/480 |

LIBERO has no across-condition family support: all 40 family folds fall back to one step. LIBERO-Plus has three unsupported singleton-family folds. These support limitations are reported rather than omitted.

The independent category and severity cross-tabs agree with the narrative. At ten steps, Plus robot-state, camera, and layout conditions have respectively 8/2, 1/5 and 7/0 rescues/regressions; severity 2 and 3 have 11/4 and 5/3.

## Interpretation checks

The narrative explicitly distinguishes retrospective failure/rescue labels from inference-time information. It includes regressions among one-step successes, so lookup performance is not evaluated only on known failures.

The lookup confidence intervals condition on already-selected assignments and do not refit the rule in each bootstrap sample. Overlapping training folds, repeated layout states, shared families, post-hoc taxonomy/rule choice and multiple comparisons prevent confirmatory generalization claims. Condition-held-out validation does not establish new-family transfer. The positive type interval is therefore described as exploratory and selection-uncorrected.

The decision to retain perturbation type as a hypothesis for an independent predeclared study is proportionate to these limitations. No new study is launched by this analysis.

## Executed validation commands

Targeted lint passed with the repository-locked Ruff:

```bash
.venv/bin/ruff check examples/frozen_flow/explore_groups.py examples/frozen_flow/render_exploratory.py
```

The two documented analysis commands regenerate the artifacts from immutable completed inputs:

```bash
.venv/bin/python examples/frozen_flow/explore_groups.py --manifest examples/frozen_flow/protocol.json --results examples/frozen_flow/results/d16dba54a815 --output examples/frozen_flow/results/d16dba54a815/exploratory
.venv/bin/python examples/frozen_flow/render_exploratory.py
```

Independent read-only Python checks, separate from the implementation's own audit, performed the exhaustive comparisons and synthetic tests listed above. All passed. No GPU or simulator execution was used for this review.
