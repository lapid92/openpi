# Exploratory fixed-step allocation analysis

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

| Group | Pairs | Conditions | 1-step failures | 2 R/L | 4 R/L | 10 R/L |
|---|---|---|---|---|---|---|
| Camera Viewpoints | 160 | 16 | 1 | 1/7 | 1/6 | 1/5 |
| Objects Layout | 160 | 16 | 7 | 6/4 | 6/1 | 7/0 |
| Robot Initial States | 160 | 16 | 18 | 8/2 | 9/4 | 8/2 |

| Group | Pairs | Conditions | 1-step failures | 2 R/L | 4 R/L | 10 R/L |
|---|---|---|---|---|---|---|
| 2 | 240 | 24 | 14 | 11/4 | 10/3 | 11/4 |
| 3 | 240 | 24 | 12 | 4/9 | 6/8 | 5/3 |

Net differences at ten steps, with exploratory 95% condition-cluster intervals:

| Group | Net pp | 95% interval |
|---|---|---|
| perturbation_type: Camera Viewpoints | -2.50 | [-5.62, 0.62] |
| perturbation_type: Objects Layout | 4.38 | [0.62, 10.00] |
| perturbation_type: Robot Initial States | 3.75 | [0.62, 7.50] |
| severity: 2 | 2.92 | [-1.25, 7.50] |
| severity: 3 | 0.83 | [-0.83, 2.50] |

These labels were not randomized across task families: type and family are partly confounded. Severity is not a monotonic difficulty or benefit score: severity 2 has 14 failures and severity 3 has 12. All groups were examined after outcomes existed; intervals are unadjusted for multiple comparisons.

### Task families

| Group | Pairs | Conditions | 1-step failures | 2 R/L | 4 R/L | 10 R/L |
|---|---|---|---|---|---|---|
| KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 20 | 2 | 5 | 4/3 | 4/1 | 5/0 |
| KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 20 | 2 | 0 | 0/0 | 0/0 | 0/0 |
| LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 40 | 4 | 5 | 0/1 | 2/1 | 1/2 |
| LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 40 | 4 | 2 | 2/0 | 2/1 | 2/1 |
| open_the_middle_drawer_of_the_cabinet | 60 | 6 | 2 | 1/0 | 0/1 | 0/0 |
| open_the_top_drawer_and_put_the_bowl_inside | 20 | 2 | 1 | 1/0 | 1/0 | 1/0 |
| pick_up_the_alphabet_soup_and_place_it_in_the_basket | 60 | 6 | 2 | 2/1 | 2/1 | 2/0 |
| pick_up_the_bbq_sauce_and_place_it_in_the_basket | 20 | 2 | 0 | 0/1 | 0/0 | 0/0 |
| pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 50 | 5 | 1 | 1/1 | 1/0 | 1/1 |
| pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 10 | 1 | 0 | 0/0 | 0/0 | 0/0 |
| pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 10 | 1 | 0 | 0/0 | 0/0 | 0/0 |
| pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 40 | 4 | 0 | 0/0 | 0/0 | 0/0 |
| pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 10 | 1 | 0 | 0/1 | 0/2 | 0/1 |
| pick_up_the_cream_cheese_and_place_it_in_the_basket | 40 | 4 | 8 | 4/2 | 4/2 | 4/1 |
| put_the_bowl_on_the_stove | 40 | 4 | 0 | 0/3 | 0/2 | 0/1 |

The moka-pot/stove family has five one-step failures across two layout conditions; ten steps rescues all five without a loss. Both conditions have only **one distinct initial state**, repeated with ten noise seeds each. This is a two-condition pattern, not twenty independent scene configurations. Conversely, soup-and-tomato basket cases contain both a robot-state condition with rescues and a camera condition with losses: a family label alone misses the interaction.

### Conditions and task identities

[All conditions](CONDITIONS.md) includes every tested condition, with full task identity, type, severity, denominator and R/L. “Promising” means observed rescues in at least one extra arm and no losses in any extra arm; “harmful-only” means losses and no rescues; “mixed” means both. These are retrospective descriptions, not selection criteria or validated recommendations. All neutral conditions are retained.

[Individual outcomes](CASES.md) lists every one-step failure and every case with any regression, separately by benchmark, with links to each arm's exact raw line. [All 880 paired cases](all_paired_cases.csv) also includes all-success ties. [Grouped results](grouped_results.csv) supplies every family, task identity, perturbation type, severity, condition grouping, with joint metadata, including per-arm net intervals and rescue fractions. Task identity includes the variation suffix; it is effectively a condition identifier here and does not itself provide independent-condition replication.

## Does a grouping transfer to held-out conditions?

For each condition, use only other conditions with the same group label to tabulate paired net gains vs one step. Choose the largest positive-gain arm, break ties toward fewer steps, and otherwise choose one step. Apply that arm to **all ten held-out cases**, not just known failures. This is a retrospective lookup calculation on existing fixed-arm records, not a trained predictor or executed adaptive policy. No extra trajectories were generated.

| Plus lookup | Supported conditions /48 | Success /480 | R/L vs1 | Net pp vs1 [descriptive CI] | Net successes vs fixed10 |
|---|---|---|---|---|---|
| family | 45 | 456 | 8/6 | 0.42 [-1.04,2.08] | -7 |
| perturbation_type | 48 | 462 | 13/5 | 1.67 [0.21,3.33] | -1 |
| severity | 48 | 459 | 15/10 | 1.04 [-1.04,3.12] | -4 |

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
