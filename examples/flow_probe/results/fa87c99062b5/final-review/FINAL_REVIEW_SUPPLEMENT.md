# Final reviewed interpretation: frozen two-evaluation ranking

Decision: **STOP; do not build an adaptive controller.** Both benchmarks fail the single prespecified four-step, 75% extra maximum-horizon compute-budget gate. The signal does not establish useful rescue selection with protection against regressions. This rejects this prespecified selector under this protocol, not every possible training-free signal.

All 9,760 fixed-arm episodes completed: 760 standard cases and 1,680 Plus cases, each at 1/2/4/10 steps. The 44 smoke episodes are excluded. Both policy and residual head remained frozen. Earlier analyzed rescue-study cases were excluded from decisive validation. Standard cases use new state/noise tuples on the same 40 tasks; Plus uses 168 unseen conditions, not 168 independent families.

## Matched outcomes

Rescue means one-step failure and higher-step success; regression is the reverse. Labels remain separate for each higher count.

| Benchmark | Higher count | Rescue | Regression | Unchanged success | Unchanged failure |
|---|---:|---:|---:|---:|---:|
| libero | 2 | 18 | 24 | 706 | 12 |
| libero | 4 | 20 | 16 | 714 | 10 |
| libero | 10 | 22 | 18 | 712 | 8 |

libero: 24/30 one-step failures have at least one higher-count success. Full four-bit patterns are retained in the metrics; a rescue at one count need not persist.

| Benchmark | Higher count | Rescue | Regression | Unchanged success | Unchanged failure |
|---|---:|---:|---:|---:|---:|
| libero_plus | 2 | 46 | 51 | 1469 | 114 |
| libero_plus | 4 | 61 | 54 | 1466 | 99 |
| libero_plus | 10 | 67 | 65 | 1455 | 93 |

libero_plus: 85/160 one-step failures have at least one higher-count success. Full four-bit patterns are retained in the metrics; a rescue at one count need not persist.

## Primary ranking and uncertainty

Intervals are 95% percentile intervals from 10,000 paired cluster bootstrap replicates (seed 20261005), resampling standard tasks and Plus suite/families. Ranking and budgets are recomputed in each replicate. AP is average precision over the entire PR curve; all non-rescues, including regressions, are negatives.

### libero

- AP 2.709% (95% CI 1.485 to 5.337%), against rescue prevalence 2.632%. AP minus prevalence interval: -0.456 to 2.142 percentage points.
- Selected 202 cases: 4 rescues and 2 regressions, with rescues across 4 primary clusters.
- Precision 1.980% (CI 0.469 to 4.256%); recall 20.000% (CI 5.000 to 35.714%). Precision minus prevalence interval: -1.979 to 0.916 percentage points.
- Net (rescues minus regressions)/all cases 0.263 percentage points (CI -0.263 to 0.789).
- Rescue-versus-regression AUROC 0.616 (CI 0.432 to 0.816).
- Offline selected-policy velocity-evaluation count 44,202 versus one-step baseline 24,757: ratio 1.785 (CI 1.740 to 1.832). This is not an executed adaptive controller or a wall-clock speedup.

### libero_plus

- AP 4.575% (95% CI 2.837 to 6.841%), against rescue prevalence 3.631%. AP minus prevalence interval: 0.010 to 2.754 percentage points.
- Selected 426 cases: 21 rescues and 16 regressions, with rescues across 10 primary clusters.
- Precision 4.930% (CI 2.400 to 7.268%); recall 34.426% (CI 22.218 to 44.615%). Precision minus prevalence interval: -0.356 to 2.934 percentage points.
- Net (rescues minus regressions)/all cases 0.298 percentage points (CI -0.404 to 0.947).
- Rescue-versus-regression AUROC 0.540 (CI 0.400 to 0.664).
- Offline selected-policy velocity-evaluation count 107,224 versus one-step baseline 60,543: ratio 1.771 (CI 1.735 to 1.816). This is not an executed adaptive controller or a wall-clock speedup.

Plus condition-cluster sensitivity gives primary precision CI 2.876–7.005%, precision-minus-prevalence CI −0.467–2.963 percentage points, net CI −0.357–0.894 percentage points, and rescue-versus-regression AUROC CI 0.433–0.652. It does not change the conclusion.

Plus has modest all-case AP enrichment, but this does not satisfy the gate: precision is below 10%, and the intervals for precision lift and net benefit include zero; rescue-versus-regression AUROC includes 0.5. Standard also fails selected rescue count/cluster coverage. Secondary budgets, counts, metadata strata or the sigma comparator cannot override either failure. The full report and metrics retain all comparisons, equal-overhead sensitivity, PR curves, non-monotonic patterns, and task/family/axis/severity counts.

## Cost and correction

The isolated initial probe performs two extra velocity evaluations, one prefix evaluation and one head evaluation. Both velocities are charged; no hypothetical sharing discount is taken. Across all arms, the scored scan used 1,471,094 velocity evaluations including 19,520 probe evaluations; full-run preparation used another 1,072. Smoke cost is separate. The allocation lasted about 43.89 hours on four H100 GPUs (175.58 allocated GPU-hours), not a throughput or speedup measurement.

The unchanged manifest and production client execute five actions before replanning. The score deliberately retains the declared first ten forecast positions by seven coordinates. The original protocol prose incorrectly said ten replan steps, and the first smoke auditor concatenated ten predicted rather than five executed actions. This was corrected explicitly before full validation; no episodes were repeated, no tolerance was relaxed, and pinned original sources were preserved. See [preserved amendment](../../../recovery/AMENDMENT.md) and [recovery manifest](../../../recovery/recovery-manifest.json). The first review missed the prose discrepancy.

## Audit scope and limitations

Both recording audits passed and decoded every episode video. The independent raw audit passed, reproduced all point metrics and the primary flow-score intervals and decision gates, including Plus condition-cluster sensitivity. **Other arm, budget and comparator intervals were structurally checked, not independently recomputed.** Separate Test Writer checks reproduced all raw counts and point results. Exact probe-on/off output and RNG parity, initial state/noise pairing, score and frozen-head reconstruction are covered by the preserved preparation/raw evidence. No new MuJoCo warning bytes were observed; the preexisting 476-byte log and zero-byte new suffix were preserved separately.

This is initial-observation episode ranking only. It does not establish a per-chunk rule, and later fixed-step trajectories visit different states. Rare outcomes, reused standard tasks, repeated Plus families and singleton layout states limit generalization. No fitting, direction flip, subgroup selection, new predictor, training, TVM, or controller was performed. No closed-loop success or adaptive speedup is claimed.

## Provenance and evidence

- Branch: codex/pi05-two-evaluation-ranking; automatic result publication commit: 614c3d8209c5a2848511fa7d4158d1da8fab8ce2. The final supplement publication commit is recorded in the closing receipt/chat.
- Original declaration commit: cf3f6bb96fed3c4cd0f9de9bf22b1f75a1bc3fa7.
- Manifest SHA256: fa87c99062b537cac51004f251394a11c7517e17f119acd827c4ee05359240db.
- Recovery manifest SHA256: 1e1d0e9c8e4f3645793bdb5c982bc56d8c5dacb2542a7bd74b93060eeb21f2c8.
- Checkpoint SHA256: 9cd1b00d402cc0447454dad6054dcc6f019b53e498469f209d2b749d4487e1d5.
- Frozen residual-head SHA256: e41eb678b938d239fdadb8bc1c77f9466007e460ea0dd53326262b7960f27d10.
- LIBERO revision: f78abd68ee283de9f9be3c8f7e2a9ad60246e95c; Plus revision: 4976dc30028e805ff8094b55501d532c48fec182.
- [Full report](../full/full-REPORT.md), [exact commands](../full/full-COMMANDS.md), [independent audit](../full/full-independent-audit.json), [standard metrics](../full/libero-metrics.json), [Plus metrics](../full/libero_plus-metrics.json).
- [W&B full run](https://wandb.ai/arm-aair-idit/pi05-two-evaluation-ranking/runs/bn7kzge3).
- [Raw tensors, immutable v0](https://wandb.ai/arm-aair-idit/pi05-two-evaluation-ranking/artifacts/initial-flow-probes/initial-flow-probes-full-fa87c99062b5/v0).
- [Evaluation evidence, immutable v0](https://wandb.ai/arm-aair-idit/pi05-two-evaluation-ranking/artifacts/evaluation/supervisor-full-fa87c99062b5/v0).
- Full media remains on Volt; lossless small raw-record shards are published in the full results directory.
