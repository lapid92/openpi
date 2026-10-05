# Independent final-count audit

This audit independently enumerates every declared matched case from the original raw JSONL records. It does not import the production aggregate-analysis code. It verifies exact four-arm completeness, strict Boolean outcomes, absence of errors and duplicates, identities, original action-count accounting, seeded flow noise, and paired initial state, observation and simulator RNG hashes.

The declaration is pinned to `e7537584d34855c24c1a38ca11da4f7e479b9bcbfc53d2cc7ec8f7ff46eadf9d`. The audit checks 302 declared source hashes, current GPU UUID allocation, and the actual checkpoint tree hash. Publication status is separate: the completed supervisor failed at GitHub push because raw files exceeded its size limit. This audit does not convert that publication failure into missing scientific episodes.

| Benchmark | Cases | Episodes | One-step failures | First success 2 / 4 / 10 / never | Rescues at 2 / 4 / 10 | Regressions at 2 / 4 / 10 |
|---|---:|---:|---:|---|---|---|
| LIBERO | 800 | 3,200 | 32 | 17 / 3 / 4 / 8 | 17 / 18 / 20 | 14 / 17 / 10 |
| LIBERO-Plus | 1,680 | 6,720 | 149 | 42 / 23 / 4 / 80 | 42 / 53 / 53 | 37 / 50 / 55 |

Counts and full success-pattern distributions agree exactly with the existing aggregate reports. LIBERO has 37 non-monotonic cases across the population; Plus has 128.

For one-step failures, success persists through all larger **tested** arms in 15 LIBERO cases and 44 Plus cases with a larger tested arm available. Five and 21 rescued cases, respectively, lose success at a larger tested arm. Four cases in each benchmark first succeed at 10; persistence beyond that is unmeasured. The legacy `persistent_rescues` field (19 and 48) includes this vacuous terminal-arm case and must not be interpreted as confirmed persistence; use `persists_with_larger_tested` and `first10_no_larger_tested`.

## Independently recomputed uncertainty

The primary bootstrap resamples the 40 standard LIBERO tasks, and the 27 suite-plus-task-family clusters for LIBERO-Plus. The Plus condition-cluster analysis (168 conditions) is reported separately as sensitivity. The independent implementation uses NumPy multinomial cluster counts, 10,000 replicates, seed 20261005 for primary results and 20261006 for condition sensitivity. It computes ratios of episode totals; undefined conditional denominators are omitted. Intervals are exploratory 95% percentile intervals without multiplicity correction.

| Quantity among one-step failures | LIBERO estimate (95% task-cluster interval) | Plus estimate (95% family-cluster interval) |
|---|---|---|
| First success at 2 | 53.13% (34.04–83.33%) | 28.19% (19.77–38.89%) |
| First success at 4 | 9.38% (0–20.59%) | 15.44% (9.38–22.83%) |
| First success at 10 | 12.50% (0–25.81%) | 2.68% (0.64–4.95%) |
| Never succeeds | 25.00% (8.33–38.89%) | 53.69% (42.86–62.41%) |

Plus net success differences at 2 / 4 / 10 are +0.30 / +0.18 / −0.12 percentage points. Their family-cluster intervals are [−0.88,+1.51], [−0.77,+1.08], and [−1.38,+0.99] percentage points. Every interval includes zero. Rescue counts alone do not establish a population gain or an online adaptive policy.

This audit does **not** rerun media decoding. The separate `runs/libero-recording-audit.json` and `runs/libero_plus-recording-audit.json` report passing full decoding and trace checks for all 3,200 and 6,720 episodes. Failure-mode labels need their own visual evidence and reviewer provenance; these counts do not establish mechanisms.

## Reproduce exactly on the pod

```bash
/volt/code/frozen-flow-study/.venv/bin/python /volt/artifacts/rescue-characterization/final-independent-counts/reproduce.py
```

For execution independent of the orchestrating computer:

```bash
nohup /volt/code/frozen-flow-study/.venv/bin/python /volt/artifacts/rescue-characterization/final-independent-counts/reproduce.py > /volt/artifacts/rescue-characterization/final-independent-counts/execution.log 2>&1 < /dev/null &
```

Outputs: `audit.json` contains identities, raw-file SHA256 digests, counts, distributions and uncertainty; `case-vectors.jsonl` contains every matched outcome vector and exact raw file/line references. All files and computation remain on the pod.

