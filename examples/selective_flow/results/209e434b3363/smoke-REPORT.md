# Independent selective fixed-step study

Smoke validation only; excluded from scientific inference.

Benchmark: libero_plus; revision: 4976dc30028e805ff8094b55501d532c48fec182.
Manifest SHA256: 209e434b336372b37d80b7a101912b4b8daeb9f5cd71a0e9772fdfe1579e782c.
Checkpoint SHA256: 9cd1b00d402cc0447454dad6054dcc6f019b53e498469f209d2b749d4487e1d5.
Frozen residual head SHA256: e41eb678b938d239fdadb8bc1c77f9466007e460ea0dd53326262b7960f27d10.
Validated episode records: 9; error attempts: 0.
W&B: https://wandb.ai/arm-aair-idit/pi05-independent-selective-steps/runs/la6kg4vn.

Rule: 10 steps for Robot Initial States and Objects Layout; 1 for Camera Viewpoints.
Rule results select matched fixed-arm records; smoke checks the actual rule wrapper.

## Overall

3 pairs; 3 conditions; 2 families.

| Strategy | Success | Equal-case mean policy call ms | Policy call mean/p50/p95 ms | Head mean ms | Episode mean/p50/p95 ms | Velocity evaluations |
|---|---:|---:|---:|---:|---:|---:|
| fixed_1 | 3/3 | 32.96071 | 32.93578/32.93271/33.43599 | 0.62263 | 43752.82701/40289.65668/54952.20985 | 67 |
| fixed_10 | 3/3 | 51.19640 | 51.15304/51.13388/51.86074 | 0.61738 | 42580.09372/41198.12027/50823.15151 | 640 |
| metadata | 3/3 | 45.10797 | 44.77548/50.74001/51.86303 | 0.61038 | 42277.27252/40289.65668/50732.30515 | 432 |

Differences are candidate minus reference; success units are fractions. Intervals are separate 95% cluster bootstrap intervals.

**metadata_vs_fixed10**: rescues/regressions 0/0; reference failures rescued 0/0 (unavailable).
Success difference 0.00000; condition CI [0.00000, 0.00000]; family sensitivity CI [0.00000, 0.00000].
Equal-case mean policy-call difference -6.08843 ms; condition CI [-18.26530, 0.00000]; family sensitivity CI [-9.13265, 0.00000].

**metadata_vs_fixed1**: rescues/regressions 0/0; reference failures rescued 0/0 (unavailable).
Success difference 0.00000; condition CI [0.00000, 0.00000]; family sensitivity CI [0.00000, 0.00000].
Equal-case mean policy-call difference 12.14726 ms; condition CI [0.00000, 18.42645]; family sensitivity CI [9.00766, 18.42645].

**fixed10_vs_fixed1**: rescues/regressions 0/0; reference failures rescued 0/0 (unavailable).
Success difference 0.00000; condition CI [0.00000, 0.00000]; family sensitivity CI [0.00000, 0.00000].
Equal-case mean policy-call difference 18.23569 ms; condition CI [18.01533, 18.42645]; family sensitivity CI [18.14031, 18.42645].

## Camera Viewpoints

1 pairs; 1 conditions; 1 families.

| Strategy | Success | Equal-case mean policy call ms | Policy call mean/p50/p95 ms | Head mean ms | Episode mean/p50/p95 ms | Velocity evaluations |
|---|---:|---:|---:|---:|---:|---:|
| fixed_1 | 1/1 | 32.83737 | 32.83737/32.81951/33.20891 | 0.60225 | 40289.65668/40289.65668/40289.65668 | 22 |
| fixed_10 | 1/1 | 51.10267 | 51.10267/51.13937/51.58981 | 0.62209 | 41198.12027/41198.12027/41198.12027 | 230 |
| metadata | 1/1 | 32.83737 | 32.83737/32.81951/33.20891 | 0.60225 | 40289.65668/40289.65668/40289.65668 | 22 |

Differences are candidate minus reference; success units are fractions. Intervals are separate 95% cluster bootstrap intervals.

**metadata_vs_fixed10**: rescues/regressions 0/0; reference failures rescued 0/0 (unavailable).
Success difference 0.00000; condition CI unavailable; family sensitivity CI unavailable.
Equal-case mean policy-call difference -18.26530 ms; condition CI unavailable; family sensitivity CI unavailable.

**metadata_vs_fixed1**: rescues/regressions 0/0; reference failures rescued 0/0 (unavailable).
Success difference 0.00000; condition CI unavailable; family sensitivity CI unavailable.
Equal-case mean policy-call difference 0.00000 ms; condition CI unavailable; family sensitivity CI unavailable.

**fixed10_vs_fixed1**: rescues/regressions 0/0; reference failures rescued 0/0 (unavailable).
Success difference 0.00000; condition CI unavailable; family sensitivity CI unavailable.
Equal-case mean policy-call difference 18.26530 ms; condition CI unavailable; family sensitivity CI unavailable.

## Objects Layout

1 pairs; 1 conditions; 1 families.

| Strategy | Success | Equal-case mean policy call ms | Policy call mean/p50/p95 ms | Head mean ms | Episode mean/p50/p95 ms | Velocity evaluations |
|---|---:|---:|---:|---:|---:|---:|
| fixed_1 | 1/1 | 32.86492 | 32.86492/32.89330/33.19947 | 0.63238 | 56581.38243/56581.38243/56581.38243 | 28 |
| fixed_10 | 1/1 | 50.88025 | 50.88025/50.92156/51.19703 | 0.59438 | 51892.59942/51892.59942/51892.59942 | 240 |
| metadata | 1/1 | 50.88025 | 50.88025/50.92156/51.19703 | 0.59438 | 51892.59942/51892.59942/51892.59942 | 240 |

Differences are candidate minus reference; success units are fractions. Intervals are separate 95% cluster bootstrap intervals.

**metadata_vs_fixed10**: rescues/regressions 0/0; reference failures rescued 0/0 (unavailable).
Success difference 0.00000; condition CI unavailable; family sensitivity CI unavailable.
Equal-case mean policy-call difference 0.00000 ms; condition CI unavailable; family sensitivity CI unavailable.

**metadata_vs_fixed1**: rescues/regressions 0/0; reference failures rescued 0/0 (unavailable).
Success difference 0.00000; condition CI unavailable; family sensitivity CI unavailable.
Equal-case mean policy-call difference 18.01533 ms; condition CI unavailable; family sensitivity CI unavailable.

**fixed10_vs_fixed1**: rescues/regressions 0/0; reference failures rescued 0/0 (unavailable).
Success difference 0.00000; condition CI unavailable; family sensitivity CI unavailable.
Equal-case mean policy-call difference 18.01533 ms; condition CI unavailable; family sensitivity CI unavailable.

## Robot Initial States

1 pairs; 1 conditions; 1 families.

| Strategy | Success | Equal-case mean policy call ms | Policy call mean/p50/p95 ms | Head mean ms | Episode mean/p50/p95 ms | Velocity evaluations |
|---|---:|---:|---:|---:|---:|---:|
| fixed_1 | 1/1 | 33.17984 | 33.17984/33.15717/33.65786 | 0.63292 | 34387.44193/34387.44193/34387.44193 | 17 |
| fixed_10 | 1/1 | 51.60629 | 51.60629/51.46062/52.25452 | 0.64347 | 34649.56146/34649.56146/34649.56146 | 170 |
| metadata | 1/1 | 51.60629 | 51.60629/51.46062/52.25452 | 0.64347 | 34649.56146/34649.56146/34649.56146 | 170 |

Differences are candidate minus reference; success units are fractions. Intervals are separate 95% cluster bootstrap intervals.

**metadata_vs_fixed10**: rescues/regressions 0/0; reference failures rescued 0/0 (unavailable).
Success difference 0.00000; condition CI unavailable; family sensitivity CI unavailable.
Equal-case mean policy-call difference 0.00000 ms; condition CI unavailable; family sensitivity CI unavailable.

**metadata_vs_fixed1**: rescues/regressions 0/0; reference failures rescued 0/0 (unavailable).
Success difference 0.00000; condition CI unavailable; family sensitivity CI unavailable.
Equal-case mean policy-call difference 18.42645 ms; condition CI unavailable; family sensitivity CI unavailable.

**fixed10_vs_fixed1**: rescues/regressions 0/0; reference failures rescued 0/0 (unavailable).
Success difference 0.00000; condition CI unavailable; family sensitivity CI unavailable.
Equal-case mean policy-call difference 18.42645 ms; condition CI unavailable; family sensitivity CI unavailable.

## Initial sigma ranking

Higher first-chunk initial sigma predicts rescue. AUC uses only discordant fixed-1/fixed-10 pairs; this retrospective conditioning cannot validate a prospective selector.

**Overall**: 0 rescues, 0 regressions, 0 discordant conditions; AUC unavailable. Descriptive only: too few rescues, regressions or conditions.
condition_id CI unavailable; valid draws 0/10000. Single-class draws excluded.
family CI unavailable; valid draws 0/10000. Single-class draws excluded.

**Camera Viewpoints**: 0 rescues, 0 regressions, 0 discordant conditions; AUC unavailable. Descriptive only: too few rescues, regressions or conditions.
condition_id CI unavailable; valid draws 0/10000. Single-class draws excluded.
family CI unavailable; valid draws 0/10000. Single-class draws excluded.

**Objects Layout**: 0 rescues, 0 regressions, 0 discordant conditions; AUC unavailable. Descriptive only: too few rescues, regressions or conditions.
condition_id CI unavailable; valid draws 0/10000. Single-class draws excluded.
family CI unavailable; valid draws 0/10000. Single-class draws excluded.

**Robot Initial States**: 0 rescues, 0 regressions, 0 discordant conditions; AUC unavailable. Descriptive only: too few rescues, regressions or conditions.
condition_id CI unavailable; valid draws 0/10000. Single-class draws excluded.
family CI unavailable; valid draws 0/10000. Single-class draws excluded.

## Interpretation limits

- Metadata results reuse matched fixed-arm records; smoke separately checks actual wrapper equivalence.
- Primary costs use equal-case mean scored-policy-call latency; total policy time and simulator time are separate.
- The support criterion is not a conventional noninferiority margin, equivalence test or joint 95% confidence statement.
- Condition-cluster inference has family-cluster sensitivity; repeated states and few families limit generalization.
- Sigma ranking conditions on observed discordance and cannot by itself justify an episode-level selector.
- No trained selector, modified predictor, adaptive speedup or pooled benchmark rate is inferred.
- Smoke results are excluded from primary scientific interpretation.

Primary support requires success CI lower >= 0 and policy-call cost CI upper < 0. This is neither equivalence nor a joint 95% guarantee.
Full per-family results, paired-case outcomes, all cost distributions and bootstrap seeds are in the adjacent validated summary JSON.
