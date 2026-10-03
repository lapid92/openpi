# Independent selective fixed-step study

Inconclusive under the predeclared joint success/cost support criterion.

Benchmark: libero_plus; revision: 4976dc30028e805ff8094b55501d532c48fec182.
Manifest SHA256: 209e434b336372b37d80b7a101912b4b8daeb9f5cd71a0e9772fdfe1579e782c.
Checkpoint SHA256: 9cd1b00d402cc0447454dad6054dcc6f019b53e498469f209d2b749d4487e1d5.
Frozen residual head SHA256: e41eb678b938d239fdadb8bc1c77f9466007e460ea0dd53326262b7960f27d10.
Validated episode records: 1920; error attempts: 0.
W&B: https://wandb.ai/arm-aair-idit/pi05-independent-selective-steps/runs/1aowjys9.

Rule: 10 steps for Robot Initial States and Objects Layout; 1 for Camera Viewpoints.
Rule results select matched fixed-arm records; smoke checks the actual rule wrapper.

## Overall

960 pairs; 96 conditions; 31 families.

| Strategy | Success | Equal-case mean policy call ms | Policy call mean/p50/p95 ms | Head mean ms | Episode mean/p50/p95 ms | Velocity evaluations |
|---|---:|---:|---:|---:|---:|---:|
| fixed_1 | 887/960 | 32.91951 | 32.90583/32.87864/33.83978 | 0.58965 | 52299.19837/43827.01606/93281.31355 | 34117 |
| fixed_10 | 877/960 | 51.21226 | 51.19843/51.20278/52.16103 | 0.59726 | 53291.48803/44018.29557/92641.66985 | 343130 |
| metadata | 886/960 | 45.11954 | 45.19153/50.82658/52.07363 | 0.59524 | 52848.03301/43957.56289/92308.09733 | 240173 |

Differences are candidate minus reference; success units are fractions. Intervals are separate 95% cluster bootstrap intervals.

**metadata_vs_fixed10**: rescues/regressions 20/11; reference failures rescued 20/83 (0.24096).
Success difference 0.00937; condition CI [-0.00208, 0.02187]; family sensitivity CI [-0.00109, 0.01982].
Equal-case mean policy-call difference -6.09272 ms; condition CI [-7.81734, -4.38425]; family sensitivity CI [-7.17959, -4.98848].

**metadata_vs_fixed1**: rescues/regressions 22/23; reference failures rescued 22/73 (0.30137).
Success difference -0.00104; condition CI [-0.01875, 0.01667]; family sensitivity CI [-0.01792, 0.01939].
Equal-case mean policy-call difference 12.20003 ms; condition CI [10.46904, 13.90991]; family sensitivity CI [11.13274, 13.30684].

**fixed10_vs_fixed1**: rescues/regressions 33/43; reference failures rescued 33/73 (0.45205).
Success difference -0.01042; condition CI [-0.03125, 0.01146]; family sensitivity CI [-0.03158, 0.01364].
Equal-case mean policy-call difference 18.29275 ms; condition CI [18.23889, 18.34460]; family sensitivity CI [18.19808, 18.38443].

## Camera Viewpoints

320 pairs; 32 conditions; 22 families.

| Strategy | Success | Equal-case mean policy call ms | Policy call mean/p50/p95 ms | Head mean ms | Episode mean/p50/p95 ms | Velocity evaluations |
|---|---:|---:|---:|---:|---:|---:|
| fixed_1 | 298/320 | 32.89473 | 32.86809/32.84298/33.82090 | 0.58826 | 50823.67473/43390.09847/85271.56517 | 11203 |
| fixed_10 | 289/320 | 51.17289 | 51.15292/51.14989/52.04827 | 0.59446 | 52154.03979/43691.25490/86171.04319 | 114160 |
| metadata | 298/320 | 32.89473 | 32.86809/32.84298/33.82090 | 0.58826 | 50823.67473/43390.09847/85271.56517 | 11203 |

Differences are candidate minus reference; success units are fractions. Intervals are separate 95% cluster bootstrap intervals.

**metadata_vs_fixed10**: rescues/regressions 20/11; reference failures rescued 20/31 (0.64516).
Success difference 0.02813; condition CI [-0.00625, 0.06563]; family sensitivity CI [-0.00313, 0.06000].
Equal-case mean policy-call difference -18.27816 ms; condition CI [-18.37503, -18.17636]; family sensitivity CI [-18.39034, -18.15793].

**metadata_vs_fixed1**: rescues/regressions 0/0; reference failures rescued 0/22 (0.00000).
Success difference 0.00000; condition CI [0.00000, 0.00000]; family sensitivity CI [0.00000, 0.00000].
Equal-case mean policy-call difference 0.00000 ms; condition CI [0.00000, 0.00000]; family sensitivity CI [0.00000, 0.00000].

**fixed10_vs_fixed1**: rescues/regressions 11/20; reference failures rescued 11/22 (0.50000).
Success difference -0.02813; condition CI [-0.06563, 0.00625]; family sensitivity CI [-0.06000, 0.00313].
Equal-case mean policy-call difference 18.27816 ms; condition CI [18.17636, 18.37503]; family sensitivity CI [18.15793, 18.39034].

## Objects Layout

320 pairs; 32 conditions; 21 families.

| Strategy | Success | Equal-case mean policy call ms | Policy call mean/p50/p95 ms | Head mean ms | Episode mean/p50/p95 ms | Velocity evaluations |
|---|---:|---:|---:|---:|---:|---:|
| fixed_1 | 293/320 | 32.99800 | 32.99939/32.98917/33.92548 | 0.59302 | 57367.97993/45558.38011/133452.64776 | 11925 |
| fixed_10 | 292/320 | 51.28107 | 51.29583/51.29176/52.28259 | 0.60001 | 58282.76972/46375.91805/137240.28375 | 119260 |
| metadata | 292/320 | 51.28107 | 51.29583/51.29176/52.28259 | 0.60001 | 58282.76972/46375.91805/137240.28375 | 119260 |

Differences are candidate minus reference; success units are fractions. Intervals are separate 95% cluster bootstrap intervals.

**metadata_vs_fixed10**: rescues/regressions 0/0; reference failures rescued 0/28 (0.00000).
Success difference 0.00000; condition CI [0.00000, 0.00000]; family sensitivity CI [0.00000, 0.00000].
Equal-case mean policy-call difference 0.00000 ms; condition CI [0.00000, 0.00000]; family sensitivity CI [0.00000, 0.00000].

**metadata_vs_fixed1**: rescues/regressions 11/12; reference failures rescued 11/27 (0.40741).
Success difference -0.00313; condition CI [-0.04375, 0.04063]; family sensitivity CI [-0.04194, 0.04545].
Equal-case mean policy-call difference 18.28307 ms; condition CI [18.19250, 18.36500]; family sensitivity CI [18.17929, 18.37444].

**fixed10_vs_fixed1**: rescues/regressions 11/12; reference failures rescued 11/27 (0.40741).
Success difference -0.00313; condition CI [-0.04375, 0.04063]; family sensitivity CI [-0.04194, 0.04545].
Equal-case mean policy-call difference 18.28307 ms; condition CI [18.19250, 18.36500]; family sensitivity CI [18.17929, 18.37444].

## Robot Initial States

320 pairs; 32 conditions; 16 families.

| Strategy | Success | Equal-case mean policy call ms | Policy call mean/p50/p95 ms | Head mean ms | Episode mean/p50/p95 ms | Velocity evaluations |
|---|---:|---:|---:|---:|---:|---:|
| fixed_1 | 296/320 | 32.86578 | 32.84276/32.81600/33.73685 | 0.58742 | 48705.94043/38754.21042/81030.81666 | 10989 |
| fixed_10 | 296/320 | 51.18282 | 51.13991/51.16325/52.07154 | 0.59719 | 49437.65458/39266.70371/83880.44680 | 109710 |
| metadata | 296/320 | 51.18282 | 51.13991/51.16325/52.07154 | 0.59719 | 49437.65458/39266.70371/83880.44680 | 109710 |

Differences are candidate minus reference; success units are fractions. Intervals are separate 95% cluster bootstrap intervals.

**metadata_vs_fixed10**: rescues/regressions 0/0; reference failures rescued 0/24 (0.00000).
Success difference 0.00000; condition CI [0.00000, 0.00000]; family sensitivity CI [0.00000, 0.00000].
Equal-case mean policy-call difference 0.00000 ms; condition CI [0.00000, 0.00000]; family sensitivity CI [0.00000, 0.00000].

**metadata_vs_fixed1**: rescues/regressions 11/11; reference failures rescued 11/24 (0.45833).
Success difference 0.00000; condition CI [-0.03133, 0.03125]; family sensitivity CI [-0.03438, 0.03438].
Equal-case mean policy-call difference 18.31703 ms; condition CI [18.22509, 18.40228]; family sensitivity CI [18.19775, 18.42905].

**fixed10_vs_fixed1**: rescues/regressions 11/11; reference failures rescued 11/24 (0.45833).
Success difference 0.00000; condition CI [-0.03133, 0.03125]; family sensitivity CI [-0.03438, 0.03438].
Equal-case mean policy-call difference 18.31703 ms; condition CI [18.22509, 18.40228]; family sensitivity CI [18.19775, 18.42905].

## Initial sigma ranking

Higher first-chunk initial sigma predicts rescue. AUC uses only discordant fixed-1/fixed-10 pairs; this retrospective conditioning cannot validate a prospective selector.

**Overall**: 33 rescues, 43 regressions, 42 discordant conditions; AUC 0.50951. Count threshold met; interpret uncertainty and type/family sensitivity.
condition_id CI [0.36235, 0.66029]; valid draws 10000/10000. Single-class draws excluded.
family CI [0.37134, 0.65035]; valid draws 10000/10000. Single-class draws excluded.

**Camera Viewpoints**: 11 rescues, 20 regressions, 17 discordant conditions; AUC 0.38636. Count threshold met; interpret uncertainty and type/family sensitivity.
condition_id CI [0.18124, 0.57779]; valid draws 10000/10000. Single-class draws excluded.
family CI [0.22222, 0.55556]; valid draws 10000/10000. Single-class draws excluded.

**Objects Layout**: 11 rescues, 12 regressions, 12 discordant conditions; AUC 0.66667. Count threshold met; interpret uncertainty and type/family sensitivity.
condition_id CI [0.29727, 0.92308]; valid draws 9982/10000. Single-class draws excluded.
family CI [0.25000, 0.88889]; valid draws 9963/10000. Single-class draws excluded.

**Robot Initial States**: 11 rescues, 11 regressions, 13 discordant conditions; AUC 0.59504. Count threshold met; interpret uncertainty and type/family sensitivity.
condition_id CI [0.28177, 1.00000]; valid draws 10000/10000. Single-class draws excluded.
family CI [0.28571, 1.00000]; valid draws 9999/10000. Single-class draws excluded.

## Interpretation limits

- Metadata results reuse matched fixed-arm records; smoke separately checks actual wrapper equivalence.
- Primary costs use equal-case mean scored-policy-call latency; total policy time and simulator time are separate.
- The support criterion is not a conventional noninferiority margin, equivalence test or joint 95% confidence statement.
- Condition-cluster inference has family-cluster sensitivity; repeated states and few families limit generalization.
- Sigma ranking conditions on observed discordance and cannot by itself justify an episode-level selector.
- No trained selector, modified predictor, adaptive speedup or pooled benchmark rate is inferred.
- All declared main cases retained.

Primary support requires success CI lower >= 0 and policy-call cost CI upper < 0. This is neither equivalence nor a joint 95% guarantee.
Full per-family results, paired-case outcomes, all cost distributions and bootstrap seeds are in the adjacent validated summary JSON.
