# Independent historical replay audit

All **548 declared episodes across 195 selected cases are present exactly once**. No error records or missing arms were found. The declaration SHA256 is `d9cfdf88561ec89206bd78d7104671e2d72e28cf48b12c947fc59cfd2b18cbdb`.

The status file's `records: 547` is a stale progress counter. In `replay_supervise.py`, lines 125–133 update it inside the worker polling loop. Once workers finish, the final coverage audit reads all rows; completion at line 159 stores the final summary but does not refresh the progress counter. The independently counted raw total and final summary both equal 548. No episode should be rerun to address this display discrepancy.

## Historical attribution

| Original study | Exact replay arms | Non-equivalent reconstructed arms |
|---|---:|---:|
| Frozen LIBERO | 17 | 103 |
| Frozen LIBERO-Plus | 42 | 154 |
| Independent selective study, LIBERO-Plus | 57 | 175 |
| Total | 116 | 432 |

Only 42 cases have all selected arms exactly reproduced: 4 standard LIBERO, 10 frozen-study Plus and 28 independent-study Plus. These are also the 42 cases with an exact one-step arm and at least one exact larger arm.

Use `case-attribution.json` to select eligible paired cases. Use `episode-attribution.jsonl` for each arm's video/trace path, raw record line, original record reference and complete mismatch list. An exact individual arm supports that original episode's visual label; comparisons about original rescues require both relevant arms to be exact. Non-equivalent videos describe reconstructions only.

Among the 432 non-equivalent episodes, 111 also change the success outcome. All 432 differ in action and subsequent observation hashes; 296 already differ in the first action hash. No initial observation, initial-state, stabilized-state or flow-noise mismatch was found by the independent comparison. These facts localize divergence but do not establish its cause.

## Checks and limits

The independent script enumerated the declaration, enforced strict Boolean outcomes, rejected duplicate/unexpected/missing keys, verified original file and record SHA256 references, independently recomputed every equivalence label, checked 304 pinned source hashes, and reconciled the final summary.

It also reran the recording auditor for current file hashes and numeric action/state/success evidence, plus inter-arm initial-state, observation, noise and per-action simulator RNG pairing. The media gate's prior full video-decoding audit is bound to the same unchanged raw-file hashes; video decoding was not rerun here. Its summary SHA256 is `28e6e8ec8cb990ae5e1039c6a09418d667e9831c60e2c72fc8c1442d8702608e`.

The existing supervisor additionally verifies the actual checkpoint tree after replay and runs complete media decoding before declaring completion. Its stale live counter is a reporting defect; the final exact-coverage gate correctly required all 548 records.

This outcome-selected replay is not a population estimate or an online predictor. No episodes were relaunched by this audit, and no git files were modified.

## Exact reproduction command

```bash
/volt/code/frozen-flow-study/.venv/bin/python /volt/artifacts/rescue-characterization/final-independent-replay/audit_replay.py
```

Outputs remain under `/volt/artifacts/rescue-characterization/final-independent-replay/`: `audit.json`, `case-attribution.json`, `episode-attribution.jsonl`, and `execution.log`.

