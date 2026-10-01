# Frozen π₀.₅ fixed flow-step study

Smoke validation only; excluded from main inference.

Each benchmark is reported separately. LIBERO and LIBERO-Plus use the same frozen LIBERO checkpoint; no RoboCasa episodes were evaluated.

Checkpoint full content SHA256: 9cd1b00d402cc0447454dad6054dcc6f019b53e498469f209d2b749d4487e1d5.
Historical metadata identity: dad4e2fbe79cceca79b83f3e53bb59c180e55ebb6815c67bfcdcf81768d7cee8.

## Benchmark results

### libero

Smoke only; excluded from study inference.

Benchmark revision: f78abd68ee283de9f9be3c8f7e2a9ad60246e95c. 1 paired cases, 1 task/condition clusters.
[W&B](https://wandb.ai/arm-aair-idit/pi05-frozen-flow-study/runs/pa2cslun)

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 1/1 (100.00%) | — | — | — |
| 2 | 1/1 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 1/1 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 1/1 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

| Steps | Velocity evaluations/chunk | Total velocity evaluations | Policy call mean / p95 (ms) | Episode mean / p95 (s) |
|---:|---:|---:|---:|---:|
| 1 | 1 | 15 | 33.53 / 33.99 | 31.16 / 31.16 |
| 2 | 2 | 32 | 35.62 / 36.16 | 32.40 / 32.40 |
| 4 | 4 | 60 | 38.96 / 39.25 | 31.42 / 31.42 |
| 10 | 10 | 150 | 51.01 / 51.40 | 31.39 / 31.39 |

#### Task-family results

**pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 1/1 (100.00%) | — | — | — |
| 2 | 1/1 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 1/1 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 1/1 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

Per-family costs, condition outcomes, and every paired case are in the JSON summary. Raw JSONL preserves chunk records and error attempts.

### libero_plus

Smoke only; excluded from study inference.

Benchmark revision: 4976dc30028e805ff8094b55501d532c48fec182. 1 paired cases, 1 task/condition clusters.
[W&B](https://wandb.ai/arm-aair-idit/pi05-frozen-flow-study/runs/8y1lygk3)

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 1/1 (100.00%) | — | — | — |
| 2 | 1/1 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 1/1 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 1/1 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

| Steps | Velocity evaluations/chunk | Total velocity evaluations | Policy call mean / p95 (ms) | Episode mean / p95 (s) |
|---:|---:|---:|---:|---:|
| 1 | 1 | 16 | 33.72 / 34.16 | 34.03 / 34.03 |
| 2 | 2 | 32 | 35.79 / 36.34 | 31.63 / 31.63 |
| 4 | 4 | 64 | 39.59 / 40.00 | 32.65 / 32.65 |
| 10 | 10 | 160 | 51.77 / 52.28 | 33.33 / 33.33 |

#### Task-family results

**pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 1/1 (100.00%) | — | — | — |
| 2 | 1/1 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 1/1 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 1/1 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

Per-family costs, condition outcomes, and every paired case are in the JSON summary. Raw JSONL preserves chunk records and error attempts.

## Cost and uncertainty boundaries

Policy latency includes synchronized production inference and input/output transforms; it excludes Gaussian noise generation, JSON and networking. Episode time includes simulator construction, reset, stabilization, policy requests and simulation; it excludes teardown. Compilation and parity are separate preparation records.
Intervals use 10,000 condition-cluster bootstrap draws retaining all states/arms. Plus conditions may share a base family, so condition clustering can understate family dependence. Single-condition family intervals are unavailable; zero discordance does not prove equivalence.
Objects Layout supplies one state per condition repeated at ten seeds. Other selected conditions use ten state indices. Severity-2/3 Plus conditions are an exploratory subset, not the official full benchmark.
Fewer than 20 one-step failures or five rescues leaves rescue benefit unresolved. No adaptive speedup is claimed.

## RoboCasa blocker

Public candidate changyeon/pi05_robocasa_as50_jax at revision 165da7e92fdbd140c4fe3e2a4bad6f0cabcda47e contains a completed Orbax save. Its configuration, camera/state/action adapter, and simulator provenance could not be verified. It was not evaluated with the LIBERO interface. See setup/ROBOCASA_DISCOVERY.md.

## Reproduction

Execution directory: /volt/code/frozen-flow-study on Volt pod uz2ptakxucbe. The manifest pins full checkpoint files, source files, benchmark assets, tasks, states, seeds and budgets. Status JSON records exact child argument lists and W&B links.
~~~bash
setsid -f .venv/bin/python examples/frozen_flow/supervise.py --manifest examples/frozen_flow/protocol.json --output /volt/artifacts/frozen-flow-study/runs --phase smoke > /volt/artifacts/frozen-flow-study/smoke-supervisor.log 2>&1
setsid -f .venv/bin/python examples/frozen_flow/supervise.py --manifest examples/frozen_flow/protocol.json --output /volt/artifacts/frozen-flow-study/runs --phase full > /volt/artifacts/frozen-flow-study/full-supervisor.log 2>&1
~~~

Errors halt execution and remain on the pod; publication occurs only after complete audits. Publication failures are recorded in supervisor status and require publication retry, not episode replacement.
