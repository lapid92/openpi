# Frozen π₀.₅ rescue characterization

Smoke validation only; excluded from main inference.

Each benchmark is reported separately. LIBERO and LIBERO-Plus use the same frozen LIBERO checkpoint; no RoboCasa episodes were evaluated.

Checkpoint full content SHA256: 9cd1b00d402cc0447454dad6054dcc6f019b53e498469f209d2b749d4487e1d5.
Historical metadata identity: dad4e2fbe79cceca79b83f3e53bb59c180e55ebb6815c67bfcdcf81768d7cee8.

## Benchmark results

### libero

Smoke only; excluded from study inference.

Benchmark revision: f78abd68ee283de9f9be3c8f7e2a9ad60246e95c. 4 paired cases, 4 task/condition clusters.
[W&B](https://wandb.ai/arm-aair-idit/pi05-rescue-characterization/runs/ksvgwbi0)

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 4/4 (100.00%) | — | — | — |
| 2 | 4/4 (100.00%) | 0 / 0 | 0.00% [0.00%, 0.00%] | 0/0 (undefined) |
| 4 | 4/4 (100.00%) | 0 / 0 | 0.00% [0.00%, 0.00%] | 0/0 (undefined) |
| 10 | 4/4 (100.00%) | 0 / 0 | 0.00% [0.00%, 0.00%] | 0/0 (undefined) |

| Steps | Velocity evaluations/chunk | Total velocity evaluations | Policy call mean / p95 (ms) | Episode mean / p95 (s) |
|---:|---:|---:|---:|---:|
| 1 | 1 | 128 | 32.73 / 33.29 | 43.89 / 64.32 |
| 2 | 2 | 314 | 34.56 / 35.06 | 51.71 / 90.88 |
| 4 | 4 | 552 | 38.61 / 39.23 | 46.58 / 74.67 |
| 10 | 10 | 1380 | 50.56 / 51.28 | 46.74 / 74.48 |

#### First-success and non-monotonic patterns

Full success-pattern counts: {"succeed/succeed/succeed/succeed": 4}
First success among one-step failures: {}
One-step failures: 0.
Primary uncertainty uses whole base families for Plus and tasks for LIBERO. Full intervals and distributions are in libero-smoke-patterns.json.
Visual failure labels remain pending independent review; numerical rescues alone do not establish a failure mechanism.

#### Task-family results

**LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 1/1 (100.00%) | — | — | — |
| 2 | 1/1 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 1/1 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 1/1 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**open_the_middle_drawer_of_the_cabinet**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 1/1 (100.00%) | — | — | — |
| 2 | 1/1 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 1/1 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 1/1 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**pick_up_the_alphabet_soup_and_place_it_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 1/1 (100.00%) | — | — | — |
| 2 | 1/1 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 1/1 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 1/1 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

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

Benchmark revision: 4976dc30028e805ff8094b55501d532c48fec182. 7 paired cases, 7 task/condition clusters.
[W&B](https://wandb.ai/arm-aair-idit/pi05-rescue-characterization/runs/qm2v25gg)

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 7/7 (100.00%) | — | — | — |
| 2 | 7/7 (100.00%) | 0 / 0 | 0.00% [0.00%, 0.00%] | 0/0 (undefined) |
| 4 | 7/7 (100.00%) | 0 / 0 | 0.00% [0.00%, 0.00%] | 0/0 (undefined) |
| 10 | 7/7 (100.00%) | 0 / 0 | 0.00% [0.00%, 0.00%] | 0/0 (undefined) |

| Steps | Velocity evaluations/chunk | Total velocity evaluations | Policy call mean / p95 (ms) | Episode mean / p95 (s) |
|---:|---:|---:|---:|---:|
| 1 | 1 | 123 | 32.90 / 33.39 | 35.89 / 42.41 |
| 2 | 2 | 246 | 34.83 / 35.43 | 35.71 / 40.92 |
| 4 | 4 | 488 | 38.71 / 39.54 | 35.56 / 40.36 |
| 10 | 10 | 1230 | 50.81 / 51.44 | 36.06 / 42.38 |

#### First-success and non-monotonic patterns

Full success-pattern counts: {"succeed/succeed/succeed/succeed": 7}
First success among one-step failures: {}
One-step failures: 0.
Primary uncertainty uses whole base families for Plus and tasks for LIBERO. Full intervals and distributions are in libero_plus-smoke-patterns.json.
Visual failure labels remain pending independent review; numerical rescues alone do not establish a failure mechanism.

#### Task-family results

**pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 6/6 (100.00%) | — | — | — |
| 2 | 6/6 (100.00%) | 0 / 0 | 0.00% [0.00%, 0.00%] | 0/0 (undefined) |
| 4 | 6/6 (100.00%) | 0 / 0 | 0.00% [0.00%, 0.00%] | 0/0 (undefined) |
| 10 | 6/6 (100.00%) | 0 / 0 | 0.00% [0.00%, 0.00%] | 0/0 (undefined) |

**pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 1/1 (100.00%) | — | — | — |
| 2 | 1/1 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 1/1 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 1/1 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

Per-family costs, condition outcomes, and every paired case are in the JSON summary. Raw JSONL preserves chunk records and error attempts.

## Cost and uncertainty boundaries

Policy latency includes synchronized production inference and input/output transforms; it excludes Gaussian noise generation, JSON and networking. Episode time includes simulator construction, reset, stabilization, policy requests and simulation; it excludes teardown. Compilation and parity are separate preparation records.
Arm tables use condition-cluster intervals. The pattern JSON additionally gives the PRIMARY Plus family-cluster intervals. Use these for scientific conclusions; condition clustering is sensitivity. Zero discordance does not prove equivalence.
Objects Layout supplies one state per condition repeated at ten seeds. Plus uses severity3/4 across7 axes with10 seeds; standard LIBERO uses20 seeds. Conditions are a predeclared subset, not the official full benchmark.
Fewer than50 one-step failures fails the predeclared adequacy target. A descriptive recurring class requires at least10 rescues across3 families and5 conditions in the new population, with counterexamples and uncertainty. No adaptive speedup is claimed.

## Reproduction

Execution directory: /volt/code/frozen-flow-study on Volt pod uz2ptakxucbe. The manifest pins full checkpoint files, source files, benchmark assets, tasks, states, seeds and budgets. Status JSON records exact child argument lists and W&B links.
~~~bash
setsid -f .venv/bin/python examples/rescue_characterization/supervise.py --manifest examples/rescue_characterization/protocol.json --output /volt/artifacts/rescue-characterization/runs --phase smoke > /volt/artifacts/rescue-characterization/smoke-supervisor.log 2>&1
setsid -f .venv/bin/python examples/rescue_characterization/supervise.py --manifest examples/rescue_characterization/protocol.json --output /volt/artifacts/rescue-characterization/runs --phase full > /volt/artifacts/rescue-characterization/full-supervisor.log 2>&1
~~~

Errors halt execution and remain on the pod; publication occurs only after complete audits. Publication failures are recorded in supervisor status and require publication retry, not episode replacement.
