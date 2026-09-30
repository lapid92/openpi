# Frozen π₀.₅ two-stage LIBERO-Plus pilot: results

The protocol in [TWO_STAGE_PROTOCOL.md](TWO_STAGE_PROTOCOL.md) and its manifest were committed before the smoke and screening outcomes. This is an exploratory subset of LIBERO-Plus, not its official full-benchmark score. Frozen π₀.₅ and the existing 3,000-step residual head were used throughout; no model training, adaptive threshold tuning, or TVM was involved.

## Identity and execution

- Branch: `codex/pi05-libero-plus-two-stage`, based on pilot commit `3e156fa44c6cf1360878080e2ab0d019d0562a58`.
- Benchmark checkout: `4976dc30028e805ff8094b55501d532c48fec182`; asset and classification digests are in the protocol manifest.
- Protocol manifest SHA256: `ca005b65c447321cefb6fc68cb75984a0e67064bedf066a10e9bb49952fbd8c7`.
- Base checkpoint identity: `dad4e2fbe79cceca79b83f3e53bb59c180e55ebb6815c67bfcdcf81768d7cee8`; residual head SHA256: `e41eb678b938d239fdadb8bc1c77f9466007e460ea0dd53326262b7960f27d10`.
- Pod `2vzhlaphss5c`, `CUDA_VISIBLE_DEVICES=3`, physical UUID `GPU-4779cdde-a260-f8ec-da6a-6fa390bc7fd7`. The model process saw this as `cuda:0`; the isolated simulator had no GPU visible.
- Smoke: one successful goal/camera severity-1 episode at seed 3; 26 action chunks, one velocity evaluation per chunk. [W&B smoke run](https://wandb.ai/arm-aair-idit/pi05-libero-velocity-residual/runs/f7v4t4a0).
- The fixed 1, 2, 4, and 10-step scored paths matched the existing public fixed sampler exactly on a synthetic same-input GPU check (maximum final-action difference 0.0). That verification used extra evaluations outside the scored episodes. Same-pass σ logging required no extra base evaluation and did not change the fixed output.
- Focused tests: 19 passed on CPU; Ruff check and format check passed. The reviewer examined the actual diff and cleared the screening provenance regression. The screen records precede condition selection; comparison records are required to carry its SHA.

## Stage 1: one-step screening

Screening used seeds 17, 29, and 43 on each of 12 declared conditions. All 36 records are saved under [results/two_stage_2026_09_30](results/two_stage_2026_09_30). A condition qualified only at one or two successes out of three. The selected-condition file freezes the complete screen SHA256 `dc30e849ce260819a14b5dcda6930ddf5b11b3871c5809d6deba96605d295c7a`. [W&B screening run](https://wandb.ai/arm-aair-idit/pi05-libero-velocity-residual/runs/b6xyzprv).

| Task family | Perturbation | Severity 1 | Severity 3 |
| --- | --- | ---: | ---: |
| Drawer opening (LIBERO Goal) | Camera viewpoints | 3/3 | 3/3 |
| Drawer opening (LIBERO Goal) | Robot initial states | **2/3 selected** | 0/3 |
| BBQ sauce into basket (LIBERO Object) | Camera viewpoints | 3/3 | 3/3 |
| BBQ sauce into basket (LIBERO Object) | Robot initial states | 3/3 | **2/3 selected** |
| Stove and moka pot (LIBERO 10) | Camera viewpoints | 3/3 | 3/3 |
| Stove and moka pot (LIBERO 10) | Robot initial states | 3/3 | 3/3 |

## Stage 2: held-out paired comparison

The selected conditions were run at fresh seeds 101, 131, 167, and 197 with all four fixed-step arms: 32 episodes, eight matched cases. The comparison record SHA256 is `1b382c871d8efe44d03c4ba552254a811f75bc72c4b021e641b83ef48e1cd333`. Pairing validation checked initial simulator state, first observation and noise hashes, σ agreement within the predeclared tolerance, checkpoint identity, per-chunk evaluation counts, and complete episode sets. [W&B comparison run](https://wandb.ai/arm-aair-idit/pi05-libero-velocity-residual/runs/b0gcl3wr).

| Condition / family | 1 step | 2 steps | 4 steps | 10 steps | Paired wins/losses of each extra-step arm vs 1 |
| --- | ---: | ---: | ---: | ---: | --- |
| Drawer, robot state severity 1 / Goal | 4/4 | 4/4 | 4/4 | 4/4 | 0/0 |
| BBQ sauce, robot state severity 3 / Object | 4/4 | 3/4 | 3/4 | 3/4 | 0/1 |
| Combined | **8/8** | 7/8 | 7/8 | 7/8 | **0/1**, each |

All selected held-out conditions were Robot Initial States; no Camera Viewpoints condition qualified in screening. By perturbation type, the combined result is therefore the Robot Initial States row. The 95% exact binomial interval for one-step success is [0.631, 1.000]; for each other arm, [0.473, 0.997]. The paired success-rate difference is −0.125 for every extra-step arm, with exploratory condition-cluster bootstrap 95% interval [−0.25, 0.00]. There are only two selected condition clusters; these intervals are coarse.

| Fixed steps | Velocity evaluations/chunk mean, p95 | Scored policy/chunk mean, p95 (ms) | Episode wall time mean, p95 (s) | Chunks |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1, 1 | 32.45, 33.02 | 34.55, 43.62 | 232 |
| 2 | 2, 2 | 34.44, 35.05 | 39.90, 52.93 | 276 |
| 4 | 4, 4 | 38.39, 39.08 | 43.89, 56.57 | 307 |
| 10 | 10, 10 | 50.67, 51.29 | 39.45, 57.47 | 263 |

Scored policy time is synchronized model computation for an action chunk. HTTP request time and full simulator episode wall time are separately recorded in the raw records and comparison summary; episode duration also depends on the number of actions taken, so its ordering need not follow per-chunk compute. This fixed-step experiment makes no adaptive speedup claim.

The head's initial σ was logged from the same action-expert pass without changing actions. There were **zero one-step failures** in the held-out set and zero rescues, so σ ranking of rescued versus unrescued failures is **unmeasured**; no AUROC or correlation is interpreted. Demonstration-action error and fine-step numerical agreement were not measured here and should not be conflated with closed-loop success.

## Decision

The predeclared extra-step benefit gate required one arm to win at least three pairs, lose at most one, and win across at least two selected conditions. No arm won a pair. Stop residual-based threshold work and do not train another predictor from this pilot. Screening qualification at 2/3 did not generalize to one-step failures in four held-out seeds per condition. A future study would need more diverse task instances and a new, predeclared screen that yields one-step failures on held-out seeds before testing any score as a rescue predictor.

## Reproduction on the pod

The model server used the frozen checkpoint and head loaded by `examples/libero_plus/policy_server.py`, bound to localhost:8765. Check its `/health` identity against the manifest before running. The launcher reads the pinned benchmark checkout and its isolated simulator environment. The commands below were used from `/volt/data/openpi_velocity`; the server was already running during this study. Start it in a separate pod terminal/session before executing the remaining commands.

```bash
export CUDA_VISIBLE_DEVICES=3
nvidia-smi --query-gpu=uuid --format=csv,noheader -i 3
.venv/bin/python examples/libero_plus/policy_server.py --host 127.0.0.1 --port 8765
.venv/bin/python examples/libero_plus/launch_two_stage.py --stage smoke --output /volt/data/openpi_evals/libero_plus_two_stage/smoke.jsonl
.venv/bin/python examples/libero_plus/two_stage_analyze.py --stage smoke --records /volt/data/openpi_evals/libero_plus_two_stage/smoke.jsonl --output /volt/data/openpi_evals/libero_plus_two_stage/smoke_summary.json --wandb-name libero-plus-two-stage-smoke
.venv/bin/python examples/libero_plus/launch_two_stage.py --stage screen --output /volt/data/openpi_evals/libero_plus_two_stage/screen.jsonl
.venv/bin/python examples/libero_plus/launch_two_stage.py --stage select --screen-records /volt/data/openpi_evals/libero_plus_two_stage/screen.jsonl --output /volt/data/openpi_evals/libero_plus_two_stage/selection.json
.venv/bin/python examples/libero_plus/two_stage_analyze.py --stage screen --records /volt/data/openpi_evals/libero_plus_two_stage/screen.jsonl --selection /volt/data/openpi_evals/libero_plus_two_stage/selection.json --output /volt/data/openpi_evals/libero_plus_two_stage/screen_summary.json --wandb-name libero-plus-two-stage-screen
.venv/bin/python examples/libero_plus/launch_two_stage.py --stage compare --screen-records /volt/data/openpi_evals/libero_plus_two_stage/screen.jsonl --selection /volt/data/openpi_evals/libero_plus_two_stage/selection.json --output /volt/data/openpi_evals/libero_plus_two_stage/compare.jsonl
.venv/bin/python examples/libero_plus/two_stage_analyze.py --stage compare --records /volt/data/openpi_evals/libero_plus_two_stage/compare.jsonl --screen-records /volt/data/openpi_evals/libero_plus_two_stage/screen.jsonl --selection /volt/data/openpi_evals/libero_plus_two_stage/selection.json --output /volt/data/openpi_evals/libero_plus_two_stage/compare_summary.json --wandb-name libero-plus-two-stage-compare
CUDA_VISIBLE_DEVICES=3 JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q examples/libero_plus/two_stage_test.py
.venv/bin/python -m ruff check examples/libero_plus/{two_stage.py,two_stage_analyze.py,two_stage_test.py,launch_two_stage.py}
.venv/bin/python -m ruff format --check examples/libero_plus/{two_stage.py,two_stage_analyze.py,two_stage_test.py,launch_two_stage.py}
```

The raw smoke, complete screen and comparison episode records, frozen selection, and all three summaries are included in the pushed branch and the W&B artifacts. W&B also records the pinned protocol and provenance hashes.
