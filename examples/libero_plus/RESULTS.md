# LIBERO-Plus pilot results (frozen π₀.₅, no training)

Branch: `codex/pi05-libero-plus-pilot`, based on
`c3a5138be728482130bd8659136529e4a4aa137b`. The final code revision
and all reported results were produced on pod `2vzhlaphss5c`. The model
process used `CUDA_VISIBLE_DEVICES=3`, physical UUID
`GPU-4779cdde-a260-f8ec-da6a-6fa390bc7fd7`, visible as `cuda:0`;
the simulator used CPU rendering. The standard frozen `pi05_libero`
checkpoint had identity
`dad4e2fbe79cceca79b83f3e53bb59c180e55ebb6815c67bfcdcf81768d7cee8`.
The existing 3,000-step head SHA256 was
`e41eb678b938d239fdadb8bc1c77f9466007e460ea0dd53326262b7960f27d10`.
Neither was updated.

Benchmark: official [LIBERO-Plus repository](https://github.com/sylvestf/LIBERO-plus)
at commit `4976dc30028e805ff8094b55501d532c48fec182`, asset ZIP SHA256
`96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf`.
The exact six task instances, two seeds each, and fixed execution protocol
are in [`pilot_manifest.json`](pilot_manifest.json), SHA256
`e3557c0bc700e3e26cf0235b06599d1c5fddbfb2b73ea975818a4de4100c87ff`.
This is an exploratory 12-case paired subset of one base task family, not
the official full-benchmark score.

## Checks before analysis

- A real-checkpoint `/verify` check on GPU 3 returned **0.0 maximum
  action difference** from production fixed-step inference at 1 and 10
  steps. This separate smoke verification used extra evaluations and is not
  included in pilot velocity counts.
- The final-manifest smoke task (ID 2111, seed 3, one flow step) succeeded
  in 76 policy actions, 16 generated chunks and 16 velocity evaluations.
  [Smoke W&B run](https://wandb.ai/arm-aair-idit/pi05-libero-velocity-residual/runs/y765s0ke).
- All 48 pilot episodes completed without error. Analysis rejected
  missing/duplicate/error records, wrong model or benchmark identity,
  mismatched first observations, initial states or noise, incorrect fixed
  times, missing score, and unaccounted velocity calls. Maximum first-pass
  log-σ drift across compiled step schedules was **0.004442**, below the
  declared 0.01 tolerance.

The task and seed subset was committed before outcomes. After smoke and
six initial pilot episodes, byte-identical observation and noise probes
revealed a schedule-specific bfloat16/JIT first-pass score difference:
one-step σ 0.0409025 versus 0.0410609 for 2/4/10 steps on the same input.
The protocol was amended to hash first observations, use the one-step
first σ as the common case score, and permit at most 0.01 log-σ drift.
The full 48-episode pilot was restarted in a new result file; the six
earlier episodes were excluded from analysis.

## Paired closed-loop outcomes

Each category has four task-seed cases. Success means the environment
returned `done` within 220 policy actions after 10 dummy actions.
The arms share task BDDL, initial-state index, environment seed, first
observation, chunk-indexed Gaussian noise, preprocessing, five-action
execution chunks, checkpoint and hardware.

| Perturbation | 1 step | 2 steps | 4 steps | 10 steps |
|---|---:|---:|---:|---:|
| Robot initial states | 4/4 | 4/4 | 4/4 | 4/4 |
| Camera viewpoints | 4/4 | 3/4 | 4/4 | 4/4 |
| Light conditions | 4/4 | 4/4 | 4/4 | 4/4 |
| **Overall** | **12/12** | **11/12** | **12/12** | **12/12** |

Against one step, two steps had 0 paired wins, 1 loss, 11 ties; four
and ten steps each had 0 wins, 0 losses, 12 ties. The lone discordant
case was camera task ID 618, seed 11: one/four/ten steps succeeded and
two steps failed. Its one-step σ was 0.03023, ninth highest of 12 cases.
There were no positive gains against one step, so σ gain-ranking AUROC
is undefined. The decision gate **fails**: extra steps showed no
meaningful selective success benefit here. No new step predictor or
adaptive threshold was trained or tuned.

The exploratory 95% exact episode interval is [0.735, 1.000] for each
12/12 arm and [0.615, 0.998] for the 11/12 arm. The task-cluster bootstrap
interval for the paired two-versus-one success difference is
[-0.25, 0.00]; for four and ten versus one it is [0, 0].
These intervals are coarse because there are only six perturbed task
instances from two underlying black-bowl tasks. The 2→4 and 2→10
recoveries on the lone discordant case do not establish an improvement
over the one-step arm.

| Fixed steps | Evaluations/chunk, mean/p95 | Scored policy/chunk, mean/p95 | HTTP request/chunk, mean/p95 | Episode wall time, mean/p95 |
|---:|---:|---:|---:|---:|
| 1 | 1 / 1 | 32.35 / 33.16 ms | 283.94 / 492.56 ms | 32.73 / 39.76 s |
| 2 | 2 / 2 | 34.49 / 35.31 ms | 281.86 / 490.84 ms | 33.46 / 48.18 s |
| 4 | 4 / 4 | 38.95 / 41.05 ms | 286.08 / 480.17 ms | 31.61 / 36.02 s |
| 10 | 10 / 10 | 50.90 / 52.75 ms | 303.65 / 500.41 ms | 31.53 / 38.18 s |

Scored policy time includes prefix processing, fixed velocity steps,
same-pass head scoring, device synchronization and output transforms.
HTTP time also includes local JSON transfer and serialization.
Episode time includes simulator reset and steps; it varies with the
closed-loop trajectory and is not a controlled compute speedup measure.
Action-output differences and demonstration-action error were not
measured in this closed-loop pilot and are not substituted for success.

[Pilot W&B run](https://wandb.ai/arm-aair-idit/pi05-libero-velocity-residual/runs/4p7cwsea)
holds the manifest, raw JSONL, paired table, summary, uncertainty and
latency. The raw local records are
`/volt/data/openpi_evals/libero_plus_pilot/pilot_v2.jsonl`;
the summary is `/volt/data/openpi_evals/libero_plus_pilot/pilot_summary.json`.

## Commands and review

Run commands on the pod in `/volt/data/openpi_velocity`:

```bash
CUDA_VISIBLE_DEVICES=3 .venv/bin/python examples/libero_plus/policy_server.py \
  --checkpoint /root/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
  --head /volt/data/openpi_velocity_runs/long_3000_20260930/head.npz
.venv/bin/python examples/libero_plus/launch_simulator.py \
  --smoke --output /volt/data/openpi_evals/libero_plus_pilot/smoke_v2.jsonl
.venv/bin/python examples/libero_plus/launch_simulator.py \
  --output /volt/data/openpi_evals/libero_plus_pilot/pilot_v2.jsonl
.venv/bin/python examples/libero_plus/analyze.py \
  --smoke --records /volt/data/openpi_evals/libero_plus_pilot/smoke_v2.jsonl \
  --output /volt/data/openpi_evals/libero_plus_pilot/smoke_summary.json \
  --wandb-name libero-plus-no-training-smoke-v2
.venv/bin/python examples/libero_plus/analyze.py \
  --records /volt/data/openpi_evals/libero_plus_pilot/pilot_v2.jsonl \
  --output /volt/data/openpi_evals/libero_plus_pilot/pilot_summary.json \
  --wandb-name libero-plus-no-training-pilot-v2
CUDA_VISIBLE_DEVICES=3 JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q \
  examples/libero_plus/pilot_test.py examples/libero_plus/analyze_test.py
.venv/bin/python -m ruff check examples/libero_plus
.venv/bin/python -m ruff format --check examples/libero_plus
```

The Reviewer examined the actual server, client, manifest, analyzer and
tests. Blockers found and fixed: benchmark and asset identity guards,
manifest initial-state index, per-episode RNG reset, first-observation
pairing, and a CLI keyword-only argument error. The Test Writer's 25
focused tests passed; Ruff check and format passed. The full simulator
run, real-checkpoint parity, W&B smoke, and fail-closed analysis passed.
No full official LIBERO-Plus suite was run.
