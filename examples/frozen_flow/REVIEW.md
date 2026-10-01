# Pre-run implementation review and validation

Date: 2026-10-01. Pod: `uz2ptakxucbe`. Repository: `/volt/code/frozen-flow-study`.

Status: software review and synthetic validation passed. This is **not** a real-checkpoint parity, simulator smoke, or final study-record approval. No new evaluation outcomes were inspected during this review.

## Role cycle

The Leader specified the frozen 1/2/4/10-step comparison, outcome-independent task lists and maximum budget. The Developer implemented the protocol generator, simulator client and production-policy server. An independent Reviewer inspected the actual new source files and reused evaluation sources, requested the fixes below, and reread the corrected implementation. The Test Writer implemented the strict analyzer and synthetic regression tests, including tests of client partitioning/resume and the supervisor gate.

## Concrete findings and resolutions

- **Old-pod GPU identity constants:** replaced by the declared four-UUID inventory. Each server verifies and masks its physical UUID before importing JAX. Real GPU startup remains a runtime gate.
- **Checkpoint and preprocessing provenance:** content hashes cover checkpoint files, normalization assets, source files, task BDDL/init files, and installed simulator assets. Client/server startup enforces the relevant hashes.
- **Stale server protocol:** health now carries the exact manifest SHA256, checked by both client and supervisor.
- **Benchmark-specific parity:** warm/verified state is keyed by benchmark and fixed arm. LIBERO preparation cannot silently satisfy the LIBERO-Plus gate.
- **Action/noise dimensions:** the actual checked configuration has action horizon 10 and model action dimension 32. The manifest declares the shape; server asserts the model agrees and analyzer recomputes every chunk noise hash with that shape. A synthetic shape-mismatch regression caught and prevented a stale 50-step default callsite.
- **Pairing:** client and final auditor compare supplied/stabilized simulator-state hashes, first observations, checkpoint and GPU identity, and common chunk-index noise. Later observations may diverge. Explicit task instances and seeds remain fixed.
- **Resume integrity:** unknown statuses, nonboolean outcomes, duplicate completed records, invalid accounting and paired-state mismatches are rejected. Infrastructure error attempts remain in the durable record stream; only complete successful executions count toward the declared plan.
- **Full-run smoke gate:** the supervisor now rereads and strictly validates exact smoke records, verifies summary benchmark/phase/count/manifest identity and record-file content hashes. A stale or incomplete smoke artifact cannot authorize the full run.
- **Success semantics:** source inspection confirmed that both simulator wrappers' `check_success()` delegates to task `_check_success()`; `get_sim_state()` returns the flattened simulator state. Termination without task completion is an execution error.
- **Timing:** policy time includes the synchronized public inference call and transforms; episode time includes environment construction/reset/stabilization and policy/simulator steps, excluding teardown. Preparation/parity calls are kept outside scored timing. No adaptive speedup is inferred.
- **Analysis:** separate benchmark and task-family/condition summaries include all fixed arms, rescues, regressions, net differences, fraction of one-step failures rescued, and cost distributions. Cluster bootstrap retains whole paired conditions, uses 10,000 draws and seeds 20261001 + arm. One-cluster intervals are null. Each comparison with fewer than 20 one-step failures or 5 rescues is labeled unresolved.

## Executed synthetic checks

All commands below executed on the pod from the repository root:

```bash
.venv/bin/ruff --version
.venv/bin/ruff check examples/frozen_flow/*.py
.venv/bin/python --version
.venv/bin/python -m pytest --version
.venv/bin/python -m pytest -q examples/frozen_flow/test_study.py
```

Runtime versions: Python **3.11.14**, Ruff **0.11.12**, pytest **8.3.5**. Result: all study Python files passed lint; **23 tests passed in 0.32 seconds**. The locked linter's union-type and unused-directive findings were corrected without changing study design.

Coverage includes expected rescue/regression denominators; deterministic cluster intervals; exact planned completeness; duplicates; checkpoint/benchmark/manifest/GPU identity; strict boolean outcomes and integer episode keys; every-chunk noise and noise-shape identity; stabilized-state and first-observation pairing; permitted later observation divergence; finite latency and velocity accounting; error preservation; zero-failure handling; smoke separation; complete disjoint worker pairs and deterministic arm rotation; resume paired re-audit; and fail-closed supervisor smoke validation without launching any subprocess.

## Remaining runtime gates and limitations

Before scored evaluation: freeze and push the fully enumerated manifest, verify real checkpoint fixed-sampler parity at every arm on each GPU/benchmark, and pass each benchmark's simulator smoke audit. At completion: run the strict full-record auditor, verify checkpoint immutability, upload W&B artifacts and push results.

LIBERO-Plus conditions with only one supplied initial state use multiple declared seeds but cannot be interpreted as ten distinct initial states. Condition-cluster intervals do not model dependence between related conditions sharing a base task; small cluster counts and degenerate intervals limit population claims. RoboCasa requires its separately verified checkpoint and observation/action/simulator interface before any evaluation; the LIBERO policy is not a substitute.


## Final pre-freeze review addendum

The final prompt correction preserves standard LIBERO `task.language`. LIBERO-Plus uses the standard language parser on the exact base task family, excluding viewpoint/initial-state/severity filename suffixes. BDDL wording is retained only as provenance because it can differ from the established policy prompt. The Developer verified all 40 standard prompts remain unchanged and all 48 Plus prompts equal their standard base-task prompt. The Reviewer inspected the final prompt-source implementation.

The virtual BDDL filename passed to the Plus environment remains intact; the separate physical asset path is hashed. Its mapping agrees with the installed Plus wrapper's `_view_`/`_initstate_` parsing. Single-state Objects Layout availability is explicit in the enumerated specs.

The new Markdown reporter is called only after full audits and before publication. It checks manifest, benchmark, phase, checkpoint and benchmark revision against the validated summary. Three added synthetic tests verify paired counts, latency units, smoke exclusion, pending-audit rejection and stale provenance rejection. The final locked checks passed all study Python files and **23 tests in 0.32 seconds**.

The supervisor now finishes its W&B run successfully only after the result push succeeds. A publication failure therefore fails the supervisor run and remains recorded on the pod.

Final pre-run software review: approved. Real-checkpoint parity, simulator smoke and final-record validation remain required; no new evaluation outcomes were inspected for this approval.

## Independent runtime gate review after protocol freeze

Frozen manifest SHA256: `d16dba54a8157af011e872299883e071e681c0b4c6ad6bce1e5e62415f7910e6`.
Frozen checkpoint full content SHA256: `9cd1b00d402cc0447454dad6054dcc6f019b53e498469f209d2b749d4487e1d5`.

The Reviewer independently ran the strict analyzer against both completed smoke files, without starting duplicate W&B runs:

```bash
.venv/bin/python examples/frozen_flow/analyze.py --manifest examples/frozen_flow/protocol.json --benchmark libero --records /volt/artifacts/frozen-flow-study/runs/libero-smoke.jsonl --output /volt/artifacts/frozen-flow-study/runs/libero-review-smoke-summary.json --smoke
.venv/bin/python examples/frozen_flow/analyze.py --manifest examples/frozen_flow/protocol.json --benchmark libero_plus --records /volt/artifacts/frozen-flow-study/runs/libero_plus-smoke.jsonl --output /volt/artifacts/frozen-flow-study/runs/libero_plus-review-smoke-summary.json --smoke
```

Both audits passed: four declared arms per benchmark, zero error attempts, complete declared identity/provenance, matched supplied and stabilized simulator states, matched first observations, every chunk noise digest recomputed, and valid action/chunk/velocity/latency accounting.

All 32 real-checkpoint parity records were independently inspected: two benchmarks × four declared GPU UUIDs × four arms. Each expected benchmark/GPU/arm appeared exactly once; every record matched the frozen manifest and checkpoint, passed the 1e-5 tolerance with **maximum absolute action difference 0.0**, and recorded exactly twice the arm's velocity evaluations for verification. Each benchmark used 136 verification velocity evaluations, separate from scored smoke costs.

Smoke results, excluded from main study inference:

| Benchmark | Arm | Success | Chunks | Velocity evaluations | Mean policy call ms | Episode seconds |
|---|---:|---:|---:|---:|---:|---:|
| LIBERO | 1 | 1/1 | 15 | 15 | 33.534 | 31.158 |
| LIBERO | 2 | 1/1 | 16 | 32 | 35.621 | 32.399 |
| LIBERO | 4 | 1/1 | 15 | 60 | 38.956 | 31.422 |
| LIBERO | 10 | 1/1 | 15 | 150 | 51.010 | 31.389 |
| LIBERO-Plus | 1 | 1/1 | 16 | 16 | 33.723 | 34.034 |
| LIBERO-Plus | 2 | 1/1 | 16 | 32 | 35.786 | 31.629 |
| LIBERO-Plus | 4 | 1/1 | 16 | 64 | 39.587 | 32.647 |
| LIBERO-Plus | 10 | 1/1 | 16 | 160 | 51.768 | 33.329 |

The LIBERO case is `libero:libero_spatial:0`; the Plus case is `libero_plus:libero_spatial:265`. Both use the declared smoke seed 900001 and initial-state index zero. All arms of both smoke cases ran on `GPU-56335fdf-6dc8-1d64-ce39-364d4a532435`. Parity additionally verified UUIDs `GPU-b786fcd4-f878-69e6-3017-a2ae408151fd`, `GPU-967ff6df-a7e2-b65a-d051-87819b7725fc`, and `GPU-0cb759da-a716-038e-f0a9-208ec138ccc7`.

Runtime gate decision: **approve the unchanged predeclared full LIBERO and LIBERO-Plus evaluations** after smoke publication. These smoke cases provide no evidence of rescue benefit and do not change task selection, seeds, arms, budgets or stopping rules. Full paired-record validation and analysis remain pending.
