# Frozen pi05 cached sampling parity audit

This branch starts from `d7cba8a2ca55803c5afa628ee6da5b7895ffe34b`. All GPU jobs used pod `2vzhlaphss5c`, `CUDA_VISIBLE_DEVICES=3`, and UUID `GPU-4779cdde-a260-f8ec-da6a-6fa390bc7fd7`. The official frozen `pi05_libero` checkpoint identity is `dad4e2fbe79cceca79b83f3e53bb59c180e55ebb6815c67bfcdcf81768d7cee8`; the head is the existing 3,000-step checkpoint. No base parameters were updated and TVM was excluded.

## First divergence and correction

The production sampler carries float32 time through one compiled `jax.lax.while_loop` and updates `x_t + dt * velocity`. The old offline cached trace called a separately compiled suffix function from a Python loop, supplied `1-i/n` as time, and updated `x - velocity/n`. Its prefix KV cache, attention masks, preprocessing, feature projection, and action normalization agree with production. Split compilation and update arithmetic produced bfloat16 trajectory differences that were amplified by later steps; accumulated time also contributes at 10 steps. The exact low-level XLA rounding choice in the old split trace is not isolated.

The head-bearing adaptive loop had a second, precisely identified update difference. Production multiplies bfloat16 velocity by a weak Python scalar `dt`, leaving the product bfloat16. The adaptive loop used a strong float32 JAX step, promoting that product to float32. On the same batch-one input, forcing ten equal steps gave a pre-fix final maximum absolute error of `0.02500027`; the first difference appeared immediately after step zero, before any velocity discrepancy. The adaptive loop now casts its negative step to velocity dtype for the action update while keeping flow time float32. A dedicated diagnostic compile mode retains per-step states and velocities; the normal sampler does not carry those diagnostic tensors.

A source-mirroring instrumented production trace and the old cached trace were run with identical checkpoint, observation, initial noise, and schedules on held-out training episode 184. At four steps the first difference was step 1 velocity, maximum absolute error `0.015625`, with identical state and time; the old final-action error was `0.0078125`. At ten steps the first difference was step 1 state `0.001953125` and velocity `0.015625`; old final-action error was `0.5775451660` in the original QA reproduction; the final hard-gated artifact also reports `0.5775451660`. Other repeats varied (for example `0.4587097168`), showing run-to-run numerical variation in the old path. One- and two-step final actions agreed. The stepwise JSON and complete state/velocity arrays are saved as `/volt/data/pi05_adaptive/parity_stepwise.json` and `/volt/data/pi05_adaptive/parity_stepwise.npz`.

`Pi0.cached_fixed_step_trace` now calls the same cached velocity and feature helper within one compiled loop, with production time and update semantics. On episode 184, its flow times, every action state, every velocity, and final actions match the instrumented production path exactly for 1, 2, 4, and 10 steps. The corrected full diagnostic enforced a maximum absolute trace-versus-production endpoint tolerance of `1e-5`; observed maximum was `0.0` at all four step counts across 672 frames. The adaptive sampler also evaluates velocity and predicted sigma in one compiled loop, with one observation-prefix cache. After the update-dtype correction, forced fixed schedules of 1, 2, 4, and 10 steps matched production per-step states, velocities, and final actions exactly on episode 184 in the instrumented GPU run. The final hard-gated GPU regression also passed on the normal adaptive graph. Its real-checkpoint [GPU smoke run](https://wandb.ai/arm-aair-idit/pi05-libero-velocity-residual/runs/pfn2ey8a) used four evaluations and reached t=0. The focused CPU regression suite passed (33 tests), including a ten-step bfloat16 update test. It remains an opt-in, uncalibrated prototype; fixed inference continues to use the unchanged production sampler.

## Previous study audit and corrected decision gate

The previous diagnostic stored 1/2/4/10 endpoints from production `Pi0.sample_actions`, so the cached-path defect did not structurally generate its demonstration-MSE refinement labels. Cached per-step sigma and generated-state inputs were affected. Recomputed production endpoints show small run-to-run numerical differences, so historical record values should not be treated as bitwise identical. The corrected [W&B diagnostic](https://wandb.ai/arm-aair-idit/pi05-libero-velocity-residual/runs/n3a8hknt) reran the same 84 held-out training trajectories, eight frames per trajectory, with the same seed, checkpoint, head, split, and fixed schedules. Its real-data [smoke run](https://wandb.ai/arm-aair-idit/pi05-libero-velocity-residual/runs/70u0ukzj) also passed. The corrected split identity remains `425fed4d4b67e57e33d19812c725a9a11572cdfc112ab79ab0a610728a7fc4fe`.

| Predicted sigma versus demonstration-MSE benefit | 1 to 10 | 2 to 10 | 4 to 10 |
| --- | ---: | ---: | ---: |
| Spearman rho | -0.0657 | -0.1436 | -0.1648 |
| Episode-bootstrap 95% interval | [-0.1439, 0.0096] | [-0.2224, -0.0649] | [-0.2293, -0.0939] |
| High-minus-low score quartile benefit | -0.002234 | -0.001565 | -0.000320 |

The predeclared 1-to-10 gate still fails. Demonstration-action MSE is only an offline proxy. Numerical agreement with the 10-step endpoint is a separate sensitivity quantity and does not establish benefit or task success. There was no residual-based threshold tuning, full adaptive LIBERO evaluation, or speedup claim.

## Exact pod commands

```bash
cd /volt/data/openpi_velocity
nvidia-smi -i 3 --query-gpu=uuid --format=csv,noheader
CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python scripts/check_pi05_cached_parity.py
CUDA_VISIBLE_DEVICES=3 WANDB_MODE=online XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python scripts/study_velocity_residual_refinement.py \
  --head /volt/data/openpi_velocity_runs/long_3000_20260930/head.npz \
  --output /volt/data/pi05_adaptive/parity_corrected_full.json \
  --run-name cached-parity-corrected-full
CUDA_VISIBLE_DEVICES=3 WANDB_MODE=online XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python scripts/smoke_adaptive_euler.py \
  --head /volt/data/openpi_velocity_runs/long_3000_20260930/head.npz \
  --output /volt/data/pi05_adaptive/adaptive_final_smoke.json
CUDA_VISIBLE_DEVICES=3 JAX_PLATFORMS=cpu \
  .venv/bin/python -m pytest -q \
  src/openpi/models/pi0_cached_parity_test.py \
  src/openpi/models/pi0_cached_test.py \
  src/openpi/models/adaptive_euler_test.py \
  scripts/study_velocity_residual_refinement_test.py
.venv/bin/ruff check src/openpi/models/pi0.py src/openpi/models/adaptive_euler.py \
  scripts/check_pi05_cached_parity.py scripts/study_velocity_residual_refinement.py
```

## Small next experiment: predict direct step sensitivity

Keep the frozen checkpoint and train a new small head from the same `t=1` action-expert features, using matched observation and initial noise. The regression label is `log(epsilon + mean((action_10 - action_1)^2))`, with `epsilon=1e-6`, over the full generated action chunk. Generate labels with the validated fixed sampler and retain only training-demo frames with a full valid action horizon, avoiding end-of-episode padding artifacts. This target measures numerical step sensitivity, not beneficial refinement or success. A 1-to-4 and 2-to-10 sensitivity label can be kept as prespecified secondary diagnostics.

Use the 1,524 existing predictor-training trajectories only for fitting; make a new suite-stratified, trajectory-level 90/10 internal train/validation split and save its ID/hash before training. Keep the 84 previously analyzed episodes exploratory. Use the other 85 held-out training trajectories once as the untouched sensitivity test. Do not use simulator evaluation episodes for fitting or selection. Sample the same eight fixed frame fractions per episode and one deterministic noise seed per frame. That is at most 12,192 labeled frames before full-horizon filtering, requiring 11 frozen velocity evaluations per frame for the 1- and 10-step endpoints, approximately 134,112 evaluations. The current four-schedule diagnostic took about 198 seconds for 672 frames on GPU 3; budget roughly 30 to 90 minutes for label generation plus head fitting, then measure actual throughput.

Validate held-out Spearman and trajectory-bootstrap confidence intervals for predicted versus measured endpoint sensitivity, but gate any step-allocation proposal on a **separate** positive association with demonstration-MSE benefit and, later, a small matched simulator pilot. Freeze any step rule and matched fixed-step control before a final success and latency comparison. If sensitivity prediction is good but benefit ranking remains nonpositive, stop; a movement predictor alone does not justify allocating more steps.
