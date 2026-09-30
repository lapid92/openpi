# Frozen π₀.₅ residual score and step allocation: stage protocol

This stage uses the standard frozen `pi05_libero` checkpoint and the 3,000-step velocity-residual head from the previous stage. It does not involve TVM or change base weights. The official LIBERO checkpoint identity and saved split identity must match before a GPU run. Use only pod `2vzhlaphss5c`, `CUDA_VISIBLE_DEVICES=3`, and GPU UUID `GPU-4779cdde-a260-f8ec-da6a-6fa390bc7fd7`.

## Offline decision gate, declared before measurement

The 169 held-out **training-demonstration** episodes are split by episode and stratified by LIBERO suite with seed `20261001`: half per suite for the diagnostic study and the rest reserved for step-rule calibration. Neither subset contains final simulator evaluation episodes. From each diagnostic episode, sample eight predetermined frame fractions `(0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 0.93, 0.98)`. A fixed seed generates Gaussian noise and flow times. At each selected observation and shared initial noise, run the existing fixed Euler sampler at 1, 2, 4, and 10 velocity evaluations with the observation-prefix cache. A separate cached trace records each trajectory time and predicted σ; numerical disagreement between traced and production fixed endpoints is measured. Record each fixed-sampler endpoint, masked MSE to the demonstration action, and endpoint disagreement with 10 steps. Apply the demonstration action-padding mask only to demonstration-based metrics; generated action tokens are all included in inference pooling. The trace instrumentation adds diagnostic velocity passes; the reported 1/2/4/10 evaluation counts refer to the fixed sampler itself.

The primary score is σ at the common initial `(x_1,t=1)` state. The primary offline refinement proxy is masked demonstration-action MSE at 1 step minus that at 10 steps; positive means the 10-step output is closer to the demonstrated action. This is **not task success**. Report Spearman rank correlation, AUROC for positive proxy benefit, and benefit contrast between top and bottom score quartiles. Bootstrap confidence intervals resample whole episodes, not frames. The gate passes only if the 95% episode-bootstrap lower bounds for Spearman and high-minus-low benefit contrast are both above zero, and Spearman is positive in at least three of four suites. The 2-to-10 and 4-to-10 comparisons are secondary. If the gate fails, report it before any full adaptive simulator run; the next minimal target would be direct step sensitivity.

For residual calibration, sample demonstration interpolation states using the original flow-time distribution, compute the masked squared velocity residual, and compare **mean predicted σ²** with **mean residual²** in score quintiles, five time bins, and four suites. Counts and ratios are recorded. For generated trajectories, compare σ at `t=0.5` with σ on the matched demonstration interpolation at `t=0.5`, using the same observation and initial noise. There is no ground-truth straight-line velocity target at generated states, so this comparison measures input shift, not residual calibration.

Run a two-batch real-data smoke first, then the full diagnostic:

```bash
cd /volt/data/openpi_velocity
nvidia-smi -i 3 --query-gpu=uuid --format=csv,noheader
CUDA_VISIBLE_DEVICES=3 WANDB_MODE=online XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python scripts/study_velocity_residual_refinement.py \
  --head /volt/data/openpi_velocity_runs/long_3000_20260930/head.npz \
  --output /volt/data/pi05_adaptive/offline_smoke.json --max-batches 2
CUDA_VISIBLE_DEVICES=3 WANDB_MODE=online XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python scripts/study_velocity_residual_refinement.py \
  --head /volt/data/openpi_velocity_runs/long_3000_20260930/head.npz \
  --output /volt/data/pi05_adaptive/offline_full.json
```

## Final LIBERO comparison requirements

Previous local LIBERO shards use TVM configurations and different checkpoints; they are not matched baselines. If the offline gate passes, calibrate the bounded step rule only on the reserved validation episodes, freeze it, and rerun standard frozen π₀.₅ fixed 1, 2, 4, and 10-step and adaptive simulator arms. Match checkpoint, task initial states, episode seeds, Gaussian noise keyed by suite/task/episode/chunk, five executed actions per chunk, rendering/preprocessing, and GPU 3. Record success by suite, mean and tail velocity evaluations per generated chunk, and end-to-end latency. A constant-step control with similar mean evaluations is required. Batch size one is the initial inference scope; per-sample stopping counts do not imply a batched compute speedup.

## Measured offline result and decision (2026-09-30)

The [full diagnostic W&B run](https://wandb.ai/arm-aair-idit/pi05-libero-velocity-residual/runs/s7mbx4kc) processed 672 real frames from 84 held-out training trajectories; 85 other held-out trajectories remain unused for rule calibration. Diagnostic split identity: `425fed4d4b67e57e33d19812c725a9a11572cdfc112ab79ab0a610728a7fc4fe`. The frozen base identity was `dad4e2fbe79cceca79b83f3e53bb59c180e55ebb6815c67bfcdcf81768d7cee8`. Fixed endpoints came from unchanged `Pi0.sample_actions`; a parallel cached trace supplied same-pass σ. Mean trace-to-fixed endpoint MSE was zero or below `1.1e-6` across 1/2/4/10 steps. The real-data smoke is [logged separately](https://wandb.ai/arm-aair-idit/pi05-libero-velocity-residual/runs/lvj5ug52). A [repeat full diagnostic](https://wandb.ai/arm-aair-idit/pi05-libero-velocity-residual/runs/x61g54z4) reproduced the failed gate (primary Spearman `-0.066` on the same 672 frames). In that repeat, 1- and 2-step cached endpoints matched fixed outputs exactly. For 4 steps, 7/672 traces had all-token maximum absolute endpoint difference above `0.01` (maximum `0.040`); for 10 steps, 2/672 did (maximum `0.578`, on a fully valid action window). Numerical parity of cached 4/10-step trajectories with production fixed inference remains unresolved; these traces cannot support a comparative success or latency claim. The primary gate and calibration use the initial score and independent production fixed endpoints, while the generated-state check at `t=0.5` comes from the exactly matching 2-step trace.

| Offline diagnostic | 1 to 10 steps | 2 to 10 steps | 4 to 10 steps |
| --- | ---: | ---: | ---: |
| Spearman of initial σ versus demonstration-MSE benefit | -0.067 | -0.144 | -0.170 |
| Episode-bootstrap 95% interval | [-0.146, 0.008] | [-0.223, -0.066] | [-0.233, -0.098] |
| Mean masked demonstration-MSE benefit | -0.000728 | -0.000557 | -0.000179 |
| Fraction with positive benefit | 0.376 | 0.202 | 0.107 |

The primary top-minus-bottom σ-quartile benefit contrast was `-0.00227`, episode-bootstrap interval `[-0.00306, -0.00155]`; only one of four suites had a positive primary within-suite rank correlation, near zero. The **predeclared gate failed**. Longer integration often increased distance from the particular demonstration action under this offline proxy. This cannot establish that fewer steps improve simulator success. As a separate, post hoc sensitivity diagnostic, initial σ ranked the magnitude of endpoint disagreement with 10 steps (Spearman `0.583`, `0.529`, and `0.383` for 1/2/4 steps). It predicts how much the output moves, but not whether the movement helps this proxy.

Calibration compared variances directly. Across 672 demonstration interpolation states, mean predicted `σ²` was `0.00526` and mean masked squared velocity residual was `0.00323` (ratio `1.63`). Predicted-score quintile ratios were `1.31, 1.26, 1.16, 0.93, 1.83`; flow-time-bin ratios from early to late were `1.82, 2.21, 1.03, 1.09, 1.00`. Suite ratios ranged from `1.31` to `1.92`; subgroup counts and raw means are in the W&B run and `/volt/data/pi05_adaptive/offline_full.json`. At matched `t=0.5`, mean σ on generated and demonstration-interpolation states was `0.0421007` and `0.0421035`. This one-time comparison is an input-shift check only and supplies no residual target on generated states.

A separate bounded sampler is implemented but **its rule is not calibrated for success**. It uses the same cached suffix pass for velocity and σ, steps backward from `t=1` to `t=0`, and guarantees an evaluation cap with configurable positive step bounds. The existing fixed sampler remains the default. A [real-GPU sampler smoke](https://wandb.ai/arm-aair-idit/pi05-libero-velocity-residual/runs/qtjkfx6c) used a held-out training observation and an illustrative rule (`max_evals=10`, `min_step=0.05`, `max_step=0.5`, `reference_step=0.25`, `reference_sigma=0.05`): it terminated in four velocity evaluations, with about `0.041 s` warm model-call time for that single chunk. This excludes simulator, networking, and action execution, and is not a comparative latency result. Reproduce it with:

```bash
cd /volt/data/openpi_velocity
CUDA_VISIBLE_DEVICES=3 WANDB_MODE=online XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python scripts/smoke_adaptive_euler.py \
  --head /volt/data/openpi_velocity_runs/long_3000_20260930/head.npz \
  --output /volt/data/pi05_adaptive/adaptive_sampler_smoke.json
```

The failed gate stops this stage before rule tuning, a constant-step control, and full adaptive/fixed 1/2/4/10-step LIBERO simulator evaluation. No simulator success or end-to-end latency comparison is claimed. The smallest next test is to train a small head against direct numerical step sensitivity, such as fixed 1-to-10 endpoint disagreement under matched initial noise, on training demonstrations. Validate whether it predicts *beneficial* refinement on untouched trajectories and a small matched simulator pilot before calibrating any step threshold. Per-sample stopping counts do not imply a batched compute speedup.
