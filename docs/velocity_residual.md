# Frozen π₀.₅ velocity residual predictor

This first stage fits only a small log-scale head. It does not alter `sample_actions` or choose integration steps. The frozen base is the public standard `gs://openpi-assets/checkpoints/pi05_libero` checkpoint and the data source is `physical-intelligence/libero` at its LeRobot v2.0 revision. Its dataset metadata declares 1,693 training demonstrations and no evaluation split. The trainer refuses another split layout.

## Method

For each real training sample, draw Gaussian noise and `t ~ 0.999 Beta(1.5, 1) + 0.001`, matching the base π₀.₅ training code. Form `x_t = t noise + (1-t) action` and `u = noise-action`. The frozen model returns velocity and final action-expert tokens in the same forward pass. The standard checkpoint's action expert is checked as 1024 wide; ten tokens are mean-pooled over valid action steps and concatenated with a 32-wide sinusoidal time embedding. The head is `1056 → 256 → 64 → 1` with SiLU after the first two layers. Its output is `log σ`, clipped to `[-8, 6]`.

The target is `r = u-vθ`. Per example, squared residual is averaged over the valid action-horizon steps and all 32 action dimensions. LeRobot's `actions_is_pad` excludes steps beyond the demonstration end. Fully padded examples are rejected. The float32 objective is `0.5 mean(r²) exp(-2 log σ) + log σ`. The base model never enters the optimizer. No evaluation episodes are loaded or used for selection.

A seeded 90/10 split is made by demonstration ID before loading examples. The checkpoint records the complete split hash and the selected episode IDs. The `--max-*-episodes` flags restrict each side for a quick smoke run while preserving the split boundary. Training and held-out validation calibration logs compare predicted σ with residual RMS by flow-time bin and by the four LIBERO suites. Each logged subgroup also records its sample count. They are diagnostics, not task-success claims.

## Train

Run on pod `2vzhlaphss5c` only after verifying GPU 3 UUID `GPU-4779cdde-a260-f8ec-da6a-6fa390bc7fd7`:

```bash
cd /volt/data/openpi_velocity
nvidia-smi -i 3 --query-gpu=uuid --format=csv,noheader
CUDA_VISIBLE_DEVICES=3 WANDB_MODE=online .venv/bin/python scripts/train_velocity_residual.py \
  --checkpoint official --output /volt/data/pi05_residual/head.npz \
  --steps 1000 --batch-size 8 --validation-every 50 --save-every 100 --calibration-batches 32 \
  --wandb-project pi05-libero-velocity-residual
```

The head checkpoint is written atomically every 100 optimizer steps by default and again after training completes. A final held-out calibration sweep records overall and per-suite/time residual scale and predicted σ in W&B and a sibling `.calibration.json` report. An interrupted run leaves the most recent completed head checkpoint at the requested output path.

A one-step real-data smoke run can use `--steps 1 --batch-size 1 --validation-every 1 --max-train-episodes 2 --max-val-episodes 1`. It still downloads LIBERO training episode files. Synthetic data are not used. W&B must be configured in the pod environment; no credentials are stored in the repository. The full training split contains 1,524 episodes (about 32.5 GiB of parquet files). On shared pod IPs, set a Hugging Face token through pod secrets before downloading to avoid anonymous HTTP 429 rate limits; never put it in a command line or repository file.

`load_head(path, checkpoint_identity)` refuses a mismatched base checkpoint identity. The `.npz` contains only the predictor weights and JSON metadata: base path and identity, dataset revision, train/validation episode IDs and split identity, feature width, steps, GPU UUID, and W&B URL. The base checkpoint remains separate. The identity hashes Orbax metadata and manifest files; preserve the exact base checkpoint directory for reload.

## Tests

```bash
CUDA_VISIBLE_DEVICES=3 .venv/bin/python -m pytest -q src/openpi/models/velocity_residual_test.py
.venv/bin/ruff check src/openpi/models/velocity_residual.py src/openpi/models/velocity_residual_test.py scripts/train_velocity_residual.py
```

The next stage can calibrate an adaptive step rule and compare against existing fixed-step π₀.₅ LIBERO results. No task-success improvement is inferred from residual calibration alone.
