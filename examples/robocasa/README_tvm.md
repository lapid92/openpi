# Human300 π₀.₅ TVM training

Two recipes use the RoboCasa Human300 training interface and eight GPUs:

| `--init` | Source weights | New optimizer schedule | Updates |
| --- | --- | --- | --- |
| `released` | RoboCasa's released Human300 π₀.₅ checkpoint | AdamW, 1,000 warmup steps, cosine `1e-5 → 1e-6` over 20,000 updates | 20,000 |
| `base` | Public `pi05_base` checkpoint | AdamW, 1,000 warmup steps, cosine `2.5e-5 → 2.5e-6` over 100,000 updates | 100,000 |

Both use global batch 64, eight accumulated microbatches of eight examples, eight-way FSDP, EMA teacher decay 0.999, and TVM weight 0.25 after 1,000 warmup and 1,000 ramp updates. The FM term stays at weight 1. The new source-time projection starts at zero, preserving one-time checkpoint predictions at initialization. These are **new TVM runs with fresh optimizer state**, including the released-checkpoint recipe. `--resume` restores a saved run's optimizer and EMA; it does not resume the owner run's optimizer.

## Inputs and setup

Use a Python 3.11 environment with this repository's `uv.lock` and RoboCasa v1.0. The validated owner interface is OpenPI `ca4c6d710db75e276bc7c866a57bd7e4aee5b6e8`, RoboCasa `8f3c96ec8d1bfcd8126cad2bca887da98d30e997`, and robosuite `5ce6643f3092639d08f7b0f90ed1c6a84f50552c`. The existing Volt pod uses JAX 0.5.3, Transformers 4.53.2, and `huggingface-hub` 0.36.0. Install RoboCasa and robosuite in the same environment as OpenPI, and set `PYTHONPATH` so this repository's `src` is used when reusing another installation's environment.

Download the official **pretrain human** RoboCasa datasets on the training pod using `python -m robocasa.scripts.download_datasets --split pretrain --source human`. The dataset host may require Hugging Face access. The Human300 registry must resolve all 300 task directories. The owner loader uses the left agent camera, right agent camera, and eye-in-hand camera; its state ordering is relative end-effector position and rotation, base position and rotation, then gripper position. Its action ordering is end-effector position and rotation, gripper close, base motion, then control mode (12 values). The prompt is `annotation.human.task_description`. State and actions are padded to 32. The released recipe uses the released checkpoint's `assets/norm_stats.json`; the base recipe derives owner-format mean/std stats from the 300 task metadata files. Neither recipe applies quantile normalization or an additional delta action transform.

If the datasets are already mounted outside RoboCasa's default `datasets` directory, pass `--dataset-root /path/to/robocasa-datasets`, where that root contains `v1.0/pretrain`.

Stage the public base checkpoint on the pod with `python scripts/stage_pi05_base_gcs.py /path/to/pi05_base/params`. For the released source, obtain `robocasa/robocasa365_checkpoints` at revision `c484448aba1a9b60a04c9b0ca117241518ea69f3`, subdirectory `pi05_pretrain_human300/multitask_learning/75000`. Pass its `params` and `assets` directories separately. Keep dataset and checkpoint transfers on the training pod.

## Launch

From the repository root, with all eight GPUs visible:

```bash
python -m scripts.run_pi05_robocasa_tvm \
  --init released \
  --params /path/to/75000/params \
  --released-assets /path/to/75000/assets \
  --exp-name human300_released_tvm_v1 \
  --checkpoint-base-dir /durable/checkpoints \
  --assets-base-dir /durable/assets

python -m scripts.run_pi05_robocasa_tvm \
  --init base \
  --params /path/to/pi05_base/params \
  --exp-name human300_base_tvm_v1 \
  --checkpoint-base-dir /durable/checkpoints \
  --assets-base-dir /durable/assets
```

Each run writes a `provenance.json` beside its Orbax checkpoints, including source metadata and normalization file hashes. Logs report FM loss, raw TVM loss, TVM weight, gradient norm, source-time gradient norm, throughput, and GPU memory. Give every new run a unique experiment name. Pass `--resume` with the same name to restore its full training state. Checkpoint assets contain the normalization statistics used by that run.

Use a persistent checkpoint volume. To keep an independent S3 copy during a long run, start the mirror **on the training pod** in a second terminal. For the released recipe, the completed run directory is `/durable/checkpoints/pi05_robocasa_human300_tvm_released/human300_released_tvm_v1`:

```bash
python3 scripts/mirror_robocasa_tvm_s3.py \
  /durable/checkpoints/pi05_robocasa_human300_tvm_released/human300_released_tvm_v1 \
  s3://YOUR-BUCKET/robocasa-tvm/human300_released_tvm_v1 \
  --watch --until-step 20000
```

Use a Python environment with `boto3` and pod S3 credentials for the mirror; it need not be the training environment. Use the corresponding base run directory and `--until-step 100000` for the base recipe. The mirror uploads only finalized Orbax step directories and writes an S3 completion manifest after verifying each uploaded file's size. Re-running it skips steps with a completion manifest.

For a bounded two-update smoke with active TVM, add `--smoke-task AdjustToasterOvenTemperature` and use a unique experiment name. This uses eight examples per update, one real Human300 task, and sets TVM warmup and ramp to zero. For a longer loss-trend check, also pass `--smoke-steps 32`. The longer smoke uses the recipe's peak learning rate immediately so that 32 updates are informative; it saves checkpoints after the first and final updates. Compare averages over early and late updates because stochastic per-step losses need not decrease monotonically. Smoke results do not represent the 300-task production distribution.

For TVM inference, `scripts/serve_policy.py` defaults to `jump` with these two configs: each Euler step queries `F(x; t, s)` at the next time `s`. Explicit `--sampler fm_only` queries the diagonal `F(x; t, t)` for comparison. The sampler is an inference choice; `fm_only` does **not** mean the model was trained without TVM. Other configs retain their existing sampler default.
