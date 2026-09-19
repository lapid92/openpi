# LIBERO Benchmark

This example runs the LIBERO benchmark: https://github.com/Lifelong-Robot-Learning/LIBERO

Note: When updating requirements.txt in this directory, there is an additional flag `--extra-index-url https://download.pytorch.org/whl/cu113` that must be added to the `uv pip compile` command.

This example requires git submodules to be initialized. Don't forget to run:

```bash
git submodule update --init --recursive
```

## With Docker (recommended)

```bash
# Grant access to the X11 server:
sudo xhost +local:docker

# To run with the default checkpoint and task suite:
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

You can customize the loaded checkpoint by providing additional `SERVER_ARGS` (see `scripts/serve_policy.py`), and the LIBERO task suite by providing additional `CLIENT_ARGS` (see `examples/libero/main.py`).
For example:

```bash
# To load a custom checkpoint (located in the top-level openpi/ directory):
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ./my_custom_checkpoint"

# To run the libero_10 task suite:
export CLIENT_ARGS="--args.task-suite-name libero_10"
```

## Without Docker (not recommended)

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# Run the simulation
python examples/libero/main.py

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx python examples/libero/main.py
```

Terminal window 2:

```bash
# Run the server
uv run scripts/serve_policy.py --env LIBERO
```

## Results

If you want to reproduce the following numbers, you can evaluate the checkpoint at `gs://openpi-assets/checkpoints/pi05_libero/`. This
checkpoint was trained in openpi with the `pi05_libero` config.

| Model | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|-------|---------------|---------------|-------------|-----------|---------|
| π0.5 @ 30k (finetuned) | 98.8 | 98.2 | 98.0 | 92.4 | 96.85

## Audited checkpoint sweeps

The evaluation runner can sweep flow-integration settings while keeping the rollout protocol fixed. It starts one
policy server per GPU, writes one durable JSONL record per episode, and only creates `_SUCCESS` after validating all
2,000 episodes (four suites, ten tasks per suite, and 50 trials per task).

```bash
python scripts/libero_eval_runner.py \
  --artifact-manifest /volt/data/checkpoints/my-checkpoint/artifact_manifest.json \
  --repo-root /volt/code/openpi \
  --output-root /volt/data/openpi_evals \
  --server-python /volt/envs/openpi/bin/python \
  --client-python /volt/envs/libero/bin/python \
  --openpi-data-home /volt/data/openpi \
  --gpu-ids 0 1 \
  --flow-steps 1 2 3 4 5 10
```

Evaluation-only checkpoint bundles must be transferred directly between Volt and the approved S3 namespace. The sync
tool hard-codes `us-east-2`, refuses paths outside `s3://aair-users-east-2/arilap01/libero/openpi/checkpoints/`, excludes
`train_state`, verifies SHA-256 after download, and never overwrites an existing artifact or local destination.

```bash
python scripts/sync_libero_checkpoint_s3.py upload \
  --checkpoint-dir /volt/data/checkpoints/pi05_libero_tvm/run/15000 \
  --artifact-key tvm/run/step_15000-checkpoint_15000 \
  --artifact-name pi05-libero-tvm-15k \
  --model-class tvm \
  --training-run-id run \
  --checkpoint-label 15000 \
  --train-steps-completed 15000 \
  --config pi05_libero_tvm \
  --git-sha FULL_40_CHARACTER_GIT_SHA

python scripts/sync_libero_checkpoint_s3.py download \
  --artifact-key tvm/run/step_15000-checkpoint_15000/sha256_SHA_FROM_UPLOAD_OUTPUT \
  --destination /volt/data/checkpoints/pi05-libero-tvm-15k
```

The upload command prints the immutable, content-addressed artifact key required by the download command.

The official released checkpoint is read directly from `gs://openpi-assets/checkpoints/pi05_libero`; it does not need
to be copied into S3.
