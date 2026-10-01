#!/bin/bash
set -euo pipefail
root=/volt/checkpoints/openpi-assets/checkpoints/pi05_libero
while [ ! -d "$root" ]; do sleep 10; done
cd "$root"
find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > /volt/artifacts/frozen-flow-study/checkpoint-files.sha256
sha256sum /volt/artifacts/frozen-flow-study/checkpoint-files.sha256 > /volt/artifacts/frozen-flow-study/checkpoint-manifest.sha256
cd /volt/code/frozen-flow-study
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu .venv/bin/python - <<'PY'
from openpi.models.velocity_residual import checkpoint_identity
print(checkpoint_identity("/volt/checkpoints/openpi-assets/checkpoints/pi05_libero"))
PY
