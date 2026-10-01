#!/bin/bash
set -euxo pipefail
export UV_CACHE_DIR=/volt/cache/uv
export OPENPI_DATA_HOME=/volt/checkpoints
python -m pip install uv
cd /volt/code/frozen-flow-study
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --python 3.11
 git submodule update --init third_party/libero
 git clone https://github.com/sylvestf/LIBERO-plus /volt/benchmarks/libero-plus
 git -C /volt/benchmarks/libero-plus checkout 4976dc30028e805ff8094b55501d532c48fec182
uv venv --python 3.8 /volt/envs/libero
uv pip install --python /volt/envs/libero/bin/python numpy==1.22.4 robosuite==1.4.1 mujoco==2.3.7 bddl easydict imageio imageio-ffmpeg opencv-python==4.6.0.66 matplotlib==3.5.3 h5py tyro PyYAML requests torch==2.0.1 torchvision==0.15.2
uv pip install --python /volt/envs/libero/bin/python --no-deps -e third_party/libero -e packages/openpi-client
.venv/bin/python - <<'PY'
from openpi.shared import download
print(download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero"))
PY
