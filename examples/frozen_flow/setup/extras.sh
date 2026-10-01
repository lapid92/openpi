#!/bin/bash
set -euxo pipefail
export UV_CACHE_DIR=/volt/cache/uv
apt-get update
apt-get install -y libosmesa6 libgl1 libglew2.2 libegl1 libmagickwand-dev libegl1-mesa-dev
uv pip install --python /volt/envs/libero/bin/python wand scikit-image gym==0.25.2 cloudpickle==2.1.0 einops easydict future bddl==1.0.1
uv pip install --python /volt/envs/libero/bin/python --no-deps robomimic==0.2.0
