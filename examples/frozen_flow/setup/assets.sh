#!/bin/bash
set -euxo pipefail
mkdir -p /volt/benchmarks
curl -L --retry 5 https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/main/assets.zip -o /volt/benchmarks/libero-plus-assets.zip
sha256sum /volt/benchmarks/libero-plus-assets.zip > /volt/artifacts/frozen-flow-study/plus-assets.sha256
while [ ! -d /volt/benchmarks/libero-plus/libero/libero ]; do sleep 10; done
python -m zipfile -e /volt/benchmarks/libero-plus-assets.zip /volt/benchmarks/libero-plus/libero/libero

python - <<'PYPATH'
from pathlib import Path
root=Path("/volt/benchmarks/libero-plus/libero/libero")
source=root/"inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/LIBERO-plus-0/assets"
source.rename(root/"assets")
PYPATH
