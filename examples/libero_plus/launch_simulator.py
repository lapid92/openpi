"""Launch the LIBERO-Plus client in its isolated CPU simulator runtime."""

import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path("/volt/data/openpi_evals/benchmarks/libero_plus_4976dc3")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    environment = dict(os.environ)
    environment.update(json.loads((ROOT / "cpu_env.json").read_text()))
    environment["CUDA_VISIBLE_DEVICES"] = ""
    command = [
        "/volt/envs/libero-h100/bin/python",
        str(Path(__file__).with_name("pilot_client.py")),
        "--classification",
        str(ROOT / "repo/libero/libero/benchmark/task_classification.json"),
        "--output",
        args.output,
    ]
    if args.smoke:
        command.append("--smoke")
    raise SystemExit(subprocess.call(command, env=environment))


if __name__ == "__main__":
    main()
