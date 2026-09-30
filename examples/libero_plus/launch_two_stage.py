"""Launch the two-stage LIBERO-Plus evaluator in its isolated CPU runtime."""

import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path("/volt/data/openpi_evals/benchmarks/libero_plus_4976dc3")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("smoke", "screen", "select", "compare"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--screen-records")
    parser.add_argument("--selection")
    args = parser.parse_args()
    env = dict(os.environ)
    env.update(json.loads((ROOT / "cpu_env.json").read_text()))
    env["CUDA_VISIBLE_DEVICES"] = ""
    command = [
        "/volt/envs/libero-h100/bin/python",
        str(Path(__file__).with_name("two_stage.py")),
        "--stage",
        args.stage,
        "--output",
        args.output,
        "--classification",
        str(ROOT / "repo/libero/libero/benchmark/task_classification.json"),
    ]
    if args.screen_records:
        command.extend(["--screen-records", args.screen_records])
    if args.selection:
        command.extend(["--selection", args.selection])
    raise SystemExit(subprocess.call(command, env=env))


if __name__ == "__main__":
    main()
