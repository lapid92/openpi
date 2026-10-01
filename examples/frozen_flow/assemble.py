"""Assemble and content-pin the complete outcome-independent protocol."""

import argparse
import csv
import json
from pathlib import Path
import subprocess

from protocol import file_hash
from protocol import tree_hash


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--libero", required=True)
    p.add_argument("--libero-plus", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--runtime", required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = Path(a.output)
    if out.exists():
        raise RuntimeError("Protocol already exists; refusing to overwrite")
    checkpoint_hash, checkpoint_files = tree_hash(a.checkpoint)
    source = {}
    for folder in (root / "src/openpi", root / "packages/openpi-client/src", root / "examples/frozen_flow"):
        for path in sorted(folder.rglob("*.py")):
            source[str(path)] = file_hash(path)
    benchmarks = {
        "libero": json.loads(Path(a.libero).read_text()),
        "libero_plus": json.loads(Path(a.libero_plus).read_text()),
    }
    for spec in benchmarks.values():
        for path in sorted(Path(spec["root"]).rglob("*.py")):
            source[str(path)] = file_hash(path)
        asset_root = Path(spec["root"]) / "libero/libero/assets"
        digest, files = tree_hash(asset_root)
        spec["installed_assets_sha256"] = digest
        spec["installed_asset_files"] = files
        spec["installed_assets_path"] = str(asset_root)
    with (root / "examples/frozen_flow/gpu-identity.csv").open() as f:
        inventory = list(csv.DictReader(f, skipinitialspace=True))
    protocol = {
        "study": "pi05-frozen-flow-study",
        "schema_version": 1,
        "predeclared_at_utc": subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip(),
        "base_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "pod_id": "uz2ptakxucbe",
        "gpu_uuids": [x["uuid"] for x in inventory],
        "checkpoint": {
            "path": a.checkpoint,
            "sha256": checkpoint_hash,
            "files": checkpoint_files,
            "source": "gs://openpi-assets/checkpoints/pi05_libero",
            "config": "pi05_libero",
            "frozen": True,
            "shared_by": ["libero", "libero_plus"],
        },
        "noise_policy": {"shape": [10, 32], "dtype": "float32", "arm_independent": True},
        "source_sha256": source,
        "benchmarks": benchmarks,
        "flow_steps": [1, 2, 4, 10],
        "settings": {
            "render_size": 256,
            "image_size": 224,
            "wait_steps": 10,
            "replan_steps": 5,
            "max_policy_steps": {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520},
            "action_horizon": 10,
            "action_dim": 32,
            "batch_size": 1,
            "residual_head": None,
            "success": "env.check_success(); termination without success is an error",
            "noise": "numpy default_rng SHA256 first8 little-endian of compact JSON [benchmark,suite,task_name,seed,init_index,chunk_index]; float32 standard_normal(10,32)",
            "arm_order": "cyclic rotation of [1,2,4,10] by predetermined case ordinal",
            "episode_latency": "environment construction, reset, stabilization and policy/simulator steps; excludes teardown; no compilation or parity",
            "policy_latency": "synchronized public policy.infer including transforms, excluding noise generation, JSON and network",
        },
        "analysis": {
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 20261001,
            "bootstrap_seeds": {"2": 20261003, "4": 20261005, "10": 20261011},
            "cluster": "condition_id",
            "minimum_one_step_failures": 20,
            "minimum_rescues": 5,
            "no_benchmark_pooling": True,
            "interval": "95% percentile; null when fewer than two clusters",
        },
        "stopping": {
            "outcome_based": False,
            "maximum_scored_episodes": {k: v["max_episodes"] for k, v in benchmarks.items()},
            "errors": "preserve exact failed attempt, halt and diagnose; retry same planned case only",
            "full_run_gate": "all real-checkpoint parity and smoke record audits pass",
            "smoke_excluded": True,
            "no_screening": True,
        },
        "runtime": json.loads(Path(a.runtime).read_text()),
        "robocasa": {
            "status": "blocked_before_evaluation",
            "reason": "Completed public candidate save exists, but no verified checkpoint/configuration/observation-action/simulator interface provenance; see ROBOCASA_DISCOVERY.md",
        },
    }
    out.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    out.with_suffix(".sha256").write_text(file_hash(out) + "  " + out.name + "\n")
    print(
        json.dumps(
            {
                "manifest": str(out),
                "sha256": file_hash(out),
                "checkpoint": checkpoint_hash,
                "max_episodes": protocol["stopping"]["maximum_scored_episodes"],
            }
        )
    )


if __name__ == "__main__":
    main()
