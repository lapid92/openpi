"""Freeze independent selective-step protocol before any new rollout."""

import argparse
import copy
import datetime
import json
from pathlib import Path

from protocol import file_hash
from protocol import tree_hash

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(HERE / "protocol.json"))
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise RuntimeError("Refusing to overwrite frozen protocol")
    old = json.loads((ROOT / "examples/frozen_flow/protocol.json").read_text())
    protocol = copy.deepcopy(old)
    spec = json.loads((HERE / "conditions.json").read_text())
    for k in ("installed_assets_sha256", "installed_asset_files", "installed_assets_path"):
        spec[k] = old["benchmarks"]["libero_plus"][k]
    head = json.loads((HERE / "setup/head.json").read_text())
    assert file_hash(head["path"]) == head["sha256"]
    checkpoint_hash, _ = tree_hash(protocol["checkpoint"]["path"])
    assert checkpoint_hash == protocol["checkpoint"]["sha256"]
    protocol.update(
        study="pi05-independent-selective-steps",
        base_git_commit="de3e9a67f0fab9bd3a05d3410e067562c9d8a1c0",
        predeclared_at_utc=datetime.datetime.now(datetime.UTC).isoformat(),
        benchmarks={"libero_plus": spec},
        head=head,
        flow_steps=[1, 10],
        preparation_seed=900199,
        metadata_rule={"Robot Initial States": 10, "Objects Layout": 10, "Camera Viewpoints": 1},
        analysis={
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 20261003,
            "cluster": "condition_id",
            "sensitivity_cluster": "family",
            "initial_log_sigma_pair_tolerance": 1e-6,
            "sigma_min_rescues": 10,
            "sigma_min_regressions": 10,
            "sigma_min_conditions": 5,
            "sigma_direction": "Higher initial log sigma predicts rescue rather than regression; never reverse after results.",
            "sigma_endpoint": "Episode first chunk, first velocity evaluation only; rescue vs regression AUROC overall and by perturbation type.",
            "primary": "Rule minus fixed10 paired success lower95 condition-cluster CI >=0 AND rule minus fixed10 mean per-case mean scored-policy-call latency upper95 CI <0. No noninferiority margin; crossing zero is inconclusive, not equivalence.",
            "secondary": "Rule vs fixed1; fixed10 rescues/regressions by type; family-cluster sensitivity; velocity evaluations and simulator-inclusive latency separately.",
            "no_benchmark_pooling": True,
        },
        stopping={
            "outcome_based": False,
            "maximum_scored_episodes": {"libero_plus": 1920},
            "smoke_episodes": 9,
            "no_screening": True,
            "smoke_excluded": True,
            "errors": "Preserve attempts, halt on execution or integrity error; diagnose infrastructure only; retry identical declared case with full attempt history. Never replace a condition.",
            "full_run_gate": "Real-checkpoint fixed-output parity, same-pass sigma equality, direct-rule smoke replay and complete smoke raw-record audits must pass.",
            "no_training": True,
            "no_rule_tuning": True,
        },
    )
    protocol["checkpoint"]["shared_by"] = ["libero_plus"]
    protocol["settings"].update(
        residual_head=head["sha256"],
        arm_order="Cyclic rotation of [1,10] by case ordinal; smoke additionally metadata replay.",
        policy_latency="Synchronized instrumented fixed sampler including transforms and one head evaluation; head time separately; no uncounted pi evaluation; excludes noise/JSON/network.",
    )
    protocol.pop("robocasa", None)
    protocol.pop("execution_amendment", None)
    for f in HERE.rglob("*.py"):
        protocol["source_sha256"][str(f)] = file_hash(f)
    out.write_text(json.dumps(protocol, sort_keys=True, separators=(",", ":")) + "\n")
    out.with_suffix(".sha256").write_text(file_hash(out) + "  " + out.name + "\n")
    print(json.dumps({"manifest": str(out), "sha256": file_hash(out), "main": 1920, "smoke": 9}))


if __name__ == "__main__":
    main()
