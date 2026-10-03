"""Outcome-selected historical replay, never population evidence."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path("/volt/code/frozen-flow-study")
HERE = ROOT / "examples/rescue_characterization"
sys.path.insert(0, str(HERE))
from protocol import file_hash, tree_hash
from client import run_episode, append_record
from extract_existing import EXECUTION_FIELDS


def build_manifest(base_path, selection_path, output):
    base = json.loads(Path(base_path).read_text())
    selection = json.loads(Path(selection_path).read_text())
    manifest = copy.deepcopy(base)
    originals = {name: json.loads((ROOT / "examples" / name / "protocol.json").read_text())
                 for name in ("frozen_flow", "selective_flow")}
    original_hashes = {name: file_hash(ROOT / "examples" / name / "protocol.json") for name in originals}
    manifest["study"] = "pi05-historical-replay-outcome-selected"
    manifest["recording"]["root"] = str(output / "recordings")
    manifest["replay_selection_sha256"] = file_hash(selection_path)
    manifest["replay_selection_path"] = str(Path(selection_path).resolve())
    manifest["historical_manifests"] = {}
    entries = []
    by_benchmark = {}
    for case in selection["cases"]:
        old = originals[case["study"]]
        original_path = ROOT / "examples" / case["study"] / "protocol.json"
        if original_hashes[case["study"]] != case["manifest_sha256"]:
            raise ValueError("Historical manifest digest changed")
        if old["checkpoint"]["sha256"] != base["checkpoint"]["sha256"]:
            raise ValueError("Historical checkpoint differs")
        for field in EXECUTION_FIELDS:
            if old["settings"][field] != base["settings"][field]:
                raise ValueError("Replay execution differs: " + field)
        if old["noise_policy"] != base["noise_policy"] or old["settings"]["noise"] != base["settings"]["noise"]:
            raise ValueError("Replay noise policy differs")
        bench = case["benchmark"]
        if old["benchmarks"][bench]["commit"] != base["benchmarks"][bench]["commit"]:
            raise ValueError("Benchmark version differs")
        condition = next(c for c in old["benchmarks"][bench]["conditions"] if c["condition_id"] == case["condition_id"])
        planned = next(c for c in condition["cases"] if (c["seed"],c["init_index"]) == (case["seed"],case["init_index"]))
        entry = dict(case, replay_condition=condition, replay_case=planned)
        entries.append(entry)
        by_benchmark.setdefault(bench, {})[condition["condition_id"]] = condition
        manifest["historical_manifests"][case["study"]] = {"path": str(original_path), "sha256": original_hashes[case["study"]]}
    for bench in list(manifest["benchmarks"]):
        if bench not in by_benchmark:
            del manifest["benchmarks"][bench]
        else:
            manifest["benchmarks"][bench]["conditions"] = list(by_benchmark[bench].values())
    manifest["replay_cases"] = entries
    manifest["replay_budget"] = {"maximum_episodes": sum(len(c["replay_steps"]) for c in entries),
                               "maximum_wall_hours": 12, "minimum_free_disk_gib": 100,
                               "population_inference": False, "equivalence_required_for_visual_attribution": True}
    if manifest["replay_budget"]["maximum_episodes"] != selection["maximum_replay_episodes"]:
        raise ValueError("Selection budget mismatch")
    for script in (HERE / "replay.py", HERE / "replay_supervise.py"):
        if script.exists():
            manifest["source_sha256"][str(script)] = file_hash(script)
    return manifest


def compare_replay(original, replay):
    required = ("success", "policy_steps", "initial_state_sha256", "stabilized_state_sha256", "total_velocity_evaluations")
    for row in (original,replay):
        if row.get("status") != "ok" or type(row.get("success")) is not bool:
            raise ValueError("Replay comparison requires successful execution with Boolean outcome")
        if any(key not in row for key in required) or not row.get("chunks"):
            raise ValueError("Incomplete replay comparison evidence")
        for chunk in row["chunks"]:
            if any(key not in chunk for key in ("noise_sha256","action_sha256","observation_sha256","velocity_evaluations")):
                raise ValueError("Incomplete chunk comparison evidence")
    mismatches = []
    for key in ("success", "policy_steps", "initial_state_sha256", "stabilized_state_sha256", "total_velocity_evaluations"):
        if original.get(key) != replay.get(key):
            mismatches.append(key)
    a, b = original.get("chunks", []), replay.get("chunks", [])
    if len(a) != len(b):
        mismatches.append("chunk_count")
    for index, (old, new) in enumerate(zip(a,b)):
        for key in ("noise_sha256", "action_sha256", "observation_sha256", "velocity_evaluations"):
            if old.get(key) != new.get(key):
                mismatches.append(f"chunk_{index}.{key}")
    return {"exact_equivalent": not mismatches,
            "classification": "exact_replay" if not mismatches else "reconstruction-not-equivalent",
            "visual_attribution_eligible": not mismatches, "mismatches": mismatches}


def source_row(reference, cache):
    path = Path(reference["path"])
    if str(path) not in cache:
        if file_hash(path) != reference["file_sha256"]:
            raise ValueError("Historical record file changed")
        cache[str(path)] = path.read_bytes().splitlines()
    if type(reference.get("line")) is not int or not 1 <= reference["line"] <= len(cache[str(path)]):
        raise ValueError("Source line must be a valid one-based integer")
    raw = cache[str(path)][reference["line"] - 1]
    if hashlib.sha256(raw).hexdigest() != reference["record_sha256"]:
        raise ValueError("Historical episode changed")
    return json.loads(raw)



def audit_replays(rows, manifest, manifest_hash, *, complete=False, inspect_video=True):
    from collections import Counter
    from recording_audit import audit_record, audit_pairs
    selection = {c["case_id"]: c for c in manifest["replay_cases"]}
    expected = {(c["case_id"], n) for c in selection.values() for n in c["replay_steps"]}
    seen, cache, derived = set(), {}, []
    for row in rows:
        key = (row["historical_case_id"], row["flow_steps"])
        if key not in expected or key in seen:
            raise ValueError("Unexpected/duplicate historical replay arm")
        seen.add(key)
        entry = selection[row["historical_case_id"]]
        reference = entry["arms"][str(row["flow_steps"])]["source_refs"][0]
        if row["original_source"] != reference or row["manifest_sha256"] != manifest_hash:
            raise ValueError("Replay source/manifest identity mismatch")
        if row["phase"] != "historical_replay" or row["original_manifest_sha256"] != entry["manifest_sha256"]:
            raise ValueError("Replay phase/historical manifest mismatch")
        if row["historical_study"] != entry["study"] or row["gpu_uuid"] not in manifest["gpu_uuids"]:
            raise ValueError("Replay study/GPU identity mismatch")
        for field in ("benchmark","condition_id","suite","task_name","family","category","severity","seed","init_index"):
            if row.get(field) != entry[field]:
                raise ValueError("Replay case identity mismatch: " + field)
        if row.get("checkpoint_sha256") != manifest["checkpoint"]["sha256"]:
            raise ValueError("Replay checkpoint identity mismatch")
        original = source_row(reference, cache)
        if original.get("checkpoint_sha256") != row["checkpoint_sha256"]:
            raise ValueError("Original checkpoint identity mismatch")
        recomputed = compare_replay(original, row)
        if row["equivalence"] != recomputed:
            raise ValueError("Persisted equivalence label disagrees with raw original")
        # Metadata validation uses the exact original selected condition/case;
        # only phase is adapted for the existing independent recording auditor.
        audit_manifest = dict(manifest)
        spec = dict(manifest["benchmarks"][row["benchmark"]])
        condition = dict(entry["replay_condition"], cases=[entry["replay_case"]])
        spec["conditions"] = [condition]
        audit_manifest["benchmarks"] = {row["benchmark"]: spec}
        audit_record(dict(row, phase="main"), audit_manifest, inspect_video=inspect_video)
        derived.append(recomputed)
    audit_pairs(rows, allow_partial=True)
    if complete and seen != expected:
        raise ValueError("Incomplete declared historical replay coverage")
    return {"status":"passed", "completed":len(seen), "maximum_episodes":len(expected),
            "equivalence":dict(Counter(r["classification"] for r in derived)),
            "invalid_equivalence_labels":0, "population_inference":False,
            "original_sources":[{"path":p, "sha256":file_hash(Path(p))} for p in sorted(cache)],
            "video_decoding_checked":inspect_video,
            "interpretation":"Only exact replays support original-case visual labels; mismatched reconstructions are excluded."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--benchmark")
    parser.add_argument("--server")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-records", type=Path, nargs="+")
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()
    if not 0 <= args.worker_index < args.workers:
        raise ValueError("Invalid worker shard")
    manifest = json.loads(args.manifest.read_text())
    mh = file_hash(args.manifest)
    if args.audit_records:
        rows = [json.loads(line) for path in args.audit_records for line in path.read_text().splitlines() if line.strip()]
        result = audit_replays(rows, manifest, mh, complete=True, inspect_video=True)
        result["raw_files"] = [{"path":str(p),"sha256":file_hash(p)} for p in args.audit_records]
        args.audit_output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
        return
    if not args.benchmark or not args.server or not args.output:
        parser.error("Execution requires --benchmark --server --output")
    spec = manifest["benchmarks"][args.benchmark]
    from libero.libero import benchmark
    root = Path(spec["root"]).resolve()
    if root not in Path(benchmark.__file__).resolve().parents:
        raise ValueError("Wrong benchmark import")
    if subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip() != spec["commit"]:
        raise ValueError("Benchmark commit changed")
    for path, digest in manifest["source_sha256"].items():
        if file_hash(path) != digest:
            raise ValueError("Source changed: " + path)
    if tree_hash(spec["installed_assets_path"])[0] != spec["installed_assets_sha256"]:
        raise ValueError("Installed assets changed")
    entries = [c for c in manifest["replay_cases"] if c["benchmark"] == args.benchmark]
    entries = [c for i,c in enumerate(entries) if i % args.workers == args.worker_index]
    if not entries:
        return
    with urllib.request.urlopen(args.server + "/health",timeout=900) as response:
        health = json.load(response)
    if (health["gpu_uuid"] != manifest["gpu_uuids"][args.worker_index]
            or health["checkpoint_sha256"] != manifest["checkpoint"]["sha256"] or health["manifest_sha256"] != mh):
        raise ValueError("Server identity mismatch")
    for entry in entries:
        condition = entry["replay_condition"]
        for path_key, hash_key in (("bddl_asset_path","bddl_sha256"),("init_file_path","init_file_sha256")):
            if file_hash(condition[path_key]) != condition[hash_key]:
                raise ValueError("Historical task assets changed")
    first = entries[0]
    suite = benchmark.get_benchmark_dict()[first["suite"]](task_order_index=0)
    prepare = run_episode(suite, first["replay_condition"], dict(first["replay_case"],seed=990700 + args.worker_index),
                          1, manifest,args.benchmark,args.server,prepare_only=True)
    append_record(args.output.with_suffix(".prepare.jsonl"),prepare)
    with urllib.request.urlopen(args.server + "/health",timeout=900) as response:
        health = json.load(response)
    if health["warmed_steps"].get(args.benchmark) != [1,2,4,10] or health["verified_steps"].get(args.benchmark) != [1,2,4,10]:
        raise ValueError("All fixed arms must pass warmup/parity")
    completed = set()
    if args.output.exists():
        previous = [json.loads(line) for line in args.output.read_text().splitlines()]
        audit_replays(previous, manifest, mh, inspect_video=True)
        for line in args.output.read_text().splitlines():
            row = json.loads(line)
            if row["manifest_sha256"] != mh:
                raise ValueError("Resume manifest mismatch")
            if row["status"] == "ok":
                key = (row["historical_case_id"],row["flow_steps"])
                if key in completed:
                    raise ValueError("Duplicate replay completion")
                for kind in ("video","trace"):
                    capture = row["recording"]
                    if file_hash(Path(capture[kind+"_path"])) != capture[kind+"_sha256"]:
                        raise ValueError("Resume recording changed")
                completed.add(key)
    cache, suites = {}, {}
    for entry in entries:
        condition,case = entry["replay_condition"],entry["replay_case"]
        if entry["suite"] not in suites:
            suites[entry["suite"]] = benchmark.get_benchmark_dict()[entry["suite"]](task_order_index=0)
        for arm in entry["replay_steps"]:
            if (entry["case_id"],arm) in completed:
                continue
            reference = entry["arms"][str(arm)]["source_refs"][0]
            original = source_row(reference,cache)
            row = {k: entry[k] for k in ("benchmark","condition_id","suite","task_name","family","category","severity","seed","init_index")}
            row.update(flow_steps=arm, phase="historical_replay", historical_case_id=entry["case_id"],
                       historical_study=entry["study"], original_source=reference, original_manifest_sha256=entry["manifest_sha256"],
                       manifest_sha256=mh, checkpoint_sha256=health["checkpoint_sha256"], benchmark_commit=spec["commit"],
                       gpu_uuid=health["gpu_uuid"], prompt=condition["prompt"])
            try:
                row.update(run_episode(suites[entry["suite"]],condition,case,arm,manifest,args.benchmark,args.server))
                row["equivalence"] = compare_replay(original,row)
                audit_replays([row],manifest,mh,inspect_video=True)
            except Exception as error:
                row.update(status="error",error=repr(error))
                append_record(args.output,row)
                raise
            append_record(args.output,row)


if __name__ == "__main__":
    main()

