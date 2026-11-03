"""Offline, fail-closed reporting for the two RoboCasa denoising sweeps.

Input layout: ROOT/{pi05,smolvla}/seed{7,8,9}/n{1,2,4,10,15,20}/
with manifest.json and either raw ``episodes/**/*.json`` or episodes.csv.
No collection or network access occurs here.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from .protocol import SEEDS, STEPS, GROUPS as TASK_GROUPS, RESET_PROTOCOL, stable_seed
from .results import manifest_digest

MODELS = ("pi05", "smolvla")
GROUPS = (*TASK_GROUPS, "overall")


def _csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("episode_index", "denoising_steps", "scene_seed", "replan_steps",
                    "policy_seed", "policy_queries", "action_network_evaluations"):
            if key in row and row[key] != "":
                row[key] = int(row[key])
        if row.get("success") in ("True", "False"):
            row["success"] = row["success"] == "True"
    return rows


def load_cell(path: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((path / "manifest.json").read_text())
    files = sorted(path.glob("episodes/*.json")) + sorted(path.glob("shard-*/episodes/*.json"))
    if files:
        rows = [json.loads(file.read_text()) for file in files]
        csv_path = path / "episodes.csv"
        if csv_path.is_file():
            with csv_path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                csv_rows = list(reader)
                fields = reader.fieldnames or []
            if set(fields) != {key for row in rows for key in row}:
                raise ValueError(f"Raw/CSV episode columns conflict: {path}")
            def cell(value):
                if isinstance(value, (dict, list)):
                    return json.dumps(value, sort_keys=True)
                return "" if value is None else str(value)
            raw_projection = sorted(tuple(cell(row.get(k)) for k in fields) for row in rows)
            csv_projection = sorted(tuple(row[k] for k in fields) for row in csv_rows)
            if raw_projection != csv_projection:
                raise ValueError(f"Raw/CSV episode evidence conflicts: {path}")
    elif (path / "episodes.csv").is_file():
        rows = _csv_rows(path / "episodes.csv")
    else:
        raise ValueError(f"Missing episode evidence: {path}")
    return manifest, rows


def validate_cell(manifest: dict, rows: list[dict], model: str, seed: int,
                  steps: int) -> dict[tuple[str, int], dict]:
    presets = json.loads(Path(__file__).with_name("presets.json").read_text())
    pinned = dict(presets["models"][model], tasks=presets["tasks"])
    for field in ("protocol_id", "checkpoint_revision", "tasks", "split",
                  "replan_steps", "horizon_multiplier", "episodes_per_task", "code_revisions"):
        if manifest.get(field) != pinned.get(field):
            raise ValueError(f"Manifest differs from pinned {model} protocol: {field}")
    for key, expected in (("model", model), ("scene_seed", seed),
                          ("denoising_steps", steps), ("policy_seed", 0)):
        if manifest.get(key) != expected:
            raise ValueError(f"Manifest {key} differs for {model}/seed{seed}/n{steps}")
    if manifest.get("episodes_per_task") != 50 or len(manifest.get("tasks", ())) != 50:
        raise ValueError("Expected 50 tasks and 50 episodes per task")
    if manifest.get("horizon_multiplier") != 1.5:
        raise ValueError("Unexpected horizon multiplier")
    expected_split, expected_replan = ("pretrain", 5) if model == "pi05" else ("target", 50)
    if manifest.get("split") != expected_split or manifest.get("replan_steps") != expected_replan:
        raise ValueError("Unexpected split or replan period")
    tasks = {entry["task"]: entry["group"] for entry in manifest["tasks"]}
    if len(tasks) != 50 or set(tasks.values()) != set(GROUPS[:-1]):
        raise ValueError("Duplicate task or unexpected group")
    expected = {(task, idx) for task in tasks for idx in range(50)}
    digest = manifest_digest(manifest)
    by_key: dict[tuple[str, int], dict] = {}
    for row in rows:
        task, idx = row.get("task"), row.get("episode_index")
        key = (task, idx)
        if key not in expected or row.get("group") != tasks[task]:
            raise ValueError(f"Unexpected episode identity: {key}")
        for field, value in (("manifest_sha256", digest),
                             ("campaign_id", manifest["campaign_id"]),
                             ("protocol_id", manifest["protocol_id"]),
                             ("checkpoint_revision", manifest["checkpoint_revision"]),
                             ("denoising_steps", steps), ("scene_seed", seed),
                             ("replan_steps", expected_replan),
                             ("scene_reset_protocol", RESET_PROTOCOL)):
            if row.get(field) != value:
                raise ValueError(f"Episode {key}: {field} mismatch")
        if model == "smolvla":
            if row.get("model") != model or row.get("split") != expected_split or row.get("prompt") != task:
                raise ValueError(f"Episode {key}: SmolVLA protocol mismatch")
        else:
            if row.get("instruction") in (None, ""):
                raise ValueError(f"Episode {key}: missing pi0.5 instruction")
        expected_policy_seed = stable_seed(f"public_{model}_baseline", task, idx, 0)
        if row.get("policy_seed") != expected_policy_seed:
            raise ValueError(f"Episode {key}: policy seed mismatch")
        queries = row.get("policy_queries")
        if type(queries) is not int or queries <= 0 or row.get("action_network_evaluations") != queries * steps:
            raise ValueError(f"Episode {key}: measured NFE mismatch")
        if row.get("status") not in ("success", "policy_failure") or type(row.get("success")) is not bool:
            raise ValueError(f"Episode {key}: incomplete or inconsistent outcome")
        if row["success"] != (row["status"] == "success"):
            raise ValueError(f"Episode {key}: success flag mismatch")
        for field in ("model_xml_sha256", "initial_state_sha256"):
            value = row.get(field)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"Episode {key}: invalid {field}")
        if key in by_key:
            if row != by_key[key]:
                raise ValueError(f"Conflicting duplicate episode: {key}")
        else:
            by_key[key] = row
    if set(by_key) != expected:
        raise ValueError(f"Incomplete coverage: {model}/seed{seed}/n{steps}: {len(by_key)}/2500")
    return by_key


def build_report(root: Path) -> dict:
    """Validate all 36 cells and return JSON-serializable rates and hash comparisons."""
    root = Path(root)
    cells = {}
    fixed_manifests = {}
    per_seed = []
    for model in MODELS:
        for seed in SEEDS:
            for steps in STEPS:
                path = root / model / f"seed{seed}" / f"n{steps}"
                manifest, rows = load_cell(path)
                fixed = {key: value for key, value in manifest.items()
                         if key not in {"scene_seed", "denoising_steps", "campaign_id", "s3_prefix"}}
                if model in fixed_manifests and fixed != fixed_manifests[model]:
                    raise ValueError(f"Protocol manifest drift: {model}/seed{seed}/n{steps}")
                fixed_manifests[model] = fixed
                cell = validate_cell(manifest, rows, model, seed, steps)
                cells[model, seed, steps] = cell
                for group in GROUPS:
                    chosen = [row for row in cell.values() if group == "overall" or row["group"] == group]
                    wins = sum(row["success"] for row in chosen)
                    per_seed.append(dict(model=model, seed=seed, steps=steps, group=group,
                                         episodes=len(chosen), successes=wins,
                                         success_rate_pct=100 * wins / len(chosen)))
    summary = []
    for model in MODELS:
        for steps in STEPS:
            for group in GROUPS:
                selected = [row for row in per_seed if (row["model"], row["steps"], row["group"]) == (model, steps, group)]
                rates = [row["success_rate_pct"] for row in selected]
                summary.append(dict(model=model, steps=steps, group=group,
                                    **{f"seed{row['seed']}_pct": row["success_rate_pct"] for row in selected},
                                    mean_pct=statistics.mean(rates), sample_sd_pct=statistics.stdev(rates),
                                    public_reference_pct=None))
    scene_hashes = []
    for model in MODELS:
        for seed in SEEDS:
            for i, left_steps in enumerate(STEPS):
                for right_steps in STEPS[i + 1:]:
                    left, right = cells[model, seed, left_steps], cells[model, seed, right_steps]
                    record = dict(model=model, seed=seed, steps_a=left_steps, steps_b=right_steps,
                                  episodes=len(left))
                    for field in ("model_xml_sha256", "initial_state_sha256"):
                        record[f"{field}_nonmissing"] = len(left)
                        record[f"{field}_equal"] = sum(left[key][field] == right[key][field] for key in left)
                    scene_hashes.append(record)
    return dict(total_episodes=sum(len(cell) for cell in cells.values()),
                per_seed=per_seed, summary=summary, scene_hash_comparison=scene_hashes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.root)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, rows in (("per_seed.csv", report["per_seed"]),
                       ("summary.csv", report["summary"]),
                       ("scene_hash_comparison.csv", report["scene_hash_comparison"])):
        with (args.output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
