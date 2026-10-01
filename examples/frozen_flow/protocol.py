# ruff: noqa: C408
"""Predeclare a frozen fixed-step study without reading any evaluation outcomes."""

import argparse
import hashlib
import json
import pathlib
import runpy
import subprocess

STEPS = [1, 2, 4, 10]


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tree_hash(root):
    root = pathlib.Path(root)
    files = [
        {"path": str(p.relative_to(root)), "sha256": file_hash(p), "bytes": p.stat().st_size}
        for p in sorted(root.rglob("*"))
        if p.is_file()
    ]
    if not files:
        raise ValueError("Empty checkpoint")
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), files


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_benchmark(name, root, classification=None, base_task_map=None):
    # Invoke separately in each simulator environment; imports must resolve to exact root.
    from libero.libero import benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import bddl_utils
    import numpy as np
    import torch

    root = pathlib.Path(root).resolve()
    if root not in pathlib.Path(benchmark.__file__).resolve().parents:
        raise RuntimeError("Benchmark import root mismatch")
    entries = json.loads(pathlib.Path(classification).read_text()) if classification else None
    base_map = runpy.run_path(base_task_map)["libero_task_map"] if base_task_map else None

    def base_family(suite, task_name):
        if name == "libero":
            return task_name
        matches = [x for x in base_map[suite] if task_name == x or task_name.startswith(x + "_")]
        if not matches:
            raise RuntimeError("Cannot determine base family: " + task_name)
        return max(matches, key=len)

    conditions = []
    for suite_name in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        suite = benchmark.get_benchmark_dict()[suite_name](task_order_index=0)
        selected = list(range(suite.n_tasks))
        if name == "libero_plus":
            if entries is None:
                raise ValueError("Plus requires classification")
            selected = []
            for category in ("Robot Initial States", "Camera Viewpoints", "Objects Layout"):
                for severity in (2, 3):
                    candidates = [
                        x
                        for x in entries[suite_name]
                        if x["category"] == category and x["difficulty_level"] == severity
                    ]
                    families = set()
                    for item in sorted(candidates, key=lambda x: x["id"]):
                        family = base_family(suite_name, item["name"])
                        if family in families:
                            continue
                        families.add(family)
                        selected.append(next(i for i, e in enumerate(entries[suite_name]) if e["id"] == item["id"]))
                        if len(families) == 2:
                            break
        for index in selected:
            task = suite.get_task(index)
            loaded_paths = []
            original_load = torch.load

            def record_load(path, *args, _paths=loaded_paths, _load=original_load, **kwargs):
                _paths.append(str(path))
                return _load(path, *args, **kwargs)

            torch.load = record_load
            try:
                states = suite.get_task_init_states(index)
            finally:
                torch.load = original_load
            if len(loaded_paths) != 1 or len(states) < 1:
                raise RuntimeError("Unexpected initial-state loading")
            if len(states) < 10 and not (
                name == "libero_plus"
                and entries[suite_name][index]["category"] == "Objects Layout"
                and len(states) == 1
            ):
                raise RuntimeError("Unexpected shortage of distinct initial states")
            indices = list(range(10)) if len(states) >= 10 else [0] * 10
            task_dict = task._asdict()
            bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            bddl_asset = (
                pathlib.Path(str(bddl).split("_view_")[0] + ".bddl")
                if "_view_" in str(bddl) and "_initstate_" in str(bddl)
                else bddl
            )
            init_path = pathlib.Path(loaded_paths[0])
            entry = entries[suite_name][index] if entries else {}
            if entries and entry["name"] != task.name:
                raise RuntimeError("Classification task order mismatch")
            family = base_family(suite_name, task.name)
            conditions.append(
                dict(
                    condition_id=name + ":" + suite_name + ":" + str(index),
                    suite=suite_name,
                    task_index=index,
                    task_name=task.name,
                    family=family,
                    category=entry.get("category", "standard"),
                    severity=entry.get("difficulty_level", 0),
                    task=task_dict,
                    bddl_path=str(bddl),
                    prompt=task.language
                    if name == "libero"
                    else benchmark.grab_language_from_filename(suite_name, family + ".bddl"),
                    prompt_source="standard LIBERO task.language"
                    if name == "libero"
                    else "standard LIBERO base-family filename language parser; excludes variation suffix",
                    bddl_language_instruction=bddl_utils.get_problem_info(str(bddl_asset))["language_instruction"],
                    bddl_asset_path=str(bddl_asset),
                    bddl_sha256=file_hash(bddl_asset),
                    init_file_path=str(init_path),
                    init_file_sha256=file_hash(init_path),
                    available_initial_states=len(states),
                    distinct_declared_initial_states=len(set(indices)),
                    distinct_declared_initial_state_hashes=len(
                        {hashlib.sha256(np.asarray(states[i]).tobytes()).hexdigest() for i in indices}
                    ),
                    cases=[
                        dict(
                            seed=1001 + i,
                            init_index=indices[i],
                            initial_state_sha256=hashlib.sha256(np.asarray(states[indices[i]]).tobytes()).hexdigest(),
                        )
                        for i in range(10)
                    ],
                )
            )
    if name == "libero" and len(conditions) != 40:
        raise RuntimeError("Expected 40 LIBERO tasks")
    if name == "libero_plus" and not 0 < len(conditions) <= 48:
        raise RuntimeError("Invalid Plus condition count")
    first = conditions[0]
    return dict(
        root=str(root),
        commit=subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        conditions=conditions,
        classification_sha256=file_hash(classification) if classification else None,
        classification_path=str(classification) if classification else None,
        smoke_cases=[
            dict(
                condition_id=first["condition_id"],
                seed=900001,
                init_index=0,
                initial_state_sha256=first["cases"][0]["initial_state_sha256"],
            )
        ],
        max_episodes=len(conditions) * 40,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=["libero", "libero_plus"], required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--classification")
    p.add_argument("--output", required=True)
    p.add_argument("--base-task-map")
    a = p.parse_args()
    out = pathlib.Path(a.output)
    if out.exists():
        raise RuntimeError("Refuse to overwrite declared benchmark protocol")
    out.write_text(json.dumps(build_benchmark(a.benchmark, a.root, a.classification, a.base_task_map), indent=2) + "\n")


if __name__ == "__main__":
    main()
