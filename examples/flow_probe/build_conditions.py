"""Metadata-only, outcome-blind declaration of new flow-probe validation cases.

Reuses the audited initial-state and BDDL loader from frozen_flow without
editing it. Each textual selection modification must match exactly once.
"""
import argparse
import hashlib
import json
import pathlib
import runpy

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
AXES = ("Background Textures", "Robot Initial States", "Camera Viewpoints",
        "Language Instructions", "Sensor Noise", "Objects Layout", "Light Conditions")
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
PRIOR = ("frozen_flow", "selective_flow", "rescue_characterization")


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def prior_documents():
    return [(ROOT / "examples" / n / "protocol.json",
             json.loads((ROOT / "examples" / n / "protocol.json").read_text())) for n in PRIOR]


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError("Audited loader changed: expected exactly one " + repr(old))
    return source.replace(old, new)


def candidate_inventory(entries, base_map, excluded):
    """Fail closed for any stratum with fewer than three new base families."""
    inventory = []
    for suite in SUITES:
        for axis in AXES:
            for severity in (3, 4):
                candidates = [x for x in entries[suite]
                              if x["category"] == axis and x["difficulty_level"] == severity
                              and x["name"] not in excluded]
                families = set()
                chosen = []
                for item in sorted(candidates, key=lambda x: x["id"]):
                    matches = [f for f in base_map[suite]
                               if item["name"] == f or item["name"].startswith(f + "_")]
                    if not matches:
                        raise RuntimeError("Cannot identify family " + item["name"])
                    family = max(matches, key=len)
                    if family not in families:
                        families.add(family)
                        if len(chosen) < 3:
                            chosen.append({"registry_id": item["id"], "name": item["name"], "family": family})
                row = dict(suite=suite, axis=axis, severity=severity,
                           candidate_conditions=len(candidates), candidate_families=len(families), selected=chosen)
                inventory.append(row)
                if len(chosen) != 3:
                    raise RuntimeError("New-condition stratum shortage: " + json.dumps(row))
    return inventory


def audit_separation(spec, name, docs):
    previous = [(c, a) for _, d in docs for c in d["benchmarks"].get(name, {}).get("conditions", [])
                for a in c["cases"]]
    old_tuples = {(c["condition_id"], a["seed"], a["init_index"]) for c, a in previous}
    old_named = {(c["task_name"], a["seed"], a["initial_state_sha256"]) for c, a in previous}
    old_names = {c["task_name"] for c, _ in previous}
    old_hashes = {(c["suite"], c["family"], a["initial_state_sha256"]) for c, a in previous}
    # Include prior smoke tuples and state hashes in the audit. Shared smoke state
    # 30 for standard is explicitly permitted only in the new smoke partition.
    for _, d in docs:
        b = d["benchmarks"].get(name, {})
        cmap = {c["condition_id"]: c for c in b.get("conditions", [])}
        for a in b.get("smoke_cases", []):
            c = cmap[a["condition_id"]]
            old_tuples.add((c["condition_id"], a["seed"], a["init_index"]))
            old_named.add((c["task_name"], a["seed"], a["initial_state_sha256"]))
            old_hashes.add((c["suite"], c["family"], a["initial_state_sha256"]))
    keys = []
    hashes = []
    shared = []
    for c in spec["conditions"]:
        if name == "libero_plus" and c["task_name"] in old_names:
            raise RuntimeError("Prior Plus condition reused")
        for a in c["cases"]:
            key = (c["condition_id"], a["seed"], a["init_index"])
            if key in old_tuples or (c["task_name"], a["seed"], a["initial_state_sha256"]) in old_named:
                raise RuntimeError("Prior case reused")
            h = (c["suite"], c["family"], a["initial_state_sha256"])
            if h in old_hashes:
                if not (name == "libero_plus" and c["available_initial_states"] == 1):
                    raise RuntimeError("Prior within-family initial state reused: " + str(key))
                shared.append(dict(condition_id=c["condition_id"], seed=a["seed"],
                                   reason="Singleton initial state; unseen perturbation condition and seed"))
            keys.append(key)
            hashes.append(h)
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate main cases")
    cmap = {c["condition_id"]: c for c in spec["conditions"]}
    for a in spec["smoke_cases"]:
        c = cmap[a["condition_id"]]
        key = (c["condition_id"], a["seed"], a["init_index"])
        if key in set(keys) or key in old_tuples:
            raise RuntimeError("Smoke case overlaps main or previous case")
        if (c["task_name"], a["seed"], a["initial_state_sha256"]) in old_named:
            raise RuntimeError("Smoke named case overlaps prior")
    return dict(passed=True, main_cases=len(keys), unique_case_keys=len(set(keys)),
                distinct_suite_family_state_hashes=len(set(hashes)),
                prior_case_tuple_overlap=0, prior_named_seed_state_overlap=0,
                prior_condition_name_overlap=len({c["task_name"] for c in spec["conditions"]} & old_names),
                prior_within_family_state_overlap=len(shared), explicitly_shared_singleton_cases=shared,
                standard_limitation="All standard tasks were previously analyzed; only state/seed/noise cases are new.",
                plus_limitation="New conditions and seeds; singleton state conditions reuse base physical state.",
                prior_protocols=[dict(path=str(p), sha256=digest(p)) for p, _ in docs])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=("libero", "libero_plus"), required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    output = pathlib.Path(a.output)
    audit_path = output.with_suffix(".separation.json")
    if output.exists() or audit_path.exists():
        raise RuntimeError("Refuse to overwrite case declaration or audit")
    docs = prior_documents()
    old = docs[0][1]
    name = a.benchmark
    base = old["benchmarks"][name]
    excluded = {c["task_name"] for _, d in docs
                for c in d["benchmarks"].get("libero_plus", {}).get("conditions", [])}
    base_map_path = ROOT / "third_party/libero/libero/libero/benchmark/libero_suite_task_map.py"
    inventory = []
    if name == "libero_plus":
        entries = json.loads(pathlib.Path(base["classification_path"]).read_text())
        inventory = candidate_inventory(entries, runpy.run_path(str(base_map_path))["libero_task_map"], excluded)
    helper = ROOT / "examples/frozen_flow/protocol.py"
    source = helper.read_text().split("\ndef main():")[0]
    source = replace_once(source, 'for category in ("Robot Initial States", "Camera Viewpoints", "Objects Layout"):',
                          "for category in AXES:")
    source = replace_once(source, "for severity in (2, 3):", "for severity in (3, 4):")
    source = replace_once(source, 'if x["category"] == category and x["difficulty_level"] == severity',
                          'if x["category"] == category and x["difficulty_level"] == severity and x["name"] not in EXCLUDED')
    source = replace_once(source, "if len(families) == 2:", "if len(families) == 3:")
    source = replace_once(source, "not 0 < len(conditions) <= 48", "len(conditions) != 168")
    namespace = dict(AXES=AXES, EXCLUDED=excluded)
    exec(compile(source, str(helper), "exec"), namespace)
    spec = namespace["build_benchmark"](name, base["root"], base["classification_path"],
                                        str(base_map_path) if name == "libero_plus" else None)
    import numpy as np
    from libero.libero import benchmark
    smoke = []
    for n, c in enumerate(spec["conditions"]):
        suite = benchmark.get_benchmark_dict()[c["suite"]](task_order_index=0)
        states = suite.get_task_init_states(c["task_index"])
        if name == "libero":
            if len(states) < 50:
                raise RuntimeError("Expected >=50 standard states")
            indices = list(range(31, 50))
            seed_start = 6001
            smoke_index = 30
        elif len(states) == 1 and c["category"] == "Objects Layout":
            indices, seed_start, smoke_index = [0] * 10, 7001, 0
        elif len(states) >= 42:
            indices, seed_start, smoke_index = list(range(31, 41)), 7001, 41
        else:
            raise RuntimeError("Initial-state shortage: no substitute allowed")
        def state_hash(i):
            return hashlib.sha256(np.asarray(states[i]).tobytes()).hexdigest()
        c["cases"] = [dict(seed=seed_start+i, init_index=j, initial_state_sha256=state_hash(j))
                      for i, j in enumerate(indices)]
        c["distinct_declared_initial_states"] = len(set(indices))
        c["distinct_declared_initial_state_hashes"] = len({x["initial_state_sha256"] for x in c["cases"]})
        if c["category"] == "Language Instructions":
            c["prompt"] = suite.get_task(c["task_index"]).language
            c["prompt_source"] = "LIBERO-Plus task.language / perturbed BDDL language_instruction"
            if c["prompt"] != c["bddl_language_instruction"]:
                raise RuntimeError("Language perturbation mismatch")
        stratum = c["suite"] if name == "libero" else c["category"]
        if stratum not in {x["smoke_stratum"] for x in smoke}:
            smoke.append(dict(condition_id=c["condition_id"], seed=960001+n if name == "libero" else 970001+n,
                              init_index=smoke_index, initial_state_sha256=state_hash(smoke_index), smoke_stratum=stratum))
    for k in ("installed_assets_sha256", "installed_asset_files", "installed_assets_path"):
        spec[k] = base[k]
    spec["smoke_cases"] = smoke
    spec["max_episodes"] = sum(len(c["cases"]) * 4 for c in spec["conditions"])
    spec["selection_rule"] = ("Metadata only: standard all40 tasks states31:50 seeds6001:6020; Plus per suite/all7axes/"
                              "severity3,4 first3 distinct families sorted registry ID excluding union of312 prior condition "
                              "names; states31:41 or singleton layout0 seeds7001:7011. No outcomes used.")
    spec["candidate_inventory"] = inventory
    spec["excluded_prior_plus_condition_count"] = len(excluded)
    spec["audited_loader"] = dict(path=str(helper), sha256=digest(helper), selection_transform="exact-match-once replacements")
    audit = audit_separation(spec, name, docs)
    if len(spec["conditions"]) != (40 if name == "libero" else 168):
        raise RuntimeError("Unexpected condition count")
    if audit["main_cases"] != (760 if name == "libero" else 1680):
        raise RuntimeError("Unexpected case count")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        f.write(json.dumps(spec, sort_keys=True, separators=(",", ":")) + "\n")
    with audit_path.open("x") as f:
        f.write(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(benchmark=name, conditions=len(spec["conditions"]), cases=audit["main_cases"],
                         episodes=spec["max_episodes"], smoke_episodes=len(smoke)*4, output=str(output),
                         sha256=digest(output), separation_audit=str(audit_path))))


if __name__ == "__main__":
    main()

