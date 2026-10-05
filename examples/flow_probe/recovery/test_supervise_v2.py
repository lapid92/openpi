"""Recovery guards tested without loading model, supervisor, or simulator."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import supervise_v2 as recovery


def setup_files(tmp_path):
    paths = [tmp_path / name for name in ["supervise.py", "audit.py", "wrapper.py", "protocol.json"]]
    for path in paths:
        path.write_text("{}")
    original, audit, wrapper, manifest = paths
    mh = recovery.file_hash(manifest)
    gates = {}
    for field, status in [("approved_smoke", "approved_for_full"), ("smoke_audit", "passed")]:
        path = tmp_path / (field + ".json")
        path.write_text(json.dumps({"status": status, "manifest_sha256": mh}))
        gates[field] = {"path": str(path), "sha256": recovery.file_hash(path)}
    spec = {
        "schema_version": 1,
        "files": {str(p): recovery.file_hash(p) for p in paths[:3]},
        "original_manifest": {"path": str(manifest), "sha256": mh},
        **gates,
    }
    path = tmp_path / "recovery.json"
    path.write_text(json.dumps(spec))
    return path, spec, original, audit, wrapper


def test_valid_hash_gates(tmp_path):
    path, spec, original, audit, wrapper = setup_files(tmp_path)
    assert recovery.validate_recovery(path, recovery.file_hash(path), original, audit, wrapper) == spec


@pytest.mark.parametrize(
    "field", ["manifest_hash", "original", "audit", "wrapper", "protocol", "approval", "smoke_audit"]
)
def test_changed_artifacts_fail(tmp_path, field):
    path, spec, original, audit, wrapper = setup_files(tmp_path)
    expected = recovery.file_hash(path)
    if field == "manifest_hash":
        expected = "0" * 64
    else:
        targets = {
            "original": original,
            "audit": audit,
            "wrapper": wrapper,
            "protocol": Path(spec["original_manifest"]["path"]),
            "approval": Path(spec["approved_smoke"]["path"]),
            "smoke_audit": Path(spec["smoke_audit"]["path"]),
        }
        targets[field].write_text("changed")
    with pytest.raises(RuntimeError):
        recovery.validate_recovery(path, expected, original, audit, wrapper)


def test_wrong_approval_status_fails_even_with_valid_hash(tmp_path):
    path, spec, original, audit, wrapper = setup_files(tmp_path)
    gate = Path(spec["approved_smoke"]["path"])
    gate.write_text(json.dumps({"status": "rejected", "manifest_sha256": spec["original_manifest"]["sha256"]}))
    spec["approved_smoke"]["sha256"] = recovery.file_hash(gate)
    path.write_text(json.dumps(spec))
    with pytest.raises(RuntimeError):
        recovery.validate_recovery(path, recovery.file_hash(path), original, audit, wrapper)


def test_missing_required_pin_fails(tmp_path):
    path, spec, original, audit, wrapper = setup_files(tmp_path)
    del spec["files"][str(audit)]
    path.write_text(json.dumps(spec))
    with pytest.raises(RuntimeError):
        recovery.validate_recovery(path, recovery.file_hash(path), original, audit, wrapper)


def test_exact_one_substitution():
    source = 'a = HERE / "independent_audit.py"\nb = 1\n'
    assert recovery.patched_source(source) == 'a = HERE / "recovery" / "audit_v2.py"\nb = 1\n'
    for bad in ["nothing", source + source]:
        with pytest.raises(RuntimeError):
            recovery.patched_source(bad)


def test_full_only_original_manifest(tmp_path):
    _, spec, *_ = setup_files(tmp_path)
    args = ["--manifest", spec["original_manifest"]["path"], "--output", str(tmp_path), "--phase", "full"]
    assert recovery.validate_arguments(args, spec).phase == "full"
    with pytest.raises(SystemExit):
        recovery.validate_arguments(args[:-1] + ["smoke"], spec)
    args[1] = str(tmp_path / "changed.json")
    with pytest.raises(RuntimeError):
        recovery.validate_arguments(args, spec)
