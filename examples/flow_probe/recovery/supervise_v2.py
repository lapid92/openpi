"""Reviewed infrastructure-only full supervisor recovery.

Never reruns smoke. Execute only with a separately published recovery manifest
and its explicit SHA256. All original study sources and protocol remain frozen.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[1]
ORIGINAL = HERE / "supervise.py"
AUDIT = HERE / "recovery" / "audit_v2.py"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_recovery(path, expected_sha256, original=ORIGINAL, audit=AUDIT, wrapper=None):
    wrapper = Path(__file__).resolve() if wrapper is None else Path(wrapper).resolve()
    if file_hash(path) != expected_sha256:
        raise RuntimeError("Recovery manifest SHA256 mismatch")
    spec = json.loads(Path(path).read_text())
    if spec["schema_version"] != 1:
        raise RuntimeError("Unsupported recovery schema")
    files = spec["files"]
    required = {str(Path(original).resolve()), str(Path(audit).resolve()), str(wrapper)}
    if not required <= set(files):
        raise RuntimeError("Recovery manifest must pin original supervisor, replacement audit and wrapper")
    for name, digest in files.items():
        if not Path(name).is_absolute() or str(Path(name).resolve()) != name:
            raise RuntimeError("Recovery file paths must be canonical absolute paths")
        if file_hash(name) != digest:
            raise RuntimeError("Recovery file hash mismatch: " + name)
    manifest = spec["original_manifest"]
    if not Path(manifest["path"]).is_absolute() or file_hash(manifest["path"]) != manifest["sha256"]:
        raise RuntimeError("Original scientific manifest changed")
    for field, status in (("approved_smoke", "approved_for_full"), ("smoke_audit", "passed")):
        gate = spec[field]
        if not Path(gate["path"]).is_absolute() or file_hash(gate["path"]) != gate["sha256"]:
            raise RuntimeError("Recovery gate changed: " + field)
        evidence = json.loads(Path(gate["path"]).read_text())
        if evidence.get("status") != status or evidence.get("manifest_sha256") != manifest["sha256"]:
            raise RuntimeError("Recovery gate status or identity mismatch: " + field)
    return spec


def patched_source(source):
    target = 'HERE / "independent_audit.py"'
    if source.count(target) != 1:
        raise RuntimeError("Expected exactly one original independent-audit command")
    return source.replace(target, 'HERE / "recovery" / "audit_v2.py"')


def validate_arguments(arguments, spec):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", choices=["full"], required=True)
    args = parser.parse_args(arguments)
    if Path(args.manifest).resolve() != Path(spec["original_manifest"]["path"]).resolve():
        raise RuntimeError("Full supervisor must use original scientific manifest")
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovery-manifest", required=True)
    parser.add_argument("--recovery-manifest-sha256", required=True)
    recovery, arguments = parser.parse_known_args()
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    spec = validate_recovery(recovery.recovery_manifest, recovery.recovery_manifest_sha256)
    validate_arguments(arguments, spec)
    source = ORIGINAL.read_text()
    if hashlib.sha256(source.encode()).hexdigest() != spec["files"][str(ORIGINAL)]:
        raise RuntimeError("Original source changed between validation and loading")
    replacement = patched_source(source)
    print(
        json.dumps(
            {
                "recovery_manifest": str(Path(recovery.recovery_manifest).resolve()),
                "recovery_manifest_sha256": recovery.recovery_manifest_sha256,
                "original_supervisor_sha256": spec["files"][str(ORIGINAL)],
                "executed_source_sha256": hashlib.sha256(replacement.encode()).hexdigest(),
                "replacement_audit_sha256": spec["files"][str(AUDIT)],
                "replacement": "independent_audit.py -> recovery/audit_v2.py",
                "phase": "full",
            }
        ),
        flush=True,
    )
    sys.argv = [str(ORIGINAL), *arguments]
    # The original file location preserves its repository, source and output roots.
    exec(compile(replacement, str(ORIGINAL), "exec"), {"__name__": "__main__", "__file__": str(ORIGINAL)})


if __name__ == "__main__":
    main()
