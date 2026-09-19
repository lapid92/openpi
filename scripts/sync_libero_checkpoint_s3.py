"""Package and transfer evaluation-only LIBERO checkpoints through the approved S3 namespace."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib
import json
import os
import pathlib
import re
import shutil
import tarfile
import tempfile
from typing import Any


S3_BUCKET = "aair-users-east-2"
S3_PREFIX = "arilap01/libero/openpi/checkpoints/"
AWS_REGION = "us-east-2"
ARCHIVE_NAME = "checkpoint.tar"
MANIFEST_NAME = "manifest.json"
SUCCESS_NAME = "_SUCCESS"
REQUIRED_PATHS = ("params", "assets", "_CHECKPOINT_METADATA")
MANIFEST_SCHEMA_VERSION = 1

_ARTIFACT_KEY_RE = re.compile(
    r"(?:tvm|finetune)/[A-Za-z0-9][A-Za-z0-9._-]{0,127}/step_[0-9]+-checkpoint_[A-Za-z0-9._-]+"
    r"(?:/sha256_[0-9a-f]{64})?\Z"
)
_REQUIRED_METADATA_FIELDS = (
    "artifact_name",
    "model_class",
    "training_run_id",
    "checkpoint_name",
    "checkpoint_label",
    "train_steps_completed",
    "config",
    "git_sha",
)


def _require_volt() -> None:
    if not pathlib.Path("/volt").exists() or not os.environ.get("VOLT_CLUSTER_NAME"):
        raise RuntimeError("Checkpoint S3 transfer is restricted to Volt pods (/volt and VOLT_CLUSTER_NAME required).")


def _validate_artifact_key(artifact_key: str) -> str:
    if not _ARTIFACT_KEY_RE.fullmatch(artifact_key):
        raise ValueError("artifact_key must match {tvm|finetune}/<run>/step_<train>-checkpoint_<label>")
    return artifact_key


def _validate_metadata(artifact_key: str, metadata: dict[str, Any]) -> None:
    missing = [field for field in _REQUIRED_METADATA_FIELDS if field not in metadata]
    if missing:
        raise ValueError(f"Artifact metadata is missing fields: {', '.join(missing)}")
    for field in ("artifact_name", "training_run_id", "checkpoint_name", "checkpoint_label", "config"):
        if not isinstance(metadata[field], str) or not metadata[field]:
            raise ValueError(f"{field} must be a non-empty string")
    if metadata["model_class"] not in {"tvm", "finetune"}:
        raise ValueError("model_class must be tvm or finetune")
    expected_config = "pi05_libero_tvm" if metadata["model_class"] == "tvm" else "pi05_libero"
    if metadata["config"] != expected_config:
        raise ValueError(f"config must be {expected_config} for model_class={metadata['model_class']}")
    if isinstance(metadata["train_steps_completed"], bool) or not isinstance(metadata["train_steps_completed"], int):
        raise ValueError("train_steps_completed must be an integer")
    if metadata["train_steps_completed"] < 0:
        raise ValueError("train_steps_completed must be nonnegative")
    if not re.fullmatch(r"[0-9a-f]{40}", str(metadata["git_sha"])):
        raise ValueError("git_sha must be a full lowercase 40-character SHA")
    expected_prefix = (
        f"{metadata['model_class']}/{metadata['training_run_id']}/"
        f"step_{metadata['train_steps_completed']}-checkpoint_{metadata['checkpoint_label']}"
    )
    if artifact_key != expected_prefix:
        raise ValueError(f"artifact_key does not match metadata; expected {expected_prefix}")
    if metadata["checkpoint_name"] != metadata["checkpoint_label"]:
        raise ValueError("checkpoint_name and checkpoint_label must match")
    label = metadata["checkpoint_label"]
    if not label.isdigit():
        raise ValueError("checkpoint_label must be numeric")
    label_step = int(label)
    logical_step = metadata["train_steps_completed"]
    if logical_step not in {label_step, label_step + 1}:
        raise ValueError("checkpoint label and logical train step are inconsistent")


def _key(artifact_key: str, filename: str) -> str:
    artifact_key = _validate_artifact_key(artifact_key)
    key = f"{S3_PREFIX}{artifact_key}/{filename}"
    if not key.startswith(S3_PREFIX):  # Defensive guard if validation changes later.
        raise ValueError(f"Refusing key outside approved prefix: {key}")
    return key


def _get_s3_client() -> Any:
    try:
        boto3 = importlib.import_module("boto3")
    except ImportError as exc:
        raise RuntimeError("boto3 is required on the Volt pod for S3 transfer") from exc
    return boto3.client("s3", region_name=AWS_REGION)


def _validate_checkpoint_source(checkpoint_dir: pathlib.Path) -> dict[str, Any]:
    if not checkpoint_dir.is_dir():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    for relative_path in REQUIRED_PATHS:
        path = checkpoint_dir / relative_path
        if not path.exists():
            raise ValueError(f"Checkpoint is missing required path: {path}")
        if path.is_symlink():
            raise ValueError(f"Checkpoint required path must not be a symlink: {path}")
    if not (checkpoint_dir / "params").is_dir() or not (checkpoint_dir / "assets").is_dir():
        raise ValueError("Checkpoint params and assets must be directories")
    if not (checkpoint_dir / "_CHECKPOINT_METADATA").is_file():
        raise ValueError("Checkpoint _CHECKPOINT_METADATA must be a file")
    try:
        checkpoint_metadata = json.loads((checkpoint_dir / "_CHECKPOINT_METADATA").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Checkpoint _CHECKPOINT_METADATA must be valid JSON") from error
    if not isinstance(checkpoint_metadata, dict):
        raise ValueError("Checkpoint _CHECKPOINT_METADATA must contain a JSON object")
    for relative_path in REQUIRED_PATHS:
        path = checkpoint_dir / relative_path
        paths = path.rglob("*") if path.is_dir() else ()
        for child in paths:
            if child.is_symlink():
                raise ValueError(f"Checkpoint bundle must not contain symlinks: {child}")
    return checkpoint_metadata


def _create_archive(checkpoint_dir: pathlib.Path, archive_path: pathlib.Path) -> None:
    _validate_checkpoint_source(checkpoint_dir)
    with tarfile.open(archive_path, "w") as archive:
        for relative_path in REQUIRED_PATHS:
            archive.add(checkpoint_dir / relative_path, arcname=relative_path, recursive=True)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_size(checkpoint_dir: pathlib.Path) -> int:
    total = 0
    for relative_path in REQUIRED_PATHS:
        path = checkpoint_dir / relative_path
        if path.is_file():
            total += path.stat().st_size
        else:
            total += sum(child.stat().st_size for child in path.rglob("*") if child.is_file())
    return total


def _object_exists(s3: Any, key: str) -> bool:
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
    except Exception as exc:
        response = getattr(exc, "response", {})
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        code = str(response.get("Error", {}).get("Code", ""))
        if status == 404 or code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise
    return True


def _assert_remote_absent(s3: Any, artifact_key: str) -> None:
    existing = [
        name
        for name in (ARCHIVE_NAME, MANIFEST_NAME, SUCCESS_NAME)
        if _object_exists(s3, _key(artifact_key, name))
    ]
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite s3://{S3_BUCKET}/{S3_PREFIX}{artifact_key}/; found {', '.join(existing)}"
        )


def _put_object_if_absent(s3: Any, key: str, body: bytes) -> None:
    try:
        s3.put_object(Bucket=S3_BUCKET, Key=key, Body=body, IfNoneMatch="*")
    except Exception as error:
        response = getattr(error, "response", {})
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        code = str(response.get("Error", {}).get("Code", ""))
        if status == 412 or code in {"412", "PreconditionFailed"}:
            raise FileExistsError(f"Refusing to overwrite s3://{S3_BUCKET}/{key}") from error
        raise


def upload(
    checkpoint_dir: pathlib.Path,
    artifact_key: str,
    metadata: dict[str, Any],
    *,
    scratch_dir: pathlib.Path | None = None,
    s3: Any | None = None,
) -> dict[str, Any]:
    """Upload an immutable evaluation-only checkpoint bundle and return its manifest."""
    _require_volt()
    checkpoint_dir = checkpoint_dir.resolve()
    artifact_key = _validate_artifact_key(artifact_key)
    _validate_metadata(artifact_key, metadata)
    checkpoint_metadata = _validate_checkpoint_source(checkpoint_dir)
    if checkpoint_dir.name != metadata["checkpoint_label"]:
        raise ValueError("Checkpoint directory name must match checkpoint_label")
    # Some producers include a logical step in Orbax metadata while current OpenPI does not.
    # When present, it is authoritative and must agree with the declared completed step.
    if "step" in checkpoint_metadata:
        metadata_step = checkpoint_metadata["step"]
        if isinstance(metadata_step, bool) or not isinstance(metadata_step, int):
            raise ValueError("Checkpoint metadata step must be an integer")
        if metadata_step != metadata["train_steps_completed"]:
            raise ValueError("Checkpoint metadata step does not match train_steps_completed")
    s3 = s3 or _get_s3_client()
    scratch_dir = (scratch_dir or pathlib.Path("/volt/data/openpi_s3_staging")).resolve()
    scratch_dir.mkdir(parents=True, exist_ok=True)
    required_scratch_bytes = _bundle_size(checkpoint_dir) + 1024**3
    if shutil.disk_usage(scratch_dir).free < required_scratch_bytes:
        raise OSError(f"Insufficient scratch space in {scratch_dir}; need {required_scratch_bytes} bytes")

    with tempfile.TemporaryDirectory(prefix="openpi-checkpoint-package-", dir=scratch_dir) as temp_dir:
        archive_path = pathlib.Path(temp_dir) / ARCHIVE_NAME
        _create_archive(checkpoint_dir, archive_path)
        archive_sha256 = _sha256(archive_path)
        published_key = f"{artifact_key}/sha256_{archive_sha256}"
        _assert_remote_absent(s3, published_key)
        manifest = {
            **metadata,
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "artifact_key": published_key,
            "logical_artifact_key": artifact_key,
            "artifact_id": f"sha256:{archive_sha256}",
            "archive": ARCHIVE_NAME,
            "archive_sha256": archive_sha256,
            "archive_size_bytes": archive_path.stat().st_size,
            "required_paths": list(REQUIRED_PATHS),
            "excluded_paths": ["train_state"],
            "bucket": S3_BUCKET,
            "prefix": f"{S3_PREFIX}{published_key}/",
            "checkpoint_s3_uri": f"s3://{S3_BUCKET}/{S3_PREFIX}{published_key}/",
            "source_checkpoint_path": str(checkpoint_dir),
            "region": AWS_REGION,
            "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()

        # The completion marker is deliberately last: readers never consume a partial publication.
        archive_key = _key(published_key, ARCHIVE_NAME)
        s3.upload_file(str(archive_path), S3_BUCKET, archive_key)
        if s3.head_object(Bucket=S3_BUCKET, Key=archive_key).get("ContentLength") != manifest["archive_size_bytes"]:
            raise ValueError("Uploaded archive size does not match local archive")
        _put_object_if_absent(s3, _key(published_key, MANIFEST_NAME), manifest_bytes)
        _put_object_if_absent(s3, _key(published_key, SUCCESS_NAME), b"")
    return manifest


def _load_manifest(s3: Any, artifact_key: str) -> dict[str, Any]:
    body = s3.get_object(Bucket=S3_BUCKET, Key=_key(artifact_key, MANIFEST_NAME))["Body"].read()
    manifest = json.loads(body)
    expected = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_key": artifact_key,
        "archive": ARCHIVE_NAME,
        "bucket": S3_BUCKET,
        "prefix": f"{S3_PREFIX}{artifact_key}/",
        "region": AWS_REGION,
        "required_paths": list(REQUIRED_PATHS),
        "excluded_paths": ["train_state"],
        "checkpoint_s3_uri": f"s3://{S3_BUCKET}/{S3_PREFIX}{artifact_key}/",
    }
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatches:
        raise ValueError(f"Remote manifest violates checkpoint contract: {', '.join(mismatches)}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("archive_sha256", ""))):
        raise ValueError("Remote manifest has an invalid archive_sha256")
    if manifest.get("artifact_id") != f"sha256:{manifest['archive_sha256']}":
        raise ValueError("Remote manifest artifact_id does not match archive SHA256")
    if not isinstance(manifest.get("archive_size_bytes"), int) or manifest["archive_size_bytes"] < 1:
        raise ValueError("Remote manifest has an invalid archive_size_bytes")
    logical_artifact_key = manifest.get("logical_artifact_key")
    if not isinstance(logical_artifact_key, str):
        raise ValueError("Remote manifest is missing logical_artifact_key")
    if artifact_key != f"{logical_artifact_key}/sha256_{manifest['archive_sha256']}":
        raise ValueError("Remote content-addressed artifact key does not match archive_sha256")
    _validate_metadata(logical_artifact_key, manifest)
    return manifest


def _validate_archive_members(archive: tarfile.TarFile) -> None:
    roots: set[str] = set()
    seen: set[pathlib.PurePosixPath] = set()
    for member in archive.getmembers():
        member_path = pathlib.PurePosixPath(member.name)
        if member_path.is_absolute() or ".." in member_path.parts or not member_path.parts:
            raise ValueError(f"Unsafe archive member: {member.name}")
        if member.issym() or member.islnk() or member.isdev():
            raise ValueError(f"Unsupported archive member type: {member.name}")
        if member_path in seen:
            raise ValueError(f"Duplicate archive member: {member.name}")
        seen.add(member_path)
        root = member_path.parts[0]
        if root not in REQUIRED_PATHS:
            raise ValueError(f"Unexpected top-level archive path: {member.name}")
        roots.add(root)
    missing = set(REQUIRED_PATHS) - roots
    if missing:
        raise ValueError(f"Archive is missing required paths: {', '.join(sorted(missing))}")


def download(destination: pathlib.Path, artifact_key: str, *, s3: Any | None = None) -> dict[str, Any]:
    """Download, verify, and atomically install an evaluation-only checkpoint bundle."""
    _require_volt()
    artifact_key = _validate_artifact_key(artifact_key)
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite destination: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    s3 = s3 or _get_s3_client()
    if not _object_exists(s3, _key(artifact_key, SUCCESS_NAME)):
        raise FileNotFoundError(f"Checkpoint is not complete: missing {SUCCESS_NAME}")
    manifest = _load_manifest(s3, artifact_key)

    archive_head = s3.head_object(Bucket=S3_BUCKET, Key=_key(artifact_key, ARCHIVE_NAME))
    if archive_head.get("ContentLength") != manifest["archive_size_bytes"]:
        raise ValueError("Remote archive size does not match manifest")

    temp_root = pathlib.Path(tempfile.mkdtemp(prefix=f".{destination.name}.download-", dir=destination.parent))
    try:
        archive_path = temp_root / ARCHIVE_NAME
        extracted = temp_root / "checkpoint"
        extracted.mkdir()
        s3.download_file(S3_BUCKET, _key(artifact_key, ARCHIVE_NAME), str(archive_path))
        if archive_path.stat().st_size != manifest["archive_size_bytes"]:
            raise ValueError("Downloaded archive size does not match manifest")
        if _sha256(archive_path) != manifest["archive_sha256"]:
            raise ValueError("Downloaded archive SHA256 does not match manifest")
        with tarfile.open(archive_path, "r") as archive:
            _validate_archive_members(archive)
            archive.extractall(extracted)  # noqa: S202 -- every member was validated above.
        _validate_checkpoint_source(extracted)
        local_manifest = {**manifest, "checkpoint_path": str(destination)}
        (extracted / "artifact_manifest.json").write_text(
            json.dumps(local_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(extracted, destination)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    upload_parser = subparsers.add_parser("upload", help="Package and upload an evaluation checkpoint")
    upload_parser.add_argument("--checkpoint-dir", type=pathlib.Path, required=True)
    upload_parser.add_argument("--artifact-key", required=True)
    upload_parser.add_argument("--artifact-name", required=True)
    upload_parser.add_argument("--model-class", choices=("tvm", "finetune"), required=True)
    upload_parser.add_argument("--training-run-id", required=True)
    upload_parser.add_argument("--checkpoint-label", required=True)
    upload_parser.add_argument("--train-steps-completed", type=int, required=True)
    upload_parser.add_argument("--config", required=True)
    upload_parser.add_argument("--git-sha", required=True)
    upload_parser.add_argument("--scratch-dir", type=pathlib.Path, default=pathlib.Path("/volt/data/openpi_s3_staging"))
    download_parser = subparsers.add_parser("download", help="Download and atomically install a checkpoint")
    download_parser.add_argument("--destination", type=pathlib.Path, required=True)
    download_parser.add_argument("--artifact-key", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "upload":
        metadata = {
            "artifact_name": args.artifact_name,
            "model_class": args.model_class,
            "training_run_id": args.training_run_id,
            "checkpoint_name": args.checkpoint_label,
            "checkpoint_label": args.checkpoint_label,
            "train_steps_completed": args.train_steps_completed,
            "config": args.config,
            "git_sha": args.git_sha,
        }
        manifest = upload(args.checkpoint_dir, args.artifact_key, metadata, scratch_dir=args.scratch_dir)
    else:
        manifest = download(args.destination, args.artifact_key)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
