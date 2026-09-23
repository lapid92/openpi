"""Recover an evaluation checkpoint directly from retained Volt job artifacts.

The checkpoint bytes travel from Volt's artifact store to the Volt pod.  A
short-lived API token is read from stdin so it is never written to disk or
included in the process arguments.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import re
import shutil
import sys
import tempfile
import urllib.request
from typing import Any


VOLT_API_URL = "https://api.volt.arm.com/job.JobService/GetArtifacts"
REQUIRED_ROOTS = ("params", "assets")
REQUIRED_FILE = "_CHECKPOINT_METADATA"
_JOB_ID_RE = re.compile(r"[a-z0-9]{12}\Z")
_PREFIX_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,511}\Z")


def _request_json(request: urllib.request.Request) -> dict[str, Any]:
    with urllib.request.urlopen(request, timeout=60) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError("Volt API returned a non-object response")
    return value


def list_artifacts(job_id: str, source_prefix: str, token: str) -> list[dict[str, Any]]:
    if not _JOB_ID_RE.fullmatch(job_id):
        raise ValueError("Invalid Volt job ID")
    source_prefix = source_prefix.strip("/")
    if not _PREFIX_RE.fullmatch(source_prefix) or ".." in pathlib.PurePosixPath(source_prefix).parts:
        raise ValueError("Invalid artifact prefix")
    body = json.dumps({"jobId": job_id, "pageSize": 1000, "prefix": source_prefix}).encode()
    request = urllib.request.Request(
        VOLT_API_URL,
        data=body,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    response = _request_json(request)
    if response.get("nextPageToken"):
        raise ValueError("Artifact response was unexpectedly paginated")
    artifacts = response.get("artifacts", [])
    if not isinstance(artifacts, list) or not artifacts:
        raise FileNotFoundError(f"No retained artifacts found for prefix {source_prefix}")
    return artifacts


def _relative_path(asset_uri: str, source_prefix: str) -> pathlib.PurePosixPath:
    marker = f"/{source_prefix.strip('/')}/"
    if marker not in asset_uri:
        raise ValueError(f"Artifact URI is outside requested prefix: {asset_uri}")
    relative = pathlib.PurePosixPath(asset_uri.split(marker, 1)[1])
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"Unsafe artifact path: {relative}")
    return relative


def _is_evaluation_file(relative: pathlib.PurePosixPath) -> bool:
    return relative.as_posix() == REQUIRED_FILE or relative.parts[0] in REQUIRED_ROOTS


def _headers(artifact: dict[str, Any]) -> dict[str, str]:
    raw = artifact.get("presignedHeaders", {}).get("headers", {})
    return {name: ",".join(value.get("values", [])) for name, value in raw.items()}


def _download_one(artifact: dict[str, Any], destination: pathlib.Path, source_prefix: str) -> tuple[str, int]:
    relative = _relative_path(str(artifact["assetUri"]), source_prefix)
    target = destination.joinpath(*relative.parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(str(artifact["presignedUrl"]), headers=_headers(artifact))
    temporary = target.with_name(f".{target.name}.partial")
    try:
        with urllib.request.urlopen(request, timeout=300) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return relative.as_posix(), target.stat().st_size


def recover(
    job_id: str,
    source_prefix: str,
    destination: pathlib.Path,
    token: str,
    *,
    workers: int = 8,
) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite destination: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    artifacts = list_artifacts(job_id, source_prefix, token)
    selected: list[dict[str, Any]] = []
    seen: set[pathlib.PurePosixPath] = set()
    for artifact in artifacts:
        relative = _relative_path(str(artifact.get("assetUri", "")), source_prefix)
        if not _is_evaluation_file(relative):
            continue
        if relative in seen:
            raise ValueError(f"Duplicate artifact path: {relative}")
        seen.add(relative)
        if not artifact.get("presignedUrl"):
            raise ValueError(f"Artifact has no presigned URL: {relative}")
        selected.append(artifact)

    temporary_root = pathlib.Path(tempfile.mkdtemp(prefix=f".{destination.name}.recover-", dir=destination.parent))
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_download_one, item, temporary_root, source_prefix) for item in selected]
            downloaded = [future.result() for future in concurrent.futures.as_completed(futures)]
        for root in REQUIRED_ROOTS:
            if not (temporary_root / root).is_dir():
                raise ValueError(f"Recovered checkpoint is missing {root}")
        metadata_path = temporary_root / REQUIRED_FILE
        if not metadata_path.is_file():
            raise ValueError(f"Recovered checkpoint is missing {REQUIRED_FILE}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("Recovered checkpoint metadata is not a JSON object")
        os.replace(temporary_root, destination)
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    return {
        "job_id": job_id,
        "source_prefix": source_prefix,
        "destination": str(destination),
        "retained_artifact_count": len(artifacts),
        "downloaded_file_count": len(downloaded),
        "downloaded_bytes": sum(size for _, size in downloaded),
        "excluded_roots": ["train_state"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--source-prefix", required=True)
    parser.add_argument("--destination", type=pathlib.Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    token_source = parser.add_mutually_exclusive_group(required=True)
    token_source.add_argument("--token-stdin", action="store_true")
    token_source.add_argument("--token-env", metavar="NAME")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not 1 <= args.workers <= 32:
        raise ValueError("workers must be between 1 and 32")
    token = os.environ.pop(args.token_env, "").strip() if args.token_env else sys.stdin.readline().strip()
    if not token:
        raise ValueError("A Volt API token is required on stdin")
    result = recover(args.job_id, args.source_prefix, args.destination, token, workers=args.workers)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
