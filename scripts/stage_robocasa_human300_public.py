"""Stage the 300 official RoboCasa pretrain-human task archives on a training pod."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
from urllib.parse import quote

import requests

os.environ.setdefault("MUJOCO_GL", "glx")

from robocasa.utils.dataset_registry import DATASET_SOUP_REGISTRY

REPO = "nvidia/PhysicalAI-Robotics-Manipulation-Kitchen-Demos"


def retry_request(method: str, url: str, *, stream: bool = False) -> requests.Response:
    for attempt in range(5):
        try:
            response = requests.request(method, url, stream=stream, timeout=(20, 180), allow_redirects=method != "HEAD")
            response.raise_for_status()
            return response
        except (requests.RequestException, TimeoutError):
            if attempt == 4:
                raise
            time.sleep(min(2**attempt, 16))
    raise AssertionError("unreachable")


def task_archive_paths() -> list[str]:
    paths = []
    for task in DATASET_SOUP_REGISTRY["pretrain_human300"]:
        parts = Path(task["path"]).parts
        relative = Path(*parts[parts.index("v1.0") + 1 :])
        paths.append(str(relative.parent / "lerobot.tar"))
    if len(paths) != 300 or len(set(paths)) != 300:
        raise ValueError("Expected exactly 300 distinct registered Human300 task archives")
    return paths


def archive_metadata(revision: str, path: str) -> dict:
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{quote(path)}"
    response = retry_request("HEAD", url)
    size = int(response.headers.get("X-Linked-Size", response.headers.get("Content-Length", "0")))
    sha = response.headers.get("X-Linked-ETag", response.headers.get("ETag", "")).strip('"')
    if size <= 0 or len(sha) != 64:
        raise ValueError(f"Missing LFS size or SHA-256 for {path}: {size}, {sha}")
    return {"path": path, "bytes": size, "sha256": sha, "url": url}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_archive(metadata: dict, dataset_root: Path, archive_root: Path) -> str:
    relative = Path(metadata["path"])
    destination = dataset_root / "v1.0" / relative.parent / "lerobot"
    marker = destination / ".human300_stage.json"
    if marker.is_file() and (destination / "meta" / "info.json").is_file():
        prior = json.loads(marker.read_text())
        if prior["sha256"] == metadata["sha256"]:
            return "already staged"

    archive = archive_root / relative
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.is_file() or archive.stat().st_size != metadata["bytes"] or sha256(archive) != metadata["sha256"]:
        partial = archive.with_suffix(".tar.part")
        digest = hashlib.sha256()
        total = 0
        with retry_request("GET", metadata["url"], stream=True) as response, partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    digest.update(chunk)
                    total += len(chunk)
        if total != metadata["bytes"] or digest.hexdigest() != metadata["sha256"]:
            raise ValueError(f"Downloaded archive failed size/SHA-256 verification: {relative}")
        partial.replace(archive)

    temporary = destination.parent / ".lerobot.extracting"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        with tarfile.open(archive) as handle:
            members = handle.getmembers()
            if any(
                member.issym()
                or member.islnk()
                or member.isdev()
                or Path(member.name).is_absolute()
                or ".." in Path(member.name).parts
                or Path(member.name).parts[0] != "lerobot"
                for member in members
            ):
                raise ValueError(f"Unsafe archive members: {relative}")
            handle.extractall(temporary)
        extracted = temporary / "lerobot"
        if not (extracted / "meta" / "info.json").is_file():
            raise ValueError(f"Missing meta/info.json after extraction: {relative}")
        if destination.exists():
            shutil.rmtree(destination)
        extracted.replace(destination)
        marker.write_text(json.dumps({key: metadata[key] for key in ("path", "bytes", "sha256")}) + "\n")
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return "downloaded and extracted"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path, help="Root that will contain v1.0/pretrain")
    parser.add_argument("archive_root", type=Path, help="Cache for verified source tar archives")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 16:
        parser.error("--workers must be between 1 and 16")

    repo_info = retry_request("GET", f"https://huggingface.co/api/datasets/{REPO}").json()
    revision = repo_info["sha"]
    paths = task_archive_paths()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        metadata = list(pool.map(lambda path: archive_metadata(revision, path), paths))
    total_bytes = sum(item["bytes"] for item in metadata)
    print(f"Verified {len(metadata)} Human300 archives at {revision}: {total_bytes} bytes ({total_bytes / 2**30:.2f} GiB)", flush=True)
    if args.preflight_only:
        return

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(stage_archive, item, args.dataset_root, args.archive_root): item for item in metadata}
        for number, future in enumerate(as_completed(futures), start=1):
            item = futures[future]
            print(f"{number}/300 {item['path']}: {future.result()}", flush=True)

    manifest = {"source_repo": REPO, "source_revision": revision, "archive_bytes": total_bytes, "tasks": metadata}
    manifest_path = args.dataset_root / "human300_stage_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Complete: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
