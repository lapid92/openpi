"""Mirror committed Orbax steps from a training pod to S3 with completion manifests."""

import argparse
import json
from pathlib import Path
import time
from urllib.parse import urlparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("s3_uri", help="Destination, for example s3://my-bucket/experiments/run1")
    parser.add_argument("--watch", action="store_true", help="Keep mirroring new committed steps")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--until-step", type=int, help="Stop after this completed step is mirrored")
    args = parser.parse_args()
    if not Path("/volt").is_dir():
        parser.error("Run this mirror on the authorized training pod, not on a local computer")
    if args.poll_seconds < 5:
        parser.error("--poll-seconds must be at least 5")
    parsed = urlparse(args.s3_uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        parser.error("s3_uri must include a bucket and destination prefix")

    import boto3
    from botocore.exceptions import ClientError

    s3 = boto3.client("s3")
    bucket = parsed.netloc
    prefix = parsed.path.strip("/").rstrip("/")

    def exists(key: str) -> bool:
        try:
            s3.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def mirror_file(path: Path, key: str) -> dict:
        size = path.stat().st_size
        s3.upload_file(str(path), bucket, key)
        remote_size = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
        if size != remote_size:
            raise IOError(f"Size mismatch after upload: {path}")
        return {"path": str(path.relative_to(args.run_dir)), "bytes": size}

    def mirror_step(step_dir: Path) -> None:
        step = int(step_dir.name)
        marker = f"{prefix}/steps/{step}/manifest.json"
        if exists(marker):
            return
        files = sorted(path for path in step_dir.rglob("*") if path.is_file())
        if not files:
            raise RuntimeError(f"Committed checkpoint has no files: {step_dir}")
        records = [mirror_file(path, f"{prefix}/steps/{step}/{path.relative_to(step_dir)}") for path in files]
        body = json.dumps({"step": step, "files": records, "source": str(step_dir)}, indent=2).encode()
        s3.put_object(Bucket=bucket, Key=marker, Body=body, ContentType="application/json")
        print(f"Mirrored committed step {step}: {len(files)} files", flush=True)

    while True:
        provenance = args.run_dir / "provenance.json"
        if provenance.is_file() and not exists(f"{prefix}/provenance.json"):
            mirror_file(provenance, f"{prefix}/provenance.json")
        step_dirs = (
            sorted(
                (path for path in args.run_dir.iterdir() if path.is_dir() and path.name.isdecimal()),
                key=lambda path: int(path.name),
            )
            if args.run_dir.exists()
            else ()
        )
        for step_dir in step_dirs:
            # Orbax writes this after finalizing a checkpoint. Temporary directories are ignored.
            if (step_dir / "_CHECKPOINT_METADATA").is_file():
                mirror_step(step_dir)
        if not args.watch or (args.until_step is not None and exists(f"{prefix}/steps/{args.until_step}/manifest.json")):
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
