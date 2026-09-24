"""Stage and verify public Pi0.5 GCS checkpoint and LIBERO normalization assets."""

import base64
import concurrent.futures
import hashlib
import pathlib
import urllib.parse

import requests

BUCKET = "openpi-assets"
PREFIXES = ("checkpoints/pi05_base/params/", "checkpoints/pi05_libero/assets/")
DEST = pathlib.Path("/volt/data/openpi/openpi-assets")


def list_objects(prefix: str) -> list[dict]:
    objects = []
    token = None
    while True:
        params = {"prefix": prefix, "maxResults": 1000}
        if token:
            params["pageToken"] = token
        response = requests.get(f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o", params=params, timeout=60)
        response.raise_for_status()
        page = response.json()
        objects.extend(page.get("items", []))
        token = page.get("nextPageToken")
        if not token:
            return objects


def stage_one(obj: dict) -> tuple[str, int]:
    name = obj["name"]
    expected_size = int(obj["size"])
    expected_md5 = base64.b64decode(obj["md5Hash"])
    destination = DEST / name
    destination.parent.mkdir(parents=True, exist_ok=True)

    def valid(path: pathlib.Path) -> bool:
        if not path.is_file() or path.stat().st_size != expected_size:
            return False
        digest = hashlib.md5()  # noqa: S324 - GCS provides MD5 for transfer integrity.
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        return digest.digest() == expected_md5

    if not valid(destination):
        temporary = destination.with_name(destination.name + ".partial")
        url = f"https://storage.googleapis.com/{BUCKET}/{urllib.parse.quote(name)}"
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for block in response.iter_content(chunk_size=8 * 1024 * 1024):
                    if block:
                        output.write(block)
        if not valid(temporary):
            raise ValueError(f"GCS size/MD5 mismatch: {name}")
        temporary.replace(destination)
    return name, expected_size


def main() -> None:
    objects = [obj for prefix in PREFIXES for obj in list_objects(prefix)]
    print(f"Staging {len(objects)} GCS objects, {sum(int(obj['size']) for obj in objects)} bytes", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for name, size in pool.map(stage_one, objects):
            print(f"verified {name} {size}", flush=True)


if __name__ == "__main__":
    main()
