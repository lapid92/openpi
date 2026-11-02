"""Stage the public pi05_base Orbax params on a training pod, with size checks."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path

import gcsfs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    source = "openpi-assets/checkpoints/pi05_base/params"
    fs = gcsfs.GCSFileSystem(token="anon")
    files = fs.find(source)
    if not files or not any(path.endswith("/_METADATA") for path in files):
        raise RuntimeError("Public pi05_base checkpoint metadata is missing")
    print(f"Staging {len(files)} public checkpoint files", flush=True)

    def copy(path: str) -> str:
        relative = path.removeprefix(source + "/")
        target = args.destination / relative
        expected = fs.info(path)["size"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == expected:
            return f"cached {relative}"
        partial = target.with_name(target.name + ".partial")
        fs.get_file(path, str(partial))
        if partial.stat().st_size != expected:
            raise IOError(f"Size mismatch: {relative}")
        os.replace(partial, target)
        return f"downloaded {relative}"

    with ThreadPoolExecutor(max_workers=8) as pool:
        for result in as_completed([pool.submit(copy, path) for path in files]):
            print(result.result(), flush=True)
    print("pi05_base staging complete", flush=True)


if __name__ == "__main__":
    main()
