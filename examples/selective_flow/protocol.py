# ruff: noqa: C408
"""Predeclare a frozen fixed-step study without reading any evaluation outcomes."""

import hashlib
import json
import pathlib

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
