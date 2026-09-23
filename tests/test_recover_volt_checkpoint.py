from __future__ import annotations

import pathlib

import pytest

from scripts import recover_volt_checkpoint as recovery


PREFIX = "openpi_checkpoints/run/29999"


def test_relative_path_accepts_requested_prefix() -> None:
    uri = f"s3://private/jobs/id/artifacts/workers/0/{PREFIX}/params/_METADATA"
    assert recovery._relative_path(uri, PREFIX) == pathlib.PurePosixPath("params/_METADATA")


@pytest.mark.parametrize(
    "uri",
    [
        "s3://private/other/params/_METADATA",
        f"s3://private/jobs/id/{PREFIX}/../secret",
    ],
)
def test_relative_path_rejects_unsafe_or_unrelated_uri(uri: str) -> None:
    with pytest.raises(ValueError):
        recovery._relative_path(uri, PREFIX)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("_CHECKPOINT_METADATA", True),
        ("params/d/abc", True),
        ("assets/stats.json", True),
        ("train_state/d/abc", False),
    ],
)
def test_evaluation_file_filter(path: str, expected: bool) -> None:
    assert recovery._is_evaluation_file(pathlib.PurePosixPath(path)) is expected


def test_list_artifacts_validates_job_id_before_network() -> None:
    with pytest.raises(ValueError, match="job ID"):
        recovery.list_artifacts("not-valid", PREFIX, "token")
