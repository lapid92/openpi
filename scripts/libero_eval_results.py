"""Validate LIBERO episode JSONL and write one audit-friendly summary CSV row.

The input format is intentionally explicit: every JSON object is one episode and
contains the complete evaluation identity/protocol metadata.  This makes files
from separate suite clients independently attributable and lets this module
reject accidental mixtures before publishing a summary.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import pathlib
import re
import tempfile
from collections.abc import Iterable, Sequence
from typing import Any


SCHEMA_VERSION = 1
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
TASKS_PER_SUITE = 10
TRIALS_PER_TASK = 50
EPISODES_PER_SUITE = TASKS_PER_SUITE * TRIALS_PER_TASK
EPISODES_PER_EVAL = len(SUITES) * EPISODES_PER_SUITE

IDENTITY_FIELDS = (
    "eval_id",
    "artifact_id",
    "artifact_name",
    "checkpoint_name",
    "checkpoint_path",
    "checkpoint_s3_uri",
    "model_class",
    "training_run_id",
    "train_steps_completed",
    "checkpoint_label",
    "config",
    "git_sha",
    "flow_steps",
    "eval_seed",
    "replan_steps",
    "trials_per_task",
    "status",
)

EPISODE_FIELDS = (
    "schema_version",
    *IDENTITY_FIELDS,
    "suite",
    "task_id",
    "episode_idx",
    "success",
    "exception",
    "started_at_utc",
    "finished_at_utc",
    "client_log_path",
)

SUMMARY_FIELDS = (
    "schema_version",
    *IDENTITY_FIELDS[:-1],
    "episodes_spatial",
    "successes_spatial",
    "episodes_object",
    "successes_object",
    "episodes_goal",
    "successes_goal",
    "episodes_libero10",
    "successes_libero10",
    "episodes_total",
    "successes_total",
    "failures_total",
    "success_rate",
    "eval_started_at_utc",
    "eval_finished_at_utc",
    "client_log_spatial",
    "client_log_object",
    "client_log_goal",
    "client_log_libero10",
    "status",
)

_INTEGER_FIELDS = ("train_steps_completed", "flow_steps", "eval_seed", "replan_steps", "trials_per_task")
_NONEMPTY_STRING_FIELDS = (
    "eval_id",
    "artifact_id",
    "artifact_name",
    "checkpoint_name",
    "checkpoint_path",
    "model_class",
    "training_run_id",
    "checkpoint_label",
    "config",
    "git_sha",
    "client_log_path",
    "status",
)
_GIT_SHA_RE = re.compile(r"[0-9a-fA-F]{40}\Z")


class ValidationError(ValueError):
    """Raised when episode results cannot form a trustworthy complete result."""


def _parse_timestamp(value: Any, field: str, source: str) -> dt.datetime:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{source}: {field} must be a non-empty RFC3339 timestamp")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = dt.datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValidationError(f"{source}: invalid {field} RFC3339 timestamp {value!r}") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValidationError(f"{source}: {field} must include a UTC offset")
    return parsed.astimezone(dt.timezone.utc)


def _format_timestamp(value: dt.datetime) -> str:
    value = value.astimezone(dt.timezone.utc)
    timespec = "microseconds" if value.microsecond else "seconds"
    return value.isoformat(timespec=timespec).replace("+00:00", "Z")


def _require_int(record: dict[str, Any], field: str, source: str, *, minimum: int | None = None) -> int:
    value = record[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{source}: {field} must be an integer")
    if minimum is not None and value < minimum:
        raise ValidationError(f"{source}: {field} must be >= {minimum}")
    return value


def _validate_episode(record: Any, source: str) -> tuple[dt.datetime, dt.datetime]:
    if not isinstance(record, dict):
        raise ValidationError(f"{source}: episode must be a JSON object")
    missing = [field for field in EPISODE_FIELDS if field not in record]
    if missing:
        raise ValidationError(f"{source}: missing required fields: {', '.join(missing)}")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValidationError(f"{source}: unsupported schema_version {record['schema_version']!r}")

    for field in _NONEMPTY_STRING_FIELDS:
        if not isinstance(record[field], str) or not record[field]:
            raise ValidationError(f"{source}: {field} must be a non-empty string")
    if record["checkpoint_s3_uri"] is not None and not isinstance(record["checkpoint_s3_uri"], str):
        raise ValidationError(f"{source}: checkpoint_s3_uri must be a string or null")
    if record["checkpoint_s3_uri"] and not record["checkpoint_s3_uri"].startswith("s3://"):
        raise ValidationError(f"{source}: non-empty checkpoint_s3_uri must start with s3://")
    if not _GIT_SHA_RE.fullmatch(record["git_sha"]):
        raise ValidationError(f"{source}: git_sha must be a full 40-character hexadecimal SHA")

    _require_int(record, "train_steps_completed", source, minimum=0)
    _require_int(record, "flow_steps", source, minimum=1)
    _require_int(record, "eval_seed", source)
    _require_int(record, "replan_steps", source, minimum=1)
    trials = _require_int(record, "trials_per_task", source, minimum=1)
    if trials != TRIALS_PER_TASK:
        raise ValidationError(f"{source}: trials_per_task must be {TRIALS_PER_TASK}, got {trials}")

    if record["suite"] not in SUITES:
        raise ValidationError(f"{source}: unknown suite {record['suite']!r}")
    task_id = _require_int(record, "task_id", source)
    episode_idx = _require_int(record, "episode_idx", source)
    if task_id not in range(TASKS_PER_SUITE):
        raise ValidationError(f"{source}: task_id must be in [0, {TASKS_PER_SUITE - 1}]")
    if episode_idx not in range(TRIALS_PER_TASK):
        raise ValidationError(f"{source}: episode_idx must be in [0, {TRIALS_PER_TASK - 1}]")
    if type(record["success"]) is not bool:
        raise ValidationError(f"{source}: success must be a JSON boolean")
    if record["status"] != "complete":
        raise ValidationError(f"{source}: only status='complete' can be summarized")
    if record.get("exception") not in (None, ""):
        raise ValidationError(f"{source}: complete episode contains exception {record['exception']!r}")

    started = _parse_timestamp(record["started_at_utc"], "started_at_utc", source)
    finished = _parse_timestamp(record["finished_at_utc"], "finished_at_utc", source)
    if finished < started:
        raise ValidationError(f"{source}: finished_at_utc precedes started_at_utc")
    return started, finished


def read_episode_jsonl(paths: Sequence[str | os.PathLike[str]]) -> list[dict[str, Any]]:
    """Read JSONL paths and return validated episode objects."""
    if not paths:
        raise ValidationError("at least one input JSONL path is required")
    records: list[dict[str, Any]] = []
    for path_value in paths:
        path = pathlib.Path(path_value)
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                source = f"{path}:{line_number}"
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValidationError(f"{source}: invalid JSON: {error.msg}") from error
                _validate_episode(record, source)
                records.append(record)
    if not records:
        raise ValidationError("input JSONL contains no episode records")
    return records


def aggregate_records(records: Iterable[dict[str, Any]]) -> dict[str, str | int]:
    """Validate a complete evaluation and return a deterministic summary row."""
    records = list(records)
    if not records:
        raise ValidationError("no episode records supplied")

    reference = records[0]
    for index, record in enumerate(records, start=1):
        _validate_episode(record, f"record {index}")
        inconsistent = [field for field in IDENTITY_FIELDS if record[field] != reference[field]]
        if inconsistent:
            raise ValidationError(
                f"record {index}: inconsistent evaluation metadata: {', '.join(inconsistent)}"
            )

    seen: set[tuple[str, int, int]] = set()
    suite_successes = dict.fromkeys(SUITES, 0)
    suite_counts = dict.fromkeys(SUITES, 0)
    suite_logs: dict[str, set[str]] = {suite: set() for suite in SUITES}
    starts: list[dt.datetime] = []
    finishes: list[dt.datetime] = []
    for index, record in enumerate(records, start=1):
        key = (record["suite"], record["task_id"], record["episode_idx"])
        if key in seen:
            raise ValidationError(f"record {index}: duplicate episode key {key!r}")
        seen.add(key)
        suite = record["suite"]
        suite_counts[suite] += 1
        suite_successes[suite] += int(record["success"])
        suite_logs[suite].add(record["client_log_path"])
        starts.append(_parse_timestamp(record["started_at_utc"], "started_at_utc", f"record {index}"))
        finishes.append(_parse_timestamp(record["finished_at_utc"], "finished_at_utc", f"record {index}"))

    if len(records) != EPISODES_PER_EVAL:
        raise ValidationError(f"incomplete evaluation: expected {EPISODES_PER_EVAL} episodes, got {len(records)}")
    expected_keys = {
        (suite, task_id, episode_idx)
        for suite in SUITES
        for task_id in range(TASKS_PER_SUITE)
        for episode_idx in range(TRIALS_PER_TASK)
    }
    if seen != expected_keys:
        missing = sorted(expected_keys - seen)
        extra = sorted(seen - expected_keys)
        raise ValidationError(f"episode grid mismatch: missing={missing[:5]!r}, extra={extra[:5]!r}")
    for suite, logs in suite_logs.items():
        if len(logs) != 1:
            raise ValidationError(f"suite {suite} has inconsistent client_log_path values: {sorted(logs)!r}")

    successes_total = sum(suite_successes.values())
    summary: dict[str, str | int] = {
        "schema_version": SCHEMA_VERSION,
        **{field: reference[field] for field in IDENTITY_FIELDS[:-1]},
        "episodes_spatial": suite_counts["libero_spatial"],
        "successes_spatial": suite_successes["libero_spatial"],
        "episodes_object": suite_counts["libero_object"],
        "successes_object": suite_successes["libero_object"],
        "episodes_goal": suite_counts["libero_goal"],
        "successes_goal": suite_successes["libero_goal"],
        "episodes_libero10": suite_counts["libero_10"],
        "successes_libero10": suite_successes["libero_10"],
        "episodes_total": len(records),
        "successes_total": successes_total,
        "failures_total": len(records) - successes_total,
        "success_rate": f"{successes_total / len(records):.6f}",
        "eval_started_at_utc": _format_timestamp(min(starts)),
        "eval_finished_at_utc": _format_timestamp(max(finishes)),
        "client_log_spatial": next(iter(suite_logs["libero_spatial"])),
        "client_log_object": next(iter(suite_logs["libero_object"])),
        "client_log_goal": next(iter(suite_logs["libero_goal"])),
        "client_log_libero10": next(iter(suite_logs["libero_10"])),
        "status": "complete",
    }
    return summary


def write_summary_csv(
    summary: dict[str, Any], output_path: str | os.PathLike[str], *, replace: bool = False
) -> pathlib.Path:
    """Atomically publish a versioned RFC4180 CSV containing one summary row."""
    output = pathlib.Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not replace:
        raise FileExistsError(f"refusing to replace existing summary: {output}")

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=output.parent, prefix=f".{output.name}.", delete=False
        ) as stream:
            temporary_name = stream.name
            writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS, dialect="excel", extrasaction="raise")
            writer.writeheader()
            writer.writerow(summary)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary_name, output)
        else:
            # A hard link publishes the fully flushed file atomically and, unlike
            # a check followed by replace, cannot race another publisher into
            # overwriting an existing audit result.
            os.link(temporary_name, output)
            os.unlink(temporary_name)
        temporary_name = None
    finally:
        if temporary_name is not None:
            pathlib.Path(temporary_name).unlink(missing_ok=True)
    return output


def aggregate_jsonl_to_csv(
    input_paths: Sequence[str | os.PathLike[str]],
    output_path: str | os.PathLike[str],
    *,
    replace: bool = False,
) -> pathlib.Path:
    return write_summary_csv(aggregate_records(read_episode_jsonl(input_paths)), output_path, replace=replace)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Episode JSONL files from LIBERO clients")
    parser.add_argument("--output", required=True, help="Destination summary CSV")
    parser.add_argument("--replace", action="store_true", help="Atomically replace an existing output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        output = aggregate_jsonl_to_csv(args.inputs, args.output, replace=args.replace)
    except (OSError, ValidationError) as error:
        raise SystemExit(f"error: {error}") from error
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
