"""Run reproducible LIBERO evaluations for one checkpoint on one or more GPUs.

The runner starts one policy server per GPU, assigns flow-step settings to the
available workers, and evaluates the four standard LIBERO suites sequentially
against each server. Every episode is written to JSONL by examples/libero/main.py.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import dataclasses
import datetime
import fcntl
import hashlib
import json
import os
import pathlib
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request

try:
    from scripts import libero_eval_results
except ModuleNotFoundError:  # Direct execution via `python scripts/libero_eval_runner.py`.
    import libero_eval_results


DEFAULT_FLOWS = (1, 2, 3, 4, 5, 10)
DEFAULT_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


@dataclasses.dataclass(frozen=True)
class Artifact:
    artifact_id: str
    artifact_name: str
    model_class: str
    training_run_id: str
    checkpoint_label: str
    checkpoint_name: str
    checkpoint_path: str
    checkpoint_s3_uri: str
    train_steps_completed: int
    config: str
    git_sha: str

    @classmethod
    def from_json(cls, path: pathlib.Path) -> Artifact:
        payload = json.loads(path.read_text(encoding="utf-8"))
        required = {
            "artifact_id",
            "artifact_name",
            "model_class",
            "training_run_id",
            "checkpoint_label",
            "checkpoint_name",
            "checkpoint_path",
            "train_steps_completed",
            "config",
            "git_sha",
        }
        missing = sorted(required - payload.keys())
        if missing:
            raise ValueError(f"Artifact manifest is missing fields: {missing}")
        artifact = cls(
            artifact_id=str(payload["artifact_id"]),
            artifact_name=str(payload["artifact_name"]),
            model_class=str(payload["model_class"]),
            training_run_id=str(payload["training_run_id"]),
            checkpoint_label=str(payload["checkpoint_label"]),
            checkpoint_name=str(payload["checkpoint_name"]),
            checkpoint_path=str(payload["checkpoint_path"]),
            checkpoint_s3_uri=str(payload.get("checkpoint_s3_uri", "")),
            train_steps_completed=payload["train_steps_completed"],
            config=str(payload["config"]),
            git_sha=str(payload["git_sha"]),
        )
        artifact.validate()
        return artifact

    def validate(self) -> None:
        for field in (self.artifact_name, self.training_run_id, self.checkpoint_label, self.checkpoint_name):
            if not re.fullmatch(r"[A-Za-z0-9._-]{1,160}", field):
                raise ValueError(f"Unsafe artifact path component: {field!r}")
        if self.model_class not in {"tvm", "finetune", "official"}:
            raise ValueError("model_class must be tvm, finetune, or official")
        if isinstance(self.train_steps_completed, bool) or not isinstance(self.train_steps_completed, int):
            raise ValueError("train_steps_completed must be an integer")
        if self.train_steps_completed < 0:
            raise ValueError("train_steps_completed must be nonnegative")
        if not re.fullmatch(r"[0-9a-f]{40}", self.git_sha):
            raise ValueError("git_sha must be a full lowercase 40-character SHA")
        expected_config = "pi05_libero_tvm" if self.model_class == "tvm" else "pi05_libero"
        if self.config != expected_config:
            raise ValueError(f"Expected config {expected_config} for model_class={self.model_class}")
        if self.model_class == "official":
            if self.checkpoint_s3_uri:
                raise ValueError("Official checkpoint must not be copied to S3")
            if self.checkpoint_path != "gs://openpi-assets/checkpoints/pi05_libero":
                raise ValueError("Official checkpoint must use gs://openpi-assets/checkpoints/pi05_libero")
            if not re.fullmatch(r"[A-Za-z0-9._-]{1,160}", self.artifact_id):
                raise ValueError("Official artifact_id is not a safe path component")
        else:
            digest_match = re.fullmatch(r"sha256:([0-9a-f]{64})", self.artifact_id)
            if digest_match is None:
                raise ValueError("S3 checkpoint artifact_id must be sha256:<64 lowercase hex characters>")
            expected_uri = (
                "s3://aair-users-east-2/arilap01/libero/openpi/checkpoints/"
                f"{self.model_class}/{self.training_run_id}/"
                f"step_{self.train_steps_completed}-checkpoint_{self.checkpoint_label}/"
                f"sha256_{digest_match.group(1)}/"
            )
            if self.checkpoint_s3_uri != expected_uri:
                raise ValueError("checkpoint_s3_uri does not match the immutable artifact provenance")
            checkpoint_path = pathlib.Path(self.checkpoint_path).resolve()
            if not checkpoint_path.is_relative_to("/volt"):
                raise ValueError("Downloaded S3 checkpoint_path must be under /volt")


@dataclasses.dataclass(frozen=True)
class Protocol:
    seed: int = 7
    replan_steps: int = 5
    trials_per_task: int = 50
    suites: tuple[str, ...] = DEFAULT_SUITES

    @property
    def protocol_id(self) -> str:
        return f"libero4_seed{self.seed}_replan{self.replan_steps}_trials{self.trials_per_task}"


@dataclasses.dataclass(frozen=True)
class Runtime:
    repo_root: pathlib.Path
    output_root: pathlib.Path
    server_python: pathlib.Path
    client_python: pathlib.Path
    openpi_data_home: pathlib.Path
    server_ready_timeout_sec: int
    client_timeout_sec: int
    xla_mem_fraction: float
    save_video: bool
    attempt_id: str


def absolute_preserving_symlink(path: pathlib.Path) -> pathlib.Path:
    """Make a CLI path absolute without resolving a virtualenv symlink."""
    return pathlib.Path(os.path.abspath(path))


def make_eval_id(protocol: Protocol, artifact_id: str, flow_steps: int) -> str:
    identity = f"{protocol.protocol_id}|{artifact_id}|flow={flow_steps}"
    return hashlib.sha256(identity.encode()).hexdigest()[:20]


def build_server_command(runtime: Runtime, artifact: Artifact, flow_steps: int, port: int) -> list[str]:
    return [
        str(runtime.server_python),
        "scripts/serve_policy.py",
        "--env",
        "LIBERO",
        "--port",
        str(port),
        "--num-steps",
        str(flow_steps),
        "policy:checkpoint",
        "--policy.config",
        artifact.config,
        "--policy.dir",
        artifact.checkpoint_path,
    ]


def build_client_command(
    runtime: Runtime,
    artifact: Artifact,
    protocol: Protocol,
    flow_steps: int,
    port: int,
    suite: str,
    eval_id: str,
    results_jsonl: pathlib.Path,
    video_root: pathlib.Path,
) -> list[str]:
    command = [
        str(runtime.client_python),
        "examples/libero/main.py",
        "--args.host",
        "127.0.0.1",
        "--args.port",
        str(port),
        "--args.task-suite-name",
        suite,
        "--args.num-trials-per-task",
        str(protocol.trials_per_task),
        "--args.replan-steps",
        str(protocol.replan_steps),
        "--args.seed",
        str(protocol.seed),
        "--args.video-out-path",
        str(video_root / suite),
        "--args.results-jsonl",
        str(results_jsonl),
        "--args.eval-id",
        eval_id,
        "--args.artifact-id",
        artifact.artifact_id,
        "--args.artifact-name",
        artifact.artifact_name,
        "--args.model-class",
        artifact.model_class,
        "--args.training-run-id",
        artifact.training_run_id,
        "--args.checkpoint-name",
        artifact.checkpoint_name,
        "--args.checkpoint-label",
        artifact.checkpoint_label,
        "--args.checkpoint-path",
        artifact.checkpoint_path,
        "--args.checkpoint-s3-uri",
        artifact.checkpoint_s3_uri,
        "--args.flow-steps",
        str(flow_steps),
        "--args.git-sha",
        artifact.git_sha,
    ]
    command += [
        "--args.train-steps-completed",
        str(artifact.train_steps_completed),
        "--args.config",
        artifact.config,
        "--args.client-log-path",
        str(results_jsonl.parent / f"client_{suite}.log"),
    ]
    if not runtime.save_video:
        command.append("--args.no-save-video")
    return command


def wait_for_server(port: int, process: subprocess.Popen, timeout_sec: int) -> None:
    deadline = time.monotonic() + timeout_sec
    health_url = f"http://127.0.0.1:{port}/healthz"
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"Policy server exited before becoming ready (exit code {return_code})")
        try:
            with urllib.request.urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
    raise TimeoutError(f"Policy server did not become ready within {timeout_sec}s: {health_url}")


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def run_client(
    command: list[str],
    log_path: pathlib.Path,
    runtime: Runtime,
    gpu_id: int,
) -> int:
    client_environment = os.environ.copy()
    libero_pythonpath = str(runtime.repo_root / "third_party" / "libero")
    inherited_pythonpath = client_environment.get("PYTHONPATH")
    if inherited_pythonpath:
        libero_pythonpath = os.pathsep.join((libero_pythonpath, inherited_pythonpath))
    client_environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu_id),
            "MUJOCO_GL": "egl",
            # Pinned robosuite validates this against the physical IDs listed
            # in CUDA_VISIBLE_DEVICES before EGL performs device selection.
            "MUJOCO_EGL_DEVICE_ID": str(gpu_id),
            "PYOPENGL_PLATFORM": "egl",
            "PYTHONPATH": libero_pythonpath,
        }
    )
    with log_path.open("w", encoding="utf-8") as client_log:
        process = subprocess.Popen(
            command,
            cwd=runtime.repo_root,
            env=client_environment,
            stdout=client_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=runtime.client_timeout_sec)
        except BaseException:
            stop_process(process)
            raise


def run_flow(
    runtime: Runtime,
    artifact: Artifact,
    protocol: Protocol,
    flow_steps: int,
    gpu_id: int,
    port: int,
) -> pathlib.Path:
    eval_id = make_eval_id(protocol, artifact.artifact_id, flow_steps)
    flow_root = runtime.output_root / protocol.protocol_id / artifact.artifact_id / f"flow_{flow_steps}"
    flow_root.mkdir(parents=True, exist_ok=True)
    for marker in sorted(flow_root.glob("*/_SUCCESS"), reverse=True):
        candidate = marker.parent
        summary = candidate / "summary.csv"
        episodes = candidate / "episodes.jsonl"
        if summary.is_file() and episodes.is_file():
            libero_eval_results.aggregate_records(libero_eval_results.read_episode_jsonl([episodes]))
            return candidate

    lock_path = flow_root / ".evaluation.lock"
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Evaluation is already claimed: {lock_path}") from error
        os.ftruncate(lock_fd, 0)
        os.write(lock_fd, f"{runtime.attempt_id}\n".encode())
        attempt_dir = flow_root / runtime.attempt_id
        success_marker = attempt_dir / "_SUCCESS"
        attempt_dir.mkdir(parents=True, exist_ok=False)
        results_jsonl = attempt_dir / "episodes.jsonl"
        server_log_path = attempt_dir / "server.log"
        metadata_path = attempt_dir / "run_metadata.json"
        metadata = {
            "schema_version": 1,
            "eval_id": eval_id,
            "protocol_id": protocol.protocol_id,
            "artifact": dataclasses.asdict(artifact),
            "flow_steps": flow_steps,
            "gpu_id": gpu_id,
            "port": port,
            "started_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "status": "running",
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        try:
            server_environment = os.environ.copy()
            server_environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(gpu_id),
                    "OPENPI_DATA_HOME": str(runtime.openpi_data_home),
                    "XLA_PYTHON_CLIENT_MEM_FRACTION": str(runtime.xla_mem_fraction),
                }
            )
            server_command = build_server_command(runtime, artifact, flow_steps, port)
            with server_log_path.open("w", encoding="utf-8") as server_log:
                server = subprocess.Popen(
                    server_command,
                    cwd=runtime.repo_root,
                    env=server_environment,
                    stdout=server_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    wait_for_server(port, server, runtime.server_ready_timeout_sec)
                    for suite in protocol.suites:
                        client_log_path = attempt_dir / f"client_{suite}.log"
                        client_command = build_client_command(
                            runtime,
                            artifact,
                            protocol,
                            flow_steps,
                            port,
                            suite,
                            eval_id,
                            results_jsonl,
                            attempt_dir / "videos",
                        )
                        return_code = run_client(client_command, client_log_path, runtime, gpu_id)
                        if return_code != 0:
                            raise RuntimeError(f"LIBERO client failed for {suite} with exit code {return_code}")
                finally:
                    stop_process(server)

            libero_eval_results.aggregate_jsonl_to_csv([results_jsonl], attempt_dir / "summary.csv")
            metadata.update(
                {
                    "finished_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                    "status": "complete",
                    "server_command": server_command,
                }
            )
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            success_marker.write_text("", encoding="utf-8")
            return attempt_dir
        except BaseException as error:
            metadata.update(
                {
                    "error": f"{type(error).__name__}: {error}",
                    "finished_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                    "status": "failed",
                }
            )
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            raise
    finally:
        os.close(lock_fd)


def run_flow_group(
    runtime: Runtime,
    artifact: Artifact,
    protocol: Protocol,
    flow_steps: list[int],
    gpu_id: int,
    port: int,
) -> list[tuple[int, pathlib.Path]]:
    """Run one worker's flow settings sequentially on its exclusive GPU and port."""
    return [(flow, run_flow(runtime, artifact, protocol, flow, gpu_id, port)) for flow in flow_steps]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--repo-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--server-python", type=pathlib.Path, required=True)
    parser.add_argument("--client-python", type=pathlib.Path, required=True)
    parser.add_argument("--openpi-data-home", type=pathlib.Path, required=True)
    parser.add_argument("--flow-steps", type=int, nargs="+", default=DEFAULT_FLOWS)
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[0])
    parser.add_argument("--base-port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--trials-per-task", type=int, default=50)
    parser.add_argument("--server-ready-timeout-sec", type=int, default=900)
    parser.add_argument("--client-timeout-sec", type=int, default=21600)
    parser.add_argument("--xla-mem-fraction", type=float, default=0.85)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--attempt-id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not pathlib.Path("/volt").exists() or not os.environ.get("VOLT_CLUSTER_NAME"):
        raise RuntimeError("LIBERO evaluation runner is restricted to Volt pods")
    if not args.gpu_ids:
        raise ValueError("At least one GPU ID is required")
    if len(set(args.flow_steps)) != len(args.flow_steps):
        raise ValueError("Flow-step values must be unique")
    if any(step < 1 for step in args.flow_steps):
        raise ValueError("Flow-step values must be positive")
    if args.trials_per_task != 50:
        raise ValueError("The audited LIBERO protocol requires exactly 50 trials per task")
    attempt_id = args.attempt_id or datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,80}", attempt_id):
        raise ValueError("attempt_id must be a safe path component")

    artifact = Artifact.from_json(args.artifact_manifest)
    protocol = Protocol(seed=args.seed, replan_steps=args.replan_steps, trials_per_task=args.trials_per_task)
    runtime = Runtime(
        repo_root=args.repo_root.resolve(),
        output_root=args.output_root.resolve(),
        # Preserve virtual-environment entry points. Path.resolve() follows the
        # interpreter symlink to the system Python and silently drops the venv.
        server_python=absolute_preserving_symlink(args.server_python),
        client_python=absolute_preserving_symlink(args.client_python),
        openpi_data_home=args.openpi_data_home.resolve(),
        server_ready_timeout_sec=args.server_ready_timeout_sec,
        client_timeout_sec=args.client_timeout_sec,
        xla_mem_fraction=args.xla_mem_fraction,
        save_video=args.save_video,
        attempt_id=attempt_id,
    )

    assignments = [[] for _ in args.gpu_ids]
    for index, flow in enumerate(args.flow_steps):
        assignments[index % len(args.gpu_ids)].append(flow)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpu_ids)) as executor:
        futures = {
            executor.submit(
                run_flow_group,
                runtime,
                artifact,
                protocol,
                flows,
                args.gpu_ids[index],
                args.base_port + index,
            ): flows
            for index, flows in enumerate(assignments)
            if flows
        }
        for future in concurrent.futures.as_completed(futures):
            for flow, output in future.result():
                print(f"flow_steps={flow} output={output}")


if __name__ == "__main__":
    main()
