import dataclasses
import datetime
import fcntl
import json
import os
import pathlib
import subprocess

import pytest

from scripts import libero_eval_runner


@pytest.fixture(autouse=True)
def _datetime_utc_compat(monkeypatch):
    """The project targets 3.11; keep these tests runnable under the host's older Python."""
    if not hasattr(libero_eval_runner.datetime, "UTC"):
        monkeypatch.setattr(
            libero_eval_runner.datetime,
            "UTC",
            datetime.timezone.utc,  # noqa: UP017
            raising=False,
        )
    if not hasattr(pathlib.Path, "is_relative_to"):

        def is_relative_to(path, other):
            try:
                path.relative_to(other)
            except ValueError:
                return False
            return True

        monkeypatch.setattr(pathlib.Path, "is_relative_to", is_relative_to, raising=False)


def _artifact() -> libero_eval_runner.Artifact:
    digest = "b" * 64
    return libero_eval_runner.Artifact(
        artifact_id=f"sha256:{digest}",
        artifact_name="pi05-libero-tvm-15k",
        model_class="tvm",
        training_run_id="matched-run",
        checkpoint_label="15000",
        checkpoint_name="15000",
        checkpoint_path="/volt/data/checkpoints/15000",
        checkpoint_s3_uri=(
            "s3://aair-users-east-2/arilap01/libero/openpi/checkpoints/"
            f"tvm/matched-run/step_15000-checkpoint_15000/sha256_{digest}/"
        ),
        train_steps_completed=15000,
        config="pi05_libero_tvm",
        git_sha="a" * 40,
    )


def _runtime() -> libero_eval_runner.Runtime:
    return libero_eval_runner.Runtime(
        repo_root=pathlib.Path("/volt/code/openpi"),
        output_root=pathlib.Path("/volt/data/evals"),
        server_python=pathlib.Path("/volt/envs/openpi/bin/python"),
        client_python=pathlib.Path("/volt/envs/libero/bin/python"),
        openpi_data_home=pathlib.Path("/volt/data/openpi"),
        server_ready_timeout_sec=900,
        client_timeout_sec=21600,
        xla_mem_fraction=0.85,
        save_video=False,
        attempt_id="attempt-1",
    )


def _assert_lock_available(path: pathlib.Path) -> None:
    lock_fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(lock_fd)


def test_build_server_command_sets_flow_steps_and_checkpoint() -> None:
    command = libero_eval_runner.build_server_command(_runtime(), _artifact(), flow_steps=5, port=8001)

    assert command[0] == "/volt/envs/openpi/bin/python"
    assert command[command.index("--num-steps") + 1] == "5"
    assert command[command.index("--policy.config") + 1] == "pi05_libero_tvm"
    assert command[command.index("--policy.dir") + 1] == "/volt/data/checkpoints/15000"


def test_build_client_command_records_fixed_protocol() -> None:
    protocol = libero_eval_runner.Protocol()
    command = libero_eval_runner.build_client_command(
        _runtime(),
        _artifact(),
        protocol,
        flow_steps=2,
        port=8000,
        suite="libero_goal",
        eval_id="eval-1",
        results_jsonl=pathlib.Path("/volt/data/evals/episodes.jsonl"),
        video_root=pathlib.Path("/volt/data/evals/videos"),
    )

    assert command[command.index("--args.task-suite-name") + 1] == "libero_goal"
    assert command[command.index("--args.num-trials-per-task") + 1] == "50"
    assert command[command.index("--args.replan-steps") + 1] == "5"
    assert command[command.index("--args.seed") + 1] == "7"
    assert command[command.index("--args.flow-steps") + 1] == "2"
    assert "--args.no-save-video" in command


def test_eval_id_changes_with_flow_steps() -> None:
    protocol = libero_eval_runner.Protocol()
    first = libero_eval_runner.make_eval_id(protocol, "artifact", 1)
    second = libero_eval_runner.make_eval_id(protocol, "artifact", 2)

    assert first != second
    assert first == libero_eval_runner.make_eval_id(protocol, "artifact", 1)


def test_trained_artifact_requires_matching_content_addressed_s3_provenance() -> None:
    _artifact().validate()

    with pytest.raises(ValueError, match="immutable artifact provenance"):
        dataclasses.replace(_artifact(), checkpoint_s3_uri="").validate()
    with pytest.raises(ValueError, match="must be under /volt"):
        dataclasses.replace(_artifact(), checkpoint_path="/volt/../tmp/checkpoint").validate()
    with pytest.raises(ValueError, match="immutable artifact provenance"):
        dataclasses.replace(_artifact(), artifact_id=f"sha256:{'c' * 64}").validate()

    wrong_run_uri = _artifact().checkpoint_s3_uri.replace("/tvm/matched-run/", "/finetune/wrong-run/")
    with pytest.raises(ValueError, match="immutable artifact provenance"):
        dataclasses.replace(_artifact(), checkpoint_s3_uri=wrong_run_uri).validate()


def test_official_artifact_requires_canonical_gcs_and_no_s3() -> None:
    artifact = dataclasses.replace(
        _artifact(),
        artifact_id="official-pi05-libero",
        artifact_name="official-pi05-libero",
        model_class="official",
        training_run_id="official-release",
        checkpoint_label="released",
        checkpoint_name="pi05_libero",
        checkpoint_path="gs://openpi-assets/checkpoints/pi05_libero",
        checkpoint_s3_uri="",
        train_steps_completed=0,
        config="pi05_libero",
    )
    artifact.validate()

    with pytest.raises(ValueError, match="must not be copied to S3"):
        dataclasses.replace(artifact, checkpoint_s3_uri=_artifact().checkpoint_s3_uri).validate()


def test_run_client_binds_gpu_and_cleans_up_on_timeout(tmp_path, monkeypatch) -> None:
    runtime = dataclasses.replace(_runtime(), repo_root=tmp_path, client_timeout_sec=17)
    created = []
    stopped = []
    monkeypatch.setenv("PYTHONPATH", "/existing/pythonpath")

    class FakeProcess:
        def wait(self, timeout):
            assert timeout == 17
            raise subprocess.TimeoutExpired(cmd="client", timeout=timeout)

    fake_process = FakeProcess()

    def fake_popen(command, **kwargs):
        created.append((command, kwargs))
        return fake_process

    monkeypatch.setattr(libero_eval_runner.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(libero_eval_runner, "stop_process", stopped.append)

    with pytest.raises(subprocess.TimeoutExpired):
        libero_eval_runner.run_client(["client", "--flag"], tmp_path / "client.log", runtime, gpu_id=3)

    assert created[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == "3"
    assert created[0][1]["env"]["MUJOCO_GL"] == "egl"
    assert created[0][1]["env"]["MUJOCO_EGL_DEVICE_ID"] == "3"
    assert created[0][1]["env"]["PYOPENGL_PLATFORM"] == "egl"
    assert created[0][1]["env"]["PYTHONPATH"] == os.pathsep.join(
        (str(tmp_path / "third_party" / "libero"), "/existing/pythonpath")
    )
    assert created[0][1]["start_new_session"] is True
    assert stopped == [fake_process]


def test_run_client_sets_standalone_libero_pythonpath_when_not_inherited(tmp_path, monkeypatch) -> None:
    runtime = dataclasses.replace(_runtime(), repo_root=tmp_path)
    captured = {}
    monkeypatch.delenv("PYTHONPATH", raising=False)

    class FakeProcess:
        def wait(self, timeout):
            assert timeout == runtime.client_timeout_sec
            return 0

    def fake_popen(command, **kwargs):
        del command
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(libero_eval_runner.subprocess, "Popen", fake_popen)

    assert libero_eval_runner.run_client(["client"], tmp_path / "client.log", runtime, gpu_id=0) == 0
    assert captured["env"]["PYTHONPATH"] == str(tmp_path / "third_party" / "libero")


def test_absolute_preserving_symlink_keeps_virtualenv_entry_point(tmp_path) -> None:
    system_python = tmp_path / "system-python"
    system_python.touch()
    venv_python = tmp_path / "venv-python"
    venv_python.symlink_to(system_python)

    result = libero_eval_runner.absolute_preserving_symlink(venv_python)

    assert result == venv_python.absolute()
    assert result != venv_python.resolve()


def test_run_flow_publishes_complete_metadata_then_success_marker(tmp_path, monkeypatch) -> None:
    runtime = dataclasses.replace(_runtime(), repo_root=tmp_path, output_root=tmp_path / "output")
    protocol = libero_eval_runner.Protocol()
    client_suites = []
    stopped = []

    server = object()
    monkeypatch.setattr(libero_eval_runner.subprocess, "Popen", lambda *args, **kwargs: server)
    monkeypatch.setattr(libero_eval_runner, "wait_for_server", lambda *args: None)
    monkeypatch.setattr(libero_eval_runner, "stop_process", stopped.append)

    def fake_run_client(command, log_path, runtime_arg, gpu_id):
        del log_path, runtime_arg, gpu_id
        client_suites.append(command[command.index("--args.task-suite-name") + 1])
        return 0

    def fake_aggregate(inputs, output):
        assert inputs[0].name == "episodes.jsonl"
        output.write_text("summary\n", encoding="utf-8")

    monkeypatch.setattr(libero_eval_runner, "run_client", fake_run_client)
    monkeypatch.setattr(libero_eval_runner.libero_eval_results, "aggregate_jsonl_to_csv", fake_aggregate)

    attempt = libero_eval_runner.run_flow(runtime, _artifact(), protocol, 2, gpu_id=1, port=8001)

    assert client_suites == list(libero_eval_runner.DEFAULT_SUITES)
    assert stopped == [server]
    assert (attempt / "summary.csv").is_file()
    assert (attempt / "_SUCCESS").is_file()
    metadata = json.loads((attempt / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "complete"
    assert "finished_at_utc" in metadata
    _assert_lock_available(attempt.parent / ".evaluation.lock")


def test_run_flow_records_failure_and_releases_lock(tmp_path, monkeypatch) -> None:
    runtime = dataclasses.replace(_runtime(), repo_root=tmp_path, output_root=tmp_path / "output")
    protocol = libero_eval_runner.Protocol()
    server = object()
    monkeypatch.setattr(libero_eval_runner.subprocess, "Popen", lambda *args, **kwargs: server)
    monkeypatch.setattr(libero_eval_runner, "wait_for_server", lambda *args: None)
    monkeypatch.setattr(libero_eval_runner, "stop_process", lambda process: None)
    monkeypatch.setattr(libero_eval_runner, "run_client", lambda *args: 9)

    with pytest.raises(RuntimeError, match=r"client failed.*exit code 9"):
        libero_eval_runner.run_flow(runtime, _artifact(), protocol, 3, gpu_id=0, port=8000)

    flow_root = runtime.output_root / protocol.protocol_id / _artifact().artifact_id / "flow_3"
    attempt = flow_root / runtime.attempt_id
    metadata = json.loads((attempt / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"
    assert "exit code 9" in metadata["error"]
    assert "finished_at_utc" in metadata
    assert not (attempt / "_SUCCESS").exists()
    _assert_lock_available(flow_root / ".evaluation.lock")


def test_run_flow_revalidates_completed_attempt_without_starting_server(tmp_path, monkeypatch) -> None:
    runtime = dataclasses.replace(_runtime(), output_root=tmp_path / "output")
    protocol = libero_eval_runner.Protocol()
    flow_root = runtime.output_root / protocol.protocol_id / _artifact().artifact_id / "flow_5"
    completed = flow_root / "old-attempt"
    completed.mkdir(parents=True)
    (completed / "_SUCCESS").write_text("", encoding="utf-8")
    (completed / "summary.csv").write_text("summary\n", encoding="utf-8")
    (completed / "episodes.jsonl").write_text("episode\n", encoding="utf-8")
    sentinel = [object()]
    monkeypatch.setattr(libero_eval_runner.libero_eval_results, "read_episode_jsonl", lambda paths: sentinel)
    monkeypatch.setattr(
        libero_eval_runner.libero_eval_results,
        "aggregate_records",
        lambda records: {"validated": records is sentinel},
    )
    monkeypatch.setattr(
        libero_eval_runner.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("server must not start for a validated completed attempt"),
    )

    assert libero_eval_runner.run_flow(runtime, _artifact(), protocol, 5, gpu_id=0, port=8000) == completed


def test_run_flow_refuses_an_existing_flow_claim(tmp_path) -> None:
    runtime = dataclasses.replace(_runtime(), output_root=tmp_path / "output")
    protocol = libero_eval_runner.Protocol()
    flow_root = runtime.output_root / protocol.protocol_id / _artifact().artifact_id / "flow_1"
    flow_root.mkdir(parents=True)
    lock_path = flow_root / ".evaluation.lock"
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already claimed"):
            libero_eval_runner.run_flow(runtime, _artifact(), protocol, 1, gpu_id=0, port=8000)
    finally:
        os.close(lock_fd)
