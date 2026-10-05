"""Detached pod supervisor with fail-closed gates, live W&B and result publication."""

import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rescue_characterization"))
import subprocess
import time
import urllib.request

from protocol import file_hash
from protocol import tree_hash

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PROJECT = "pi05-two-evaluation-ranking"
BRANCH = "codex/pi05-two-evaluation-ranking"


def now():
    return datetime.datetime.now(datetime.UTC).isoformat()


def validate_smoke_gate(manifest, manifest_path, output):
    output = Path(output)
    mh = file_hash(manifest_path)
    gate = json.loads((output / "smoke-independent-audit.json").read_text())
    prior = json.loads((output / "smoke-status.json").read_text())
    if gate.get("status") != "passed" or gate.get("manifest_sha256") != mh:
        raise RuntimeError("Independent smoke gate absent or wrong identity")
    if prior.get("status") != "published" or prior.get("manifest_sha256") != mh:
        raise RuntimeError("Smoke phase must be published")
    expected = {
        "record_files": {str(output / (b + "-smoke.jsonl")) for b in manifest["benchmarks"]},
        "metrics_files": {str(output / (b + "-smoke-metrics.json")) for b in manifest["benchmarks"]},
        "preparation_files": {
            str(output / (b + "-smoke-prepare-worker-" + str(i) + ".jsonl"))
            for b in manifest["benchmarks"]
            for i in range(4)
        },
    }
    for field, paths in expected.items():
        entries = gate.get(field, [])
        if len(entries) != len(paths) or {x["path"] for x in entries} != paths:
            raise RuntimeError("Incomplete independent smoke gate: " + field)
        for entry in entries:
            if file_hash(entry["path"]) != entry["sha256"]:
                raise RuntimeError("Changed independent smoke evidence")


def shard_records(source, destination, limit=35_000_000):
    """Preserve original byte stream exactly, split only between JSONL rows."""
    destination.mkdir(parents=True, exist_ok=True)
    entries, source_hash, reconstructed_hash = [], hashlib.sha256(), hashlib.sha256()
    total_lines = total_bytes = 0
    writer = None
    path = None
    count = size = 0

    def close():
        if writer is not None:
            writer.close()
            entries.append({"path": str(path.name), "bytes": size, "lines": count, "sha256": file_hash(path)})

    with Path(source).open("rb") as stream:
        for line in stream:
            if len(line) > limit:
                raise RuntimeError("One JSONL row exceeds shard limit")
            json.loads(line)
            if writer is None or size + len(line) > limit:
                close()
                path = destination / (Path(source).stem + f"-part-{len(entries):03d}.jsonl")
                writer = path.open("xb")
                count = size = 0
            writer.write(line)
            source_hash.update(line)
            total_lines += 1
            total_bytes += len(line)
            count += 1
            size += len(line)
    close()
    for entry in entries:
        with (destination / entry["path"]).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                reconstructed_hash.update(block)
    if source_hash.digest() != reconstructed_hash.digest():
        raise RuntimeError("Lossless shard verification failed")
    return {
        "source_path": str(source),
        "source_sha256": source_hash.hexdigest(),
        "bytes": total_bytes,
        "lines": total_lines,
        "shards": entries,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--phase", choices=["smoke", "full"], required=True)
    a = p.parse_args()
    manifest_path = Path(a.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    mh = file_hash(manifest_path)
    output = Path(a.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / (a.phase + "-status.json")).exists():
        raise RuntimeError("Existing phase state: inspect preserved attempt; never auto-restart")
    lock = (output / "supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    runtime = manifest["runtime"]
    py = runtime["inference_python"]
    env = dict(
        os.environ,
        **runtime["inference_common_env"],
        WANDB_MODE="online",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    for path, digest in manifest["source_sha256"].items():
        if file_hash(path) != digest:
            raise RuntimeError("Source changed: " + path)
    for spec in manifest["benchmarks"].values():
        digest, _ = tree_hash(spec["installed_assets_path"])
        if digest != spec["installed_assets_sha256"]:
            raise RuntimeError("Installed simulator assets changed")
    if tree_hash(manifest["checkpoint"]["path"])[0] != manifest["checkpoint"]["sha256"]:
        raise RuntimeError("Checkpoint changed")
    if file_hash(manifest["head"]["path"]) != manifest["head"]["sha256"]:
        raise RuntimeError("Head changed")
    if subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip() != BRANCH:
        raise RuntimeError("Unexpected publication branch")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Tracked checkout must be committed before execution")
    if a.phase == "full":
        validate_smoke_gate(manifest, manifest_path, output)
    import wandb

    run = wandb.init(
        project=PROJECT,
        name=a.phase + "-" + mh[:12],
        job_type="supervisor",
        config={
            "manifest_sha256": mh,
            "pod": manifest["pod_id"],
            "git_commit": head,
            "phase": a.phase,
            "maximum_episodes": manifest["stopping"]["maximum_scored_episodes"],
        },
    )
    if not run.url:
        raise RuntimeError("Online W&B tracking unavailable")
    diagnostic_path = ROOT / "MUJOCO_LOG.TXT"
    diagnostic_before = {
        "exists": diagnostic_path.exists(),
        "bytes": diagnostic_path.stat().st_size if diagnostic_path.exists() else 0,
        "sha256": file_hash(diagnostic_path) if diagnostic_path.exists() else None,
        "inode": diagnostic_path.stat().st_ino if diagnostic_path.exists() else None,
    }
    if diagnostic_before["exists"]:
        preserved_log = output / (a.phase + "-MUJOCO-preexisting.txt")
        with diagnostic_path.open("rb") as source, preserved_log.open("xb") as destination:
            shutil.copyfileobj(source, destination)
        if file_hash(preserved_log) != diagnostic_before["sha256"]:
            raise RuntimeError("Preexisting diagnostic log changed during preservation")
        diagnostic_before["preserved_path"] = str(preserved_log)
        diagnostic_before["preserved_sha256"] = file_hash(preserved_log)
    state = {
        "phase": a.phase,
        "status": "running",
        "started_at": now(),
        "git_commit": head,
        "manifest_sha256": mh,
        "wandb_url": run.url,
        "commands": [],
    }
    state["simulator_diagnostic_before"] = diagnostic_before
    children = []
    logs = []
    begun = time.monotonic()
    wall_hours = 8 if a.phase == "smoke" else 96

    def interrupted(signum, frame):
        raise RuntimeError("Supervisor interrupted by signal " + str(signum))

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def save():
        state["updated_at"] = now()
        temp = output / (a.phase + "-status.tmp")
        temp.write_text(json.dumps(state, indent=2) + "\n")
        temp.replace(output / (a.phase + "-status.json"))

    def capture_simulator_diagnostics():
        # Preserve the inherited log untouched; only an authenticated appended suffix
        # is attributed to this phase. Replacement/truncation is explicitly reported.
        current_exists = diagnostic_path.exists()
        info = {
            "path": str(diagnostic_path),
            "before": diagnostic_before,
            "exists_after": current_exists,
            "identity_changed": False,
            "new_warning_count": None,
            "capture_at": now(),
        }
        if not current_exists:
            info["identity_changed"] = diagnostic_before["exists"]
            if not diagnostic_before["exists"]:
                info["new_warning_count"] = 0
        else:
            stat = diagnostic_path.stat()
            info.update(bytes_after=stat.st_size, sha256_after=file_hash(diagnostic_path), inode_after=stat.st_ino)
            prefix = hashlib.sha256()
            with diagnostic_path.open("rb") as stream:
                remaining = diagnostic_before["bytes"]
                while remaining:
                    block = stream.read(min(1024 * 1024, remaining))
                    if not block:
                        break
                    prefix.update(block)
                    remaining -= len(block)
                preserved = not diagnostic_before["exists"] or (
                    not remaining
                    and prefix.hexdigest() == diagnostic_before["sha256"]
                    and stat.st_ino == diagnostic_before["inode"]
                )
                info["identity_changed"] = not preserved
                if not preserved:
                    destination = output / (a.phase + "-MUJOCO-boundary-unknown.txt")
                    shutil.copy2(diagnostic_path, destination)
                    info.update(
                        boundary_unknown_path=str(destination),
                        boundary_unknown_sha256=file_hash(destination),
                        warning_attribution="unknown; current file preserved without assigning warnings to this phase",
                    )
                if preserved:
                    suffix = stream.read()
                    destination = output / (a.phase + "-MUJOCO-new.txt")
                    destination.write_bytes(suffix)
                    info.update(
                        new_bytes=len(suffix),
                        suffix_path=str(destination),
                        suffix_sha256=file_hash(destination),
                        new_warning_count=suffix.upper().count(b"WARNING"),
                    )
        state["simulator_diagnostics"] = info
        save()

    def launch(command, label, custom_env=env):
        state["commands"].append({"label": label, "argv": list(map(str, command))})
        save()
        log = (output / (a.phase + "-" + label + ".log")).open("x")
        logs.append(log)
        proc = subprocess.Popen(
            list(map(str, command)),
            cwd=ROOT,
            env=custom_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        children.append(proc)
        state["commands"][-1]["pid"] = proc.pid
        state["commands"][-1]["environment_overrides"] = {
            k: custom_env[k]
            for k in set(runtime["inference_common_env"])
            | set(runtime["simulator_common_env"])
            | {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "WANDB_MODE"}
            if k in custom_env
        }
        save()
        return proc

    def check_processes(procs):
        while any(x.poll() is None for x in procs):
            if time.monotonic() - begun > wall_hours * 3600:
                raise RuntimeError("Declared wall-time budget exhausted")
            if shutil.disk_usage(output).free < manifest["stopping"]["minimum_free_disk_gib"] * 1024**3:
                raise RuntimeError("Declared disk reserve reached")
            if any(x.poll() not in (None, 0) for x in procs):
                raise RuntimeError("Worker failed; see exact command logs")
            if any(x.poll() is not None for x in servers):
                raise RuntimeError("Policy server exited")
            counts = {}
            for path in output.glob("*-smoke.jsonl" if a.phase == "smoke" else "*-main-worker-*.jsonl"):
                with path.open() as f:
                    counts[path.stem] = sum(1 for line in f if line.strip())
            run.log({"elapsed_seconds": time.monotonic() - begun, "completed_or_error_records": sum(counts.values())})
            state["record_counts"] = counts
            save()
            time.sleep(20)
        if any(x.returncode != 0 for x in procs):
            raise RuntimeError("Worker failed; see exact command logs")

    def client(bench, index, phase):
        custom = dict(env, **runtime["simulator_common_env"], **runtime[bench])
        target = output / (
            bench
            + (
                "-smoke.jsonl"
                if phase == "smoke"
                else "-" + (a.phase + "-prepare" if phase == "prepare" else phase) + "-worker-" + str(index) + ".jsonl"
            )
        )
        if target.exists():
            raise RuntimeError("Refusing existing worker records: " + str(target))
        cmd = [
            runtime["simulator_python"],
            HERE / "client.py",
            "--manifest",
            manifest_path,
            "--benchmark",
            bench,
            "--server",
            "http://127.0.0.1:" + str(8940 + index),
            "--output",
            target,
        ]
        if phase == "prepare":
            cmd += ["--prepare-only", "--worker-index", str(index), "--workers", "4"]
        elif phase == "smoke":
            cmd += ["--smoke"]
        else:
            cmd += ["--worker-index", str(index), "--workers", "4"]
        return launch(cmd, bench + "-" + phase + "-" + str(index), custom)

    def analyze(bench, *, smoke):
        records = (
            [output / (bench + "-smoke.jsonl")]
            if smoke
            else [output / (bench + "-main-worker-" + str(i) + ".jsonl") for i in range(4)]
        )
        cmd = [
            py,
            HERE / "analysis.py",
            "--manifest",
            manifest_path,
            "--benchmark",
            bench,
            "--records",
            *records,
            "--output",
            output / (bench + ("-smoke-metrics.json" if smoke else "-metrics.json")),
        ]
        if smoke:
            cmd.append("--smoke")
        check_processes([launch(cmd, bench + "-metrics")])
        capture = [
            runtime["simulator_python"],
            HERE.parent / "rescue_characterization" / "recording_audit.py",
            "--manifest",
            manifest_path,
            "--records",
            *records,
            "--output",
            output / (bench + ("-smoke-recording-audit.json" if smoke else "-recording-audit.json")),
        ]
        check_processes([launch(capture, bench + "-recording-audit")])

    servers = []
    try:
        for i, uuid in enumerate(manifest["gpu_uuids"]):
            servers.append(
                launch(
                    [
                        py,
                        HERE / "policy_server.py",
                        "--manifest",
                        manifest_path,
                        "--gpu-uuid",
                        uuid,
                        "--port",
                        str(8940 + i),
                    ],
                    "server-" + str(i),
                )
            )
        deadline = time.monotonic() + 1800
        for i, server in enumerate(servers):
            while True:
                if server.poll() is not None:
                    raise RuntimeError("Server startup failed")
                try:
                    with urllib.request.urlopen("http://127.0.0.1:" + str(8940 + i) + "/health", timeout=5) as r:
                        health = json.load(r)
                    if (
                        health["gpu_uuid"] != manifest["gpu_uuids"][i]
                        or health["checkpoint_sha256"] != manifest["checkpoint"]["sha256"]
                        or health["manifest_sha256"] != mh
                        or health["head_sha256"] != manifest["head"]["sha256"]
                    ):
                        raise RuntimeError("Server identity differs from declared protocol")
                    break
                except (OSError, TimeoutError):
                    if time.monotonic() > deadline:
                        raise RuntimeError("Server startup deadline exceeded") from None
                    time.sleep(5)
        for bench in manifest["benchmarks"]:
            check_processes([client(bench, i, "prepare") for i in range(4)])
            if a.phase == "smoke":
                check_processes([client(bench, 0, "smoke")])
                analyze(bench, smoke=True)
            else:
                check_processes([client(bench, i, "main") for i in range(4)])
                analyze(bench, smoke=False)
        independent = output / (a.phase + "-independent-audit.json")
        audit_command = [
            py,
            HERE / "independent_audit.py",
            "--manifest",
            manifest_path,
            "--phase",
            "smoke" if a.phase == "smoke" else "main",
            "--run-dir",
            output,
            "--output",
            independent,
        ]
        check_processes([launch(audit_command, "independent-audit")])
        audit = json.loads(independent.read_text())
        if audit.get("status") != "passed" or audit.get("manifest_sha256") != mh:
            raise RuntimeError("Independent audit did not pass with current identity")
        preparation_cost = {
            "records": 0,
            "velocity_evaluations": 0,
            "probe_velocity_evaluations": 0,
            "head_evaluations": 0,
            "prefix_evaluations": 0,
        }
        for bench in manifest["benchmarks"]:
            for i in range(4):
                path = output / (bench + "-" + a.phase + "-prepare-worker-" + str(i) + ".jsonl")
                rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
                if len(rows) != 1 or rows[0]["status"] != "verified":
                    raise RuntimeError("Incomplete preparation record")
                row = rows[0]
                preparation_cost["records"] += 1
                preparation_cost["velocity_evaluations"] += row["warmup"]["velocity_evaluations"]
                preparation_cost["probe_velocity_evaluations"] += row["warmup"]["probe_velocity_evaluations"]
                preparation_cost["head_evaluations"] += row["warmup"]["head_evaluations"]
                preparation_cost["prefix_evaluations"] += row["warmup"]["prefix_evaluations"]
                for verification in row["verification"]:
                    preparation_cost["velocity_evaluations"] += verification["velocity_evaluations_for_verification"]
                    preparation_cost["probe_velocity_evaluations"] += verification["probe_velocity_evaluations"]
                    preparation_cost["head_evaluations"] += verification["head_evaluations"]
                    preparation_cost["prefix_evaluations"] += verification["prefix_evaluations"]
        state["preparation_cost"] = preparation_cost
        capture_simulator_diagnostics()
        if a.phase == "full":
            report_command = [
                py,
                HERE / "report.py",
                "--manifest",
                manifest_path,
                "--results",
                output,
                "--output",
                output / "full-REPORT.md",
            ]
            check_processes([launch(report_command, "final-report")])
        # The experiment actually executes a separate probe in every scored arm.
        probe_artifact = wandb.Artifact("initial-flow-probes-" + a.phase + "-" + mh[:12], type="initial-flow-probes")
        probe_index = []
        for bench in manifest["benchmarks"]:
            paths = (
                [output / (bench + "-smoke.jsonl")]
                if a.phase == "smoke"
                else [output / (bench + "-main-worker-" + str(i) + ".jsonl") for i in range(4)]
            )
            for path in paths:
                with path.open() as stream:
                    for line in stream:
                        row = json.loads(line)
                        probe = row["initial_probe"]
                        raw = Path(probe["raw_path"])
                        if file_hash(raw) != probe["raw_sha256"]:
                            raise RuntimeError("Raw probe hash changed")
                        relative = "probes/" + probe["raw_sha256"] + ".npz"
                        probe_artifact.add_file(str(raw), name=relative)
                        probe_index.append(
                            {
                                "benchmark": bench,
                                "condition_id": row["condition_id"],
                                "seed": row["seed"],
                                "init_index": row["init_index"],
                                "flow_steps": row["flow_steps"],
                                "raw_path": str(raw),
                                "raw_sha256": probe["raw_sha256"],
                                "artifact_relative_path": relative,
                            }
                        )
        uploaded_probes = run.log_artifact(probe_artifact)
        uploaded_probes.wait()
        state["probe_artifact"] = uploaded_probes.qualified_name
        (output / (a.phase + "-probe-artifact-index.json")).write_text(
            json.dumps(
                {
                    "artifact": uploaded_probes.qualified_name,
                    "manifest_sha256": mh,
                    "phase": a.phase,
                    "entries": probe_index,
                },
                indent=2,
            )
            + "\n"
        )
        save()
        if tree_hash(manifest["checkpoint"]["path"])[0] != manifest["checkpoint"]["sha256"]:
            raise RuntimeError("Post-evaluation checkpoint content changed")
        if file_hash(manifest["head"]["path"]) != manifest["head"]["sha256"]:
            raise RuntimeError("Post-evaluation head changed")
        for path, digest in manifest["source_sha256"].items():
            if file_hash(path) != digest:
                raise RuntimeError("Source changed during evaluation: " + path)
        import shlex

        fence = chr(96) * 3
        (output / (a.phase + "-COMMANDS.md")).write_text(
            "# Executed commands\n\n"
            + "\n\n".join(
                "## " + item["label"] + "\n\n" + fence + "bash\n" + shlex.join(item["argv"]) + "\n" + fence
                for item in state["commands"]
            )
            + "\n"
        )
        state["status"] = "validated"
        state["elapsed_seconds"] = time.monotonic() - begun
        state["four_gpu_allocation_hours"] = 4 * state["elapsed_seconds"] / 3600
        save()
        dest = HERE / "results" / mh[:12] / a.phase
        dest.mkdir(parents=True, exist_ok=False)
        shard_index = []
        for path in sorted(output.glob("*")):
            is_phase_record = (
                (path.name.endswith("-smoke.jsonl") or "-smoke-prepare-worker-" in path.name)
                if a.phase == "smoke"
                else ("-main-worker-" in path.name or "-full-prepare-worker-" in path.name)
            )
            if path.is_file() and path.suffix == ".jsonl" and is_phase_record:
                shard_index.append(shard_records(path, dest / "raw-shards"))
            elif (path.is_file() and path.suffix in (".json", ".md")) or (
                path.is_file()
                and path.name
                in (
                    a.phase + "-MUJOCO-new.txt",
                    a.phase + "-MUJOCO-boundary-unknown.txt",
                    a.phase + "-MUJOCO-preexisting.txt",
                )
            ):
                if path.stat().st_size >= 35_000_000:
                    raise RuntimeError("Small artifact exceeds publication cap: " + str(path))
                shutil.copy2(path, dest / path.name)
        (dest / "raw-shards-index.json").write_text(json.dumps(shard_index, indent=2) + "\n")
        artifact = wandb.Artifact("supervisor-" + a.phase + "-" + mh[:12], type="evaluation")
        artifact.add_dir(str(dest))
        artifact.add_file(str(manifest_path), name="protocol.json")
        saved = run.log_artifact(artifact)
        saved.wait()
        state["wandb_artifact"] = saved.qualified_name
        save()
        shutil.copy2(output / (a.phase + "-status.json"), dest / (a.phase + "-status.json"))
        subprocess.run(["git", "add", str(dest.relative_to(ROOT))], cwd=ROOT, check=True)
        staged = subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True).splitlines()
        if any(not path.startswith(str(dest.relative_to(ROOT)) + "/") for path in staged):
            raise RuntimeError("Unrelated staged files prohibit result commit")
        subprocess.run(["git", "commit", "-m", "Record frozen two-evaluation ranking " + a.phase], cwd=ROOT, check=True)
        subprocess.run(["git", "push", "origin", "HEAD:refs/heads/" + BRANCH], cwd=ROOT, check=True)
        run.summary["publication_status"] = "published"
        run.finish()
        state["status"] = "published"
        state["published_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        save()
    except BaseException as exc:
        state["status"] = "failed"
        state["error"] = repr(exc)
        save()
        run.finish(exit_code=1)
        raise
    finally:
        for proc in children:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        for proc in children:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
        for log in logs:
            log.close()
        capture_simulator_diagnostics()


if __name__ == "__main__":
    main()
