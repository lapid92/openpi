"""Detached pod supervisor with fail-closed gates, live W&B and result publication."""

import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request

from protocol import file_hash
from protocol import tree_hash

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PROJECT = "pi05-frozen-flow-study"


def now():
    return datetime.datetime.now(datetime.UTC).isoformat()


def validate_smoke_gate(manifest, manifest_path, output):
    from analyze import validate

    mh = file_hash(manifest_path)
    output = Path(output)
    for benchmark in manifest["benchmarks"]:
        records = output / (benchmark + "-smoke.jsonl")
        summary = json.loads((output / (benchmark + "-smoke-summary.json")).read_text())
        rows = [json.loads(line) for line in records.read_text().splitlines() if line.strip()]
        found, _ = validate(manifest, rows, benchmark, mh, smoke=True)
        if (
            summary["audit"] != "passed"
            or summary["manifest_sha256"] != mh
            or summary["benchmark"] != benchmark
            or summary["phase"] != "smoke"
            or summary["completed_episodes"] != len(found)
        ):
            raise RuntimeError("Smoke summary identity mismatch")
        recorded = {entry["path"]: entry["sha256"] for entry in summary["record_files"]}
        if recorded.get(str(records)) != file_hash(records):
            raise RuntimeError("Smoke record hash mismatch")


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
    lock = (output / "supervisor.lock").open("w")
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
    state = {
        "phase": a.phase,
        "status": "running",
        "started_at": now(),
        "git_commit": head,
        "manifest_sha256": mh,
        "wandb_url": run.url,
        "commands": [],
    }
    children = []
    logs = []
    begun = time.monotonic()

    def save():
        state["updated_at"] = now()
        temp = output / (a.phase + "-status.tmp")
        temp.write_text(json.dumps(state, indent=2) + "\n")
        temp.replace(output / (a.phase + "-status.json"))

    def launch(command, label, custom_env=env):
        state["commands"].append({"label": label, "argv": list(map(str, command))})
        save()
        log = (output / (label + ".log")).open("a")
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
        return proc

    def check_processes(procs):
        while any(x.poll() is None for x in procs):
            if any(x.poll() not in (None, 0) for x in procs):
                raise RuntimeError("Worker failed; see exact command logs")
            if any(x.poll() is not None for x in servers):
                raise RuntimeError("Policy server exited")
            counts = {}
            for path in output.glob("*-worker-*.jsonl"):
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
            bench + ("-smoke.jsonl" if phase == "smoke" else "-" + phase + "-worker-" + str(index) + ".jsonl")
        )
        cmd = [
            runtime["simulator_python"],
            HERE / "client.py",
            "--manifest",
            manifest_path,
            "--benchmark",
            bench,
            "--server",
            "http://127.0.0.1:" + str(8800 + index),
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
        target = output / (bench + ("-smoke-summary.json" if smoke else "-summary.json"))
        cmd = [
            py,
            HERE / "analyze.py",
            "--manifest",
            manifest_path,
            "--benchmark",
            bench,
            "--records",
            *records,
            "--output",
            target,
            "--wandb-project",
            PROJECT,
            "--wandb-name",
            bench + ("-smoke-" if smoke else "-full-") + mh[:12],
        ]
        if smoke:
            cmd += ["--smoke"]
        check_processes([launch(cmd, bench + ("-smoke-analysis" if smoke else "-analysis"))])

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
                        str(8800 + i),
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
                    with urllib.request.urlopen("http://127.0.0.1:" + str(8800 + i) + "/health", timeout=5) as r:
                        health = json.load(r)
                    if (
                        health["gpu_uuid"] != manifest["gpu_uuids"][i]
                        or health["checkpoint_sha256"] != manifest["checkpoint"]["sha256"]
                        or health["manifest_sha256"] != mh
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
        if tree_hash(manifest["checkpoint"]["path"])[0] != manifest["checkpoint"]["sha256"]:
            raise RuntimeError("Post-evaluation checkpoint content changed")
        report_cmd = [
            py,
            HERE / "report.py",
            "--manifest",
            manifest_path,
            "--results",
            output,
            "--output",
            output / (a.phase + "-REPORT.md"),
        ]
        if a.phase == "smoke":
            report_cmd.append("--smoke")
        check_processes([launch(report_cmd, a.phase + "-report")])
        state["status"] = "validated"
        state["elapsed_seconds"] = time.monotonic() - begun
        state["four_gpu_allocation_hours"] = 4 * state["elapsed_seconds"] / 3600
        save()
        artifact = wandb.Artifact("supervisor-" + a.phase + "-" + mh[:12], type="evaluation")
        for path in output.glob("*"):
            if path.is_file() and path.suffix in (".json", ".jsonl", ".md"):
                artifact.add_file(str(path))
        artifact.add_file(str(manifest_path))
        run.log_artifact(artifact)
        # Preserve all records and diagnostic attempts, including preparations, on the fork.
        dest = HERE / "results" / mh[:12]
        dest.mkdir(parents=True, exist_ok=True)
        for path in output.glob("*"):
            if path.is_file() and path.suffix in (".json", ".jsonl", ".md"):
                shutil.copy2(path, dest / path.name)
        subprocess.run(["git", "add", str(dest.relative_to(ROOT))], cwd=ROOT, check=True)
        if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=False).returncode:
            subprocess.run(
                ["git", "commit", "-m", "Record frozen flow study " + a.phase + " validation and results"],
                cwd=ROOT,
                check=True,
            )
        subprocess.run(["git", "push", "origin", "HEAD:refs/heads/codex/pi05-frozen-flow-study"], cwd=ROOT, check=True)
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
                proc.terminate()
        for proc in children:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
