"""Detached four-GPU historical replay supervisor; never launches the main scan."""
import argparse
from collections import Counter
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT=Path("/volt/code/frozen-flow-study")
HERE=ROOT/"examples/rescue_characterization"
sys.path.insert(0,str(HERE))
from protocol import file_hash, tree_hash
from replay import build_manifest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--base-manifest",type=Path)
    parser.add_argument("--selection",type=Path,default=HERE/"old-replay-selection.json")
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=Path("/volt/artifacts/rescue-characterization/replay"))
    parser.add_argument("--build-only",action="store_true")
    args=parser.parse_args()
    def terminate(signum, frame):
        raise SystemExit("Supervisor received signal " + str(signum))
    signal.signal(signal.SIGTERM,terminate)
    signal.signal(signal.SIGINT,terminate)
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    if args.build_only:
        if not args.base_manifest:
            parser.error("--build-only requires --base-manifest")
        if args.manifest.exists():
            raise ValueError("Refusing to overwrite a replay declaration")
        manifest=build_manifest(args.base_manifest,args.selection,output)
        args.manifest.write_text(json.dumps(manifest,sort_keys=True,separators=(",",":"))+"\n")
        print(json.dumps({"manifest":str(args.manifest),"sha256":file_hash(args.manifest),"budget":manifest["replay_budget"]}))
        return
    manifest=json.loads(args.manifest.read_text())
    mh=file_hash(args.manifest)
    lock=(output/"supervisor.lock").open("w")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for path,digest in manifest["source_sha256"].items():
        if file_hash(path)!=digest:
            raise ValueError("Pinned source changed: "+path)
    if file_hash(manifest["replay_selection_path"])!=manifest["replay_selection_sha256"]:
        raise ValueError("Replay selection changed")
    gpu_rows=subprocess.check_output(["nvidia-smi","--query-gpu=uuid,name","--format=csv,noheader"],text=True).splitlines()
    gpu_map={row.split(",")[0].strip():row.split(",",1)[1].strip() for row in gpu_rows}
    if set(gpu_map)!=set(manifest["gpu_uuids"]) or any("H100" not in name for name in gpu_map.values()):
        raise ValueError("Expected four declared H100 UUIDs")
    # Fail closed if any GPU is in use by the priority scan or another process.
    active=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip()
    if active:
        raise RuntimeError("GPUs are busy; replay must wait for explicit orchestration")
    runtime=manifest["runtime"]
    env=dict(os.environ,**runtime["inference_common_env"],WANDB_MODE="online",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    os.environ.update({k:env[k] for k in ("WANDB_MODE","HF_HOME","OPENPI_DATA_HOME","UV_CACHE_DIR","XDG_CACHE_HOME") if k in env})
    import wandb
    run=wandb.init(project="pi05-rescue-characterization",name="historical-replay-"+mh[:12],job_type="historical-replay",
                   config={"manifest_sha256":mh,"maximum_episodes":manifest["replay_budget"]["maximum_episodes"],
                           "population_evidence":False,"gpu_uuids":manifest["gpu_uuids"]})
    if not run.url:
        raise RuntimeError("Online W&B tracking unavailable")
    state={"status":"running","manifest_sha256":mh,"wandb_url":run.url,
           "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
           "commands":[],"gpu_inventory":gpu_map,"started_at":datetime.datetime.now(datetime.UTC).isoformat()}
    children,logs,servers=[],[],[]
    begun=time.monotonic()
    def save():
        state["updated_at"]=datetime.datetime.now(datetime.UTC).isoformat()
        tmp=output/"status.tmp"
        tmp.write_text(json.dumps(state,indent=2)+"\n")
        tmp.replace(output/"status.json")
    def launch(cmd,label,custom_env=env):
        state["commands"].append({"label":label,"argv":list(map(str,cmd))})
        save()
        log=(output/(label+".log")).open("a")
        logs.append(log)
        process=subprocess.Popen(list(map(str,cmd)),cwd=ROOT,env=custom_env,stdin=subprocess.DEVNULL,
                                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(process)
        return process
    def budget():
        if time.monotonic()-begun>manifest["replay_budget"]["maximum_wall_hours"]*3600:
            raise RuntimeError("Replay wall-time limit reached")
        import shutil
        if shutil.disk_usage(output).free<manifest["replay_budget"]["minimum_free_disk_gib"]*1024**3:
            raise RuntimeError("Replay storage reserve reached")
    try:
        save()
        for i,uuid in enumerate(manifest["gpu_uuids"]):
            servers.append(launch([runtime["inference_python"],HERE/"policy_server.py","--manifest",args.manifest,
                                   "--gpu-uuid",uuid,"--port",str(8940+i)],"server-"+str(i)))
        deadline=time.monotonic()+1800
        for i,server in enumerate(servers):
            while True:
                budget()
                if server.poll() is not None:
                    raise RuntimeError("Replay policy server exited")
                try:
                    with urllib.request.urlopen("http://127.0.0.1:"+str(8940+i)+"/health",timeout=5) as response:
                        health=json.load(response)
                    if (health["gpu_uuid"]!=manifest["gpu_uuids"][i] or health["checkpoint_sha256"]!=manifest["checkpoint"]["sha256"]
                            or health["manifest_sha256"]!=mh):
                        raise ValueError("Replay server identity mismatch")
                    break
                except (OSError,TimeoutError):
                    if time.monotonic()>deadline:
                        raise RuntimeError("Replay policy startup deadline exceeded") from None
                    time.sleep(5)
        for bench in manifest["benchmarks"]:
            procs=[]
            for i in range(4):
                custom=dict(env,**runtime["simulator_common_env"],**runtime[bench])
                procs.append(launch([runtime["simulator_python"],HERE/"replay.py","--manifest",args.manifest,
                    "--benchmark",bench,"--server","http://127.0.0.1:"+str(8940+i),"--worker-index",str(i),"--workers","4",
                    "--output",output/(bench+"-worker-"+str(i)+".jsonl")],bench+"-worker-"+str(i),custom))
            while any(p.poll() is None for p in procs):
                budget()
                if any(p.poll() not in (None,0) for p in procs) or any(p.poll() is not None for p in servers):
                    raise RuntimeError("Replay worker/server failed; preserved logs and attempts")
                record_count=sum(path.read_bytes().count(b"\n") for path in output.glob("*-worker-*.jsonl") if not path.name.endswith(".prepare.jsonl"))
                run.log({"episodes_recorded":record_count,"elapsed_seconds":time.monotonic()-begun})
                state["records"]=record_count
                save()
                time.sleep(20)
            if any(p.returncode!=0 for p in procs):
                raise RuntimeError("Replay worker failed")
        rows=[json.loads(line) for path in output.glob("*-worker-*.jsonl") if not path.name.endswith(".prepare.jsonl") for line in path.read_text().splitlines()]
        good=[r for r in rows if r["status"]=="ok"]
        expected={(c["case_id"],n) for c in manifest["replay_cases"] for n in c["replay_steps"]}
        observed=[(r["historical_case_id"],r["flow_steps"]) for r in good]
        if len(observed)!=len(set(observed)) or set(observed)!=expected:
            raise ValueError("Replay coverage mismatch")
        if tree_hash(manifest["checkpoint"]["path"])[0]!=manifest["checkpoint"]["sha256"]:
            raise ValueError("Post-replay checkpoint changed")
        raw_paths=[p for p in output.glob("*-worker-*.jsonl") if not p.name.endswith(".prepare.jsonl")]
        audit_command=[runtime["simulator_python"],HERE/"replay.py","--manifest",args.manifest,
                       "--audit-records",*raw_paths,"--audit-output",output/"summary.json"]
        audit_proc=launch(audit_command,"independent-final-audit",dict(env,**runtime["simulator_common_env"]))
        while audit_proc.poll() is None:
            budget()
            time.sleep(5)
        if audit_proc.returncode != 0:
            raise RuntimeError("Independent final replay audit failed")
        summary=json.loads((output/"summary.json").read_text())
        artifact=wandb.Artifact("historical-replay-"+mh[:12],type="retrospective-replay")
        for path in [args.manifest,output/"summary.json",*raw_paths]:
            artifact.add_file(str(path),name=path.name)
        run.log_artifact(artifact)
        run.summary.update(summary)
        state.update(status="completed",summary=summary)
        save()
    except BaseException as error:
        state.update(status="failed",error=repr(error))
        save()
        raise
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
        for child in children:
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL)
                child.wait(timeout=10)
        for log in logs:
            log.close()
        run.finish(exit_code=0 if state["status"]=="completed" else 1)


if __name__=="__main__":
    main()

