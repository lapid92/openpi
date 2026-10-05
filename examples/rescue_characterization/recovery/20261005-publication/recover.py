"""Explicit publication recovery; never reruns the completed 9920-episode scan."""
import argparse,datetime,fcntl,hashlib,json,os
from pathlib import Path
import shutil,signal,subprocess,sys,time
from shard_records import digest,shard,verify,LIMIT

ROOT=Path("/volt/code/frozen-flow-study")
HERE=ROOT/"examples/rescue_characterization"
ART=Path("/volt/artifacts/rescue-characterization")
REC=ART/"recovery-20261005"
RUNS=ART/"runs"
PIN="e7537584d34855c24c1a38ca11da4f7e479b9bcbfc53d2cc7ec8f7ff46eadf9d"
DEST=HERE/"results"/PIN[:12]
REP_REC=HERE/"recovery"/"20261005-publication"
BRANCH="refs/heads/codex/pi05-rescue-characterization"
BACKUP="refs/heads/codex/recovery-backup-ae903b8"
PY=ROOT/".venv/bin/python"
STATE=REC/"state.json"

def now():return datetime.datetime.now(datetime.UTC).isoformat()
def git(*args):return subprocess.check_output(["git",*args],cwd=ROOT,text=True).strip()
def save(state):
 state["updated_at"]=now();tmp=STATE.with_suffix(".tmp")
 tmp.write_text(json.dumps(state,indent=2)+"\n");tmp.replace(STATE)
def require(test,message):
 if not test:raise RuntimeError(message)
def pinned():
 m=json.loads((HERE/"protocol.json").read_text())
 require(digest(HERE/"protocol.json")==PIN,"Main manifest changed")
 for name in ("protocol.json","replay-protocol.json"):
  value=json.loads((HERE/name).read_text())
  for path,sha in value["source_sha256"].items():require(digest(path)==sha,"Pinned source changed: "+path)
 return m
def idle(m):
 uuids=set(subprocess.check_output(["nvidia-smi","--query-gpu=uuid","--format=csv,noheader"],text=True).split())
 require(uuids==set(m["gpu_uuids"]),"GPU UUID allocation changed")
 require(not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip(),"GPU work active")
def remote():return git("ls-remote","origin",BRANCH).split()[0]
def guard_large(parent):
 data=git("rev-list","--objects","HEAD","^"+parent)
 proc=subprocess.run(["git","cat-file","--batch-check=%(objecttype) %(objectsize) %(rest)"],cwd=ROOT,
                     input=data+"\n",text=True,stdout=subprocess.PIPE,check=True)
 for line in proc.stdout.splitlines():
  fields=line.split(" ",2)
  if len(fields)>=2 and fields[0]=="blob":
   require(int(fields[1])<40_000_000,"Unpublished blob still >=40MB: "+line)
def command(state,argv,label):
 state["stage"]=label
 item={"label":label,"argv":list(map(str,argv)),"started_at":now()}
 state["commands"].append(item);save(state)
 with (REC/(label+".log")).open("a") as log:
  proc=subprocess.Popen(item["argv"],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  state["child_pid"]=proc.pid;save(state)
  code=proc.wait()
 item.update(returncode=code,ended_at=now());state.pop("child_pid",None);save(state)
 require(code==0,"Command failed; preserved log: "+label)
def archive_failure(state):
 backup=REC/"original-failure";backup.mkdir(exist_ok=False)
 for source in [ART/"pipeline-status.json",RUNS/"full-status.json",ART/"full-pipeline-supervisor.log"]:
  require(source.exists(),"Missing original failure evidence: "+str(source))
  target=backup/source.name;shutil.copy2(source,target)
  state.setdefault("original_evidence",[]).append({"source":str(source),"copy":str(target),"sha256":digest(target)})
 (backup/"original-commit.txt").write_text(git("show","--stat","--oneline",state["original_commit"])+"\n")
 (backup/"tracked-tree.txt").write_text(git("ls-tree","-r","--long",state["original_commit"])+"\n")
def snapshot_raw():
 index={"schema_version":1,"purpose":"Exact byte reconstruction of raw JSONL publication snapshots",
        "original_commit":git("rev-parse",BACKUP),"backup_ref_local_only":BACKUP,"source_runs":str(RUNS),
        "reconstruct_command":"python examples/rescue_characterization/recovery/20261005-publication/shard_records.py --index examples/rescue_characterization/results/"+PIN[:12]+"/raw-shards/index.json --output-dir /volt/artifacts/rescue-characterization/reconstructed-main",
        "files":[]}
 for path in sorted(DEST.glob("*-main-worker-*.jsonl")):
  source=RUNS/path.name
  require(digest(path)==digest(source),"Snapshot differs from durable raw source")
  folder=DEST/"raw-shards"/path.stem
  record=shard(path,folder)
  record.update(original_snapshot_path=str(path.relative_to(ROOT)),source_path=str(source),parts_directory=path.stem)
  index["files"].append(record)
  verify(record,folder)
  # Both original runs bytes and backup commit are already preserved.
  path.unlink()
 require(len(index["files"])==8,"Expected all eight main worker raw files")
 target=DEST/"raw-shards/index.json";target.write_text(json.dumps(index,indent=2)+"\n")
 return target
def prepare():
 require(not STATE.exists(),"Recovery state already exists; inspect before explicit further recovery")
 m=pinned();idle(m)
 require(not git("diff","--name-only") and not git("diff","--cached","--name-only"),"Tracked/staged changes present")
 original=git("rev-parse","HEAD");parent=git("rev-parse","HEAD^");upstream=remote()
 require(original.startswith("ae903b8") and parent==upstream and upstream.startswith("ddb0e077"),"Unpublished history differs from audited failure")
 require(all(x.startswith("examples/rescue_characterization/results/"+PIN[:12]+"/") for x in git("diff","--name-only",parent,original).splitlines()),"Unpublished commit contains unexpected source changes")
 failed=json.loads((RUNS/"full-status.json").read_text())
 require(failed["status"]=="failed" and "git" in failed["error"] and "push" in failed["error"],"Failure is not publication-only")
 for bench,n in (("libero",3200),("libero_plus",6720)):
  a=json.loads((RUNS/(bench+"-summary.json")).read_text());b=json.loads((RUNS/(bench+"-recording-audit.json")).read_text())
  require(a["audit"]=="passed" and a["completed_episodes"]==n and a["manifest_sha256"]==PIN,"Main evaluation evidence incomplete")
  require(b["status"]=="passed" and b["records"]==n and b["manifest_sha256"]==PIN,"Recording audit evidence incomplete")
 state={"status":"preparing","original_commit":original,"remote_parent":parent,"backup_ref":BACKUP,
        "manifest_sha256":PIN,"started_at":now(),"commands":[],"original_pipeline_rerun":False,"main_evaluations_rerun":False}
 existing_backup=subprocess.run(["git","rev-parse","--verify",BACKUP],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
 require(existing_backup.returncode!=0 or existing_backup.stdout.strip()==original,"Backup ref already points elsewhere")
 save(state);git("update-ref",BACKUP,original);archive_failure(state);save(state)
 index=snapshot_raw()
 counts=ART/"final-independent-counts"
 if counts.exists():
  for item in sorted(counts.iterdir()):
   if item.is_file() and item.suffix in (".json",".jsonl",".py",".md",".log"):
    require(item.stat().st_size<LIMIT,"Independent count artifact too large")
    target=DEST/"final-independent-counts"/item.name;target.parent.mkdir(exist_ok=True);shutil.copy2(item,target)
 REP_REC.mkdir(parents=True,exist_ok=False)
 for name in ("recover.py","shard_records.py","README.md"):
  shutil.copy2(REC/name,REP_REC/name)
 shutil.copytree(REC/"original-failure",REP_REC/"original-failure")
 (REP_REC/"PREPARATION.json").write_text(json.dumps({**state,"shard_index":str(index.relative_to(ROOT)),
    "shard_index_sha256":digest(index),"original_raw_retained":str(RUNS)},indent=2)+"\n")
 git("add",str(DEST.relative_to(ROOT)),str(REP_REC.relative_to(ROOT)))
 git("commit","--amend","--no-edit")
 require(git("rev-parse","HEAD^")==parent,"Amend changed parent")
 guard_large(parent);pinned()
 state.update(status="prepared_review_required",prepared_commit=git("rev-parse","HEAD"),shard_index_sha256=digest(index),
              recovery_script_sha256=digest(REC/"recover.py"),sharder_sha256=digest(REC/"shard_records.py"))
 save(state);print(json.dumps(state,indent=2))
def publish(state,label,message,paths=()):
 old=remote()
 for source,dest in paths:
  dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
 if paths:
  git("add",*[str(dest.relative_to(ROOT)) for _,dest in paths])
  if git("diff","--cached","--name-only"):git("commit","-m",message)
 guard_large(old)
 command(state,["git","push","origin","HEAD:"+BRANCH],label)
 require(remote()==git("rev-parse","HEAD"),"Published remote differs from local HEAD")
 state["last_published_commit"]=git("rev-parse","HEAD");save(state)
def continue_work():
 state=json.loads(STATE.read_text())
 require(state["status"]=="prepared_review_required","Continuation requires explicit reviewed prepared state; no automatic restart")
 require(git("rev-parse","HEAD")==state["prepared_commit"] and remote()==state["remote_parent"],"Publication base changed")
 require(digest(REC/"recover.py")==state["recovery_script_sha256"] and digest(REC/"shard_records.py")==state["sharder_sha256"],"Reviewed recovery source changed")
 require(not (ART/"replay/status.json").exists(),"Replay already attempted; refuse duplicate launch")
 require(not (ART/"independent-final-audit.json").exists(),"Independent main audit already exists; inspect before recovery")
 m=pinned();idle(m)
 for record in json.loads((DEST/"raw-shards/index.json").read_text())["files"]:
  verify(record,DEST/"raw-shards"/record["parts_directory"])
  require(digest(record["source_path"])==record["original_sha256"],"Durable raw original changed")
 require(digest(DEST/"raw-shards/index.json")==state["shard_index_sha256"],"Shard index changed")
 try:
  state["status"]="publication_recovery_running";save(state)
  publish(state,"publish-lossless-main-shards","Publish lossless main raw shards")
  # Correct only operational publication state; failed originals and errors remain archived.
  full=json.loads((RUNS/"full-status.json").read_text())
  full.update(status="published",published_commit=state["last_published_commit"],publication_recovery={
   "original_failed_commit":state["original_commit"],"backup_ref":BACKUP,"failed_status_path":str(REC/"original-failure/full-status.json"),
   "recovered_at":now(),"recovery_state":str(STATE),"no_evaluation_rerun":True})
  full["original_publication_error"]=full.pop("error")
  (RUNS/"full-status.json").write_text(json.dumps(full,indent=2)+"\n")
  state["status"]="independent_main_audit_running";save(state)
  command(state,[PY,HERE/"independent_audit.py","--phase","main"],"independent-main-audit")
  audit=ART/"independent-final-audit.json";result=json.loads(audit.read_text())
  require(result["status"]=="passed" and result["manifest_sha256"]==PIN,"Independent main gate failed")
  publish(state,"publish-independent-main-audit","Record independent main audit after publication recovery",
          [(audit,DEST/audit.name),(RUNS/"full-status.json",DEST/"full-status.json")])
  idle(m);pinned()
  require(not (ART/"replay/status.json").exists(),"Replay already attempted; refusing duplicate")
  state["status"]="historical_replay_running";state["replay_launch_authorized_once_at"]=now();save(state)
  command(state,[PY,"-u",HERE/"replay_supervise.py","--manifest",HERE/"replay-protocol.json","--output",ART/"replay"],"historical-replay-once")
  replay=ART/"replay";require(json.loads((replay/"status.json").read_text())["status"]=="completed","Replay final audit did not complete")
  rd=HERE/"results/historical-replay";rd.mkdir(parents=True,exist_ok=False)
  shard_index={"schema_version":1,"source_runs":str(replay),"files":[]}
  for source in sorted(replay.iterdir()):
   if not source.is_file() or source.suffix not in (".json",".jsonl",".md"):continue
   if source.stat().st_size>=40_000_000:
    require(source.suffix==".jsonl","Non-JSONL artifact too large for safe publication")
    entry=shard(source,rd/"raw-shards"/source.stem)
    entry.update(source_path=str(source),parts_directory=source.stem)
    shard_index["files"].append(entry)
   else:shutil.copy2(source,rd/source.name)
  if shard_index["files"]:(rd/"raw-shards/index.json").write_text(json.dumps(shard_index,indent=2)+"\n")
  git("add",str(rd.relative_to(ROOT)));git("commit","-m","Record independently audited historical replay artifacts")
  publish(state,"publish-historical-replay","Publish audited historical replay artifacts")
  state["status"]="numeric_and_replay_complete_review_pending";save(state)
  publish(state,"publish-recovery-provenance","Preserve publication recovery commands and provenance",[(STATE,REP_REC/"FINAL-STATE.json")])
  state["final_commit"]=git("rev-parse","HEAD");save(state)
 except BaseException as exc:
  current=json.loads(STATE.read_text())
  if current.get("status")=="interrupted":state=current
  else:state["status"]="failed";state["error"]=repr(exc)
  save(state);raise
def main():
 p=argparse.ArgumentParser();p.add_argument("mode",choices=("prepare","continue"));a=p.parse_args()
 REC.mkdir(parents=True,exist_ok=True)
 locks=[]
 for path in (REC/"recovery.lock",ART/"pipeline.lock",RUNS/"supervisor.lock"):
  f=path.open("a");fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 def interrupted(signum,frame):
  if STATE.exists():
   state=json.loads(STATE.read_text());state.update(status="interrupted",error="Signal "+str(signum)+"; inspect any recorded child PID before further recovery");save(state)
  raise SystemExit(128+signum)
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 prepare() if a.mode=="prepare" else continue_work()
if __name__=="__main__":main()

