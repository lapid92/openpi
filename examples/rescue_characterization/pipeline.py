"""Detached sequencing: independent smoke gate, full scan, final audit, historical replay."""
import datetime,fcntl,json,os,pathlib,shutil,subprocess,time,signal
ROOT=pathlib.Path("/volt/code/frozen-flow-study")
HERE=ROOT/"examples/rescue_characterization"
ART=pathlib.Path("/volt/artifacts/rescue-characterization")
RUNS=ART/"runs"
PY=ROOT/".venv/bin/python"
PIN="e7537584d34855c24c1a38ca11da4f7e479b9bcbfc53d2cc7ec8f7ff46eadf9d"
def main():
 lock=(ART/"pipeline.lock").open("w")
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (ART/"pipeline-status.json").exists():raise RuntimeError("Existing pipeline state; explicit recovery required, refusing duplicate launch")
 state={"status":"waiting_for_smoke","started_at":datetime.datetime.now(datetime.UTC).isoformat(),"manifest_sha256":PIN,"commands":[]}
 def save():
  state["updated_at"]=datetime.datetime.now(datetime.UTC).isoformat()
  tmp=ART/"pipeline-status.tmp";tmp.write_text(json.dumps(state,indent=2)+"\n");tmp.replace(ART/"pipeline-status.json")
 def run(argv,label):
  state["stage"]=label;state["commands"].append({"label":label,"argv":list(map(str,argv))});save()
  with (ART/(label+".log")).open("a") as log:
   subprocess.run(list(map(str,argv)),cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
 def idle():
  deadline=time.monotonic()+180
  while time.monotonic()<deadline:
   with (RUNS/"supervisor.lock").open("a") as f:
    try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:time.sleep(5);continue
   active=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip()
   if not active:return
   time.sleep(5)
  raise RuntimeError("Previous GPU supervisor still active")
 def publish(paths,message):
  dest=HERE/"results"/PIN[:12];dest.mkdir(parents=True,exist_ok=True)
  for p in paths:shutil.copy2(p,dest/p.name)
  subprocess.run(["git","add",str(dest.relative_to(ROOT))],cwd=ROOT,check=True)
  if subprocess.run(["git","diff","--cached","--quiet"],cwd=ROOT).returncode:
   subprocess.run(["git","commit","-m",message],cwd=ROOT,check=True)
  subprocess.run(["git","push","origin","HEAD:refs/heads/codex/pi05-rescue-characterization"],cwd=ROOT,check=True)
 def interrupted(signum, frame):
  state["status"]="interrupted";state["error"]="Signal "+str(signum)+"; inspect child supervisor before any recovery"
  save()
  raise SystemExit(128+signum)
 signal.signal(signal.SIGTERM,interrupted)
 signal.signal(signal.SIGINT,interrupted)
 try:
  save();deadline=time.monotonic()+6*3600
  while True:
   p=RUNS/"smoke-status.json"
   if p.exists():
    smoke=json.loads(p.read_text())
    if smoke.get("status")=="failed":raise RuntimeError("Smoke failed; see smoke-status.json")
    if smoke.get("status")=="published":break
   if time.monotonic()>deadline:raise RuntimeError("Smoke publication wait exceeded6h")
   time.sleep(30)
  state["status"]="independent_smoke_audit";save()
  run([PY,HERE/"independent_audit.py","--phase","smoke"],"independent-smoke-gate")
  audit=json.loads((ART/"independent-smoke-audit.json").read_text())
  if audit["status"]!="passed" or audit["manifest_sha256"]!=PIN:raise RuntimeError("Independent smoke gate failed")
  publish([ART/"independent-smoke-audit.json"],"Record independent smoke gate before full rescue scan")
  idle()
  state["status"]="full_scan_running";save()
  run([PY,"-u",HERE/"supervise.py","--manifest",HERE/"protocol.json","--output",RUNS,"--phase","full"],"full-pipeline-supervisor")
  if json.loads((RUNS/"full-status.json").read_text()).get("status")!="published":raise RuntimeError("Full scan not published")
  state["status"]="independent_final_audit";save()
  run([PY,HERE/"independent_audit.py","--phase","main"],"independent-final-gate")
  publish([ART/"independent-final-audit.json"],"Record independent full rescue-scan audit")
  idle()
  state["status"]="historical_replay_running";save()
  run([PY,"-u",HERE/"replay_supervise.py","--manifest",HERE/"replay-protocol.json","--output",ART/"replay"],"historical-replay-supervisor")
  r=ART/"replay"
  if json.loads((r/"status.json").read_text()).get("status")!="completed":raise RuntimeError("Historical replay not completed")
  dest=HERE/"results"/"historical-replay";dest.mkdir(parents=True,exist_ok=True)
  for p in r.glob("*"):
   if p.is_file() and p.suffix in (".json",".jsonl",".md"):shutil.copy2(p,dest/p.name)
  subprocess.run(["git","add",str(dest.relative_to(ROOT))],cwd=ROOT,check=True)
  if subprocess.run(["git","diff","--cached","--quiet"],cwd=ROOT).returncode:subprocess.run(["git","commit","-m","Record exact-equivalence audited historical replays"],cwd=ROOT,check=True)
  subprocess.run(["git","push","origin","HEAD:refs/heads/codex/pi05-rescue-characterization"],cwd=ROOT,check=True)
  state["status"]="numeric_and_replay_complete_review_pending";save()
 except BaseException as exc:
  if state["status"]!="interrupted":state["status"]="failed";state["error"]=repr(exc)
  save();raise
if __name__=="__main__":main()
