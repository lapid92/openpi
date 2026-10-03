"""Freeze manifest once, after review and before any new simulations."""
import copy,datetime,json,pathlib,subprocess
from protocol import file_hash,tree_hash
ROOT=pathlib.Path(__file__).resolve().parents[2]
HERE=pathlib.Path(__file__).resolve().parent
def main():
 out=HERE/"protocol.json"
 if out.exists():raise RuntimeError("Refuse frozen manifest overwrite")
 old=json.loads((ROOT/"examples/frozen_flow/protocol.json").read_text())
 p=copy.deepcopy(old)
 p["study"]="pi05-rescue-characterization"
 p["predeclared_at_utc"]=datetime.datetime.now(datetime.UTC).isoformat()
 p["base_git_commit"]=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
 p["benchmarks"]={b:json.loads((HERE/(b+"-conditions.json")).read_text()) for b in ("libero","libero_plus")}
 p.pop("robocasa",None);p.pop("execution_amendment",None)
 p["stopping"].update(maximum_scored_episodes={b:s["max_episodes"] for b,s in p["benchmarks"].items()},smoke_episodes=44,maximum_full_wall_hours=96,minimum_free_disk_gib=100,outcome_based=False,no_training=True)
 p["analysis"].update(bootstrap_seed=20261003,primary_cluster={"libero":"condition_id","libero_plus":"family"},sensitivity_cluster="condition_id",first_success="Among one-step failures: first success in2,4,10 or never. Missing arms unknown; all16 vectors retained.",repeatability="Descriptive recurring class requires >=10 rescues across >=3 distinct base families and >=5 conditions in new data; otherwise limited evidence. Count regressions and never-rescued controls, no causal visual attribution.",minimum_one_step_failures=50)
 p["recording"]={"root":"/volt/artifacts/rescue-characterization/media","every_episode":True,"fps":20,"frames":"initial stabilized observation and every executed action observation; agent and wrist views side by side","numeric":"all predicted chunks, executed actions, eef/gripper state, per-step NumPy RNG hashes; no additional environment calls"}
 p["budget"]={"main_episodes":9920,"smoke_episodes":44,"estimated_full_wall_hours":[40,70],"estimated_allocated_H100_hours":[160,280],"estimated_video_trace_storage_gib":[30,150],"basis":"Previous1920episodes took7.24h on4H100; linear9920 scale37.4h, extra harder failures and video overhead widen estimate; no early stopping on outcomes."}
 p["settings"]["episode_latency"]+="; includes video and trace instrumentation, not comparable to old episode timing"
 p["review"]={"rubric":"approach, grasp, manipulation, placement, recovery, unclear; labels require visible events and trace time references","blind":"Random opaque clip IDs with separate restricted mapping; reviewer sees no step arm; context can imperfectly reveal outcome. Mark unreviewed, do not fabricate labels.","selection":"All rescues/regressions and all never-rescued failures retained. Prioritize a condition/family-matched never-rescued sample for independent visual review; report coverage."}
 assert tree_hash(p["checkpoint"]["path"])[0]==p["checkpoint"]["sha256"]
 for path,digest in old["source_sha256"].items():
  if file_hash(path)!=digest:raise RuntimeError("Old source changed: "+path)
 for spec in p["benchmarks"].values():
  if tree_hash(spec["installed_assets_path"])[0]!=spec["installed_assets_sha256"]:raise RuntimeError("Assets changed")
 for f in sorted(HERE.glob("*.py")):p["source_sha256"][str(f)]=file_hash(f)
 out.write_text(json.dumps(p,sort_keys=True,separators=(",",":"))+"\n")
 out.with_suffix(".sha256").write_text(file_hash(out)+"  protocol.json\n")
 print(json.dumps({"sha256":file_hash(out),"episodes":p["stopping"]["maximum_scored_episodes"],"smoke":44}))
if __name__=="__main__":main()
