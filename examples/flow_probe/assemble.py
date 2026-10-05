"""Freeze identities and prospective decisions after independent review."""
import copy
import datetime
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"rescue_characterization"))
from protocol import file_hash,tree_hash
def main():
    output=HERE/"protocol.json"
    if output.exists():raise RuntimeError("Refuse frozen manifest overwrite")
    old=json.loads((HERE.parent/"rescue_characterization/protocol.json").read_text())
    p=copy.deepcopy(old)
    p["study"]="pi05-two-evaluation-ranking"
    p["base_git_commit"]="74160dfe761ab96045ec62811a6676552f05a0e8"
    p["predeclared_at_utc"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
    p["branch"]="codex/pi05-two-evaluation-ranking"
    p["benchmarks"]={b:json.loads((HERE/(b+"-conditions.json")).read_text()) for b in ("libero","libero_plus")}
    p["head"]=json.loads((HERE.parent/"selective_flow/setup/head.json").read_text())
    p["probe_artifact_dir"]="/volt/artifacts/flow-probe/probes"
    p["recording"]["root"]="/volt/artifacts/flow-probe/media"
    p["stopping"].update(maximum_scored_episodes={b:s["max_episodes"] for b,s in p["benchmarks"].items()},maximum_full_wall_hours=96,maximum_smoke_wall_hours=8,minimum_free_disk_gib=100,no_training=True,outcome_based=False)
    p["budget"]={"main_episodes":9760,"smoke_episodes":44,"estimated_full_wall_hours":[44,72],"allocated_H100_hours":[176,288],"basis":"Audited9920episode scan43.92hours; initial-only isolated probe adds2velocityevals per episode. No extrapolatedspeedupclaim."}
    p["signal"]={"name":"midpoint-relative-velocity-rms-v1","times":[1.,.5],"midpoint":"z-0.5*v1","formula":"RMS(vmid-v1)/max(RMS(v1),1e-6)","shape":[10,7],"dtype":"float32","higher_predicts_rescue":True,"chunk_index":0,"probe_evaluations":2,"changes_actions":False}
    p["analysis"]={"bootstrap_replicates":10000,"bootstrap_seed":20261005,"primary_cluster":{"libero":"suite/family","libero_plus":"suite/family"},"sensitivity_cluster":"condition_id","primary_steps":4,"primary_extra_cap_budget":.75,"extra_cap_budgets":[.25,.5,.75,1.],"comparator_overhead":{"score":2,"sigma":1,"severity":0},"decision":"Bothbenchmarks pass all PROTOCOL.md criteria else STOP; neverbuildcontrollerinthisstudy"}
    p["review"]={"require_independent_smoke":True,"full_outcome_and_metric_audit":True,"visual_review":"Not required for scalar ranking; preserve allvideos/traces for diagnostic errors."}
    p["protocol_document_sha256"]=file_hash(HERE/"PROTOCOL.md")
    p["source_sha256"]=dict(old["source_sha256"])
    for path,digest in p["source_sha256"].items():
        if file_hash(path)!=digest:raise RuntimeError("Pinned earlier source changed: "+path)
    for path in sorted(HERE.glob("*.py")):p["source_sha256"][str(path)]=file_hash(path)
    for b in p["benchmarks"]:
        for suffix in ("-conditions.json","-conditions.separation.json"):
            path=HERE/(b+suffix)
            if path.exists():p["source_sha256"][str(path)]=file_hash(path)
    p["source_sha256"][str(HERE/"PROTOCOL.md")]=file_hash(HERE/"PROTOCOL.md")
    if tree_hash(p["checkpoint"]["path"])[0]!=p["checkpoint"]["sha256"]:raise RuntimeError("Checkpointchanged")
    if file_hash(p["head"]["path"])!=p["head"]["sha256"]:raise RuntimeError("Headchanged")
    for spec in p["benchmarks"].values():
        if tree_hash(spec["installed_assets_path"])[0]!=spec["installed_assets_sha256"]:raise RuntimeError("Assets changed")
    with output.open("x") as f:json.dump(p,f,sort_keys=True,separators=(",",":"));f.write("\n")
    output.with_suffix(".sha256").write_text(file_hash(output)+"  protocol.json\n")
    print(json.dumps({"sha256":file_hash(output),"episodes":p["stopping"]["maximum_scored_episodes"]}))
if __name__=="__main__":main()
