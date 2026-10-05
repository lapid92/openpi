"""Descriptive selected-sample classes; never infer population rates or missing labels."""
import argparse,json
from collections import defaultdict,Counter
from pathlib import Path

def diagnostic(r):
 return r["cohort"]=="new" and r["benchmark"]=="libero" and r["condition_id"]=="libero:libero_10:5" and r["seed"]==3009 and r["init_index"]==18

def label(r): return r["baseline"]["label"]
def agreed_where_available(r,fields):
 overlap=r.get("independent_overlap")
 return not overlap or (overlap["label"]["confidence"] in {"high","medium"} and all(label(r)[f]==overlap["label"][f] for f in fields))
def adjudication_supports(r,outcome,stage=None):
 a=r.get("adjudication")
 return not a or (a["label"]["confidence"] in {"high","medium"} and a["label"]["observed_outcome"]==outcome and (stage is None or a["label"]["primary_stage"]==stage))
def clear_failure(r):
 l=label(r)
 return not diagnostic(r) and adjudication_supports(r,"not completed",l["primary_stage"]) and l["observed_outcome"]=="not completed" and l["primary_stage"] in {"approach","grasp","manipulation","placement","recovery"} and l["confidence"] in {"high","medium"} and agreed_where_available(r,["observed_outcome","primary_stage"])
def clear_completion(r):
 l=label(r)
 return adjudication_supports(r,"completed") and l["observed_outcome"]=="completed" and l["confidence"] in {"high","medium"} and agreed_where_available(r,["observed_outcome"])
def summarize(members):
 return {"cases":len({r["group_id"] for r in members}),"families":len({(r["suite"],r["family"]) for r in members}),"conditions":len({r["condition_id"] for r in members}),"distinct_initial_state_hashes":len({r["initial_state_sha256"] for r in members if r.get("initial_state_sha256")}),"distinct_stabilized_state_hashes":len({r["stabilized_state_sha256"] for r in members if r.get("stabilized_state_sha256")}),"groups":sorted({r["group_id"] for r in members}),"clip_ids":sorted({r["clip_id"] for r in members})}
def main():
 p=argparse.ArgumentParser();p.add_argument("--joined",required=True,type=Path);p.add_argument("--output",required=True,type=Path);a=p.parse_args()
 rows=json.loads(a.joined.read_text());groups=defaultdict(dict)
 for r in rows:groups[(r["cohort"],r["group_id"])][r["arm"]]=r
 classes=defaultdict(list);candidates=[];raw_disagreements=[];flags=[]
 for r in rows:
  if diagnostic(r):flags.append({"clip_id":r["clip_id"],"group_id":r["group_id"],"reason":"Preserved MuJoCo instability warning; not solely policy-attributable"})
  visual=label(r)["observed_outcome"]
  if visual!="unclear" and (visual=="completed")!=r["raw_success"]:
   raw_disagreements.append({"cohort":r["cohort"],"clip_id":r["clip_id"],"raw_success":r["raw_success"],"visible_outcome":visual,"source":r["source"]})
 for (cohort,gid),arms in groups.items():
  if 1 not in arms:continue
  first=arms[1];study=first.get("study","unspecified")
  for step,other in sorted(arms.items()):
   if step==1 or not first["paired_claim_eligible"] or not other["paired_claim_eligible"]:continue
   if not first["raw_success"] and other["raw_success"]:
    stage=label(first)["primary_stage"]
    base=(cohort,study,first["benchmark"],step,stage)
    classes[base+("baseline_failure_stage",)].append(first)
    if clear_failure(first) and clear_completion(other):
     classes[base+("clear_outcomes_no_observed_disagreement",)].append(first)
     candidates.append({"cohort":cohort,"study":study,"benchmark":first["benchmark"],"step":step,"stage":stage,"group_id":gid,"one":first["clip_id"],"other":other["clip_id"],"family":first["family"],"condition_id":first["condition_id"]})
   elif first["raw_success"] and not other["raw_success"]:
    classes[(cohort,study,first["benchmark"],step,label(other)["primary_stage"],"regression_failure_stage")].append(other)
   elif first["selection_class"] in {"never_rescued","failed_at_1_and_10_intermediates_unmeasured"}:
    name="never_rescued_control" if first["selection_class"]=="never_rescued" else "failed_at_1_and_10_control_intermediates_unmeasured"
    classes[(cohort,study,first["benchmark"],step,label(other)["primary_stage"],name)].append(other)
 out=[]
 for k,m in sorted(classes.items()):
  c=summarize(m)
  out.append(dict(cohort=k[0],study=k[1],benchmark=k[2],step=k[3],stage=k[4],evidence_rule=k[5],**c,meets_declared_new_rescue_threshold=k[0]=="new" and k[5]=="clear_outcomes_no_observed_disagreement" and k[4] in {"approach","grasp","manipulation","placement","recovery"} and c["cases"]>=10 and c["families"]>=3 and c["conditions"]>=5))
 result={"interpretation":"Outcome-stratified qualitative sample only; counts are not population rates, intervals, or causal mechanisms. Primary and overlap labels are retained. Clear-outcome rule requires medium/high confidence in baseline and any overlap label and no observed stage/outcome disagreement where overlap exists; it does not imply every pair was independently double-reviewed. Alternatives remain in label table.","classes":out,"clear_rescue_pairs":candidates,"visual_vs_numeric_disagreements":raw_disagreements,"simulator_diagnostic_flags":flags,"baseline_clips":len(rows)}
 with a.output.open("x") as f:json.dump(result,f,indent=2);f.write("\n")
if __name__=="__main__":main()
