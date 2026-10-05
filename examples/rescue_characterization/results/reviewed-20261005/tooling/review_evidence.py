"""Strict post-label join and blind cross-review planning. Never infer missing labels."""
import argparse
from collections import Counter,defaultdict
import hashlib,json,math,datetime
from pathlib import Path

ART=Path("/volt/artifacts/rescue-characterization")
EXPECTED={"new":272,"historical":116}
STAGES={"approach","grasp","manipulation","placement","recovery","unclear","completed"}
OUTCOMES={"completed","not completed","unclear"}
CONFIDENCE={"high","medium","low"}
REQUIRED=("clip_id","observed_outcome","primary_stage","confidence","decisive_indices","observation","interpretation","full_video_viewed")

def require(value,message):
 if not value:raise ValueError(message)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write_new(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open("x") as f:json.dump(value,f,indent=2);f.write("\n")
def key(cohort,clip):return cohort+"/"+clip
def source_rows(refs):
 wanted=defaultdict(dict);out={}
 for ref in refs:wanted[ref["path"]][ref["line"]]=ref
 for name,lines in wanted.items():
  require(all(sha(name)==r["file_sha256"] for r in [next(iter(lines.values()))]),"Raw source hash differs")
  require(len({r["file_sha256"] for r in lines.values()})==1,"Contradictory source file hashes")
  with open(name,"rb") as f:
   for n,raw in enumerate(f,1):
    if n in lines:
     body=raw.rstrip(b"\r\n")
     require(hashlib.sha256(body).hexdigest()==lines[n]["record_sha256"],"Raw record hash differs")
     out[(name,n)]=json.loads(body)
  require(all((name,n) in out for n in lines),"Missing referenced raw line")
 return out
def catalog(art=ART):
 result={};evidence=[];old_manifests={}
 for cohort,total in EXPECTED.items():
  root=art/("visual-review-"+cohort);mapping_path=root/"private/unblinding.json"
  packets_path=root/"blind/packets.json"
  mappings=read(mapping_path);packets=read(packets_path)
  public={c["clip_id"]:g["group_id"] for g in packets for c in g["clips"]}
  require(len(public)==total and len(mappings)==total,"Unexpected packet size")
  require({m["clip_id"] for m in mappings}==set(public),"Packet/mapping identity differs")
  refs=[m["source"] if cohort=="new" else m["original_source"] for m in mappings]
  if cohort=="historical":refs += [m["replay_source"] for m in mappings]
  raws=source_rows(refs)
  if cohort=="historical":
   coverage_path=root/"private/coverage.json";coverage=read(coverage_path)
   summary_path=art/"replay/summary.json"
   require(sha(summary_path)==coverage["summary_sha256"],"Historical replay summary changed")
   summary=read(summary_path)
   require(summary["status"]=="passed" and summary["equivalence"]=={"exact_replay":116,"reconstruction-not-equivalent":432},"Historical exact gate differs")
   evidence += [{"path":str(p),"sha256":sha(p)} for p in (coverage_path,summary_path)]
  for m in mappings:
   require(public[m["clip_id"]]==m["group_id"],"Group mapping differs")
   ref=m["source"] if cohort=="new" else m["original_source"]
   row=raws[(ref["path"],ref["line"])]
   require(row["status"]=="ok" and row["flow_steps"]==m["arm"] and row["success"]==m["success"],"Raw outcome/arm identity differs")
   eligible=True
   if cohort=="historical":
    rr=m["replay_source"];replay=raws[(rr["path"],rr["line"])]
    require(replay["equivalence"]["classification"]=="exact_replay" and replay["equivalence"]["visual_attribution_eligible"] is True,"Ineligible historical replay")
    require(replay["original_source"]==ref and replay["flow_steps"]==m["arm"],"Replay mapping differs")
    eligible=m["coverage"]["full_declared_case_exact"]
   severity=row.get("severity",m.get("severity"))
   if cohort=="historical" and severity is None:
    study=m["historical_study"]
    if study not in old_manifests:
     path=Path("/volt/code/frozen-flow-study/examples")/study/"protocol.json"
     require(sha(path)==row["manifest_sha256"],"Historical metadata manifest changed")
     old_manifests[study]=read(path)
     evidence.append({"path":str(path),"sha256":sha(path)})
    conditions=old_manifests[study]["benchmarks"][row["benchmark"]]["conditions"]
    condition=next(c for c in conditions if c["condition_id"]==row["condition_id"])
    require(condition["task_name"]==row["task_name"],"Historical task metadata differs")
    severity=condition["severity"]
   kind=m["selection_class"] if cohort=="new" else m["coverage"]["original_selection_class"]
   original_kind=kind
   if cohort=="historical" and m.get("historical_study")=="selective_flow" and kind=="never_rescued":kind="failed_at_1_and_10_intermediates_unmeasured"
   result[key(cohort,m["clip_id"])]=dict(cohort=cohort,clip_id=m["clip_id"],group_id=m["group_id"],arm=m["arm"],
    raw_success=row["success"],selection_class=kind,original_selection_class=original_kind,benchmark=row["benchmark"],suite=row["suite"],family=row["family"],
    condition_id=row["condition_id"],category=row["category"],severity=severity,
    seed=row["seed"],init_index=row["init_index"],initial_state_sha256=row["initial_state_sha256"],stabilized_state_sha256=row["stabilized_state_sha256"],study=m.get("historical_study","rescue_characterization"),tested_arms=[1,10] if m.get("historical_study")=="selective_flow" else [1,2,4,10],paired_claim_eligible=eligible,source=ref,max_frame_index=row["policy_steps"],
    replay_source=m.get("replay_source"),recording=m["recording"])
  evidence += [{"path":str(p),"sha256":sha(p)} for p in (mapping_path,packets_path)]
 return result,evidence

def initial_assignments(new_packets,historical_packets,new_reviewers,historical_reviewers):
 require(len(new_reviewers)==2 and len(historical_reviewers)==2,"Exactly two baseline reviewers per cohort required")
 result={}
 for cohort,path,reviewers in (("new",new_packets,new_reviewers),("historical",historical_packets,historical_reviewers)):
  packets=sorted(read(path),key=lambda g:g["group_id"]);cut=34 if cohort=="new" else (len(packets)+1)//2
  if cohort=="new":require(len(packets)==68,"Expected fixed new packet grouping")
  for i,g in enumerate(packets):
   for c in g["clips"]:result[key(cohort,c["clip_id"])]=reviewers[int(i>=cut)]
 return {"assignments":result,"new_group_rule":"primary first34, secondary last34 sorted by opaque group_id",
         "historical_group_rule":"first ceil(n/2), remainder; explicit reviewer IDs required"}

def normalize_label(raw):
 row=dict(raw)
 aliases={"decisive_indices":"decisive_frame_action_indices","observation":"literal_observation",
          "reviewed_at":"reviewed_at_utc"}
 for canonical,alias in aliases.items():
  if canonical not in row and alias in raw:row[canonical]=raw[alias]
 indices=row.get("decisive_indices")
 if isinstance(indices,list) and indices and all(isinstance(v,dict) for v in indices):
  require(all(type(v.get("frame")) is int and v.get("last_executed_action")==v["frame"]-1 for v in indices),"Frame/action dictionary convention differs")
  row["decisive_indices"]=[v["frame"] for v in indices]
 if row.get("observed_outcome")=="not_completed":row["observed_outcome"]="not completed"
 coverage=raw.get("viewing_coverage",raw.get("actual_viewing_coverage"))
 require(isinstance(coverage,dict),"Explicit actual viewing coverage required")
 coverage=dict(coverage)
 if "additional_video_frames" not in coverage and "dense_frame_action_indices" in coverage:
  coverage["additional_video_frames"]=coverage["dense_frame_action_indices"]
 if "continuous_full_video" not in coverage and "full_video_viewed" in coverage:
  coverage["continuous_full_video"]=coverage["full_video_viewed"]
 if "alternative_label" in raw and raw["alternative_label"] is None:row["alternative_label"]=""
 row["viewing_coverage"]=coverage
 frames=[]
 for name in ("contact_sheet_frames","additional_video_frames"):
  values=coverage.get(name,[])
  require(isinstance(values,list) and all(type(n) is int and n>=0 for n in values),"Coverage frame indices must be explicit integer lists")
  frames.extend(values)
 row["viewed_indices"]=sorted(set(frames))
 reported="continuous_full_video" in coverage
 if reported:require(type(coverage["continuous_full_video"]) is bool,"continuous_full_video must be Boolean")
 row["full_video_viewed"]=coverage.get("continuous_full_video",False)
 row["full_video_viewed_reported"]=reported
 # full_video_necessary is a recommendation, never evidence that viewing occurred.
 return row

def validate_label(row):
 require(all(k in row for k in REQUIRED),"Missing rubric fields: "+str([k for k in REQUIRED if k not in row]))
 require(row["primary_stage"] in STAGES and row["observed_outcome"] in OUTCOMES and row["confidence"] in CONFIDENCE,"Invalid rubric category")
 require(type(row["full_video_viewed"]) is bool,"full_video_viewed must be Boolean")
 require(row.get("viewed_indices") or row["full_video_viewed"],"Actual viewed frame indices or explicit full-video coverage required")
 require(all(n in row["viewed_indices"] for n in row["decisive_indices"]) or row["full_video_viewed"],"Decisive frames absent from declared viewing coverage")
 require(isinstance(row.get("alternative_label"),(str,list)),"Alternative label field required; explicit empty string/list allowed")
 times=row.get("decisive_timestamps_seconds")
 require(isinstance(times,list) and all(isinstance(t,(int,float)) and not isinstance(t,bool) and math.isfinite(t) and t>=0 for t in times),"Explicit finite decisive timestamps required")
 require(isinstance(row.get("reviewed_at"),str),"Review timestamp required")
 datetime.datetime.fromisoformat(row["reviewed_at"].replace("Z","+00:00"))
 require(isinstance(row["decisive_indices"],list) and all(type(n) is int and n>=0 for n in row["decisive_indices"]),"Invalid decisive indices")
 require(len(row["decisive_indices"])>0 and len(times)==len(row["decisive_indices"]) and all(abs(n/20-t)<=0.026 for n,t in zip(row["decisive_indices"],times)),"Decisive timestamps must match viewed frames at 20fps")
 require(isinstance(row["observation"],str) and row["observation"].strip(),"Literal observation required")
 require(isinstance(row["interpretation"],str),"Interpretation must be separate text")
 require(row["observed_outcome"]!="completed" or row["primary_stage"] in ("completed","unclear"),"Completed outcome cannot receive a failure stage")
 require(row["primary_stage"]!="completed" or row["observed_outcome"]=="completed","Completed stage contradicts outcome")
 if row.get("stage_not_observed"):
  require(row["primary_stage"]=="unclear","Unobserved stage must be explicitly labelled unclear; no inferred fill")
 require(not any(k in row for k in ("arm","flow_steps","raw_success","selection_class")),"Blind label contains forbidden unblinded metadata")
 return row

def load_labels(specs):
 result={};files=[]
 for spec in specs:
  require(set(("cohort","reviewer","path")).issubset(spec),"Label source must name cohort/reviewer/path")
  path=Path(spec["path"]);count=0;file_sha=sha(path)
  for n,line in enumerate(path.read_text().splitlines(),1):
   if not line.strip():continue
   raw=json.loads(line);row=validate_label(normalize_label(raw));k=key(spec["cohort"],row["clip_id"])
   require(k not in result,"Duplicate baseline clip: "+k)
   result[k]={"label":row,"original_label":raw,"reviewer":spec["reviewer"],"label_source":{"path":str(path),"line":n,"file_sha256":file_sha}}
   count+=1
  files.append({**spec,"sha256":file_sha,"labels":count})
 return result,files

def lock_baseline(spec,assignment,catalog_data,evidence,output):
 labels,files=load_labels(spec)
 require(set(labels)==set(catalog_data),"Baseline missing/extra clips: missing="+str(len(set(catalog_data)-set(labels)))+", extra="+str(len(set(labels)-set(catalog_data))))
 require(set(assignment)==set(catalog_data),"Assignment coverage differs")
 for k,r in labels.items():
  require(r["reviewer"]==assignment[k],"Baseline reviewer assignment differs: "+k)
  require(all(n<=catalog_data[k]["max_frame_index"] for n in r["label"]["viewed_indices"]+r["label"]["decisive_indices"]),"Viewing/frame indices exceed clip length")
 lock={"status":"baseline_locked","files":files,"catalog_evidence":evidence,"assignments":assignment,
       "clips":len(labels),"cohort_counts":dict(Counter(k.split("/")[0] for k in labels))}
 write_new(output,lock)
 return lock

def cross_plan(cat,assignments,routes,baseline=None):
 require(set(assignments)==set(cat),"Complete assignment map required")
 families=defaultdict(list)
 for k,m in cat.items():families[(m["cohort"],m["benchmark"],m["suite"],m["family"])].append(k)
 chosen={}
 for members in families.values():
  k=min(members,key=lambda x:hashlib.sha256(("20261003:"+x).encode()).hexdigest())
  chosen[k]={"family"}
 if baseline is not None:
  require(set(baseline)==set(cat),"Baseline must be complete for low/unclear supplement")
  for k,r in baseline.items():
   label=r["label"]
   if label["confidence"]=="low" or label["primary_stage"]=="unclear" or label["observed_outcome"]=="unclear":
    chosen.setdefault(k,set()).add("low_or_unclear")
 public=[];private=[]
 for k in sorted(chosen):
  original=assignments[k];reviewer=routes.get(original)
  require(reviewer is not None and reviewer!=original,"Independent swapped reviewer route required for "+original)
  public.append({"cohort":cat[k]["cohort"],"clip_id":cat[k]["clip_id"],"reviewer":reviewer})
  private.append({"key":k,"reasons":sorted(chosen[k]),"initial_reviewer":original,"independent_reviewer":reviewer})
 return {"assignments":public},{"coverage":private,"selected_family_clusters":len(families),"target_clips":len(public)}

def check_lock(lock,cat,evidence):
 require(lock["status"]=="baseline_locked" and lock["clips"]==len(cat),"Baseline lock invalid")
 for f in lock["files"]:require(sha(f["path"])==f["sha256"],"Locked baseline file changed")
 require(lock["catalog_evidence"]==evidence,"Private/public catalog changed after lock")
 labels,_=load_labels(lock["files"])
 require(set(labels)==set(cat),"Locked baseline incomplete")
 return labels

def aggregate(joined):
 result={}
 for field in ("benchmark","selection_class","family","condition_id","category","severity"):
  counts=defaultdict(Counter)
  for row in joined:
   lab=row["baseline"]["label"];group="/".join((row["cohort"],str(row[field])))
   counts[group].update(clips=1)
   counts[group]["stage:"+lab["primary_stage"]]+=1
   counts[group]["confidence:"+lab["confidence"]]+=1
  result[field]={k:dict(v) for k,v in sorted(counts.items())}
 return result

def comparison_kind(a,b):
 if not a["raw_success"] and b["raw_success"]:return "rescue"
 if a["raw_success"] and not b["raw_success"]:return "regression"
 if not a["raw_success"] and not b["raw_success"]:
  if a["selection_class"]=="never_rescued":return "never_rescued_control"
  if a["selection_class"]=="failed_at_1_and_10_intermediates_unmeasured":return "failed_at_1_and_10_control_intermediates_unmeasured"
  return "both_fail"
 return "both_succeed"

def join_report(cat,baseline,overlap,plan,adjudications):
 joined=[{**m,"baseline":baseline[k],"independent_overlap":overlap.get(k),"adjudication":adjudications.get(k)} for k,m in sorted(cat.items())]
 planned={key(x["cohort"],x["clip_id"]):x["reviewer"] for x in plan["assignments"]}
 require(set(overlap)<=set(planned),"Unplanned overlap labels")
 for k,r in overlap.items():
  require(r["reviewer"]==planned[k] and r["reviewer"]!=baseline[k]["reviewer"],"Overlap reviewer not independent")
  require(all(n<=cat[k]["max_frame_index"] for n in r["label"]["viewed_indices"]+r["label"]["decisive_indices"]),"Overlap frame exceeds clip length")
 require(set(adjudications)<=set(overlap),"Adjudication requires preserved independent overlap")
 pairs=[];bygroup=defaultdict(dict)
 for row in joined:bygroup[(row["cohort"],row["group_id"])][row["arm"]]=row
 for (cohort,group),arms in bygroup.items():
  if 1 not in arms:continue
  for n in (2,4,10):
   if n not in arms:continue
   a,b=arms[1],arms[n]
   if not (a["paired_claim_eligible"] and b["paired_claim_eligible"]):continue
   kind=comparison_kind(a,b)
   pairs.append({"cohort":cohort,"group_id":group,"step":n,"comparison":kind,
    "benchmark":a["benchmark"],"family":a["family"],"condition_id":a["condition_id"],"selection_class":a["selection_class"],
    "one_step_stage":a["baseline"]["label"]["primary_stage"],"other_stage":b["baseline"]["label"]["primary_stage"],
    "clip_ids":[a["clip_id"],b["clip_id"]],"sources":[a["source"],b["source"]]})
 agreement={}
 for field in ("observed_outcome","primary_stage","confidence"):
  n=len(overlap);matches=sum(baseline[k]["label"][field]==v["label"][field] for k,v in overlap.items())
  confusion=Counter(baseline[k]["label"][field]+" -> "+v["label"][field] for k,v in overlap.items())
  agreement[field]={"reviewed":n,"agreed":matches,"raw_agreement":matches/n if n else None,"confusion":dict(confusion)}
 low={k for k,v in baseline.items() if v["label"]["confidence"]=="low" or v["label"]["primary_stage"]=="unclear" or v["label"]["observed_outcome"]=="unclear"}
 family_sets=defaultdict(set)
 for k,m in cat.items():family_sets[(m["cohort"],m["benchmark"],m["suite"],m["family"])].add(k)
 family_covered=sum(bool(keys&set(overlap)) for keys in family_sets.values())
 complete=set(overlap)==set(planned) and low<=set(overlap) and family_covered==len(family_sets)
 historical_groups=[list(arms.values()) for (cohort,_),arms in bygroup.items() if cohort=="historical"]
 full_historical=sum(all(row["paired_claim_eligible"] for row in group) for group in historical_groups)
 partial_historical=len(historical_groups)-full_historical
 summary={"status":"complete" if complete else "cross_review_incomplete","baseline_clips":len(baseline),
  "planned_overlap":len(planned),"completed_overlap":len(overlap),"missing_overlap":sorted(set(planned)-set(overlap)),
  "low_or_unclear_clips":len(low),"low_or_unclear_cross_reviewed":len(low&set(overlap)),
  "selected_family_clusters":len(family_sets),"cross_reviewed_family_clusters":family_covered,
  "agreement":agreement,"adjudications":len(adjudications),"historical_full_cases":full_historical,"historical_partial_cases":partial_historical,
  "historical_excluded_arms":432,"interpretation":"Outcome-stratified qualitative review; no population rates or unobserved stage inference",
  "baseline_labels_preserved":True,"distributions":aggregate(joined)}
 return joined,pairs,summary

def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest="mode",required=True)
 a=sub.add_parser("initial-plan");a.add_argument("--new-reviewers",nargs=2,required=True);a.add_argument("--historical-reviewers",nargs=2,required=True);a.add_argument("--output",required=True,type=Path)
 a=sub.add_parser("lock");a.add_argument("--sources",required=True,type=Path);a.add_argument("--assignments",required=True,type=Path);a.add_argument("--output",required=True,type=Path)
 a=sub.add_parser("cross-plan");a.add_argument("--assignments",required=True,type=Path);a.add_argument("--routes",required=True,type=Path);a.add_argument("--lock",type=Path);a.add_argument("--output",required=True,type=Path);a.add_argument("--private-output",required=True,type=Path)
 a=sub.add_parser("join");a.add_argument("--lock",required=True,type=Path);a.add_argument("--overlap-sources",required=True,type=Path);a.add_argument("--cross-plan",required=True,type=Path);a.add_argument("--adjudications",type=Path);a.add_argument("--output",required=True,type=Path)
 args=p.parse_args()
 if args.mode=="initial-plan":
  result=initial_assignments(ART/"visual-review-new/blind/packets.json",ART/"visual-review-historical/blind/packets.json",args.new_reviewers,args.historical_reviewers)
  write_new(args.output,result);return
 cat,evidence=catalog()
 if args.mode=="lock":lock_baseline(read(args.sources),read(args.assignments)["assignments"],cat,evidence,args.output)
 elif args.mode=="cross-plan":
  labels=check_lock(read(args.lock),cat,evidence) if args.lock else None
  public,private=cross_plan(cat,read(args.assignments)["assignments"],read(args.routes),labels)
  write_new(args.output,public);write_new(args.private_output,private)
 else:
  baseline=check_lock(read(args.lock),cat,evidence)
  overlaps,_=load_labels(read(args.overlap_sources))
  adjudications=read(args.adjudications) if args.adjudications else {}
  for k,v in adjudications.items():
   require("rationale" in v and v["rationale"].strip(),"Adjudication needs rationale")
   v["original_label"]=v["label"]
   v["label"]=validate_label(normalize_label(v["label"]))
   require(v["label"]["clip_id"]==k.split("/",1)[1],"Adjudication clip mismatch")
  joined,pairs,summary=join_report(cat,baseline,overlaps,read(args.cross_plan),adjudications)
  summary["baseline_lock_sha256"]=sha(args.lock)
  summary["baseline_lock_path"]=str(args.lock)
  summary["cross_plan_sha256"]=sha(args.cross_plan)
  summary["overlap_source_files"]=[{**f,"sha256":sha(f["path"])} for f in read(args.overlap_sources)]
  args.output.mkdir(parents=True,exist_ok=False)
  for name,value in (("label-table.json",joined),("paired-stages.json",pairs),("review-summary.json",summary)):
   write_new(args.output/name,value)
if __name__=="__main__":main()

