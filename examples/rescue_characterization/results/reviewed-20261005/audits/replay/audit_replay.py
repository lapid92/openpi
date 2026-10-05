"""Independent replay coverage/equivalence audit; no evaluation launches."""
import collections,hashlib,json,pathlib,sys,importlib.util
ROOT=pathlib.Path("/volt/code/frozen-flow-study");HERE=ROOT/"examples/rescue_characterization"
RUN=pathlib.Path("/volt/artifacts/rescue-characterization/replay")
OUT=pathlib.Path("/volt/artifacts/rescue-characterization/final-independent-replay")
def digest(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def need(ok,msg):
 if not ok:raise ValueError(msg)
def run():
 OUT.mkdir(exist_ok=True,parents=True)
 mp=HERE/"replay-protocol.json";m=json.load(open(mp));mh=digest(mp)
 status=json.load(open(RUN/"status.json"));summary=json.load(open(RUN/"summary.json"))
 need(mh==status["manifest_sha256"],"Manifest/status mismatch")
 need(status["status"]=="completed" and summary["status"]=="passed","Replay did not finish")
 need(digest(m["replay_selection_path"])==m["replay_selection_sha256"],"Selection changed")
 for p,h in m["source_sha256"].items():need(digest(p)==h,"Pinned replay source mismatch "+p)
 cases={c["case_id"]:c for c in m["replay_cases"]}
 need(len(cases)==len(m["replay_cases"]),"Duplicate selected case")
 expected={(c["case_id"],s) for c in cases.values() for s in c["replay_steps"]}
 need(len(expected)==m["replay_budget"]["maximum_episodes"]==548,"Declaration budget mismatch")
 raw_paths=[RUN/(b+"-worker-"+str(i)+".jsonl") for b in ("libero","libero_plus") for i in range(4)]
 summary_hashes={r["path"]:r["sha256"] for r in summary["raw_files"]}
 cache={};seen=set();counts=collections.Counter();by=collections.defaultdict(collections.Counter);findings=[];pairrows=[];raw_evidence=[];mismatch_kinds=collections.Counter()
 sys.path.insert(0,str(HERE));import recording_audit
 for path in raw_paths:
  h=digest(path);need(summary_hashes[str(path)]==h,"Final media audit raw binding mismatch")
  n=0
  for line,text in enumerate(open(path),1):
   r=json.loads(text);n+=1;k=(r["historical_case_id"],r["flow_steps"])
   need(k in expected and k not in seen,"Unexpected/duplicate replay arm");seen.add(k)
   need(r["status"]=="ok" and type(r["success"]) is bool,"Execution error/nonbool")
   c=cases[k[0]];ref=c["arms"][str(k[1])]["source_refs"][0]
   need(r["original_source"]==ref and r["manifest_sha256"]==mh,"Wrong source/manifest")
   need(r["historical_study"]==c["study"] and r["original_manifest_sha256"]==c["manifest_sha256"],"Historical identity")
   for field in ("benchmark","condition_id","suite","task_name","family","category","severity","seed","init_index"):
    need(r[field]==c[field],"Case identity "+field)
   original_path=ref["path"]
   if original_path not in cache:
    need(digest(original_path)==ref["file_sha256"],"Original file changed")
    cache[original_path]=pathlib.Path(original_path).read_bytes().splitlines()
   raw=cache[original_path][ref["line"]-1];need(hashlib.sha256(raw).hexdigest()==ref["record_sha256"],"Original row hash mismatch")
   old=json.loads(raw);need(old["status"]=="ok" and type(old["success"]) is bool,"Invalid old record")
   mismatches=[]
   for field in ("success","policy_steps","initial_state_sha256","stabilized_state_sha256","total_velocity_evaluations"):
    if old[field]!=r[field]:mismatches.append(field)
   if len(old["chunks"])!=len(r["chunks"]):mismatches.append("chunk_count")
   for i,(a,b) in enumerate(zip(old["chunks"],r["chunks"])):
    for field in ("noise_sha256","action_sha256","observation_sha256","velocity_evaluations"):
     if a[field]!=b[field]:mismatches.append("chunk_"+str(i)+"."+field)
   equivalent=not mismatches
   label="exact_replay" if equivalent else "reconstruction-not-equivalent"
   need(r["equivalence"]==dict(exact_equivalent=equivalent,classification=label,visual_attribution_eligible=equivalent,mismatches=mismatches),"Stored equivalence label incorrect")
   counts[label]+=1;by[c["study"]+"/"+c["benchmark"]][label]+=1
   mismatch_kinds.update(set(x.split(".")[-1] if x.startswith("chunk_") and "." in x else x for x in mismatches))
   adapted=dict(m);spec=dict(m["benchmarks"][r["benchmark"]]);spec["conditions"]=[dict(c["replay_condition"],cases=[c["replay_case"]])]
   adapted["benchmarks"]={r["benchmark"]:spec}
   recording_audit.audit_record(dict(r,phase="main"),adapted,inspect_video=False)
   pairrows.append(r)
   findings.append(dict(case_id=k[0],study=c["study"],benchmark=c["benchmark"],category=c["category"],flow_steps=k[1],
                        original_success=old["success"],replay_success=r["success"],equivalent=equivalent,
                        mismatches=mismatches,record=dict(path=str(path),line=line),original_source=ref,
                        video_path=r["recording"]["video_path"],trace_path=r["recording"]["trace_path"]))
  raw_evidence.append(dict(path=str(path),sha256=h,records=n))
 need(seen==expected,"Missing declared replays")
 recording_audit.audit_pairs(pairrows,allow_partial=True)
 need(dict(counts)==summary["equivalence"] and len(seen)==summary["completed"],"Final summary mismatch")
 casegroups=collections.defaultdict(list)
 for f in findings:casegroups[f["case_id"]].append(f)
 elig=[]
 for cid,fs in casegroups.items():
  eq=[f["flow_steps"] for f in fs if f["equivalent"]]
  elig.append(dict(case_id=cid,study=cases[cid]["study"],benchmark=cases[cid]["benchmark"],category=cases[cid]["category"],
                   selected_steps=cases[cid]["replay_steps"],equivalent_steps=eq,all_selected_arms_equivalent=len(eq)==len(fs),
                   one_step_and_higher_equivalent=1 in eq and any(s>1 for s in eq)))
 result=dict(status="passed",manifest_sha256=mh,declared_cases=len(cases),declared_episodes=len(expected),actual_episodes=len(seen),
             reported_live_counter=status["records"],final_summary_counter=summary["completed"],raw_files=raw_evidence,
             equivalence=dict(counts),by_study_benchmark={k:dict(v) for k,v in by.items()},mismatch_categories=dict(mismatch_kinds),
             fully_equivalent_selected_cases=sum(c["all_selected_arms_equivalent"] for c in elig),
             one_step_plus_higher_equivalent_cases=sum(c["one_step_and_higher_equivalent"] for c in elig),
             counter_explanation="replay_supervise.py updates state.records only inside worker polling loop. Completion assigns status/summary but never refreshes records; 547 is stale last progress snapshot, exact raw/final audited coverage is548.",
             source_files_verified=len(m["source_sha256"]),media_hashes_and_numeric_traces_reverified=True,
             media_decoding_rerun=False,prior_full_decode_audit_hash=digest(RUN/"summary.json"),
             media_gate="Prior successful complete video decoding binds identical SHA256 raw files; each current recording file/hash/numeric trace rechecked. Pairing/RNG audit rerun.",
             attribution_rule="Only individually exact arms may support original-episode labels; original paired comparisons require both relevant arms exact. All non-equivalent videos are reconstruction-only, even when outcomes happen to agree.")
 (OUT/"audit.json").write_text(json.dumps(result,indent=2)+"\n")
 (OUT/"episode-attribution.jsonl").write_text("".join(json.dumps(f)+"\n" for f in findings))
 (OUT/"case-attribution.json").write_text(json.dumps(elig,indent=2)+"\n")
 print(json.dumps({k:v for k,v in result.items() if k!="raw_files"},indent=2))
if __name__=="__main__":run()

