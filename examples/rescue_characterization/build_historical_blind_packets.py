"""Historical media adapter: retain all exact clips, exclude divergent replays."""
import argparse
from collections import Counter,defaultdict
import hashlib,json,os
from pathlib import Path
import sys
import numpy as np
ROOT=Path("/volt/code/frozen-flow-study")
HERE=ROOT/"examples/rescue_characterization"
sys.path.insert(0,str(HERE))
from build_blind_packets import filehash,rank,sheet,require
from replay import audit_replays

def classify_coverage(entries,rows):
    by_case=defaultdict(dict)
    for row in rows:
        by_case[row["historical_case_id"]][row["flow_steps"]]=row
    coverage=[];counts=defaultdict(Counter)
    for entry in entries:
        arms=by_case[entry["case_id"]]
        expected=set(entry["replay_steps"])
        require(set(arms)==expected,"Historical coverage mismatch")
        exact=sorted(n for n,r in arms.items() if r["equivalence"]["classification"]=="exact_replay")
        pairs=[[1,n] for n in (2,4,10) if 1 in exact and n in exact]
        kind=entry["selection_reason"]
        stratum="/".join((entry["study"],entry["benchmark"],kind))
        counts[stratum].update(cases=1,declared_clips=len(expected),exact_clips=len(exact),
                              excluded_clips=len(expected)-len(exact),fully_exact_cases=int(set(exact)==expected),
                              partial_exact_cases=int(bool(exact) and set(exact)!=expected))
        coverage.append({"case_id":entry["case_id"],"historical_study":entry["study"],"benchmark":entry["benchmark"],
                         "original_selection_class":kind,"declared_arms":sorted(expected),"exact_arms":exact,
                         "excluded_arms":sorted(expected-set(exact)),"full_declared_case_exact":set(exact)==expected,
                         "exact_comparison_pairs":pairs,"paired_failure_mode_claim_eligible":set(exact)==expected,
                         "partial_case_scope":"Standalone clip observations only; no paired failure-mode claim unless full declared case is exact"})
    return coverage,{k:dict(v) for k,v in sorted(counts.items())}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--summary",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--blind-key",required=True)
    a=p.parse_args()
    require(not a.output.exists(),"Refuse packet overwrite")
    manifest=json.loads(a.manifest.read_text());mh=filehash(a.manifest)
    summary=json.loads(a.summary.read_text())
    require(summary["status"]=="passed" and summary["video_decoding_checked"] is True,"Replay full media audit must have passed")
    rows=[];refs={};sources=[]
    for item in summary["raw_files"]:
        path=Path(item["path"]);require(filehash(path)==item["sha256"],"Replay raw file changed after audit")
        sources.append(item)
        for number,raw in enumerate(path.read_bytes().splitlines(),1):
            row=json.loads(raw);require(row["manifest_sha256"]==mh,"Replay manifest mismatch")
            key=(row["historical_case_id"],row["flow_steps"])
            require(key not in refs,"Duplicate historical arm")
            refs[key]={"path":str(path),"line":number,"file_sha256":item["sha256"],"record_sha256":hashlib.sha256(raw).hexdigest()}
            rows.append(row)
    # Recompute source-equivalence and numeric trace integrity; recorded full video
    # decoding is bound by unchanged summary raw-file hashes above.
    audited=audit_replays(rows,manifest,mh,complete=True,inspect_video=False)
    require(audited["completed"]==summary["completed"] and audited["equivalence"]==summary["equivalence"],"Replay audit summary mismatch")
    coverage,counts=classify_coverage(manifest["replay_cases"],rows)
    coverage_by_id={c["case_id"]:c for c in coverage}
    eligible=[r for r in rows if r["equivalence"]["classification"]=="exact_replay"
              and r["equivalence"]["exact_equivalent"] is True and r["equivalence"]["visual_attribution_eligible"] is True]
    require(len(eligible)==summary["equivalence"].get("exact_replay",0),"Exact eligibility mismatch")
    excluded=[r for r in rows if r not in eligible]
    blind=a.output/"blind";private=a.output/"private"
    blind.mkdir(parents=True);private.mkdir(mode=0o700)
    grouped=defaultdict(list)
    for row in eligible:grouped[row["historical_case_id"]].append(row)
    packets=[];unblind=[]
    for case_id,group in sorted(grouped.items(),key=lambda pair:rank(a.blind_key,pair[0])):
        group_id="G"+rank(a.blind_key,case_id)[:16]
        packet={"group_id":group_id,"task_instruction":group[0]["prompt"],"clips":[]}
        for row in sorted(group,key=lambda r:rank(a.blind_key,[case_id,r["flow_steps"]])):
            arm=row["flow_steps"];clip_id="C"+rank(a.blind_key,[case_id,arm])[:20]
            capture=row["recording"]
            for kind in ("video","trace"):
                require(filehash(capture[kind+"_path"])==capture[kind+"_sha256"],"Exact replay media changed")
            os.link(capture["video_path"],blind/(clip_id+".mp4"))
            metrics=sheet(capture["video_path"],capture["trace_path"],clip_id,blind/(clip_id+".png"),capture["video_frames"])
            with np.load(capture["trace_path"],allow_pickle=False) as z:
                np.savez_compressed(blind/(clip_id+".npz"),actions=z["actions"],eef_gripper_states=z["eef_gripper_states"])
            packet["clips"].append({"clip_id":clip_id,"contact_sheet":clip_id+".png","video":clip_id+".mp4",
                                    "numeric_trace":clip_id+".npz","metrics":metrics})
            unblind.append({"group_id":group_id,"clip_id":clip_id,"arm":arm,"historical_case_id":case_id,
                            "historical_study":row["historical_study"],"success":row["success"],
                            "replay_source":refs[(case_id,arm)],"original_source":row["original_source"],
                            "equivalence":row["equivalence"],"recording":capture,"coverage":coverage_by_id[case_id]})
        packets.append(packet)
    (blind/"packets.json").write_text(json.dumps(packets,indent=2)+"\n")
    rubric=(HERE/"BLIND_REVIEW.md").read_text()
    rubric+="\nPacket scope: describe only the provided clips. Some groups may have unshown counterparts; never infer unshown behavior. The coordinator will separately determine which paired comparisons are supported after labels are finalized.\n"
    (blind/"RUBRIC.md").write_text(rubric)
    (private/"unblinding.json").write_text(json.dumps(unblind,indent=2)+"\n")
    exclusion=[{"replay_source":refs[(r["historical_case_id"],r["flow_steps"])],"original_source":r["original_source"],
                "historical_case_id":r["historical_case_id"],"arm":r["flow_steps"],"equivalence":r["equivalence"]} for r in excluded]
    (private/"exclusions.json").write_text(json.dumps(exclusion,indent=2)+"\n")
    (private/"coverage.json").write_text(json.dumps({"selection":"All exact replay clips; no additional outcome sampling.",
        "scope":"Historical outcome-selected replay evidence, never population rates.",
        "fully_exact_cases_only_for_paired_failure_mode_claims":True,"counts":counts,"cases":coverage,
        "sources":sources,"manifest_sha256":mh,"summary_sha256":filehash(a.summary),"recomputed_audit":audited},indent=2)+"\n")
    print(json.dumps({"status":"completed","output":str(a.output),"groups":len(packets),"eligible_clips":len(eligible),"excluded_clips":len(excluded)}))

if __name__=="__main__":main()

