"""Audit saved rollout media and executed numeric traces, without simulation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import re
from collections import defaultdict
import numpy as np

def file_hash(path):
    digest=hashlib.sha256()
    with open(path,"rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()

def require(value,message):
    if not value:
        raise ValueError(message)

def audit_record(row, manifest, inspect_video=True):
    require(row.get("status")=="ok","Execution error cannot be audited as failure")
    if "benchmarks" in manifest:
        spec=manifest["benchmarks"][row["benchmark"]]
        condition=next((c for c in spec["conditions"] if c["condition_id"]==row["condition_id"]),None)
        require(condition is not None,"Undeclared condition")
        for field in ("suite","task_name","family","category","severity","prompt"):
            require(field in row and row[field]==condition[field],"Manifest metadata mismatch: "+field)
        require(row["benchmark_commit"]==spec["commit"],"Benchmark commit mismatch")
        cases=condition["cases"] if row["phase"]=="main" else [c for c in spec["smoke_cases"] if c["condition_id"]==row["condition_id"]]
        selected=[c for c in cases if c["seed"]==row["seed"] and c["init_index"]==row["init_index"]]
        require(len(selected)==1,"Undeclared case")
        require(row["initial_state_sha256"]==selected[0]["initial_state_sha256"],"Declared initial state mismatch")
        require(row["flow_steps"] in manifest["flow_steps"],"Undeclared arm")
    for field in ("checkpoint_sha256","initial_state_sha256","stabilized_state_sha256","manifest_sha256"):
        if "benchmarks" in manifest:
            require(bool(re.fullmatch(r"[0-9a-f]{64}",str(row.get(field,"")))),"Missing or malformed "+field)
    if "checkpoint" in manifest:
        require(row["checkpoint_sha256"]==manifest["checkpoint"]["sha256"],"Declared checkpoint mismatch")
    capture=row["recording"]
    for kind in ("video","trace"):
        require(bool(re.fullmatch(r"[0-9a-f]{64}",str(capture.get(kind+"_sha256","")))),"Missing or malformed "+kind+" hash")
        require(Path(capture[kind+"_path"]).stat().st_size>0,"Empty artifact")
        require(file_hash(capture[kind+"_path"])==capture[kind+"_sha256"],kind+" hash mismatch")
    with np.load(capture["trace_path"],allow_pickle=False) as stored:
        arrays={name:stored[name] for name in stored.files}
    steps=row["policy_steps"]
    actions=arrays["actions"]; states=arrays["eef_gripper_states"]
    rng=arrays["rng_sha256"]; successes=arrays["success"]
    require(steps>0 and actions.shape==(steps,7),"Executed action shape/count mismatch")
    require(states.ndim==2 and states.shape[0]==steps,"State trace length mismatch")
    require(np.isfinite(actions).all() and np.isfinite(states).all(),"Nonfinite trace")
    require(rng.shape==(steps,) and all(re.fullmatch(r"[0-9a-f]{64}",str(x)) for x in rng),"RNG trace mismatch")
    require(successes.shape==(steps,) and successes.dtype==np.dtype(bool),"Success trace mismatch")
    require(not successes[:-1].any() and bool(successes[-1])==row["success"],"Terminal success mismatch")
    require(capture["executed_actions"]==steps,"Executed action metadata mismatch")
    require(capture["video_frames"]==steps+1,"Frame metadata mismatch")
    replan=manifest["settings"]["replan_steps"]
    chunks=row["chunks"]
    require(len(chunks)==(steps+replan-1)//replan,"Action chunk count mismatch")
    prefix=[]; matched_dtypes=set()
    for index,chunk in enumerate(chunks):
        for field in ("noise_sha256","observation_sha256","action_sha256","simulator_rng_sha256"):
            require(bool(re.fullmatch(r"[0-9a-f]{64}",str(chunk.get(field,"")))),"Missing or malformed chunk "+field)
        require(chunk["chunk_index"]==index,"Chunk index mismatch")
        require(chunk["velocity_evaluations"]==row["flow_steps"],"Fixed sampler mismatch")
        values=np.asarray(chunk["actions"])
        require(values.shape==(manifest["noise_policy"]["shape"][0],7) and np.isfinite(values).all(),"Chunk action shape mismatch")
        # JSON does not preserve dtype. Require the original byte hash to match an exact float representation.
        matches=[dtype for dtype in ("float32","float64") if hashlib.sha256(np.asarray(chunk["actions"],dtype=dtype).tobytes()).hexdigest()==chunk["action_sha256"]]
        require(bool(matches),"Original server action digest mismatch")
        matched_dtypes.update(matches)
        require(chunk["simulator_rng_sha256"]==str(rng[index*replan]),"Chunk RNG does not match action trace")
        prefix.extend(chunk["actions"][:replan])
    require(np.array_equal(actions,np.asarray(prefix[:steps])),"Executed actions differ from queued chunk prefixes")
    require(row["total_velocity_evaluations"]==sum(c["velocity_evaluations"] for c in chunks),"Total sampler accounting mismatch")
    if inspect_video:
        result=json.loads(subprocess.check_output(["ffprobe","-v","error","-count_frames","-select_streams","v:0","-show_entries","stream=width,height,nb_read_frames","-of","json",capture["video_path"]],text=True))
        stream=result["streams"][0]
        require(int(stream["nb_read_frames"])==steps+1,"Decoded video frame count mismatch")
        require(stream["width"]==2*manifest["settings"]["render_size"] and stream["height"]==manifest["settings"]["render_size"],"Video dimensions mismatch")
    return {"rng":rng,"steps":steps,"action_digest_dtypes":sorted(matched_dtypes)}

def audit_records(rows,manifest,inspect_video=True,allow_partial=False):
    groups=defaultdict(dict); dtypes=set()
    for row in rows:
        key=(row["benchmark"],row["condition_id"],row["seed"],row["init_index"])
        require(row["flow_steps"] not in groups[key],"Duplicate arm")
        result=audit_record(row,manifest,inspect_video)
        dtypes.update(result["action_digest_dtypes"])
        groups[key][row["flow_steps"]]=(row,result)
    audit_pairs(rows,allow_partial=allow_partial)
    return {"status":"passed","records":len(rows),"matched_cases":len(groups),"action_digest_dtypes":sorted(dtypes),"video_decoding_checked":inspect_video,"allow_partial":allow_partial}


def audit_pairs(rows,allow_partial=True):
    groups=defaultdict(dict)
    for row in rows:
        key=(row["benchmark"],row["condition_id"],row["seed"],row["init_index"])
        require(row["flow_steps"] not in groups[key],"Duplicate arm")
        with np.load(row["recording"]["trace_path"],allow_pickle=False) as trace:
            groups[key][row["flow_steps"]]=(row,{"rng":trace["rng_sha256"].copy()})
    for arms in groups.values():
        require(allow_partial or set(arms)=={1,2,4,10},"Incomplete four-arm case")
        reference,trace=next(iter(arms.values()))
        for row,result in arms.values():
            for field in ("initial_state_sha256","stabilized_state_sha256","checkpoint_sha256","benchmark_commit","manifest_sha256"):
                require(row[field]==reference[field],"Pair identity mismatch: "+field)
            length=min(len(trace["rng"]),len(result["rng"]))
            require(np.array_equal(trace["rng"][:length],result["rng"][:length]),"Per-action simulator RNG pairing mismatch")
            require(row["chunks"][0]["observation_sha256"]==reference["chunks"][0]["observation_sha256"],"Initial observation mismatch")
            for a,b in zip(row["chunks"],reference["chunks"]):
                require(a["noise_sha256"]==b["noise_sha256"],"Flow noise prefix mismatch")
    return {"matched_cases":len(groups)}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest",required=True,type=Path)
    parser.add_argument("--records",nargs="+",required=True,type=Path)
    parser.add_argument("--allow-partial",action="store_true")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text())
    rows=[json.loads(line) for path in args.records for line in path.read_text().splitlines() if line.strip()]
    for row in rows:
        require(row["manifest_sha256"]==file_hash(args.manifest),"Declared manifest hash mismatch")
    result=audit_records(rows,manifest,allow_partial=args.allow_partial)
    result["manifest_sha256"]=file_hash(args.manifest)
    result["record_files"]=[{"path":str(path),"sha256":file_hash(path)} for path in args.records]
    if args.output:
        args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()

