"""Independent detached gate; never launches evaluations or modifies pinned source."""
import argparse
from collections import Counter,defaultdict
import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path("/volt/code/frozen-flow-study")
HERE=ROOT/"examples/rescue_characterization"
ARTIFACTS=Path("/volt/artifacts/rescue-characterization")
PIN="e7537584d34855c24c1a38ca11da4f7e479b9bcbfc53d2cc7ec8f7ff46eadf9d"
STEPS=(1,2,4,10)

def require(value,message):
    if not value:raise ValueError(message)

def digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
    return h.hexdigest()

def load_rows(path):
    rows=[]
    for n,line in enumerate(path.read_bytes().splitlines(),1):
        require(bool(line.strip()),f"Empty raw line {path}:{n}")
        row=json.loads(line);rows.append(row)
    return rows

def counts(groups):
    patterns=Counter();first=Counter();rescue=Counter();regress=Counter()
    f1=s1=nonmono=persistent=nonpersistent=0
    vectors=[]
    for key,arms in sorted(groups.items()):
        require(set(arms)==set(STEPS),"Unavailable first/intermediate arm")
        values=[arms[s]["success"] for s in STEPS]
        require(all(type(v) is bool for v in values),"Outcome is not Boolean")
        pattern="/".join("succeed" if v else "fail" for v in values)
        patterns[pattern]+=1
        nonmono+=any(values[i] and not values[j] for i in range(4) for j in range(i+1,4))
        if values[0]:s1+=1
        else:
            f1+=1
            firststep=next((str(s) for s in STEPS[1:] if arms[s]["success"]),"never")
            first[firststep]+=1
            if firststep!="never":
                if all(arms[s]["success"] for s in STEPS if s>=int(firststep)):persistent+=1
                else:nonpersistent+=1
        for s in STEPS[1:]:
            rescue[str(s)]+=not values[0] and arms[s]["success"]
            regress[str(s)]+=values[0] and not arms[s]["success"]
        vectors.append({"key":list(key),"success_vector":values,"pattern":pattern})
    return dict(cases=len(groups),one_step_failures=f1,one_step_successes=s1,patterns=dict(patterns),
                first_success_among_one_step_failures=dict(first),nonmonotonic=nonmono,
                persistent_rescues=persistent,nonpersistent_rescues=nonpersistent,
                rescues=dict(rescue),regressions=dict(regress),vectors=vectors)

def independent_noise(row,index,shape):
    key=[row["benchmark"],row["suite"],row["task_name"],row["seed"],row["init_index"],index]
    seed=int.from_bytes(hashlib.sha256(json.dumps(key,ensure_ascii=True,separators=(",",":")).encode()).digest()[:8],"little")
    values=np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
    return hashlib.sha256(values.tobytes()).hexdigest()

def run(phase,manifest_path,runs):
    manifest_hash=digest(manifest_path)
    require(manifest_hash==PIN,"Manifest differs from independently pinned declaration")
    m=json.loads(manifest_path.read_text())
    status_path=runs/("smoke-status.json" if phase=="smoke" else "full-status.json")
    status=json.loads(status_path.read_text())
    require(status.get("status")=="published","Supervisor has not published successfully")
    require(status["manifest_sha256"]==PIN,"Published manifest mismatch")
    require(bool(status.get("published_commit")) and bool(status.get("wandb_url")),"Publication/W&B evidence missing")
    require(m["flow_steps"]==list(STEPS),"Fixed arms changed")
    for path,expected in m["source_sha256"].items():
        require(digest(path)==expected,"Pinned source mismatch: "+path)
    actual_uuids={x.strip() for x in subprocess.check_output(["nvidia-smi","--query-gpu=uuid","--format=csv,noheader"],text=True).splitlines()}
    require(actual_uuids==set(m["gpu_uuids"]),"Current GPU UUID allocation mismatch")
    checkpoint=Path(m["checkpoint"]["path"])
    checkpoint_files=[]
    for p in sorted(checkpoint.rglob("*")):
        if p.is_file():checkpoint_files.append({"path":str(p.relative_to(checkpoint)),"sha256":digest(p),"bytes":p.stat().st_size})
    checkpoint_hash=hashlib.sha256(json.dumps(checkpoint_files,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    require(checkpoint_hash==m["checkpoint"]["sha256"],"Checkpoint content mismatch")
    source=HERE/"recording_audit.py"
    spec=importlib.util.spec_from_file_location("pinned_recording_audit",source)
    recording=importlib.util.module_from_spec(spec);spec.loader.exec_module(recording)
    parity=[];prior_parity=[];out={};evidence=[status_path,manifest_path]
    for benchmark in ("libero","libero_plus"):
        b=m["benchmarks"][benchmark]
        require(subprocess.check_output(["git","-C",b["root"],"rev-parse","HEAD"],text=True).strip()==b["commit"],"Benchmark checkout mismatch")
        conditions={c["condition_id"]:c for c in b["conditions"]}
        selected=[(conditions[x["condition_id"]],x) for x in b["smoke_cases"]] if phase=="smoke" else [(c,x) for c in b["conditions"] for x in c["cases"]]
        expected={(c["condition_id"],x["seed"],x["init_index"],s):(c,x) for c,x in selected for s in STEPS}
        require(len(expected)==len(selected)*4,"Duplicate declaration")
        files=[runs/(benchmark+"-smoke.jsonl")] if phase=="smoke" else [runs/(benchmark+"-main-worker-"+str(i)+".jsonl") for i in range(4)]
        rows=[];found={};groups=defaultdict(dict)
        for path in files:
            evidence.append(path)
            for row in load_rows(path):
                require(row.get("status")=="ok","Error attempt present")
                require(type(row.get("success")) is bool,"Raw success not real bool")
                for field in ("seed","init_index","flow_steps","policy_steps"):require(type(row.get(field)) is int,"Invalid integer field "+field)
                key=(row["condition_id"],row["seed"],row["init_index"],row["flow_steps"])
                require(key in expected and key not in found,"Unexpected/duplicate raw arm")
                c,x=expected[key]
                checks=dict(benchmark=benchmark,phase=phase,manifest_sha256=PIN,checkpoint_sha256=checkpoint_hash,
                            benchmark_commit=b["commit"],initial_state_sha256=x["initial_state_sha256"])
                checks.update({field:c[field] for field in ("suite","task_name","family","category","severity","prompt")})
                for field,value in checks.items():require(row.get(field)==value,"Identity mismatch "+field)
                require(row["gpu_uuid"] in actual_uuids,"Unexpected recorded GPU")
                if phase=="main":
                    require(row["gpu_uuid"]==m["gpu_uuids"][files.index(path)],"Worker GPU allocation mismatch")
                n=row["policy_steps"];h=m["settings"]["replan_steps"]
                require(0<n<=m["settings"]["max_policy_steps"][row["suite"]],"Episode length outside declaration")
                require(len(row["chunks"])==(n+h-1)//h,"Original action chunk count mismatch")
                for i,chunk in enumerate(row["chunks"]):
                    require(chunk["chunk_index"]==i and chunk["velocity_evaluations"]==row["flow_steps"],"Sampler action accounting mismatch")
                    require(chunk["noise_sha256"]==independent_noise(row,i,m["noise_policy"]["shape"]),"Raw seeded noise digest mismatch")
                require(row["total_velocity_evaluations"]==len(row["chunks"])*row["flow_steps"],"Total action evaluation mismatch")
                found[key]=row;groups[key[:3]][key[3]]=row;rows.append(row)
        require(set(found)==set(expected),"Missing declared episodes")
        for arms in groups.values():
            reference=arms[1]
            for row in arms.values():
                for field in ("initial_state_sha256","stabilized_state_sha256"):
                    require(row[field]==reference[field],"Raw pair initial/stabilized mismatch")
                require(row["chunks"][0]["observation_sha256"]==reference["chunks"][0]["observation_sha256"],"Raw initial observation mismatch")
                for a,c in zip(row["chunks"],reference["chunks"]):
                    require(a["noise_sha256"]==c["noise_sha256"],"Raw noise pair mismatch")
        # Independently checks raw evidence; reuses pinned media parser only for actual decoding/trace integrity.
        media=recording.audit_records(rows,m,inspect_video=True,allow_partial=False)
        independent=counts(groups)
        suffix="-smoke-patterns.json" if phase=="smoke" else "-patterns.json"
        pattern_path=runs/(benchmark+suffix);evidence.append(pattern_path)
        published=json.loads(pattern_path.read_text())
        cohort=published[m["study"]+"/"+phase+"/"+benchmark]
        for field in ("cases","one_step_failures","one_step_successes","patterns","first_success_among_one_step_failures","nonmonotonic"):
            require(cohort[field]==independent[field],"Independent count disagreement: "+field)
        require(cohort["complete_four_arm"]==len(groups),"Published incomplete cases")
        for s in ("2","4","10"):
            require(cohort["steps"][s]["rescues"]==independent["rescues"][s],"Rescue reconciliation mismatch")
            require(cohort["steps"][s]["regressions"]==independent["regressions"][s],"Regression reconciliation mismatch")
        for worker,uuid in enumerate(m["gpu_uuids"]):
            p=runs/(benchmark+"-prepare-worker-"+str(worker)+".jsonl");evidence.append(p)
            records=load_rows(p)
            validate_preparation_count(records,phase)
            for attempt,r in enumerate(records):
                require(r["status"]=="verified" and r["phase"]=="preparation","Unverified preparation")
                require(r["gpu_uuid"]==uuid and r["benchmark"]==benchmark and r["manifest_sha256"]==PIN and r["checkpoint_sha256"]==checkpoint_hash,"Preparation identity mismatch")
                require(r["warmup"]["pass_"] is True,"Warmup failure")
                checks=r["verification"]
                require(len(checks)==4 and {v["flow_steps"] for v in checks}==set(STEPS),"Incomplete actual sampler parity")
                for v in checks:
                    require(v["pass"] is True and v["gpu_uuid"]==uuid and v["checkpoint_sha256"]==checkpoint_hash,"Failed parity identity")
                    error=v["max_abs_action_difference"];tolerance=v["tolerance"]
                    require(math.isfinite(error) and 0<=error<=tolerance<=1e-5,"Parity outside numerical tolerance")
                    require(v["velocity_evaluations_for_verification"]==2*v["flow_steps"],"Parity evaluation accounting mismatch")
                    item={"benchmark":benchmark,"gpu_uuid":uuid,"flow_steps":v["flow_steps"],"max_abs_action_difference":error,"preparation_record_index":attempt}
                    if attempt==len(records)-1:parity.append(item)
                    else:prior_parity.append(item)
        independent["episodes"]=len(rows);independent["recording_audit"]=media
        out[benchmark]=independent
    require(sum(x["episodes"] for x in out.values())==(44 if phase=="smoke" else 9920),"Total episode budget mismatch")
    require(len(parity)==32,"Expected exactly32 current actual sampler parity checks")
    require(len(prior_parity)==(0 if phase=="smoke" else 32),"Unexpected prior sampler parity count")
    return dict(status="passed",phase=phase,manifest_sha256=PIN,published_commit=status["published_commit"],
                wandb_url=status["wandb_url"],checkpoint_sha256=checkpoint_hash,gpu_uuids=sorted(actual_uuids),
                source_files_verified=len(m["source_sha256"]),parity_checks=parity,prior_smoke_parity_checks=prior_parity,benchmarks=out,
                evidence_files=[{"path":str(p),"sha256":digest(p)} for p in evidence],
                methodology="Independent expected-case enumeration, seeded-noise reproduction and counts; pinned recording_audit reused for trace/video decoding only",
                limitations="No video failure-mode labels or online prediction inferred by this integrity gate")

def validate_preparation_count(records,phase):
    expected=1 if phase=="smoke" else 2
    require(len(records)==expected,"Missing/duplicate preparation: expected "+str(expected))

def self_test():
    validate_preparation_count([{}],"smoke")
    validate_preparation_count([{},{}],"main")
    for phase,records in (("smoke",[{},{}]),("main",[{}]),("main",[{}, {}, {}])):
        try:validate_preparation_count(records,phase)
        except ValueError:pass
        else:raise AssertionError("Incorrect preparation count accepted")

    rows={1:{"success":False},2:{"success":True},4:{"success":False},10:{"success":True}}
    r=counts({("condition",1,0):rows})
    require(r["first_success_among_one_step_failures"]=={"2":1} and r["nonpersistent_rescues"]==1,"nonmonotonic test")
    try:counts({("condition",1,0):{1:{"success":False},10:{"success":True}}})
    except ValueError:pass
    else:raise AssertionError("Partial arm accepted")
    try:counts({("condition",1,0):{s:{"success":1} for s in STEPS}})
    except ValueError:pass
    else:raise AssertionError("Nonboolean accepted")
    print("Independent gate self-tests passed")

def main():
    p=argparse.ArgumentParser();p.add_argument("--phase",choices=("smoke","main"));p.add_argument("--self-test",action="store_true")
    p.add_argument("--manifest",type=Path,default=HERE/"protocol.json");p.add_argument("--runs",type=Path,default=ARTIFACTS/"runs");p.add_argument("--output",type=Path)
    a=p.parse_args()
    if a.self_test:self_test();return
    if not a.phase:p.error("--phase required")
    output=a.output or ARTIFACTS/("independent-smoke-audit.json" if a.phase=="smoke" else "independent-final-audit.json")
    try:result=run(a.phase,a.manifest,a.runs)
    except Exception as error:
        result={"status":"failed","phase":a.phase,"error":repr(error),"manifest_sha256":digest(a.manifest)}
        code=1
    else:code=0
    result["created_at_utc"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
    result["audit_script_sha256"]=digest(__file__)
    temp=output.with_suffix(".tmp");temp.write_text(json.dumps(result,indent=2)+"\n");temp.replace(output)
    print(json.dumps({k:v for k,v in result.items() if k not in ("benchmarks","evidence_files","parity_checks","prior_smoke_parity_checks")},indent=2))
    raise SystemExit(code)

if __name__=="__main__":main()

