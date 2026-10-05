"""Independent CPU audit of paired raw episodes, probe tensors and point metrics.

Does not import the production scoring or analysis implementations. No rollouts.
"""
import argparse, collections, hashlib, json, math
from pathlib import Path
import numpy as np

ARMS=(1,2,4,10)
LABELS=("rescue","regression","unchanged_success","unchanged_failure")
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def need(condition,message):
    if not condition:raise ValueError(message)
def label(a,b):
    need(type(a) is bool and type(b) is bool,"Outcome must be boolean")
    return {(False,True):"rescue",(True,False):"regression",(True,True):"unchanged_success",(False,False):"unchanged_failure"}[a,b]
def independent_rank(scores,labels):
    n=len(scores);positives=sum(l=="rescue" for l in labels)
    thresholds,inverse,counts=np.unique(np.asarray(scores),return_inverse=True,return_counts=True)
    pos_counts=np.bincount(inverse,weights=np.asarray(labels)=="rescue",minlength=len(thresholds))
    thresholds=thresholds[::-1];counts=counts[::-1];pos_counts=pos_counts[::-1]
    totals=np.cumsum(counts);tps=np.cumsum(pos_counts)
    ap=float(np.sum(pos_counts/positives*tps/totals)) if positives else 0.
    pr=[dict(threshold=float(t),selected=int(n),precision=float(tp/n),recall=float(tp/positives) if positives else None) for t,n,tp in zip(thresholds,totals,tps)]
    def auc(negatives):
        p=[s for s,l in zip(scores,labels) if l=="rescue"];q=[s for s,l in zip(scores,labels) if l in negatives]
        if not p or not q:return None
        # Searchsorted counts ties and lower scores independently of production rank sums.
        q=np.sort(q);p=np.asarray(p)
        return float(np.sum(np.searchsorted(q,p,side="left")+.5*(np.searchsorted(q,p,side="right")-np.searchsorted(q,p,side="left")))/(len(p)*len(q)))
    return dict(ap=ap if positives else None,auroc=auc(set(LABELS)-{"rescue"}),rescue_regression_auroc=auc({"regression"}),prevalence=positives/n,ap_lift=ap-positives/n if positives else None,pr=pr)
def independent_budget(cases,k,field,budget,overhead):
    n=len(cases);cap=sum(c["cap_chunks"] for c in cases);capacity=budget*cap-overhead*n
    chosen=[];spent=0
    for c in sorted(cases,key=lambda x:(-x[field],x["case_id"])):
        extra=(k-1)*c["cap_chunks"]
        if spent+extra>capacity:break
        chosen.append(c);spent+=extra
    counts=collections.Counter(label(c["success"]["1"],c["success"][str(k)]) for c in chosen)
    total=sum(label(c["success"]["1"],c["success"][str(k)])=="rescue" for c in cases)
    base=sum(c["chunks"]["1"] for c in cases)
    actual=base+overhead*n+sum(k*c["chunks"][str(k)]-c["chunks"]["1"] for c in chosen)
    p=counts["rescue"]/len(chosen) if chosen else None
    return dict(selected=len(chosen),selected_fraction=len(chosen)/n,**{l:counts[l] for l in LABELS},precision=p,recall=counts["rescue"]/total if total else None,precision_lift=p-total/n if p is not None else None,net=(counts["rescue"]-counts["regression"])/n,selected_rescue_clusters=len({c["cluster"] for c in chosen if label(c["success"]["1"],c["success"][str(k)])=="rescue"}),planned_extra_evaluations=spent+overhead*n,baseline_cap=cap,extra_budget=budget*cap,overhead_feasible=capacity>=0,actual_counterfactual_evaluations=actual,actual_baseline_evaluations=base,actual_ratio=actual/base if base else None)
def compare(actual,expected,path=""):
    if isinstance(expected,dict):
        for k,v in expected.items():need(k in actual,"Missing metric "+path+"/"+k);compare(actual[k],v,path+"/"+k)
    elif isinstance(expected,list):
        need(len(actual)==len(expected),"Length "+path)
        for i,(a,e) in enumerate(zip(actual,expected)):compare(a,e,path+"/"+str(i))
    elif expected is None:need(actual is None,"Expected null "+path)
    elif isinstance(expected,(int,float)) and not isinstance(expected,bool):need(math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-12),"Metric difference "+path+repr((actual,expected)))
    else:need(actual==expected,"Metric difference "+path)
def independent_intervals(cases,cluster,replicates=10000,seed=20261005):
    groups={}
    for c in cases:groups.setdefault(c[cluster],[]).append(c)
    keys=sorted(groups);rng=np.random.default_rng(seed);values=collections.defaultdict(list)
    for _ in range(replicates):
        sample=[c for j in rng.integers(0,len(keys),size=len(keys)) for c in groups[keys[j]]]
        labels=[label(c["success"]["1"],c["success"]["4"]) for c in sample]
        rank=independent_rank([c["score"] for c in sample],labels)
        budget=independent_budget(sample,4,"score",.75,2)
        for k in ["ap","ap_lift","auroc","rescue_regression_auroc","prevalence"]:values["4/"+k].append(rank[k])
        for k in ["precision","recall","precision_lift","net","selected","rescue","regression","actual_ratio"]:values["4/0.75/"+k].append(budget[k])
    out={}
    for key,rows in values.items():
        finite=[x for x in rows if x is not None and math.isfinite(x)]
        out[key]=dict(low=float(np.percentile(finite,2.5)) if finite else None,high=float(np.percentile(finite,97.5)) if finite else None,undefined=replicates-len(finite),replicates=replicates)
    return out

def head_numpy(features,head):
    # Independent frozen three-layer head evaluation at t=1; no optimization.
    need(features.shape==(1,10,1024),"Head feature shape")
    fraction=np.linspace(0,1,16,dtype=np.float32)
    period=np.float32(.004)*np.float32(1000)**fraction
    angles=np.float32(1)/period*np.float32(2*np.pi)
    h=np.concatenate([features.mean(axis=1,dtype=np.float32),np.concatenate([np.sin(angles),np.cos(angles)])[None]],axis=-1)
    for i in range(3):
        h=h@head[f"layer_{i}/kernel"]+head[f"layer_{i}/bias"]
        if i<2:h=h/(1+np.exp(-h))
    return float(np.clip(h[0,0],-8,6))
def audit_probe(p,r,m,mh,head):
    for f,v in [("manifest_sha256",mh),("checkpoint_sha256",m["checkpoint"]["sha256"]),("head_sha256",m["head"]["sha256"]),("gpu_uuid",r["gpu_uuid"]),("chunk_index",0),("velocity_evaluations",2),("head_evaluations",1),("prefix_evaluations",1),("rng_unchanged",True)]:
        need(p[f]==v,"Probe identity "+f)
    need(p["noise_sha256"]==r["chunks"][0]["noise_sha256"],"Probe noise hash mismatch")
    need(sha(p["raw_path"])==p["raw_sha256"],"Raw probe hash")
    with np.load(p["raw_path"],allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
    need(set(arrays)=={"states","velocities","times","noise","first_action_features"},"Probe arrays")
    for k,a in arrays.items():
        need(a.dtype==np.float32 and np.isfinite(a).all(),"Probe dtype/nonfinite "+k)
        need(ah(a)==p["array_sha256"][k],"Array hash "+k)
    x,v,t,z=arrays["states"],arrays["velocities"],arrays["times"],arrays["noise"]
    need(x.shape==(3,1,10,32) and v.shape==(2,1,10,32) and z.shape==(10,32),"Probe shape")
    need(np.array_equal(t,np.array([1,.5],np.float32)) and np.array_equal(x[0,0],z),"Times/noise")
    need(ah(z)==p["noise_sha256"],"Raw noise hash")
    np.testing.assert_allclose(x[1],x[0]-.5*v[0],rtol=1e-6,atol=1e-6)
    with np.errstate(all="raise"):
        first=v[0,0,:,:7];delta=v[1,0,:,:7]-first
        den=np.sqrt(np.mean(first*first,dtype=np.float32));num=np.sqrt(np.mean(delta*delta,dtype=np.float32))
        score=float(np.float32(num/np.maximum(den,np.float32(1e-6))))
    need(score==p["score"] and float(num)==p["numerator_rms"] and float(den)==p["denominator_rms"],"Score reproduction")
    need(p["times"]==[1,.5] and p["dimension_slice"]==[0,7] and p["horizon_slice"]==[0,10] and p["ranking_direction"]=="descending","Score specification")
    logs=head_numpy(arrays["first_action_features"],head)
    need(abs(logs-p["first_log_sigma"])<=1e-5,"Independent head log sigma mismatch "+str((logs,p["first_log_sigma"])))
    need(float(np.exp(np.float32(p["first_log_sigma"])))==p["first_sigma"],"Sigma exponent")
    return abs(logs-p["first_log_sigma"])
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--manifest",required=True);parser.add_argument("--run-dir",required=True);parser.add_argument("--phase",choices=["smoke","main","full"],required=True);parser.add_argument("--output",required=True);a=parser.parse_args()
    m=json.loads(Path(a.manifest).read_text());mh=sha(a.manifest);root=Path(a.run_dir);smoke=a.phase=="smoke";phase="smoke" if smoke else "main";prep_phase="smoke" if smoke else "full"
    record_paths=[root/f"{b}-smoke.jsonl" for b in m["benchmarks"]] if smoke else [root/f"{b}-main-worker-{w}.jsonl" for b in m["benchmarks"] for w in range(4)]
    prep_paths=[root/f"{b}-{prep_phase}-prepare-worker-{w}.jsonl" for b in m["benchmarks"] for w in range(4)]
    metric_paths=[root/(f"{b}-smoke-metrics.json" if smoke else f"{b}-metrics.json") for b in m["benchmarks"]]
    need(sha(m["head"]["path"])==m["head"]["sha256"],"Head identity")
    for path,digest in m["source_sha256"].items():need(sha(path)==digest,"Pinned source "+path)
    with np.load(m["head"]["path"],allow_pickle=False) as f:head={k:f[k] for k in f.files if k!="metadata"}
    prepared=set();prep_nfe=0
    for path in prep_paths:
        rows=[json.loads(l) for l in path.read_text().splitlines() if l.strip()];need(len(rows)==1,"Preparation rows")
        r=rows[0];need(r["status"]=="verified" and r["manifest_sha256"]==mh and r["checkpoint_sha256"]==m["checkpoint"]["sha256"],"Preparation identity")
        need(r["gpu_uuid"] in m["gpu_uuids"],"Preparation UUID")
        need((r["benchmark"],r["gpu_uuid"]) not in prepared,"Duplicate preparation UUID");prepared.add((r["benchmark"],r["gpu_uuid"]))
        need({v["flow_steps"] for v in r["verification"]}==set(ARMS) and len(r["verification"])==4,"Preparation arms")
        need(r["warmup"]["pass"] is True and r["warmup"]["velocity_evaluations"]==50 and r["warmup"]["probe_velocity_evaluations"]==16 and r["warmup"]["head_evaluations"]==8 and r["warmup"]["prefix_evaluations"]==16,"Warmup accounting")
        prep_nfe+=r["warmup"]["velocity_evaluations"]
        for v in r["verification"]:
            for field,expected in [("checkpoint_sha256",m["checkpoint"]["sha256"]),("head_sha256",m["head"]["sha256"]),("gpu_uuid",r["gpu_uuid"]),("chunk_index",0),("probe_velocity_evaluations",4),("head_evaluations",2),("prefix_evaluations",6)]:
                need(v[field]==expected,"Verification identity/accounting "+field)
            for flag in ["pass","fixed_action_hash_bitwise_equal","rng_progression_unchanged","probe_repeat_bitwise_equal"]:need(v[flag] is True,"Parity flag "+flag)
            need(v["cached_public_max_abs_difference"]<=1e-5 and v["cached_public_tolerance"]==1e-5,"Cached/public parity")
            need(v["velocity_evaluations_for_verification"]==4*v["flow_steps"]+4,"Verification NFE")
            prep_nfe+=v["velocity_evaluations_for_verification"]
    grouped={};head_max=0;scored_nfe=0
    for path in record_paths:
        for line in path.read_text().splitlines():
            if not line.strip():continue
            r=json.loads(line);b=r["benchmark"];spec=m["benchmarks"][b];conds={c["condition_id"]:c for c in spec["conditions"]};c=conds[r["condition_id"]]
            expected={ (v["condition_id"],v["seed"],v["init_index"]):v["initial_state_sha256"] for v in spec["smoke_cases"]} if smoke else {(c["condition_id"],v["seed"],v["init_index"]):v["initial_state_sha256"] for c in spec["conditions"] for v in c["cases"]}
            key=(r["condition_id"],r["seed"],r["init_index"]);arm=r["flow_steps"]
            need(key in expected and arm in ARMS and type(r["success"]) is bool,"Case or outcome")
            need(r["initial_state_sha256"]==expected[key],"Declared state hash")
            need(r["status"]=="ok" and r["phase"]==phase and r["manifest_sha256"]==mh,"Row identity")
            need(r["checkpoint_sha256"]==m["checkpoint"]["sha256"] and r["benchmark_commit"]==spec["commit"] and r["gpu_uuid"] in m["gpu_uuids"],"Frozen identity")
            chunks=r["chunks"];need(bool(chunks),"Empty chunks")
            need([ch["chunk_index"] for ch in chunks]==list(range(len(chunks))),"Chunk ordering")
            for i,ch in enumerate(chunks):
                need(ch["velocity_evaluations"]==arm,"Fixed NFE")
                noise_key=[b,c["suite"],c["task_name"],r["seed"],r["init_index"],i]
                seed=int.from_bytes(hashlib.sha256(json.dumps(noise_key,ensure_ascii=True,separators=(",",":")).encode()).digest()[:8],"little")
                noise=np.random.default_rng(seed).standard_normal((10,32)).astype(np.float32)
                need(ah(noise)==ch["noise_sha256"],"Matched deterministic noise")
                actions=np.asarray(ch["actions"]);need(actions.shape==(10,7) and np.isfinite(actions).all(),"Action shape/finiteness")
                need(ah(actions)==ch["action_sha256"],"Action hash")
            need(r["total_velocity_evaluations"]==arm*len(chunks) and r["total_probe_velocity_evaluations"]==2 and r["total_actual_velocity_evaluations"]==arm*len(chunks)+2,"Actual NFE")
            scored_nfe+=r["total_actual_velocity_evaluations"]
            head_max=max(head_max,audit_probe(r["initial_probe"],r,m,mh,head))
            rec=r["recording"]
            for f in ["video","trace"]:need(sha(rec[f+"_path"])==rec[f+"_sha256"],"Media hash "+f)
            with np.load(rec["trace_path"],allow_pickle=False) as tr:
                need(len(tr["actions"])==r["policy_steps"]==rec["executed_actions"],"Action trace length")
                np.testing.assert_array_equal(tr["actions"],np.concatenate([ch["actions"] for ch in chunks])[:r["policy_steps"]])
                need(bool(tr["success"][-1])==r["success"],"Trace outcome")
            need(rec["video_frames"]==r["policy_steps"]+1,"Frame accounting")
            g=grouped.setdefault((b,*key),{});need(arm not in g,"Duplicate arm");g[arm]=r
    audits={}
    for b,spec in m["benchmarks"].items():
        expected={(v["condition_id"],v["seed"],v["init_index"]) for v in spec["smoke_cases"]} if smoke else {(c["condition_id"],v["seed"],v["init_index"]) for c in spec["conditions"] for v in c["cases"]}
        need({key[1:] for key in grouped if key[0]==b}==expected,"Missing declared cases")
        cases=[];patterns=collections.Counter()
        for key in sorted(k for k in grouped if k[0]==b):
            arms=grouped[key];need(set(arms)==set(ARMS),"Missing arm");ref=arms[1];c=next(c for c in spec["conditions"] if c["condition_id"]==key[1])
            for r in arms.values():
                for f in ["initial_state_sha256","stabilized_state_sha256","gpu_uuid"]:need(r[f]==ref[f],"Pair "+f)
                need(r["chunks"][0]["observation_sha256"]==ref["chunks"][0]["observation_sha256"],"Initial observation")
                for x,y in zip(r["chunks"],ref["chunks"]):need(x["noise_sha256"]==y["noise_sha256"] and x["simulator_rng_sha256"]==y["simulator_rng_sha256"],"Paired streams")
                for f in ["array_sha256","input_sha256","score","first_sigma","first_log_sigma"]:need(r["initial_probe"][f]==ref["initial_probe"][f],"Paired probe "+f)
            p=ref["initial_probe"];identity=hashlib.sha256(("flow-probe-tie-v1"+json.dumps(list(key),separators=(",",":"))).encode()).hexdigest()
            cases.append(dict(case_id=identity,score=p["score"],sigma=p["first_sigma"],severity=c["severity"],success={str(k):arms[k]["success"] for k in ARMS},chunks={str(k):len(arms[k]["chunks"]) for k in ARMS},cap_chunks=math.ceil(m["settings"]["max_policy_steps"][c["suite"]]/m["settings"]["replan_steps"]),cluster=c["suite"]+"/"+c["family"],condition_id=c["condition_id"]))
            patterns["".join("S" if arms[k]["success"] else "F" for k in ARMS)]+=1
        metric_path=root/(f"{b}-smoke-metrics.json" if smoke else f"{b}-metrics.json");metrics=json.loads(metric_path.read_text())
        need(metrics["manifest_sha256"]==mh and metrics["cases"]==len(cases) and metrics["episodes"]==4*len(cases),"Metric identity")
        compare(metrics["patterns"],dict(patterns),"patterns")
        for field,overhead in [("score",2),("sigma",1),("severity",0)]:
            for k in (2,4,10):
                labels=[label(c["success"]["1"],c["success"][str(k)]) for c in cases];point=metrics["comparators"][field]["point"][str(k)]
                compare(point,independent_rank([c[field] for c in cases],labels),field+"/"+str(k))
                compare(point["labels"],{l:labels.count(l) for l in LABELS})
                for budget in [.25,.5,.75,1.]:compare(point["budgets"][str(budget)],independent_budget(cases,k,field,budget,overhead),field+"/budget")
        reps=200 if smoke else 10000
        independent_ci=independent_intervals(cases,"cluster",reps)
        compare(metrics["comparators"]["score"]["intervals"],independent_ci,"primary score intervals")
        if b=="libero_plus":
            sensitivity=independent_intervals(cases,"condition_id",reps)
            compare(metrics["comparators"]["score"]["condition_sensitivity"],sensitivity,"condition score intervals")
        if not smoke:
            all_labels=[label(c["success"]["1"],c["success"]["4"]) for c in cases]
            budget=independent_budget(cases,4,"score",.75,2)
            checks=[all_labels.count("rescue")>=10,all_labels.count("regression")>=10,budget["rescue"]>=10,budget["selected_rescue_clusters"]>=5,budget["precision"] is not None and budget["precision"]>=.1,budget["recall"] is not None and budget["recall"]>=.2]
            for key,threshold in [("4/0.75/precision_lift",0),("4/0.75/net",0),("4/rescue_regression_auroc",.5)]:
                v=independent_ci[key];checks.append(v["low"] is not None and v["low"]>threshold and v["undefined"]/v["replicates"]<=.05)
            need(metrics["decision_gate"]["pass_"]==all(checks),"Independent decision gate")
            need(metrics["decision_gate"]["decision"]==("propose_separate_test_only_if_both_benchmarks_pass" if all(checks) else "STOP_no_controller"),"Independent decision message")
        for comparator in metrics["comparators"].values():
            for block in ["intervals","condition_sensitivity"]:
                for v in comparator.get(block,{}).values():
                    need(v["replicates"]==reps and 0<=v["undefined"]<=reps,"Interval metadata")
                    need((v["low"] is None and v["high"] is None) or (math.isfinite(v["low"]) and math.isfinite(v["high"]) and v["low"]<=v["high"]),"Interval range")
        audits[b]=dict(cases=len(cases),episodes=4*len(cases),patterns=dict(patterns),point_metrics_reproduced=True,independent_primary_intervals=independent_ci,interval_scope="Flow score: all k4 ranking metrics and B0.75 budget metrics; Plus condition sensitivity also reproduced. Other intervals structurally checked only.")
    receipt=dict(status="passed",phase=phase,primary_intervals_reproduced=True,decision_gate_reproduced=not smoke,manifest_sha256=mh,benchmarks=audits,scored_velocity_evaluations=scored_nfe,preparation_velocity_evaluations=prep_nfe,max_independent_head_log_sigma_difference=head_max,intervals_independently_reproduced="Flow primary gate metrics and Plus condition sensitivity; other intervals metadata only",limitation="No independent reproduction of secondary arm/budget/comparator intervals.",record_files=[dict(path=str(p),sha256=sha(p)) for p in record_paths],preparation_files=[dict(path=str(p),sha256=sha(p)) for p in prep_paths],metrics_files=[dict(path=str(p),sha256=sha(p)) for p in metric_paths])
    with Path(a.output).open("x") as f:json.dump(receipt,f,indent=2,allow_nan=False)
    print(json.dumps(receipt))
if __name__=="__main__":main()

