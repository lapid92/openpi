"""Prespecified ranking, metadata-budget prefix and paired cluster uncertainty."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
LABELS=("rescue","regression","unchanged_success","unchanged_failure")
BUDGETS=(.25,.5,.75,1.)
STEPS=(2,4,10)
def classification(s1,sk):
    return ("unchanged_success" if sk else "regression") if s1 else ("rescue" if sk else "unchanged_failure")
def auc(scores,pos):
    scores=np.asarray(scores,float);pos=np.asarray(pos,bool)
    n=int(pos.sum());m=len(pos)-n
    if not n or not m:return None
    order=np.argsort(scores,kind="stable");s=scores[order];p=pos[order]
    ends=np.r_[np.flatnonzero(s[1:]!=s[:-1])+1,len(s)]
    cum=np.cumsum(p);a=np.diff(np.r_[0,cum[ends-1]]);b=np.diff(np.r_[0,ends])-a
    neg_before=np.cumsum(b)-b
    return float(np.sum(a*(neg_before+.5*b))/(n*m))
def ranking_metrics(scores,labels):
    scores=np.asarray(scores,float)
    if not np.all(np.isfinite(scores)):raise ValueError("Nonfinite score")
    labels=np.asarray(labels);p=labels=="rescue";n=int(p.sum())
    if len(scores)!=len(labels) or not len(scores):raise ValueError("Invalid ranking lengths")
    order=np.argsort(-scores,kind="stable");s=scores[order];y=p[order]
    ends=np.r_[np.flatnonzero(s[1:]!=s[:-1])+1,len(s)]
    tp=np.cumsum(y)[ends-1];rec=tp/n if n else np.full(len(tp),np.nan);prec=tp/ends
    ap=float(np.sum(np.diff(np.r_[0.,rec])*prec)) if n else None
    mask=(labels=="rescue")|(labels=="regression")
    return dict(ap=ap,auroc=auc(scores,p),rescue_regression_auroc=auc(scores[mask],p[mask]),
                prevalence=float(n/len(labels)),ap_lift=None if ap is None else ap-n/len(labels),
                pr=[dict(threshold=float(s[e-1]),precision=float(t/e),recall=float(t/n) if n else None,selected=int(e)) for e,t in zip(ends,tp)])
def budget_metrics(cases,k,field,budget,overhead):
    n=len(cases);cap=sum(c["cap_chunks"] for c in cases);available=budget*cap-overhead*n
    ranked=sorted(enumerate(cases),key=lambda ic:(-float(ic[1][field]),ic[1]["case_id"]))
    chosen=[];chosen_indices=set();used=0
    for index,c in ranked:
        cost=(k-1)*c["cap_chunks"]
        if used+cost>available:break
        chosen.append(c);chosen_indices.add(index);used+=cost
    counts={l:0 for l in LABELS}
    for c in chosen:counts[classification(c["success"]["1"],c["success"][str(k)])]+=1
    total=sum(classification(c["success"]["1"],c["success"][str(k)])=="rescue" for c in cases)
    precision=counts["rescue"]/len(chosen) if chosen else None
    actual=overhead*n+sum(c["chunks"][str(k)]*k if i in chosen_indices else c["chunks"]["1"] for i,c in enumerate(cases))
    base=sum(c["chunks"]["1"] for c in cases)
    return dict(selected=len(chosen),selected_fraction=len(chosen)/n,**counts,
                precision=precision,recall=counts["rescue"]/total if total else None,
                precision_lift=None if precision is None else precision-total/n,
                net=(counts["rescue"]-counts["regression"])/n,
                selected_rescue_clusters=len({c["cluster"] for c in chosen if classification(c["success"]["1"],c["success"][str(k)])=="rescue"}),
                planned_extra_evaluations=used+overhead*n,baseline_cap=cap,extra_budget=budget*cap,
                overhead_feasible=available>=0,actual_counterfactual_evaluations=actual,
                actual_baseline_evaluations=base,actual_ratio=actual/base if base else None)
def summary(cases,field,overhead,include_pr=True):
    out={}
    for k in STEPS:
        labels=[classification(c["success"]["1"],c["success"][str(k)]) for c in cases]
        r=ranking_metrics([c[field] for c in cases],labels)
        if not include_pr:r.pop("pr")
        r["labels"]={l:labels.count(l) for l in LABELS}
        r["budgets"]={str(b):budget_metrics(cases,k,field,b,overhead) for b in BUDGETS}
        out[str(k)]=r
    return out
def flat(result):
    out={}
    for k,r in result.items():
        for m in ("ap","ap_lift","auroc","rescue_regression_auroc","prevalence"):out[k+"/"+m]=r[m]
        for b,v in r["budgets"].items():
            for m in ("precision","recall","precision_lift","net","selected","rescue","regression","actual_ratio"):
                out[k+"/"+b+"/"+m]=v[m]
    return out
def bootstrap(cases,field,overhead,cluster,replicates,seed):
    groups={}
    for c in cases:groups.setdefault(c[cluster],[]).append(c)
    keys=sorted(groups);rng=np.random.default_rng(seed);values={}
    for _ in range(replicates):
        sampled=[c for j in rng.integers(0,len(keys),size=len(keys)) for c in groups[keys[j]]]
        for key,v in flat(summary(sampled,field,overhead,False)).items():values.setdefault(key,[]).append(v)
    result={}
    for key,vs in values.items():
        valid=[v for v in vs if v is not None and math.isfinite(v)]
        result[key]=dict(low=float(np.percentile(valid,2.5)) if valid else None,
                        high=float(np.percentile(valid,97.5)) if valid else None,
                        undefined=replicates-len(valid),replicates=replicates)
    return result
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_cases(manifest,paths,benchmark,smoke=False):
    rows=[]
    for p in paths:rows.extend(json.loads(l) for l in Path(p).read_text().splitlines() if l.strip())
    mh=digest(manifest);m=json.loads(Path(manifest).read_text());spec=m["benchmarks"][benchmark]
    cond={c["condition_id"]:c for c in spec["conditions"]}
    expected={(x["condition_id"],x["seed"],x["init_index"]) for x in spec["smoke_cases"]} if smoke else {(c["condition_id"],x["seed"],x["init_index"]) for c in cond.values() for x in c["cases"]}
    declared={(c["condition_id"],x["seed"],x["init_index"]):x["initial_state_sha256"] for c in cond.values() for x in c["cases"]}
    if smoke:declared={(x["condition_id"],x["seed"],x["init_index"]):x["initial_state_sha256"] for x in spec["smoke_cases"]}
    grouped={}
    for r in rows:
        if r["status"]!="ok" or r["manifest_sha256"]!=mh or r["phase"]!=("smoke" if smoke else "main") or r["benchmark"]!=benchmark:raise ValueError("Bad record identity/status")
        key=(r["condition_id"],r["seed"],r["init_index"]);arm=r["flow_steps"]
        if key not in expected or arm not in (1,2,4,10) or arm in grouped.setdefault(key,{}):raise ValueError("Unexpected/duplicate case arm")
        if r["checkpoint_sha256"]!=m["checkpoint"]["sha256"] or r["benchmark_commit"]!=spec["commit"] or r["gpu_uuid"] not in m["gpu_uuids"]:raise ValueError("Frozen identity mismatch")
        if r["total_probe_velocity_evaluations"]!=2 or r["total_actual_velocity_evaluations"]!=r["total_velocity_evaluations"]+2:raise ValueError("Probe accounting mismatch")
        if any(c["velocity_evaluations"]!=arm for c in r["chunks"]) or r["total_velocity_evaluations"]!=arm*len(r["chunks"]):raise ValueError("Fixed accounting mismatch")
        if type(r["success"]) is not bool or r["initial_state_sha256"]!=declared[key]:raise ValueError("Outcome/state declaration mismatch")
        if not r["chunks"] or [x["chunk_index"] for x in r["chunks"]]!=list(range(len(r["chunks"]))):raise ValueError("Invalid chunk order")
        grouped[key][arm]=r
    if set(grouped)!=expected or any(set(a)!={1,2,4,10} for a in grouped.values()):raise ValueError("Incomplete outcomes")
    cases=[]
    for key,arms in sorted(grouped.items()):
        c=cond[key[0]];ref=arms[1]
        for r in arms.values():
            for f in ("initial_state_sha256","stabilized_state_sha256","checkpoint_sha256","gpu_uuid"):
                if r[f]!=ref[f]:raise ValueError("Pair identity mismatch "+f)
            if r["chunks"][0]["observation_sha256"]!=ref["chunks"][0]["observation_sha256"]:raise ValueError("Initial observation mismatch")
            for x,y in zip(r["chunks"],ref["chunks"]):
                for f in ("noise_sha256","simulator_rng_sha256"):
                    if x[f]!=y[f]:raise ValueError("Noise/RNG pair mismatch")
            p=r["initial_probe"]
            if p!=ref["initial_probe"]: # equality audited on scientific values below; filenames legitimately differ
                for f in ("score","first_sigma","first_log_sigma","array_sha256","input_sha256","noise_sha256","head_sha256","checkpoint_sha256","manifest_sha256"):
                    if p[f]!=ref["initial_probe"][f]:raise ValueError("Probe score mismatch "+f)
        p=ref["initial_probe"]
        case_id=hashlib.sha256(("flow-probe-tie-v1"+json.dumps([benchmark,*key],separators=(",",":"))).encode()).hexdigest()
        cases.append(dict(case_id=case_id,benchmark=benchmark,condition_id=key[0],seed=key[1],init_index=key[2],family=c["family"],suite=c["suite"],category=c["category"],severity=c["severity"],
                          cluster=c["suite"]+"/"+c["family"],score=p["score"],sigma=p["first_sigma"],
                          success={str(k):arms[k]["success"] for k in arms},chunks={str(k):len(arms[k]["chunks"]) for k in arms},
                          cap_chunks=math.ceil(m["settings"]["max_policy_steps"][c["suite"]]/m["settings"]["replan_steps"])))
    return m,cases,rows
def gate(point,ci):
    r=point["4"];b=r["budgets"]["0.75"]
    checks={"adequate_rescues":r["labels"]["rescue"]>=10,"adequate_regressions":r["labels"]["regression"]>=10,
            "selected_rescues":b["rescue"]>=10,"selected_clusters":b["selected_rescue_clusters"]>=5,
            "precision":b["precision"] is not None and b["precision"]>=.1,"recall":b["recall"] is not None and b["recall"]>=.2}
    for key,threshold in [("4/0.75/precision_lift",0),("4/0.75/net",0),("4/rescue_regression_auroc",.5)]:
        v=ci[key];checks[key]=v["low"] is not None and v["low"]>threshold and v["undefined"]/v["replicates"]<=.05
    return dict(pass_=all(checks.values()),checks=checks,decision="propose_separate_test_only_if_both_benchmarks_pass" if all(checks.values()) else "STOP_no_controller")
def main():
    p=argparse.ArgumentParser();p.add_argument("--manifest",required=True);p.add_argument("--records",nargs="+",required=True);p.add_argument("--output",required=True);p.add_argument("--benchmark",required=True);p.add_argument("--smoke",action="store_true");a=p.parse_args()
    m,cases,rows=load_cases(a.manifest,a.records,a.benchmark,a.smoke)
    reps=200 if a.smoke else 10000
    out=dict(status="complete",phase="smoke" if a.smoke else "main",manifest_sha256=digest(a.manifest),benchmark=a.benchmark,cases=len(cases),episodes=len(rows),records=[dict(path=x,sha256=digest(x)) for x in a.records],actual_scored_velocity_evaluations=sum(r["total_actual_velocity_evaluations"] for r in rows),patterns={},comparators={})
    for c in cases:
        v="".join("S" if c["success"][str(k)] else "F" for k in (1,2,4,10));out["patterns"][v]=out["patterns"].get(v,0)+1
    for field,overhead in [("score",2),("sigma",1),("severity",0)]:
        pt=summary(cases,field,overhead);ci=bootstrap(cases,field,overhead,"cluster",reps,20261005)
        out["comparators"][field]=dict(point=pt,primary_cluster="suite/family",intervals=ci)
        if a.benchmark=="libero_plus":out["comparators"][field]["condition_sensitivity"]=bootstrap(cases,field,overhead,"condition_id",reps,20261005)
        if field!="score":out["comparators"][field]["equal_probe_overhead_sensitivity"]=summary(cases,field,2)
    out["descriptive_strata"]={}
    for field in ("cluster","condition_id","category","severity"):
        strata={}
        for c in cases:strata.setdefault(str(c[field]),[]).append(c)
        out["descriptive_strata"][field]={key:dict(cases=len(cs),comparisons={str(k):{label:sum(classification(c["success"]["1"],c["success"][str(k)])==label for c in cs) for label in LABELS} for k in STEPS}) for key,cs in strata.items()}
    out["decision_gate"]=dict(status="excluded_smoke") if a.smoke else gate(out["comparators"]["score"]["point"],out["comparators"]["score"]["intervals"])
    target=Path(a.output)
    with target.with_suffix(".cases.jsonl").open("x") as f:
        for c in cases:f.write(json.dumps(c,sort_keys=True)+"\n")
    with target.open("x") as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps({"output":str(target),"cases":len(cases),"episodes":len(rows),"decision_gate":out["decision_gate"]}))
if __name__=="__main__":main()
