"""Independent final aggregate reproduction; no production analysis imports."""
import collections,hashlib,json,pathlib,datetime,subprocess
import numpy as np
ROOT=pathlib.Path("/volt/code/frozen-flow-study")
RUNS=pathlib.Path("/volt/artifacts/rescue-characterization/runs")
OUT=pathlib.Path("/volt/artifacts/rescue-characterization/final-independent-counts")
PIN="e7537584d34855c24c1a38ca11da4f7e479b9bcbfc53d2cc7ec8f7ff46eadf9d"
STEPS=(1,2,4,10)
def digest(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def require(ok,msg):
 if not ok:raise ValueError(msg)
def stats(cases):
 n=len(cases);f=sum(not c["vector"][0] for c in cases);s=n-f
 first=collections.Counter(c["first_success"] for c in cases if not c["vector"][0])
 patterns=collections.Counter(c["pattern"] for c in cases)
 return dict(cases=n,one_step_failures=f,one_step_successes=s,first_success=dict(first),patterns=dict(patterns),
  nonmonotonic=sum(c["nonmonotonic"] for c in cases),
  persistent_rescues=sum(c["persists"] is True for c in cases if not c["vector"][0]),
  nonpersistent_rescues=sum(c["persists"] is False for c in cases if not c["vector"][0]),
  persists_with_larger_tested=sum(c["persists"] is True and c["first_success"]!="10" for c in cases if not c["vector"][0]),
  first10_no_larger_tested=sum(c["first_success"]=="10" for c in cases if not c["vector"][0]),
  steps={str(step):dict(rescues=sum(not c["vector"][0] and c["vector"][j] for c in cases),
                       regressions=sum(c["vector"][0] and not c["vector"][j] for c in cases)) for j,step in enumerate(STEPS) if j})
def cluster(cases,field,seed=20261005,reps=10000):
 groups=collections.defaultdict(list)
 for c in cases:groups[(c["suite"],c[field])].append(c)
 # Independent cluster-level sufficient statistics and multinomial sampling.
 arr=[]
 for cs in groups.values():
  z=stats(cs);arr.append([z["cases"],z["one_step_failures"],z["one_step_successes"]]+[z["first_success"].get(str(k),0) for k in (2,4,10,"never")]+[z["steps"][str(k)]["rescues"] for k in STEPS[1:]]+[z["steps"][str(k)]["regressions"] for k in STEPS[1:]])
 arr=np.array(arr,dtype=float);rng=np.random.default_rng(seed)
 weights=rng.multinomial(len(arr),np.repeat(1/len(arr),len(arr)),size=reps)
 draws=weights@arr
 metrics={}
 for j,k in enumerate((2,4,10,"never")):
  metrics["first_"+str(k)+"_among_all"]=draws[:,3+j]/draws[:,0]
  eligible=draws[:,1]>0;metrics["first_"+str(k)+"_among_failures"]=draws[eligible,3+j]/draws[eligible,1]
 for j,k in enumerate(STEPS[1:]):
  f=draws[:,1]>0;s=draws[:,2]>0
  metrics[str(k)+"_rescue_rate_among_failures"]=draws[f,7+j]/draws[f,1]
  metrics[str(k)+"_regression_rate_among_successes"]=draws[s,10+j]/draws[s,2]
  metrics[str(k)+"_net_success_difference"]=(draws[:,7+j]-draws[:,10+j])/draws[:,0]
 return dict(cluster=field,clusters=len(arr),replicates=reps,seed=seed,
  method="95% percentile bootstrap over suite+cluster, independent NumPy multinomial implementation; ratio of totals; undefined conditional denominators omitted; no multiplicity correction",
  intervals={k:dict(lower=float(np.quantile(v,.025)),upper=float(np.quantile(v,.975)),valid_replicates=len(v)) for k,v in metrics.items()},
  cluster_case_counts={str(k):len(v) for k,v in groups.items()})
def main():
 OUT.mkdir(exist_ok=True,parents=True)
 manifest_path=ROOT/"examples/rescue_characterization/protocol.json"
 require(digest(manifest_path)==PIN,"Manifest hash changed");m=json.load(open(manifest_path))
 source_mismatches=[p for p,h in m["source_sha256"].items() if digest(p)!=h]
 require(not source_mismatches,"Pinned source changed: "+str(source_mismatches))
 result=dict(status="passed",manifest_sha256=PIN,created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_files_verified=len(m["source_sha256"]),benchmarks={},raw_files=[])
 actual_gpus={x.strip() for x in subprocess.check_output(["nvidia-smi","--query-gpu=uuid","--format=csv,noheader"],text=True).splitlines()}
 require(actual_gpus==set(m["gpu_uuids"]),"Current GPU allocation differs")
 cp=pathlib.Path(m["checkpoint"]["path"])
 files=[dict(path=str(p.relative_to(cp)),sha256=digest(p),bytes=p.stat().st_size) for p in sorted(cp.rglob("*")) if p.is_file()]
 tree=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(",",":")).encode()).hexdigest()
 require(tree==m["checkpoint"]["sha256"],"Actual checkpoint bytes changed")
 result["actual_gpu_uuids"]=sorted(actual_gpus)
 result["actual_checkpoint_sha256"]=tree
 all_vectors=[]
 for bench in ("libero","libero_plus"):
  b=m["benchmarks"][bench];expected={(c["condition_id"],x["seed"],x["init_index"],arm):(c,x) for c in b["conditions"] for x in c["cases"] for arm in STEPS}
  found={};groups=collections.defaultdict(dict)
  for worker in range(4):
   path=RUNS/(bench+"-main-worker-"+str(worker)+".jsonl")
   result["raw_files"].append(dict(path=str(path),sha256=digest(path)))
   for line,raw in enumerate(open(path),1):
    r=json.loads(raw);key=(r["condition_id"],r["seed"],r["init_index"],r["flow_steps"])
    require(key in expected and key not in found,"Unexpected/duplicate row")
    require(r.get("status")=="ok" and type(r.get("success")) is bool,"Invalid outcome/error")
    require(all(type(r[k]) is int for k in ("seed","init_index","flow_steps","policy_steps")),"Invalid key/count type")
    c,x=expected[key]
    ident=dict(phase="main",benchmark=bench,benchmark_commit=b["commit"],manifest_sha256=PIN,checkpoint_sha256=m["checkpoint"]["sha256"],gpu_uuid=m["gpu_uuids"][worker],initial_state_sha256=x["initial_state_sha256"])
    ident.update({k:c[k] for k in ("suite","task_name","family","category","severity","prompt")})
    for k,v in ident.items():require(r.get(k)==v,"Identity mismatch "+k)
    n=r["policy_steps"];replan=m["settings"]["replan_steps"]
    require(0<n<=m["settings"]["max_policy_steps"][r["suite"]],"Invalid rollout length")
    require(len(r["chunks"])==(n+replan-1)//replan,"Chunk count mismatch")
    require(r["total_velocity_evaluations"]==len(r["chunks"])*r["flow_steps"],"Evaluation sum mismatch")
    for i,ch in enumerate(r["chunks"]):
     require(ch["chunk_index"]==i and ch["velocity_evaluations"]==r["flow_steps"],"Sampler count mismatch")
     nk=[bench,r["suite"],r["task_name"],r["seed"],r["init_index"],i]
     seed=int.from_bytes(hashlib.sha256(json.dumps(nk,separators=(",",":")).encode()).digest()[:8],"little")
     noise=np.random.default_rng(seed).standard_normal(m["noise_policy"]["shape"]).astype(np.float32)
     require(hashlib.sha256(noise.tobytes()).hexdigest()==ch["noise_sha256"],"Deterministic noise mismatch")
    r["_source"]=dict(path=str(path),line=line)
    found[key]=r;groups[key[:3]][key[3]]=r
  require(set(found)==set(expected),"Incomplete population")
  cases=[]
  for key,arms in sorted(groups.items()):
   require(set(arms)==set(STEPS),"Incomplete arms")
   ref=arms[1]
   for arm,r in arms.items():
    for f in ("initial_state_sha256","stabilized_state_sha256"):require(r[f]==ref[f],"Pair mismatch "+f)
    require(r["chunks"][0]["observation_sha256"]==ref["chunks"][0]["observation_sha256"],"Initial observation mismatch")
    for ch,rc in zip(r["chunks"],ref["chunks"]):
     require(ch["noise_sha256"]==rc["noise_sha256"] and ch["simulator_rng_sha256"]==rc["simulator_rng_sha256"],"Noise/RNG pairing mismatch")
   v=[arms[k]["success"] for k in STEPS];wins=[i for i,x in enumerate(v) if x]
   c={k:ref[k] for k in ("benchmark","condition_id","suite","task_name","family","category","severity","prompt","seed","init_index","initial_state_sha256")}
   c.update(vector=v,pattern="/".join("succeed" if x else "fail" for x in v),
            first_success=str(STEPS[wins[0]]) if wins else "never",
            persists=all(v[wins[0]:]) if wins else None,
            nonmonotonic=any(v[i] and not v[j] for i in range(4) for j in range(i+1,4)),
            raw_records={str(k):arms[k]["_source"] for k in STEPS})
   cases.append(c)
  st=stats(cases);require(len(cases)==(800 if bench=="libero" else 1680),"Wrong final case count")
  published=json.load(open(RUNS/(bench+"-patterns.json")))[m["study"]+"/main/"+bench]
  for field in ("cases","one_step_failures","one_step_successes","patterns","nonmonotonic"):require(st[field]==published[field],"Published counts mismatch "+field)
  require(st["first_success"]==published["first_success_among_one_step_failures"],"First success mismatch")
  for arm in ("2","4","10"):
   for field in ("rescues","regressions"):require(st["steps"][arm][field]==published["steps"][arm][field],"Published rescue/regression mismatch")
  st["episodes"]=len(found);st["primary_uncertainty"]=cluster(cases,"family" if bench=="libero_plus" else "condition_id")
  st["condition_sensitivity"]=cluster(cases,"condition_id",seed=20261006)
  st["family_sensitivity"]=cluster(cases,"family",seed=20261007)
  st["distributions"]={}
  for field in ("suite","family","category","severity","condition_id"):
   by=collections.defaultdict(list)
   for c in cases:by[str(c[field])].append(c)
   st["distributions"][field]={k:stats(cs) for k,cs in by.items()}
  result["benchmarks"][bench]=st;all_vectors.extend(cases)
 (OUT/"case-vectors.jsonl").write_text("".join(json.dumps(c)+"\n" for c in all_vectors))
 (OUT/"audit.json").write_text(json.dumps(result,indent=2)+"\n")
 print(json.dumps({k:{f:v[f] for f in ("cases","episodes","one_step_failures","first_success","steps","nonmonotonic","persistent_rescues","nonpersistent_rescues")} for k,v in result["benchmarks"].items()},indent=2))
if __name__=="__main__":main()

