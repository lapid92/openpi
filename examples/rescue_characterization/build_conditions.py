"""Outcome-independent expanded case declaration; no result reads."""
import argparse, copy, hashlib, json, os, pathlib, runpy, subprocess
ROOT=pathlib.Path(__file__).resolve().parents[2]
HERE=pathlib.Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser(); p.add_argument("--benchmark",required=True); a=p.parse_args()
 name=a.benchmark
 old=json.loads((ROOT/"examples/frozen_flow/protocol.json").read_text())
 independent=json.loads((ROOT/"examples/selective_flow/protocol.json").read_text())
 spec=old["benchmarks"][name]
 source=(ROOT/"examples/frozen_flow/protocol.py").read_text()
 source=source[:source.index("\ndef main():")]
 source=source.replace('for category in ("Robot Initial States", "Camera Viewpoints", "Objects Layout"):','for category in ("Background Textures", "Robot Initial States", "Camera Viewpoints", "Language Instructions", "Sensor Noise", "Objects Layout", "Light Conditions"):')
 source=source.replace('for severity in (2, 3):','for severity in (3, 4):')
 source=source.replace('if x["category"] == category and x["difficulty_level"] == severity','if x["category"] == category and x["difficulty_level"] == severity and x["name"] not in EXCLUDED')
 source=source.replace('if len(families) == 2:', 'if len(families) == 3:')
 source=source.replace('not 0 < len(conditions) <= 48', 'len(conditions) != 168')
 # Original builder loads ten old states; replace case definitions after building below.
 ns={"EXCLUDED":{c["task_name"] for doc in (old,independent) for c in doc["benchmarks"]["libero_plus"]["conditions"]}}
 exec(compile(source,"frozen_build_benchmark","exec"),ns)
 specnew=ns["build_benchmark"](name,spec["root"],spec["classification_path"],str(ROOT/"third_party/libero/libero/libero/benchmark/libero_suite_task_map.py") if name=="libero_plus" else None)
 from libero.libero import benchmark
 import numpy as np
 smoke=[]
 for n,c in enumerate(specnew["conditions"]):
  suite=benchmark.get_benchmark_dict()[c["suite"]](task_order_index=0)
  states=suite.get_task_init_states(c["task_index"])
  indices=list(range(10,30)) if name=="libero" else (list(range(20,30)) if len(states)>=30 else [0]*10)
  if name=="libero_plus" and len(states)<30 and len(states)!=1: raise RuntimeError("Undeclared state shortage")
  c["cases"]=[dict(seed=(3001+i if name=="libero" else 4001+i),init_index=j,initial_state_sha256=hashlib.sha256(np.asarray(states[j]).tobytes()).hexdigest()) for i,j in enumerate(indices)]
  c["distinct_declared_initial_states"]=len(set(indices))
  c["distinct_declared_initial_state_hashes"]=len({x["initial_state_sha256"] for x in c["cases"]})
  if c["category"]=="Language Instructions":
   c["prompt"]=suite.get_task(c["task_index"]).language
   c["prompt_source"]="LIBERO-Plus task.language / perturbed BDDL language_instruction"
   if c["prompt"]!=c["bddl_language_instruction"]: raise RuntimeError("Language perturbation mismatch")
  smoke_key=c["suite"] if name=="libero" else c["category"]
  if smoke_key not in {x["smoke_stratum"] for x in smoke}:
   j=30 if len(states)>30 else 0
   smoke.append(dict(condition_id=c["condition_id"],seed=930001+n,init_index=j,initial_state_sha256=hashlib.sha256(np.asarray(states[j]).tobytes()).hexdigest(),smoke_stratum=smoke_key))
 for k in ("installed_assets_sha256","installed_asset_files","installed_assets_path"): specnew[k]=spec[k]
 specnew["smoke_cases"]=smoke
 specnew["max_episodes"]=sum(len(c["cases"])*4 for c in specnew["conditions"])
 specnew["selection_rule"]="All40 standard tasks states10:30 seeds3001:3020; Plus per suite/all7axes/severity3,4 first3 distinct families by registry ID excluding all144 previous condition names, states20:30 or one-state0 seeds4001:4010."
 if len(specnew["conditions"])!=(40 if name=="libero" else 168): raise RuntimeError("Unexpected condition count")
 out=HERE/(name+"-conditions.json")
 if out.exists(): raise RuntimeError("Refuse overwrite")
 out.write_text(json.dumps(specnew,sort_keys=True,separators=(",",":"))+"\n")
 print(name,len(specnew["conditions"]),specnew["max_episodes"])
if __name__=="__main__": main()
