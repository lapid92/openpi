"""Reuse audited fixed episode execution; retain observational initial probe separately."""
import importlib.util
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/"rescue_characterization"
sys.path.insert(0,str(BASE))
spec=importlib.util.spec_from_file_location("rescue_client",BASE/"client.py")
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
original_post=base.post_json
original_run=base.run_episode
probes=[]
def post(url,payload):
    response=original_post(url,dict(payload,probe_role=("preparation" if not url.endswith("/infer") else "episode")))
    if url.endswith("/infer") and payload["chunk_index"]==0:
        probe=response.get("initial_probe")
        if not probe:raise RuntimeError("Initial probe missing")
        probes.append(probe)
    elif url.endswith("/infer") and response.get("initial_probe") is not None:
        raise RuntimeError("Later-chunk probe forbidden")
    return response
def run(*args,**kwargs):
    probes.clear()
    row=original_run(*args,**kwargs)
    if row.get("status")=="ok":
        if len(probes)!=1:raise RuntimeError("Exactly one initial probe required")
        row["initial_probe"]=probes[0]
        row["total_probe_velocity_evaluations"]=2
        row["total_actual_velocity_evaluations"]=row["total_velocity_evaluations"]+2
    return row
base.post_json=post
base.run_episode=run
if __name__=="__main__":base.main()
