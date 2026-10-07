from pathlib import Path
import json,hashlib,subprocess,datetime
import wandb
root=Path('/volt/code/frozen-flow-study')
out=root/'examples/flow_probe/results/fa87c99062b5/final-review'
b=Path('/volt/artifacts/flow-probe')
assert json.loads((out/'final-review.json').read_text())['status']=='approved_for_final_publication'
assert json.loads((out/'final-test-review.json').read_text())['status']=='approved'
assert hashlib.sha256((out/'FINAL_REVIEW_SUPPLEMENT.md').read_bytes()).hexdigest()=='fcb59571bda47c7a5b46957363b8c508c63cedf662cfdf2c145f236a8389ce80'
with (b/'final-publication-start.json').open('x') as f:
 json.dump({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'evaluations_launched':0,'command':'.venv/bin/python -u /volt/artifacts/flow-probe/publish-final-review.py'},f,indent=2)
index={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()}
(out/'SHA256.json').write_text(json.dumps(index,indent=2))
run=wandb.init(project='pi05-two-evaluation-ranking',job_type='final-independent-review',config={'manifest_sha256':'fa87c99062b537cac51004f251394a11c7517e17f119acd827c4ee05359240db','decision':'STOP_no_controller','evaluations_launched':0})
artifact=wandb.Artifact('final-independent-review-fa87c99062b5',type='research-report')
artifact.add_dir(str(out))
logged=run.log_artifact(artifact);logged.wait()
receipt={'status':'published_to_wandb','artifact':logged.qualified_name,'artifact_digest':logged.digest,'wandb_url':run.url,'decision':'STOP_no_controller','evaluations_launched':0,'base_results_commit':'614c3d8209c5a2848511fa7d4158d1da8fab8ce2'}
(out/'PUBLICATION.json').write_text(json.dumps(receipt,indent=2))
subprocess.run(['git','add','-f',str(out.relative_to(root))],cwd=root,check=True)
staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=root,text=True).splitlines()
assert all(x.startswith(str(out.relative_to(root))+'/') for x in staged)
subprocess.run(['git','commit','-m','Publish final independent flow-ranking review and STOP conclusion'],cwd=root,check=True)
subprocess.run(['git','push','origin','HEAD:codex/pi05-two-evaluation-ranking'],cwd=root,check=True)
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/pi05-two-evaluation-ranking'],cwd=root,text=True).split()[0]
assert head==remote
api_art=wandb.Api().artifact(logged.qualified_name,type='research-report')
assert api_art.state=='COMMITTED'
receipt.update(status='published_verified',commit=head,artifact_state=api_art.state)
(b/'final-publication-receipt.json').write_text(json.dumps(receipt,indent=2))
run.summary.update(receipt);run.finish()
print(json.dumps(receipt))
