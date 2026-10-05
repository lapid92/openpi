"""Publish preserved smoke evidence after versioned audit correction. No evaluations."""
import datetime,fcntl,hashlib,importlib.util,json,shutil,subprocess
from pathlib import Path
ROOT=Path('/volt/code/frozen-flow-study');HERE=ROOT/'examples/flow_probe';BASE=Path('/volt/artifacts/flow-probe');RUN=BASE/'runs';REC=BASE/'recovery'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 with Path(p).open('x') as f:json.dump(d,f,indent=2)
def main():
 manifest=HERE/'protocol.json';mh=sha(manifest);m=json.loads(manifest.read_text())
 if mh!='fa87c99062b537cac51004f251394a11c7517e17f119acd827c4ee05359240db':raise RuntimeError('Wrong original manifest')
 approval=BASE/'reviewer/smoke-evidence-review.json';a=json.loads(approval.read_text())
 if a['status']!='approved_for_smoke_publication' or a['manifest_sha256']!=mh:raise RuntimeError('Reviewer approval required')
 audit=REC/'smoke-independent-audit-v2.json';d=json.loads(audit.read_text())
 if d['status']!='passed' or d['manifest_sha256']!=mh:raise RuntimeError('Audit missing')
 for field in ('record_files','preparation_files','metrics_files'):
  for x in d[field]:
   if sha(x['path'])!=x['sha256']:raise RuntimeError('Audited evidence changed')
 for p,h in m['source_sha256'].items():
  if sha(p)!=h:raise RuntimeError('Original pinned source changed')
 for x in (RUN/'full-status.json',BASE/'full-launch.json',REC/'publication-start.json'):
  if x.exists():raise RuntimeError('Existing execution/publication state; inspect before any recovery')
 lock=(RUN/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU processes remain')
 if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='codex/pi05-two-evaluation-ranking':raise RuntimeError('Wrong branch')
 write(REC/'publication-start.json',{'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest_sha256':mh,'approval_sha256':sha(approval),'audit_sha256':sha(audit),'evaluations_launched':0})
 spec=importlib.util.spec_from_file_location('original_supervisor',HERE/'supervise.py');sup=importlib.util.module_from_spec(spec);spec.loader.exec_module(sup)
 if sup.tree_hash(m['checkpoint']['path'])[0]!=m['checkpoint']['sha256'] or sha(m['head']['path'])!=m['head']['sha256']:raise RuntimeError('Frozen parameters changed')
 target=RUN/'smoke-independent-audit.json'
 if target.exists():raise RuntimeError('Existing canonical audit')
 shutil.copy2(audit,target)
 dest=HERE/'results'/mh[:12]/'smoke-recovered';dest.mkdir(parents=True,exist_ok=False)
 for p in sorted(RUN.iterdir()):
  if p.is_file() and p.suffix in ('.json','.md','.txt') and p.stat().st_size<35000000:shutil.copy2(p,dest/p.name)
 index=[]
 for p in sorted(RUN.glob('*.jsonl')):index.append(sup.shard_records(p,dest/'raw-shards'))
 write(dest/'raw-shards-index.json',index)
 shutil.copytree(REC,dest/'recovery-evidence')
 shutil.copy2(approval,dest/'smoke-evidence-review.json')
 shutil.copy2(HERE/'recovery/AMENDMENT.md',dest/'AMENDMENT.md')
 import wandb
 run=wandb.init(project='pi05-two-evaluation-ranking',job_type='smoke-publication-recovery',config={'manifest_sha256':mh,'evaluations_launched':0,'original_failed_run':'o5n6vjot'})
 receipt={'manifest_sha256':mh,'evaluations_launched':0,'original_failed_run':'o5n6vjot','wandb_url':run.url}
 try:
  artifact=wandb.Artifact('initial-flow-probes-smoke-'+mh[:12],type='initial-flow-probes');entries=[];added=set()
  for b in m['benchmarks']:
   for line in (RUN/(b+'-smoke.jsonl')).read_text().splitlines():
    r=json.loads(line);p=r['initial_probe'];path=p['raw_path'];h=p['raw_sha256']
    if sha(path)!=h:raise RuntimeError('Raw probe changed')
    relative='probes/'+h+'.npz'
    if h not in added:artifact.add_file(path,name=relative);added.add(h)
    entries.append({**{k:r[k] for k in ('benchmark','condition_id','seed','init_index','flow_steps')},'raw_path':path,'raw_sha256':h,'artifact_relative_path':relative})
  logged=run.log_artifact(artifact);logged.wait();receipt['probe_artifact']=logged.qualified_name
  write(dest/'smoke-probe-artifact-index.json',{'artifact':logged.qualified_name,'entries':entries,'manifest_sha256':mh})
  artifact=wandb.Artifact('supervisor-smoke-recovered-'+mh[:12],type='evaluation');artifact.add_dir(str(dest));artifact.add_file(str(manifest),name='protocol.json')
  logged=run.log_artifact(artifact);logged.wait();receipt['evidence_artifact']=logged.qualified_name
  receipt['status']='published_to_wandb'
  write(dest/'PUBLICATION.json',receipt)
  subprocess.run(['git','add','-f',str(dest.relative_to(ROOT))],cwd=ROOT,check=True)
  staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()
  if any(not x.startswith(str(dest.relative_to(ROOT))+'/') for x in staged):raise RuntimeError('Unrelated staged changes')
  subprocess.run(['git','commit','-m','Publish preserved smoke evidence after explicit audit correction'],cwd=ROOT,check=True)
  subprocess.run(['git','push','origin','HEAD:codex/pi05-two-evaluation-ranking'],cwd=ROOT,check=True)
  receipt.update(status='published',published_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
  write(REC/'publication-receipt.json',receipt)
  state=json.loads((RUN/'smoke-status.json').read_text())
  if state['status']!='failed':raise RuntimeError('Unexpected original state')
  state['original_failure']=state.pop('error');state.update(status='published',recovery=receipt,original_status_path=str(REC/'original-smoke-status.json'),published_commit=receipt['published_commit'],updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
  temp=RUN/'smoke-status-recovered.tmp';temp.write_text(json.dumps(state,indent=2));temp.replace(RUN/'smoke-status.json')
  run.summary.update(receipt);run.finish()
 except BaseException:
  run.finish(exit_code=1);raise
 print(json.dumps(receipt))
if __name__=='__main__':main()
