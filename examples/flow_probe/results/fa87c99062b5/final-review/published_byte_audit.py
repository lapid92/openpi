import pathlib,json,hashlib,subprocess,wandb,datetime
root=pathlib.Path('/volt/artifacts/flow-probe');run=root/'runs';s=json.loads((run/'full-status.json').read_text());commit=s['published_commit'];dest=pathlib.Path('/volt/code/frozen-flow-study/examples/flow_probe/results/fa87c99062b5/full');repo=pathlib.Path('/volt/code/frozen-flow-study')
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
m=json.load(open(repo/'examples/flow_probe/protocol.json'))
for p,h in m['source_sha256'].items():assert sha(p)==h,p
rm=json.load(open(repo/'examples/flow_probe/recovery/recovery-manifest.json'))
for p,h in rm['files'].items():assert sha(p)==h,p
for k in ['original_manifest','approved_smoke','smoke_audit']:assert sha(rm[k]['path'])==rm[k]['sha256']
idx=json.loads((dest/'raw-shards-index.json').read_text());total=0;shards=0
for entry in idx:
 h=hashlib.sha256();n=0
 for part in entry['shards']:
  p=dest/'raw-shards'/part['path'];data=subprocess.check_output(['git','show',commit+':'+str(p.relative_to(repo))],cwd=repo)
  assert hashlib.sha256(data).hexdigest()==part['sha256'];h.update(data);n+=len(data.splitlines());shards+=1
 assert h.hexdigest()==entry['source_sha256']==sha(entry['source_path']) and n==entry['lines']
 total+=n
audit=json.loads((run/'full-independent-audit.json').read_text())
for k in ['record_files','preparation_files','metrics_files']:
 for e in audit[k]:assert sha(e['path'])==e['sha256']
for n in ['full-independent-audit.json','full-REPORT.md','libero-metrics.json','libero_plus-metrics.json','libero-recording-audit.json','libero_plus-recording-audit.json','full-COMMANDS.md']:
 p=dest/n;assert subprocess.check_output(['git','show',commit+':'+str(p.relative_to(repo))],cwd=repo)==p.read_bytes()==(run/n).read_bytes()
api=wandb.Api();arts=[]
for f in ['probe_artifact','wandb_artifact']:
 a=api.artifact(s[f]);assert a.state=='COMMITTED';arts.append({'name':a.qualified_name,'state':a.state,'entries':len(a.manifest.entries),'digest':a.digest})
pidx=json.loads((dest/'full-probe-artifact-index.json').read_text());a=api.artifact(s['probe_artifact'])
for e in pidx['entries']:
 assert e['artifact_relative_path'] in a.manifest.entries
 assert sha(e['raw_path'])==e['raw_sha256']
result={'status':'passed','reviewed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commit':commit,'source_hashes':len(m['source_sha256']),'recovery_hashes':len(rm['files']),'git_shards':shards,'source_indexes':len(idx),'source_lines':total,'probe_rows':len(pidx['entries']),'artifacts':arts,'remote':subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/pi05-two-evaluation-ranking'],cwd=repo,text=True).strip()}
(root/'reviewer/published-byte-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))

