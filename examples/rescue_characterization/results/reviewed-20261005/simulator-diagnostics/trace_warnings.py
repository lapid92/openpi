import json,hashlib,shutil
from pathlib import Path
root=Path('/volt/artifacts/rescue-characterization')
out=root/'simulator-diagnostics'
findings=[]
for p in list((root/'runs').glob('*.log'))+list((root/'replay').glob('*.log')):
 lines=p.read_text(errors='replace').splitlines()
 for i,line in enumerate(lines):
  if 'QACC' not in line and 'simulation is unstable' not in line:continue
  following=None
  for j in range(i+1,len(lines)):
   if lines[j].startswith('{'):
    try:
     r=json.loads(lines[j])
     if 'flow_steps' in r and 'condition_id' in r:
      following={k:r.get(k) for k in ['benchmark','condition_id','seed','init_index','flow_steps','success','policy_steps','status','initial_state_sha256','stabilized_state_sha256','recording']}
      break
    except json.JSONDecodeError:pass
  findings.append({'log':str(p),'log_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'warning_line':i+1,'warning':line,'following_episode_summary':following,'summary_line':j+1 if following else None})
raws=[]
for p in (root/'runs').glob('*-main-worker-*.jsonl'):
 for lineno,line in enumerate(p.read_text().splitlines(),1):
  r=json.loads(line)
  if r['benchmark']=='libero' and r['condition_id']=='libero:libero_10:5' and r['seed']==3009 and r['init_index']==18:
   raws.append({'path':str(p),'line':lineno,'record_sha256':hashlib.sha256(line.encode()).hexdigest(),**{k:r[k] for k in ['flow_steps','success','status','policy_steps','recording']}})
d={'warnings':findings,'affected_raw_records':raws,'episode_error_rows':sum(r['status']!='ok' for r in raws),'interpretation':'Four simulator instability warnings correspond to all four arms of one predeclared standard case. Preserve the case in all primary counts; do not attribute these failures solely to the policy. No reruns or substitutions. Warning association uses sequential per-worker logs.'}
(out/'warning-audit.json').write_text(json.dumps(d,indent=2)+'\n')
shutil.copy2('/volt/code/frozen-flow-study/MUJOCO_LOG.TXT',out/'MUJOCO_LOG.TXT')
(out/'warning-excerpts.txt').write_text('\n'.join(f"{x['log']}:{x['warning_line']} {x['warning']}\nNext summary: {json.dumps(x['following_episode_summary'])}" for x in findings)+'\n')
print(json.dumps({'warnings':len(findings),'affected_raw_arms':[{k:v for k,v in r.items() if k in ['flow_steps','success','status','policy_steps','line']} for r in raws]}))
