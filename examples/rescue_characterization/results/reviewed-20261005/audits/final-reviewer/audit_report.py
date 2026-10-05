import json,re,hashlib,subprocess,collections
from pathlib import Path
A=Path('/volt/artifacts/rescue-characterization');report=A/'final-report.md';text=report.read_text()
read=lambda p:json.loads(p.read_text())
cases={b:[json.loads(l) for l in (A/'runs'/f'{b}-patterns.cases.jsonl').read_text().splitlines()] for b in ['libero','libero_plus']}
summ={b:next(iter(read(A/'runs'/f'{b}-patterns.json').values())) for b in cases}
for s in summ.values():
 if s.get('primary_uncertainty')=='family_sensitivity':s['uncertainty']=s['family_sensitivity']
checked=0
section='';subsection=''
for line in text.splitlines():
 if line.startswith('### '):section=line
 if line.startswith('**') and line.endswith('**'):subsection=line
 if not line.startswith('|'):continue
 cells=[x.strip() for x in line.strip('|').split('|')]
 if section=='### First successful tested count among one-step failures' and cells[0] in cases:
  b,step=cells[:2];s=summ[b];n=s['first_success_among_one_step_failures'][step];ci=s['uncertainty']['intervals']['first_'+step+'_among_failures']
  assert int(cells[2])==n
  assert cells[3]==f"{100*n/s['one_step_failures']:.2f}%"
  assert cells[4].removesuffix('%')==f"[{100*ci['lower']:.2f}, {100*ci['upper']:.2f}]"
  assert cells[5]==f"{100*n/s['cases']:.2f}%";checked+=1
 elif section=='### Rescues and regressions versus one step' and cells[0] in cases:
  b,step=cells[:2];s=summ[b];d=s['steps'][step];ci=s['uncertainty']['intervals']
  assert cells[2]==f"{s['one_step_successes']+d['rescues']-d['regressions']}/{s['cases']}"
  assert cells[3]==f"{d['rescues']}/{s['one_step_failures']} ({100*d['rescue_rate_among_failures']:.2f}%)"
  assert cells[5]==f"{d['regressions']}/{s['one_step_successes']} ({100*d['regression_rate_among_successes']:.2f}%)"
  assert cells[6]==f"{100*d['net_success_difference']:+.2f}"
  for pos,key in [(4,step+'_rescue_rate_among_failures'),(7,step+'_net_success_difference')]:
   c=ci[key];assert cells[pos].removesuffix('%')==f"[{100*c['lower']:.2f}, {100*c['upper']:.2f}]"
  checked+=1
 elif section=='### All observed new success vectors' and cells[0].startswith(('fail/','succeed/')):
  assert [int(x) for x in cells[1:]]==[summ[b]['patterns'][cells[0]] for b in cases];checked+=1
 elif section=='### Task, axis and severity concentration' and len(cells)==6 and cells[0] not in ['Suite','category','severity','Family (suite)'] and not cells[0].startswith('-'):
  b='libero_plus' if 'plus' in subsection.lower() else 'libero'
  field='suite' if 'suite' in subsection else 'category' if 'category' in subsection else 'severity' if 'severity' in subsection else None
  if field:
   subset=[x for x in cases[b] if str(x[field])==cells[0]]
   assert len(subset)==int(cells[1])
   fail=[x for x in subset if not x['success_vector'][0]]
   assert len(fail)==int(cells[2])
   first=collections.Counter(x['first_success'] for x in fail)
   assert cells[3]==' / '.join(str(first[k]) for k in ['2','4','10','never'])
   for pos,k in [(4,'rescue_steps'),(5,'regression_steps')]:
    assert cells[pos]==' / '.join(str(sum(step in x[k] for x in subset)) for step in [2,4,10]),(b,field,cells)
   checked+=1
  elif 'concentration' in subsection:
   name,suite=cells[0].rsplit(' (',1);suite=suite[:-1];subset=[x for x in cases[b] if x['family']==name and x['suite']==suite]
   fail=[x for x in subset if not x['success_vector'][0]]
   assert [int(x) for x in cells[1:]]==[len(subset),len(fail),sum(bool(x['rescue_steps']) for x in fail),sum(not x['rescue_steps'] for x in fail),len({x['initial_state_sha256'] for x in fail})];checked+=1
cache={};github=0;relative=[]
for url in re.findall(r'\]\(([^)]+)\)',text):
 if url.startswith('https://github.com/lapid92/openpi/blob/'):
  rev,path=url.split('/blob/',1)[1].split('/',1);path,sep,anchor=path.partition('#')
  if (rev,path) not in cache:cache[(rev,path)]=subprocess.check_output(['git','-C','/volt/code/frozen-flow-study','show',rev+':'+path])
  if anchor:
   n=int(anchor.removeprefix('L')); assert 0<n<=len(cache[(rev,path)].splitlines())
  github+=1
 elif not url.startswith('https://'):relative.append(url)
r={'status':'numerical_tables_and_immutable_targets_passed','report_sha256':hashlib.sha256(report.read_bytes()).hexdigest(),'table_rows_checked':checked,'github_link_occurrences_resolved':github,'relative_targets_for_package_audit':sorted(set(relative))}
(A/'final-reviewer/report-audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
