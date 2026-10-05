import sys,json,hashlib,subprocess,collections
from pathlib import Path
A=Path('/volt/artifacts/rescue-characterization')
T=A/'review-tooling'; O=A/'final-reviewer'
sys.path.insert(0,str(T))
import review_evidence as re
cat,evidence=re.catalog()
lock=re.read(A/'review-control/baseline-lock.json')
base=re.check_lock(lock,cat,evidence)
groups=collections.defaultdict(list)
for k,r in cat.items(): groups[(r['cohort'],r['group_id'])].append(r)
checks={}
fields=['benchmark','suite','family','condition_id','seed','init_index','initial_state_sha256','stabilized_state_sha256','study']
for key,rows in groups.items():
 for field in fields:
  assert len({str(r[field]) for r in rows})==1,(key,field)
 assert len({r['arm'] for r in rows})==len(rows)
 if key[0]=='new':assert {r['arm'] for r in rows}=={1,2,4,10}
checks['groups_metadata_and_arm_uniqueness']='passed'
plan=re.read(A/'review-control/final-blind-plan.json')
planned={re.key(r['cohort'],r['clip_id']) for r in plan['assignments']}
low={k for k,v in base.items() if v['label']['confidence']=='low' or v['label']['primary_stage']=='unclear' or v['label']['observed_outcome']=='unclear'}
families=collections.defaultdict(set)
for k,v in cat.items(): families[(v['cohort'],v['benchmark'],v['suite'],v['family'])].add(k)
assert low<=planned
assert all(planned&v for v in families.values())
checks['planned_all_low_unclear_and_family_coverage']='passed'
warn=re.read(A/'simulator-diagnostics/warning-audit.json')
assert len(warn['warnings'])==4
assert {w['following_episode_summary']['flow_steps'] for w in warn['warnings']}=={1,2,4,10}
for w in warn['warnings']:
 assert re.sha(w['log'])==w['log_sha256']
 lines=Path(w['log']).read_text().splitlines()
 assert lines[w['warning_line']-1]==w['warning']
 raw=json.loads(lines[w['summary_line']-1])
 assert (raw['condition_id'],raw['seed'],raw['init_index'],raw['success'],raw['policy_steps'])==('libero:libero_10:5',3009,18,False,520)
for r in warn['affected_raw_records']:
 line=Path(r['path']).read_text().splitlines()[r['line']-1]
 assert hashlib.sha256(line.encode()).hexdigest()==r['record_sha256']
checks['warning_log_and_raw_identity']='passed'
joined,pairs,summary=re.join_report(cat,base,{},plan,{})
assert (summary['historical_full_cases'],summary['historical_partial_cases'])==(42,4)
assert len([x for x in cat.values() if x['cohort']=='historical'])==116
checks['exact_historical_42full_4partial_116arms']='passed'
out=O/'preliminary-label-table.json'
out.write_text(json.dumps(joined))
links=O/'preliminary-links.json'
subprocess.run([sys.executable,str(T/'link_evidence.py'),'--joined',str(out),'--output',str(links)],check=True)
cache={}
for item in re.read(links):
 for ref,url in [(item['source'],item['immutable_raw_url']),(item['replay_source'],item['immutable_replay_url'])]:
  if ref is None:continue
  tail=url.split('/blob/')[1];rev,pathline=tail.split('/',1);path,num=pathline.rsplit('#L',1)
  key=(rev,path)
  if key not in cache:cache[key]=subprocess.check_output(['git','-C','/volt/code/frozen-flow-study','show',rev+':'+path]).splitlines()
  assert hashlib.sha256(cache[key][int(num)-1]).hexdigest()==ref['record_sha256'],url
checks['all_388_raw_and_116_replay_immutable_links_hash_checked']='passed'
result={'status':'preliminary_read_only_audit_passed_not_final_approval','checks':checks,'baseline':len(base),'low_unclear':len(low),'family_clusters':len(families),'planned_overlap':len(planned),'new_groups':sum(k[0]=='new' for k in groups),'historical_groups':sum(k[0]=='historical' for k in groups),'reviewed_tool_hashes':{p.name:re.sha(p) for p in T.glob('*.py')},'report_draft_sha256':re.sha(A/'report-draft.md')}
(O/'preliminary-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
