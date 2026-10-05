import json,hashlib,collections,csv,sys
from pathlib import Path
A=Path('/volt/artifacts/rescue-characterization');R=A/(sys.argv[1] if len(sys.argv)>1 else 'review-final');O=A/'final-reviewer'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=read(R/'label-table.json');summary=read(R/'review-summary.json');coverage=read(R/'coverage-details.json')
assert len(rows)==388
lock=read(A/'review-control/baseline-lock.json')
assert sha(A/'review-control/baseline-lock.json')==summary['baseline_lock_sha256']
for f in lock['files']:assert sha(Path(f['path']))==f['sha256']
for f in summary['overlap_source_files']:assert sha(Path(f['path']))==f['sha256']
cache={}
for r in rows:
 for field in ['baseline','independent_overlap']:
  if not r.get(field):continue
  ref=r[field]['label_source'];p=Path(ref['path'])
  assert sha(p)==ref['file_sha256']
  if str(p) not in cache:cache[str(p)]=[json.loads(x) for x in p.read_text().splitlines()]
  assert cache[str(p)][ref['line']-1]==r[field]['original_label']
  assert r[field]['label']['clip_id']==r['clip_id']
  assert all(0<=n<=r['max_frame_index'] for n in r[field]['label']['viewed_indices'])
ovs=[r for r in rows if r.get('independent_overlap')]
assert len(ovs)==96
for field in ['observed_outcome','primary_stage','confidence']:
 assert sum(r['baseline']['label'][field]==r['independent_overlap']['label'][field] for r in ovs)==summary['agreement'][field]['agreed']
for field in ['baseline','independent_overlap']:
 labs=[r[field]['label'] for r in rows if r.get(field)]
 c=coverage['viewing_coverage'][field]
 assert c['sheet_frame_views']==sum(len(l['viewing_coverage'].get('contact_sheet_frames',[])) for l in labs)
 assert c['additional_frame_views']==sum(len(l['viewing_coverage'].get('additional_video_frames',[])) for l in labs)
 assert sum(l['full_video_viewed'] for l in labs)==0
low={r['cohort']+'/'+r['clip_id'] for r in rows if r['baseline']['label']['confidence']=='low' or r['baseline']['label']['primary_stage']=='unclear' or r['baseline']['label']['observed_outcome']=='unclear'}
keys={r['cohort']+'/'+r['clip_id'] for r in ovs}
assert len(low)==45 and low<=keys
fam=lambda r:(r['cohort'],r['benchmark'],r['suite'],r['family'])
assert {fam(r) for r in rows}=={fam(r) for r in ovs}
assert len({fam(r) for r in rows})==56
assert all(r['baseline']['reviewer']!=r['independent_overlap']['reviewer'] for r in ovs)
diff=[r for r in ovs if any(r['baseline']['label'][f]!=r['independent_overlap']['label'][f] for f in ['observed_outcome','primary_stage'])]
assert len(diff)==39
register=read(R/'disagreement-register.json')
assert {(r['cohort'],r['clip_id']) for r in register if r['substantive']}=={(r['cohort'],r['clip_id']) for r in diff}
ads=[r for r in rows if r.get('adjudication')]
assert len(ads)==4 and all(r['adjudication']['label']['observed_outcome']=='unclear' for r in ads)
raw=cache[str(A/'visual-review-new/blind/labels-overlap-primary-effective.jsonl')]
orig=[json.loads(x) for x in (A/'visual-review-new/blind/labels-overlap-primary.jsonl').read_text().splitlines()]
side=[json.loads(x) for x in (A/'visual-review-new/blind/labels-overlap-primary-revisions.jsonl').read_text().splitlines()]
assert len(raw)==len(orig)==34 and len(side)==1
changes=[(a,b) for a,b in zip(orig,raw) if a!=b]
assert len(changes)==1
a,b=changes[0];rev=side[0]
assert a==rev['original_label']==b['previous_label']
assert b['reviewed_at']==rev['revised_at']
assert all(b[k]==v for k,v in rev['revised_label'].items() if k!='reviewed_at')
assert b['revision_provenance']['sha256']==sha(A/'visual-review-new/blind/labels-overlap-primary-revisions.jsonl')
assert b['revision_provenance']['original_file_sha256']==sha(A/'visual-review-new/blind/labels-overlap-primary.jsonl')
chars=read(R/'characterization.json')
assert not any(c['meets_declared_new_rescue_threshold'] for c in chars['classes'])
lookup={(r['cohort'],r['clip_id']):r for r in rows}
for c in chars['classes']:
 members=[lookup[(c['cohort'],clip)] for clip in c['clip_ids']]
 assert len({r['group_id'] for r in members})==c['cases']
 assert len({(r['suite'],r['family']) for r in members})==c['families']
 assert len({r['condition_id'] for r in members})==c['conditions']
 assert len({r['initial_state_sha256'] for r in members})==c['distinct_initial_state_hashes']
for pair in chars['clear_rescue_pairs']:
 for sidekey in ['one','other']:
  r=lookup[(pair['cohort'],pair[sidekey])]
  assert r['baseline']['label']['confidence'] in ['medium','high']
  if r.get('independent_overlap'):assert r['independent_overlap']['label']['confidence'] in ['medium','high']
  assert not r.get('adjudication')
assert len(list(csv.DictReader((R/'labels.csv').open())))==388
result={'status':'joined_artifact_checks_passed','directory':str(R),'files':{p.name:sha(p) for p in R.iterdir() if p.is_file()},'baseline':388,'overlap':96,'low_unclear_covered':45,'families_covered':56,'agreement_outcome':62,'agreement_stage':67,'substantive_differences':39,'unclear_adjudications':4,'primary_revision_preserved_exactly_one':True,'no_new_recurring_class_qualifies':True,'viewing_coverage':coverage['viewing_coverage'],'remaining':'Review final report/package/diff; derived terminology separately checked in v3-semantics-audit.json.'}
out=O/(R.name+'-audit.json');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
