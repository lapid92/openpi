"""Make a NEW effective export; never overwrite the reviewer's original evidence."""
import json,hashlib
from pathlib import Path
ROOT=Path('/volt/artifacts/rescue-characterization')
blind=ROOT/'visual-review-new/blind'
raw_path=blind/'labels-overlap-primary.jsonl';side_path=blind/'labels-overlap-primary-revisions.jsonl'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(raw_path)=='ce19616cdba3e16e9f23a3a7c22251c1abd4e5d184d45a1c39da4ec4ac49e907'
assert sha(side_path)=='30775e2ba8ac3682b00a54aab905151fd888ee30166cc50fff2bd4b3f5dea5a0'
rows=[json.loads(x) for x in raw_path.read_text().splitlines()]
revisions=[json.loads(x) for x in side_path.read_text().splitlines()]
for rev in revisions:
 i=next(i for i,r in enumerate(rows) if r['clip_id']==rev['original_label']['clip_id'])
 assert rows[i]==rev['original_label']
 updated=dict(rev['revised_label']);updated['previous_label']=rows[i]
 updated['reviewed_at']=rev['revised_at']
 updated['revision_provenance']={'source':str(side_path),'sha256':sha(side_path),'reason':rev['reason'],'revised_at':rev['revised_at'],'original_file':str(raw_path),'original_file_sha256':sha(raw_path),'original_line':i+1,'export_note':'Canonical reviewed_at uses reviewer revision timestamp; all original fields retained in sidecar.'}
 rows[i]=updated
out=blind/'labels-overlap-primary-effective.jsonl'
with out.open('x') as f:
 for r in rows:f.write(json.dumps(r)+'\n')
spec_path=ROOT/'review-control/overlap-sources.json'
spec=json.loads(spec_path.read_text())
with (ROOT/'review-control/overlap-sources-pre-revision.json').open('x') as f:json.dump(spec,f,indent=2)
for r in spec:
 if r['cohort']=='new' and r['reviewer']=='primary':r['path']=str(out)
spec_path.write_text(json.dumps(spec,indent=2)+'\n')
print('effective_overlap_sha256',sha(out))
