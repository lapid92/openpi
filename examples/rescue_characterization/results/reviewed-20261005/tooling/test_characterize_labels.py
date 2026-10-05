import json,subprocess,sys,tempfile,unittest
from pathlib import Path
from characterize_labels import clear_failure,clear_completion,summarize
def row(g='g',arm=1,cohort='new',stage='grasp',success=False):
 l={'observed_outcome':'not completed' if not success else 'completed','primary_stage':stage if not success else 'completed','confidence':'medium'}
 return dict(cohort=cohort,group_id=g,clip_id=g+str(arm),arm=arm,benchmark='libero',suite='s',family=g,condition_id=g,seed=1,init_index=1,paired_claim_eligible=True,raw_success=success,source={},selection_class='rescue',baseline={'label':l})
class Tests(unittest.TestCase):
 def test_diagnostic_excluded_and_disagreement_preserved(self):
  r=row();self.assertTrue(clear_failure(r))
  r.update(condition_id='libero:libero_10:5',seed=3009,init_index=18);self.assertFalse(clear_failure(r))
  r=row();r['independent_overlap']={'label':dict(r['baseline']['label'],primary_stage='placement')};self.assertFalse(clear_failure(r))
  r=row();r['independent_overlap']={'label':dict(r['baseline']['label'],confidence='low')};self.assertFalse(clear_failure(r))
  r=row(success=True);self.assertTrue(clear_completion(r))
  r['independent_overlap']={'label':dict(r['baseline']['label'],observed_outcome='unclear')};self.assertFalse(clear_completion(r))
 def test_unclear_adjudication_excludes_otherwise_clear_pair(self):
  for success,fn in [(False,clear_failure),(True,clear_completion)]:
   r=row(success=success);self.assertTrue(fn(r))
   r['adjudication']={'label':dict(r['baseline']['label'],observed_outcome='unclear',confidence='low')}
   self.assertFalse(fn(r))
 def test_counts_unique_cases_and_families(self):
  r=row();s=summarize([r,r]);self.assertEqual((s['cases'],s['families'],s['conditions']),(1,1,1))
 def test_whole_cli_threshold_no_pooling_and_partial_exclusion(self):
  rows=[]
  for i in range(10):rows += [row('g'+str(i)),row('g'+str(i),arm=2,success=True)]
  rows += [row('old',cohort='historical'),row('old',arm=2,cohort='historical',success=True)]
  rows[-2]['paired_claim_eligible']=False;rows[-1]['paired_claim_eligible']=False
  with tempfile.TemporaryDirectory() as t:
   a=Path(t)/'rows.json';b=Path(t)/'result.json';a.write_text(json.dumps(rows))
   subprocess.run([sys.executable,'characterize_labels.py','--joined',str(a),'--output',str(b)],check=True)
   d=json.loads(b.read_text());self.assertEqual(len(d['clear_rescue_pairs']),10)
   self.assertTrue(all(c['cohort']=='new' and c['cases']==10 for c in d['classes']))
   self.assertEqual(sum(c['meets_declared_new_rescue_threshold'] for c in d['classes']),1)
if __name__=='__main__':unittest.main()
