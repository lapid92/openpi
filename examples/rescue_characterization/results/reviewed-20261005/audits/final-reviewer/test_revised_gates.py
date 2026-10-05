import sys,json,unittest,subprocess,tempfile
from pathlib import Path
sys.path.insert(0,'/volt/artifacts/rescue-characterization/review-tooling')
from characterize_labels import clear_failure,clear_completion
from test_characterize_labels import row
class RevisedGateTests(unittest.TestCase):
 def test_history_and_controls_never_qualify(self):
  rows=[]
  for cohort in ['new','historical']:
   for i in range(10):
    for arm,success in [(1,False),(2,True)]:
     r=row(cohort+str(i),arm=arm,cohort=cohort,success=success);r['study']=cohort;rows.append(r)
  for i in range(10):
   for arm in [1,2]:
    r=row('control'+str(i),arm=arm);r['selection_class']='never_rescued';rows.append(r)
  with tempfile.TemporaryDirectory(dir='/volt/artifacts/rescue-characterization/final-reviewer') as tmp:
   a=Path(tmp)/'in.json';b=Path(tmp)/'out.json';a.write_text(json.dumps(rows))
   subprocess.run([sys.executable,'/volt/artifacts/rescue-characterization/review-tooling/characterize_labels.py','--joined',str(a),'--output',str(b)],check=True)
   d=json.loads(b.read_text())
   qualified=[c for c in d['classes'] if c['meets_declared_new_rescue_threshold']]
   self.assertEqual(len(qualified),1)
   self.assertEqual((qualified[0]['cohort'],qualified[0]['evidence_rule']),('new','clear_outcomes_no_observed_disagreement'))
   self.assertTrue(any(c['cohort']=='historical' and c['cases']==10 for c in d['classes']))
 def test_unclear_adjudication_and_low_overlap_each_exclude(self):
  for success,fn in [(False,clear_failure),(True,clear_completion)]:
   r=row(success=success);self.assertTrue(fn(r))
   r['independent_overlap']={'label':dict(r['baseline']['label'],confidence='low')}
   self.assertFalse(fn(r))
   r.pop('independent_overlap')
   r['adjudication']={'label':dict(r['baseline']['label'],observed_outcome='unclear',primary_stage='unclear',confidence='low')}
   self.assertFalse(fn(r))
if __name__=='__main__':unittest.main(verbosity=2)
