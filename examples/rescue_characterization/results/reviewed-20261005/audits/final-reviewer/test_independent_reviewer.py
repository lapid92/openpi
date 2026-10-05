import sys,unittest,copy
sys.path.insert(0,'/volt/artifacts/rescue-characterization/review-tooling')
from review_evidence import join_report,normalize_label,validate_label,cross_plan
from characterize_labels import clear_failure,summarize

def label(clip,stage='grasp',outcome='not completed',confidence='medium'):
 return {'reviewer':'one','label':{'clip_id':clip,'primary_stage':stage,'observed_outcome':outcome,'confidence':confidence,'viewed_indices':[0],'decisive_indices':[0]}}
def row(clip,arm,success=False,selection='rescue',cohort='new',paired=True):
 return dict(cohort=cohort,clip_id=clip,group_id='g',arm=arm,raw_success=success,selection_class=selection,benchmark='libero',suite='s',family='f',condition_id='c',seed=1,init_index=1,category='standard',severity=0,paired_claim_eligible=paired,source={},max_frame_index=10)
class IndependentReviewerTests(unittest.TestCase):
 def test_all_pair_classifications(self):
  for one,other,selection,expected in [(False,True,'rescue','rescue'),(True,False,'regressed','regression'),(False,False,'rescue','both_fail'),(False,False,'never_rescued','never_rescued_control'),(True,True,'regressed','both_succeed')]:
   cat={'new/a':row('a',1,one,selection),'new/b':row('b',4,other,selection)}
   baseline={'new/a':label('a'),'new/b':label('b')}
   _,pairs,_=join_report(cat,baseline,{}, {'assignments':[]},{})
   self.assertEqual(pairs[0]['comparison'],expected)
 def test_either_partial_arm_blocks_historical_pair(self):
  for partial in ('a','b'):
   cat={'historical/a':row('a',1,False,cohort='historical'),'historical/b':row('b',10,True,cohort='historical')}
   cat['historical/'+partial]['paired_claim_eligible']=False
   _,pairs,_=join_report(cat,{k:label(v['clip_id']) for k,v in cat.items()},{},{'assignments':[]},{})
   self.assertEqual(pairs,[])
 def test_missing_one_family_overlap_not_complete(self):
  cat={'new/a':row('a',1),'new/b':row('b',1)};cat['new/b'].update(group_id='h',family='other')
  base={k:label(v['clip_id']) for k,v in cat.items()}
  over={'new/a':dict(label('a'),reviewer='two')}
  plan={'assignments':[{'cohort':'new','clip_id':'a','reviewer':'two'}]}
  _,_,summary=join_report(cat,base,over,plan,{})
  self.assertEqual(summary['status'],'cross_review_incomplete')
 def test_missing_low_confidence_overlap_not_complete(self):
  cat={'new/a':row('a',1),'new/b':row('b',2)}
  base={'new/a':label('a'),'new/b':label('b',confidence='low')}
  over={'new/a':dict(label('a'),reviewer='two')}
  plan={'assignments':[{'cohort':'new','clip_id':'a','reviewer':'two'}]}
  _,_,summary=join_report(cat,base,over,plan,{})
  self.assertEqual(summary['status'],'cross_review_incomplete')
 def test_duplicate_seed_not_counted_as_new_state(self):
  rows=[dict(row(str(i),1),group_id=str(i),initial_state_sha256='same',stabilized_state_sha256='same') for i in range(5)]
  counts=summarize(rows)
  self.assertEqual((counts['cases'],counts['distinct_initial_state_hashes'],counts['families'],counts['conditions']),(5,1,1,1))
 def test_warned_case_cannot_be_clear_policy_failure(self):
  r=dict(row('x',1),baseline=label('x'))
  self.assertTrue(clear_failure(r))
  r.update(condition_id='libero:libero_10:5',seed=3009,init_index=18)
  self.assertFalse(clear_failure(r))
if __name__=='__main__':unittest.main(verbosity=2)
