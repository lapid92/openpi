import copy,json,tempfile,unittest
from pathlib import Path
from review_evidence import normalize_label,validate_label,load_labels,lock_baseline,check_lock,cross_plan,join_report,sha,write_new

def raw(clip="CA",stage="grasp",confidence="high"):
 return dict(clip_id=clip,observed_outcome="not_completed",primary_stage=stage,confidence=confidence,
             decisive_frame_action_indices=[5],decisive_timestamps_seconds=[.25],literal_observation="Object slips at frame5.",
             interpretation="Loss of grasp.",alternative_label="unclear",full_video_necessary=True,
             viewing_coverage={"contact_sheet_frames":[0,5],"additional_video_frames":[],"continuous_full_video":False,
                               "camera_views":["agent","wrist"],"numeric_trace":False},reviewed_at="2026-10-05T09:00:00Z")
def meta(clip,success=False,arm=1,cohort="new",paired=True,family="f"):
 return dict(cohort=cohort,clip_id=clip,group_id="G",arm=arm,raw_success=success,benchmark="libero",selection_class="rescue",
  suite="s",family=family,condition_id="c",category="standard",severity=0,paired_claim_eligible=paired,source={},
  max_frame_index=10)
def labelled(r,reviewer="first"):
 return {"label":validate_label(normalize_label(r)),"reviewer":reviewer}
class Tests(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
 def tearDown(self):self.tmp.cleanup()
 def test_actual_primary_and_secondary_schema(self):
  a=validate_label(normalize_label(raw()))
  self.assertEqual(a["observed_outcome"],"not completed")
  self.assertFalse(a["full_video_viewed"]);self.assertTrue(a["full_video_necessary"])
  b=raw();b["actual_viewing_coverage"]=b.pop("viewing_coverage");b["reviewed_at_utc"]=b.pop("reviewed_at")
  self.assertEqual(validate_label(normalize_label(b))["viewed_indices"],[0,5])
 def test_missing_video_coverage_not_inferred_from_necessary(self):
  r=raw();r["viewing_coverage"].pop("continuous_full_video")
  a=validate_label(normalize_label(r))
  self.assertFalse(a["full_video_viewed"]);self.assertFalse(a["full_video_viewed_reported"])
 def test_unseen_decisive_frame_rejected(self):
  r=raw();r["decisive_frame_action_indices"]=[9]
  with self.assertRaises(ValueError):validate_label(normalize_label(r))
 def test_missing_and_nonfinite_fields_rejected(self):
  for field in ("alternative_label","decisive_timestamps_seconds","literal_observation"):
   r=raw();r.pop(field)
   with self.assertRaises(ValueError):validate_label(normalize_label(r))
  r=raw();r["decisive_timestamps_seconds"]=[float("nan")]
  with self.assertRaises(ValueError):validate_label(normalize_label(r))
 def test_unobserved_stage_must_be_unclear(self):
  r=raw();r["stage_not_observed"]=True
  with self.assertRaises(ValueError):validate_label(normalize_label(r))
 def test_duplicate_baseline_rejected(self):
  p=self.root/"labels";p.write_text(json.dumps(raw())+"\n"+json.dumps(raw())+"\n")
  with self.assertRaises(ValueError):load_labels([{"cohort":"new","reviewer":"first","path":str(p)}])
 def test_complete_lock_and_changed_file_rejected(self):
  p=self.root/"labels";p.write_text(json.dumps(raw())+"\n")
  specs=[{"cohort":"new","reviewer":"first","path":str(p)}]
  cat={"new/CA":meta("CA")}
  lock=lock_baseline(specs,{"new/CA":"first"},cat,[],self.root/"lock")
  self.assertEqual(len(check_lock(lock,cat,[])),1)
  p.write_text(p.read_text()+"\n")
  with self.assertRaises(ValueError):check_lock(lock,cat,[])
 def test_missing_baseline_and_wrong_reviewer_rejected(self):
  p=self.root/"labels";p.write_text(json.dumps(raw())+"\n")
  specs=[{"cohort":"new","reviewer":"first","path":str(p)}]
  with self.assertRaises(ValueError):lock_baseline(specs,{"new/CA":"second"},{"new/CA":meta("CA")},[],self.root/"lock")
  with self.assertRaises(ValueError):lock_baseline(specs,{"new/CA":"first","new/CB":"first"},{"new/CA":meta("CA"),"new/CB":meta("CB")},[],self.root/"lock")
 def test_crossplan_no_metadata_leak_and_low_unclear_additions(self):
  cat={"new/CA":meta("CA"),"new/CB":meta("CB")}
  labels={"new/CA":labelled(raw()),"new/CB":labelled(raw("CB","unclear","low"))}
  assignments=dict.fromkeys(cat,"first")
  public,private=cross_plan(cat,assignments,{"first":"second"},labels)
  self.assertIn("CB",[r["clip_id"] for r in public["assignments"]])
  for r in public["assignments"]:self.assertEqual(set(r),{"cohort","clip_id","reviewer"})
  with self.assertRaises(ValueError):cross_plan(cat,assignments,{"first":"first"})
 def test_historical_partial_not_paired_and_missing_overlap_visible(self):
  cat={"historical/CA":meta("CA",arm=1,cohort="historical",paired=False),
       "historical/CB":meta("CB",success=True,arm=10,cohort="historical",paired=False)}
  a=labelled(raw());b=raw("CB");b.update(observed_outcome="completed",primary_stage="completed")
  base={"historical/CA":a,"historical/CB":labelled(b)}
  plan={"assignments":[{"cohort":"historical","clip_id":"CA","reviewer":"second"}]}
  joined,pairs,summary=join_report(cat,base,{},plan,{})
  self.assertFalse(pairs);self.assertEqual(summary["status"],"cross_review_incomplete")
  self.assertEqual(summary["historical_partial_cases"],1)
 def test_agreement_preserves_baseline_and_adjudication(self):
  cat={"new/CA":meta("CA")}
  base={"new/CA":labelled(raw())};overlap={"new/CA":labelled(raw(stage="manipulation"),"second")}
  plan={"assignments":[{"cohort":"new","clip_id":"CA","reviewer":"second"}]}
  joined,pairs,summary=join_report(cat,base,overlap,plan,{"new/CA":{"rationale":"reviewed disagreement"}})
  self.assertEqual(joined[0]["baseline"]["label"]["primary_stage"],"grasp")
  self.assertEqual(summary["agreement"]["primary_stage"]["agreed"],0)
  self.assertEqual(summary["status"],"complete")
 def test_existing_output_never_overwritten(self):
  p=self.root/"out";write_new(p,{"a":1})
  with self.assertRaises(FileExistsError):write_new(p,{"b":2})
if __name__=="__main__":unittest.main(verbosity=2)



class RootReviewTests(unittest.TestCase):
 def test_failed_pair_in_rescued_case_is_not_never_control(self):
  cat={"new/CA":meta("CA",arm=1),"new/CB":meta("CB",arm=4)}
  baseline={"new/CA":labelled(raw()),"new/CB":labelled(raw("CB"))}
  _,pairs,_=join_report(cat,baseline,{},{"assignments":[]},{})
  self.assertEqual(pairs[0]["comparison"],"both_fail")
  for value in cat.values():value["selection_class"]="never_rescued"
  _,pairs,_=join_report(cat,baseline,{},{"assignments":[]},{})
  self.assertEqual(pairs[0]["comparison"],"never_rescued_control")
 def test_assignments_sort_opaque_groups(self):
  from review_evidence import initial_assignments
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/"new.json";h=Path(t)/"old.json"
   p.write_text(json.dumps([{"group_id":f"G{i:02d}","clips":[{"clip_id":f"C{i:02d}"}]} for i in reversed(range(68))]))
   h.write_text("[]")
   a=initial_assignments(p,h,["primary","secondary"],["old","old"])["assignments"]
   self.assertEqual(a["new/C00"],"primary");self.assertEqual(a["new/C33"],"primary")
   self.assertEqual(a["new/C34"],"secondary");self.assertEqual(a["new/C67"],"secondary")

class ExplicitIndexSchemaTests(unittest.TestCase):
 def test_frame_action_dictionary_preserves_observed_frame(self):
  r=raw();r["decisive_frame_action_indices"]=[{"frame":5,"last_executed_action":4}]
  self.assertEqual(validate_label(normalize_label(r))["decisive_indices"],[5])
  r["decisive_frame_action_indices"][0]["last_executed_action"]=5
  with self.assertRaises(ValueError):normalize_label(r)

class FrameAuditTests(unittest.TestCase):
 def test_mismatched_empty_or_missing_timestamp_rejected(self):
  for timestamps in ([],[.4],[.25,.3]):
   r=raw();r["decisive_timestamps_seconds"]=timestamps
   with self.assertRaises(ValueError):validate_label(normalize_label(r))
  r=raw();r["decisive_frame_action_indices"]=[];r["decisive_timestamps_seconds"]=[]
  with self.assertRaises(ValueError):validate_label(normalize_label(r))
 def test_overlap_frame_outside_clip_rejected(self):
  cat={"new/CA":meta("CA")}
  a=raw();a["decisive_frame_action_indices"]=[15];a["decisive_timestamps_seconds"]=[.75]
  a["viewing_coverage"]["additional_video_frames"]=[15]
  plan={"assignments":[{"cohort":"new","clip_id":"CA","reviewer":"second"}]}
  with self.assertRaises(ValueError):join_report(cat,{"new/CA":labelled(raw())},{"new/CA":labelled(a,"second")},plan,{})

class HistoricalPartialArmTerminologyTests(unittest.TestCase):
 def test_only_measured_one_ten_control_is_not_never(self):
  from review_evidence import comparison_kind
  a=meta("CA",cohort="historical");b=meta("CB",cohort="historical",arm=10)
  a["selection_class"]="failed_at_1_and_10_intermediates_unmeasured"
  self.assertEqual(comparison_kind(a,b),"failed_at_1_and_10_control_intermediates_unmeasured")
