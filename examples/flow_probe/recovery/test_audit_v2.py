"""Regression evidence for five-action replanning; reads preserved smoke only."""
import hashlib,json,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
class AuditRecoveryTests(unittest.TestCase):
 def test_only_declared_audit_replacement(self):
  old=(ROOT/"examples/flow_probe/independent_audit.py").read_text()
  new=(ROOT/"examples/flow_probe/recovery/audit_v2.py").read_text()
  before='np.concatenate([ch["actions"] for ch in chunks])[:r["policy_steps"]]'
  after='np.concatenate([ch["actions"][:m["settings"]["replan_steps"]] for ch in chunks])[:r["policy_steps"]]'
  self.assertEqual(old.count(before),1)
  self.assertEqual(old.replace(before,after),new)
 def test_trailing_unexecuted_forecasts_excluded(self):
  chunks=[{"actions":np.arange(70).reshape(10,7)},{"actions":np.arange(70,140).reshape(10,7)}]
  expected=np.vstack([np.arange(35).reshape(5,7),np.arange(70,84).reshape(2,7)])
  actual=np.concatenate([c["actions"][:5] for c in chunks])[:7]
  np.testing.assert_array_equal(actual,expected)
  self.assertFalse(np.array_equal(np.concatenate([c["actions"] for c in chunks])[:7],expected))
 def test_all_44_preserved_smoke_traces(self):
  manifest=json.loads((ROOT/"examples/flow_probe/protocol.json").read_text())
  self.assertEqual(manifest["settings"]["replan_steps"],5)
  rows=[]
  for benchmark in ["libero","libero_plus"]:
   path=Path("/volt/artifacts/flow-probe/runs")/(benchmark+"-smoke.jsonl")
   rows.extend(json.loads(l) for l in path.read_text().splitlines() if l)
  self.assertEqual(len(rows),44);total=0
  for row in rows:
   self.assertEqual(row["status"],"ok")
   actions=[]
   # Independent explicit loop, not production concat reconstruction.
   for chunk in row["chunks"]:
    for i in range(5):
     if len(actions)<row["policy_steps"]:actions.append(chunk["actions"][i])
   with np.load(row["recording"]["trace_path"],allow_pickle=False) as data:
    np.testing.assert_array_equal(np.asarray(actions),data["actions"])
   total+=len(actions)
  self.assertGreater(total,0)
if __name__=="__main__":unittest.main()

