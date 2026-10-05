"""Report the fixed decision gate only after independent full audit."""
import argparse,hashlib,json
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--manifest",required=True);p.add_argument("--results",required=True);p.add_argument("--output",required=True);a=p.parse_args()
 root=Path(a.results);m=json.loads(Path(a.manifest).read_text());audit=json.loads((root/"full-independent-audit.json").read_text())
 if audit["status"]!="passed" or audit["manifest_sha256"]!=sha(a.manifest):raise RuntimeError("Independent complete audit required")
 if audit.get("primary_intervals_reproduced") is not True or audit.get("decision_gate_reproduced") is not True:raise RuntimeError("Independent primary uncertainty and decision reproduction required")
 for row in audit["metrics_files"]:
  if sha(row["path"])!=row["sha256"]:raise RuntimeError("Audited metrics changed")
 metrics={b:json.loads((root/(b+"-metrics.json")).read_text()) for b in m["benchmarks"]}
 passed=all(d["decision_gate"]["pass_"] for d in metrics.values())
 text=["# Frozen two-evaluation flow ranking","", "**Decision: "+("propose a separate closed-loop test; no controller built." if passed else "STOP. The prespecified gate was not met; no adaptive controller is justified by this study.")+"**","",
 "Initial-observation ranking only; no per-chunk adaptive, causal or speedup claim. All cases are prospective state/noise tuples; standard tasks are reused, Plus conditions are unseen. Earlier rescue-study cases were not validation data.",
 "", "## Signal and cost","The fixed score is RMS(v_mid-v1)/max(RMS(v1),1e-6), float32 first ten-by-seven normalized action coordinates. Times1 and0.5; midpoint z-0.5v1. Production sampler/actions unchanged. Actual isolated probe costs TWO extra velocity evaluations plus prefix/head at the initial observation. No theoretical shared-evaluation discount is used.",
 "", "## Primary result: four steps,75% extra maximum-horizon NFE cap","",
 "| Benchmark | Cases | Rescue / regression / unchanged success / unchanged failure | Selected | Precision [95% CI] | Recall [95% CI] | Rescues / regressions selected | Gate |",
 "|---|---:|---|---:|---|---|---|---|"]
 def pct(x):return "undefined" if x is None else f"{100*x:.2f}%"
 for b,d in metrics.items():
  q=d["comparators"]["score"];r=q["point"]["4"];v=r["budgets"]["0.75"];ci=q["intervals"]
  def interval(key):
   x=ci[key];return "["+pct(x["low"])+", "+pct(x["high"])+"]"
  labels=r["labels"]
  text.append(f"| {b} | {d['cases']} | {labels['rescue']} / {labels['regression']} / {labels['unchanged_success']} / {labels['unchanged_failure']} | {v['selected']} | {pct(v['precision'])} {interval('4/0.75/precision')} | {pct(v['recall'])} {interval('4/0.75/recall')} | {v['rescue']} / {v['regression']} | {d['decision_gate']['pass_']} |")
 for b,d in metrics.items():
  text+=["",f"### {b}: all counts and budgets","",
  "| Signal | Steps | AP | Rescue prevalence | Budget | Selected | Precision | Rescues | Regressions | Net / all cases | Actual offline NFE ratio |",
  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
  for field,q2 in d["comparators"].items():
   for k,r2 in q2["point"].items():
    for budget,v2 in r2["budgets"].items():
     text.append(f"| {field} | {k} | {r2['ap']} | {r2['prevalence']} | {budget} | {v2['selected']} | {v2['precision']} | {v2['rescue']} | {v2['regression']} | {v2['net']} | {v2['actual_ratio']} |")
  text+=["",f"Full PR curves, primary cluster intervals, undefined replicate counts, condition sensitivity, equal-overhead comparisons, four-bit patterns and task/family/axis/severity denominators: [{b} metrics]({b}-metrics.json).",f"Gate checks: {json.dumps(d['decision_gate']['checks'],sort_keys=True)}",f"Actual scored study velocity evaluations across all four arms and probes: {d['actual_scored_velocity_evaluations']}.", ""]
 text+=["## Limits and provenance","",
 "The budget uses declared task horizons, never observed outcome lengths, for selection. Actual NFE ratio is an offline calculation after selection, not measured closed-loop performance. Comparator sigma charges one isolated feature evaluation; metadata charges zero. Recorded runs include the full two-evaluation probe for every arm. Different fixed policies visit different later states; no later score is aggregated.",
 "Ten thousand paired cluster-bootstrap replicates, seed20261005; standard task clusters and Plus suite/family clusters, condition sensitivity. Rare rescues, undefined replicates, singleton initial states and repeated families limit generalization. A passing gate permits only a separate test with the rule frozen here and fresh evaluation cases; failing/inconclusive gates cannot be repaired by choosing another count, budget, sign, score or subgroup.",
 "",f"Manifest SHA256: {sha(a.manifest)}",f"Checkpoint SHA256: {m['checkpoint']['sha256']}",f"Frozen head SHA256: {m['head']['sha256']}",f"Benchmark revisions: {json.dumps({b:s['commit'] for b,s in m['benchmarks'].items()})}",
 "", "[Independent raw audit](full-independent-audit.json). Exact commands and W&B publication receipts accompany the phase results. Any preserved simulator warnings/errors must be disclosed in the final human-facing review before scientific completion."]
 with Path(a.output).open("x") as f:f.write("\n".join(text)+"\n")
 print(json.dumps({"report":a.output,"gate_pass":passed}))
if __name__=="__main__":main()
