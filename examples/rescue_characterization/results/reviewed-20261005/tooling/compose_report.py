"""Compose the report from audited numerical draft and frozen visual artifacts."""
import json
from pathlib import Path
ROOT=Path('/volt/artifacts/rescue-characterization')
REVIEW=ROOT/'review-final-v3'
PREFIX='https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/'
def read(p):return json.loads(p.read_text())
def main():
 text=(ROOT/'report-draft.md').read_text()
 text=text.replace('# Frozen π₀.₅ rescue characterization — numerical/provenance draft','# Frozen π₀.₅ rescue characterization — final research report')
 text=text.replace('Draft only. Visual labels and the final research conclusion are pending. No predictor, adaptive rule, TVM, training, or adaptive speedup was used.','Completed fixed-sampler study and blinded qualitative review. No predictor, adaptive rule, TVM, training, or adaptive speedup was used.')
 text=text.replace('The strongest supported numerical statement is the existence of repeated paired rescue outcomes across multiple task families; a shared behavioral failure mode still requires blinded visual evidence.','The evidence supports repeated paired rescue outcomes across multiple task families, but the completed visual sample does not establish a single failure class rescued by a particular tested step count under the declared recurrence criterion. Visible acquisition and placement differences occur, alongside regressions, uncertain completion boundaries, and one simulator-affected case.')
 text=text.split('## Awaiting blinded visual labels')[0]
 text=text.replace('Current published branch codex/pi05-rescue-characterization at f631acc42573a5de5217cd57de4bf7c9b4bf8891.','Numerical/replay publication on branch codex/pi05-rescue-characterization is pinned at f631acc42573a5de5217cd57de4bf7c9b4bf8891. This reviewed report and its evidence package are published on the same branch; the containing Git commit identifies the final report version.')
 mappings={
 '/volt/artifacts/rescue-characterization/runs/libero-patterns.cases.jsonl':PREFIX+'e7537584d348/libero-patterns.cases.jsonl',
 '/volt/artifacts/rescue-characterization/runs/libero_plus-patterns.cases.jsonl':PREFIX+'e7537584d348/libero_plus-patterns.cases.jsonl',
 '/volt/artifacts/rescue-characterization/existing/summary.json':PREFIX+'existing/summary.json',
 '/volt/artifacts/rescue-characterization/existing/cases.jsonl':PREFIX+'existing/cases.jsonl',
 '/volt/artifacts/rescue-characterization/existing/audit.json':PREFIX+'existing/audit.json',
 '/volt/artifacts/rescue-characterization/replay/summary.json':PREFIX+'historical-replay/summary.json',
 '/volt/artifacts/rescue-characterization/replay/status.json':PREFIX+'historical-replay/status.json',
 '/volt/artifacts/rescue-characterization/independent-final-audit.json':PREFIX+'e7537584d348/independent-final-audit.json',
 '/volt/artifacts/rescue-characterization/final-independent-counts/audit.json':'audits/counts/audit.json',
 '/volt/artifacts/rescue-characterization/review-recovery.json':'audits/review-recovery.json',
 '/volt/code/frozen-flow-study/examples/rescue_characterization/results/e7537584d348/raw-shards/index.json':PREFIX+'e7537584d348/raw-shards/index.json',
 '/volt/artifacts/rescue-characterization/runs/full-status.json':PREFIX+'e7537584d348/full-status.json',
 '/volt/artifacts/rescue-characterization/recovery-20261005/state.json':'recovery/state.json'
 }
 for a,b in mappings.items():text=text.replace(']('+a+')',']('+b+')')
 assert '](/volt/' not in text
 summary=read(REVIEW/'review-summary.json');character=read(REVIEW/'characterization.json');coverage=read(REVIEW/'coverage-details.json')
 assert summary['status']=='complete' and summary['baseline_clips']==388 and summary['completed_overlap']==96
 assert not any(c['meets_declared_new_rescue_threshold'] for c in character['classes'])
 links={r['clip_id']:r for r in read(REVIEW/'evidence-links.json')}
 def raw(clip):return '['+clip+']('+links[clip]['immutable_raw_url']+')'
 text+='''## Completed visual review and disagreement

All 68 selected new cases were reviewed at all four arms (272 clips). The historical packet contains all 116 eligible exact-replay clips from 46 cases: 42 cases with every originally selected arm exact and four isolated exact arms from partial cases. Only fully exact cases enter historical paired-stage claims. The 432 non-equivalent arms remain excluded, including outcome-matching reconstructions. There are no unseen clips in these packets, but most new-population cases were not selected for qualitative review.

| New benchmark / class | Available cases | Reviewed cases | Reviewed families | Reviewed conditions | Distinct initial-state hashes |
|---|---:|---:|---:|---:|---:|
| LIBERO rescue | 24 | 12 | 12 | 12 | 12 |
| LIBERO regression | 32 | 12 | 12 | 12 | 12 |
| LIBERO never at tested counts | 8 | 8 | 5 | 5 | 8 |
| Plus rescue | 69 | 12 | 12 | 12 | 12 |
| Plus regression | 107 | 12 | 12 | 12 | 12 |
| Plus never at tested counts | 80 | 12 | 5 | 7 | 11 |

The matched-control selection found 2 standard and 10 Plus cases in selected suite/family/axis/severity strata; this is not universal exact-condition matching. Remaining controls use the declared family-balanced selection. The Plus control packet includes repeated initial-state content, illustrating why seed count is not automatically state diversity.

Baseline review inspected 4,656 contact-sheet frame views and 1,103 additional frame views; independent overlap inspected 1,152 sheet views and 301 additional views. Views can overlap across sampling passes and reviewers. No reviewer claimed continuous full-video viewing. Action/EEF/gripper consultation is documented in individual coverage fields: 181 baseline and 43 overlap records contain trace-consultation narratives. These are documentation counts, not proof that absent narratives imply no trace inspection, and commands alone do not establish contact.

All 388 baseline labels were frozen by SHA256 before joining outcomes. Independent overlap contains 96 clips: 70 new and 26 historical, covering all 56 selected cohort/benchmark/family clusters and every one of the 45 initially low-confidence or unclear clips. One overlap rating was revised after the same blind reviewer inspected additional early frames; its original row, correction sidecar, rationale, timestamps and new effective export are all retained. No baseline label was rewritten.

| Independent overlap | Clips | Outcome agreement | Primary-stage agreement | Confidence agreement |
|---|---:|---:|---:|---:|
| New | 70 | 43/70 (61.43%) | 48/70 (68.57%) | 32/70 (45.71%) |
| Historical exact replay | 26 | 19/26 (73.08%) | 19/26 (73.08%) | 14/26 (53.85%) |
| Combined review sample | 96 | 62/96 (64.58%) | 67/96 (69.79%) | 46/96 (47.92%) |

Overlap intentionally oversamples uncertain clips, so these are selected-sample agreement descriptions, not population annotation-reliability estimates. There are 39 clips with a stage and/or outcome disagreement. Four direct completed-versus-not-completed conflicts received coordinator frame checks and conservative unclear/low-confidence adjudications; the other substantive differences remain explicitly unresolved. Both independent labels survive every adjudication. Typical disagreements concern whether an object at a basket rim is supported inside, whether the correct drawer was opened, and task-relative placement directions.

Across all baseline labels, confidence is high for 116 clips, medium for 228, and low for 44. Visible outcome is completed for 194, not completed for 153, and unclear for 41. Twenty-five non-unclear visual outcomes disagree with the numeric predicate (15 new, 10 historical); the numeric outcomes remain unchanged. These disagreements limit fine-grained behavioral claims. Full confusion tables, original labels, coverage, alternatives, adjudications and immutable raw links are in [review summary](review/review-summary.json), [coverage](review/coverage-details.json), [disagreement register](review/disagreement-register.json), [full labels](review/label-table.json), [compact label CSV](review/labels.csv) and [raw evidence links](review/evidence-links.json).

## Failure-class evidence at particular tested counts

For a conservative descriptive class count, the one-step arm must have a visible failure stage and the rescued arm a visible completion, with medium/high baseline confidence. Where independent overlap exists, it must agree on the relevant stage/outcome and also have medium/high confidence. An unclear adjudication excludes the pair. The simulator-affected case is excluded from policy-only class interpretation. This rule does not mean every pair received two ratings; it removes observed disagreement rather than inventing consensus for unreviewed overlaps. Alternative labels remain in the table. Counts below are selected qualitative cases, not rescue-rate estimates.

| New benchmark | Tested steps | Approach | Grasp | Manipulation | Placement | Recovery |
|---|---:|---:|---:|---:|---:|---:|
'''
 classes={(c['benchmark'],c['step'],c['stage']):c for c in character['classes'] if c['cohort']=='new' and c['evidence_rule']=='clear_outcomes_no_observed_disagreement'}
 for bench in ['libero','libero_plus']:
  for step in [2,4,10]:
   counts=[classes.get((bench,step,stage),{}).get('cases',0) for stage in ['approach','grasp','manipulation','placement','recovery']]
   text+='| '+('LIBERO' if bench=='libero' else 'Plus')+' | '+str(step)+' | '+' | '.join(map(str,counts))+' |\n'
 text+='''
Each nonzero cell above spans the same number of selected families, conditions and distinct initial-state hashes as its case count. Cases can recur in multiple step columns and must not be summed as independent rescues. The largest class at a particular count is four standard grasp-stage cases at two steps; Plus has three grasp-stage cases at each tested larger count. The largest raw baseline-stage cell before the conservative screen is four cases for LIBERO and three for Plus at a particular count. Thus no reviewed new class meets the declared requirement of at least 10 rescues across three base families and five conditions.

This does not prove that such a class is absent from the unreviewed population. Only 12 of 24 standard rescued cases and 12 of 69 Plus rescued cases were selected, and label uncertainty is substantial. Broad labels such as grasp can also combine different visible events, including failed acquisition, lost retention, or interaction with the wrong object; they do not establish a single internal cause.

The numerical recurrence evidence is broader than these visual-class counts: new first-success-at-two cases span 14 standard families and 21 Plus families; first-success-at-four spans three and 12 respectively. Conversely, all four standard first-success-at-ten cases come from one moka-pot family, so that finding is concentrated in related task states rather than a broad ten-step class. The four Plus first-success-at-ten cases span four families but remain only four observations, and no larger tested count exists.

Historical exact-replay class counts remain separate. Under the same conservative visual screen, the earlier standard cohort supplies one grasp-stage case rescued at four and ten steps, and the earlier Plus cohort supplies one at four and ten. The independent historical Plus cohort supplies seven grasp-stage ten-step rescues across four families, six conditions and seven distinct state hashes, plus one placement-stage example. These are descriptive, selected reconstructions and do not satisfy the new-population confirmation rule. Their unmeasured two/four-step arms cannot identify first success.

The complete [class tables](review/characterization.json) also retain raw baseline-stage counts, regressions and controls. Historical independent controls are explicitly labelled **failed at measured 1 and 10, with 2/4 unmeasured**, preserving the builder's original selection metadata without implying never-rescued status over four counts.

## Directly checked examples and counterexamples

The coordinator inspected actual saved frames after labels were locked. These checks are unblinded supplementary observations, not additional independent ratings. Frame f is at f/20 seconds and follows action index f−1.

'''
 examples=[
('New bottle acquisition and reversal','LIBERO condition libero:libero_goal:2, seed3008, state17. The1step and10step bottle remains on the table atframe300 (15.0s). At2 and4steps it is lifted and positioned upright at the cabinet top byframe97 (4.85s). The numerical vector is F/S/S/F. Precise support/release and approach-versus-grasp cause remain alternatives.',['C090dc827d4cc4a9e19a5','C1bec2653e5f2bd8a1fea','C840d5738ae5cf6a31e98','C5b01e854d990fdcb590e']),
('New placement contrast','Plus butter/basket condition libero_plus:libero_object:81, Background Textures severity3, seed4007, state26. At1step the package stays high by the rim/hand atframes140,200,280 (7,10,14s); at2steps it is lower inside the basket with fingers apart at149 (7.45s). No contact-force inference.',['C9fe34110bc86af164805','C09a0e9a744cec9b09a6b']),
('Matched drawer control','Plus middle-drawer condition libero_plus:libero_goal:286, Robot Initial States severity3. Seed4008/state27 is numerically rescued at2/4/10; seed4005/state24 fails allfour. Both1step clips leave fronts flush. At2steps a drawer extends in both cases, but control views show two handle/front regions above the extended drawer. Exact drawer identification and earliest failure stage remain uncertain; gross opening alone is not task success.',['C6404f5a08cf6bbde17d5','Cb872c7c957a004c61816','C4ad67fde7962ff97214a','Cf422e63768b1475f0494']),
('Higher-count regression','Plus bowl/stove condition libero_plus:libero_goal:1558, Sensor Noise severity3, seed4007, state26. The1step wrist view shows the bowl roughly horizontal over the burner at90 (4.5s). At10steps it is steeply tilted at90/150 and displaced at300. The external camera is blurred; numeric success/failure is retained separately from final support uncertainty.',['C0ace75dd78c305554e51','C954dcd4956421e55f879']),
('Earlier exact reconstruction only','Old standard wine-bottle/rack condition libero:libero_goal:9, seed1006, state5. At1step the bottle separates from the hand and lies on the table at160/191 (8.0/9.55s); at10steps it is positioned lengthwise on the rack at186/205 (9.3/10.25s). Exact available-hash equivalence permits attribution; original historical frames were never recorded.',['C8c700c642d9c969d06ec','C13025af0b16d7eec66b4']),
('Simulator-affected control','Standard book/caddy seed3009/state18. Book and mug are visible initially but leave the external table view very early; the caddy is empty at520. The four preserved MuJoCo warnings prevent policy-only attribution. Primary population outcomes remain F/F/F/F.',['Ca38d9e6720868f30da6a'])
 ]
 for title,body,ids in examples:text+='**'+title+'.** '+body+' Raw records: '+', '.join(raw(x) for x in ids)+'.\n\n'
 text+='''Actual viewed frame receipts and observations are in [coordinator observations](coordinator/observations.json), [supplement](coordinator/observations-supplement.json) and the adjoining receipt files. Full videos and action arrays remain on Volt. Selected videos/contact sheets are published to W&B with the clip-to-source map; the [publication receipt](PUBLICATION.json) records immutable artifact versions and the run URL.

## Validation and delivery

The independent Test Writer reproduced all9,920 new outcomes and548 replay records from raw files, verified the frozen checkpoint/source identities and GPU UUIDs, and preserved every case. The full recording audit decoded all9,920 main videos and verified traces; the replay audit verifies its media/equivalence gate. The independent Reviewer verified every visual raw link (388 original plus116 replay links), label/source hashes, the single overlap revision, all required family/low-confidence coverage, all class counts/state hashes, the four simulator-warning associations, and explicit historical measured-arm scope. Current auxiliary tests include22 targeted tests plus8 independent adversarial tests; detailed commands/results and final approval are under [independent review evidence](audits/final-reviewer/).

[Baseline lock](review-control/baseline-lock.json), [final overlap plan](review-control/final-blind-plan.json), [analysis commands](review-control/finalize-commands-v3.sh), [independent count reproduction](audits/counts/README.md), [independent replay reproduction](audits/replay/README.md), [diagnostic audit](simulator-diagnostics/warning-audit.json) and [publication manifest](publication-manifest.json) provide the reproduction trail. Derived versions with misleading generic historical1/10control terminology were superseded before publication; original selection metadata, original raw files and all label revisions remain preserved. No scientific condition, source-pinned sampler, or completed evaluation was changed.

## Research conclusion and limits

**The completed evidence supports repeated retrospective case-specific rescue outcomes, but does not establish a shared visual failure class rescued by a particular tested step count under the declared10-case/3-family/5-condition criterion.** Acquisition and placement contrasts appear across several distinct families and states, so the phenomenon is not only one repeated initialization. Nevertheless, the sampled class counts are small, agreement is limited, and some apparent completions conflict with benchmark predicates. The standard first-success-at-ten result is specifically concentrated in one family.

Reject a monotonic “more steps fixes failures” interpretation: the new study contains37 standard and128 Plus non-monotonic cases, and higher counts introduce regressions. All overall net-success confidence intervals include zero. Do not claim overall superiority, equivalence, an online predictor, adaptive speedup, or a causal internal mechanism.

Treat the observed grasp/placement categories as hypotheses for a separately declared future characterization, not as an adaptive rule learned from these cases. Stronger class confirmation would require more independent visual evidence under a prespecified sampling plan and clearer task/contact annotation, ideally including human review. No such additional evaluations, tuning, or budget extension were performed here.
'''
 # Readable spacing in generated prose.
 for a,b in [('all9,920','all 9,920'),('and548','and 548'),('include22','include 22'),('plus8','plus 8'),('declared10-case','declared 10-case'),('contains37','contains 37'),('and128','and 128'),('allfour','all four'),('allfour','all four'),('all16','all 16')]:text=text.replace(a,b)
 out=ROOT/'final-report.md'
 with out.open('x') as f:f.write(text)
 print(out,len(text.encode()))
if __name__=='__main__':main()
