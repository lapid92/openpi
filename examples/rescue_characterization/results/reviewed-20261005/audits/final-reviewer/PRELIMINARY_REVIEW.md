# Independent preliminary reviewer assessment

Status: preliminary only; final joined labels, characterization, report, publication package, and actual final diff remain to review. No final approval is implied.

## Read-only judgments

Reviewed actual review_evidence.py, characterize_labels.py, link_evidence.py, publish_review.py, both test files, PROTOCOL.md, BLIND_REVIEW.md, and report-draft.md. Exact reviewed hashes are recorded in preliminary-audit.json.

The numerical draft correctly separates new populations from retrospective records, labels partial 1/10 first success unknown, discloses all net intervals include zero, reports the standard benchmark's failure-count inadequacy, and avoids population rates from the outcome-stratified visual sample. Its simulator diagnostic preserves primary counts and labels the exclusion count supplementary. Its discussion of exact replays does not assign divergence causality.

Actionable semantics before final packaging:
1. The initial characterize_labels.py applied meets_numeric_recurrence_threshold to historical, regression, and control classes as well as new rescues. The protocol qualifies a recurring rescue class only in the new population. Root acknowledged and is restricting/renaming the flag. Re-review required.
2. The initial clear-outcome filter checked baseline medium/high confidence but not overlap confidence; root separately identified and is correcting this.
3. Adjudications are preserved by the join but the initial characterization ignores them. An unclear coordinator adjudication must exclude the affected clear-rescue candidate explicitly. Root notified.
4. The final report must retain study-specific historical interpretation: aggregate distributions and paired-stages do not themselves carry all study/state metadata, which remains in the joined label table. No quantitative pooling of historical studies or inferred independent initial states is justified.

No labels, pinned manifests, original raw evidence, or root-owned tooling were edited by this reviewer.

## Independently executed validation

audit_preliminary.py checked the current actual artifacts:
- All 388 baseline labels match their hash lock.
- Group arms are unique; all new groups have exactly 1/2/4/10; cohort/study/benchmark/suite/family/condition/seed/index/initial-state/stabilized-state metadata agree within groups.
- Planned overlap covers all 45 low/unclear clips and all 56 selected family clusters, with 96 planned overlap labels.
- Historical packet has 42 fully exact groups and 4 partial groups, 116 eligible arms.
- All 388 raw immutable URLs and all 116 replay immutable URLs were resolved through git show and compared byte-line SHA256 against each original source reference. Passed.
- All four diagnostic log hashes and warning lines, sequential following summaries, and raw record hashes match the same standard book task, seed 3009/state 18, all four failed 520-action arms.

test_independent_reviewer.py: 6 independent synthetic tests passed, checking pair classifications, partial-arm exclusion, missing family/low-confidence overlap preventing completion, seed-versus-state counting, and warning exclusion from clear policy failure.
Existing auxiliary test discovery: 20 tests passed.
These tests are synthetic/read-only; no simulation, replay, supervisor, training, commit, push, or W&B publication was executed.

## Final gate outstanding

Review final complete overlap labels and preserved disagreements; final root adjudications and actual viewing evidence; new-only recurrence classes and counterexamples; exact raw links and study/state counts; complete report/package/diff; publication manifest hash consistency. Do not treat this preliminary review as closure.
