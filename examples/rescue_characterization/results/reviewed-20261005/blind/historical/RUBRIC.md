# Blind failure review workflow

This directory is an unlaunched development draft. No new outcomes were read to choose the sampling design. Run only after the independently declared population completes and passes identity, pairing and recording audits.

## Packaging
Use simulator Python (NumPy/Pillow available) with build_blind_packets.py, --manifest the frozen protocol, --records all main worker JSONL files for both benchmarks, --output a new review directory, --per-class 12, --selection-seed 20261003, and a private --blind-key. The packet builder requires all declared cases and four arms; smoke is rejected. It creates no simulator or policy process. FFmpeg/ffprobe only decode saved video.

The target is up to 12 cases per benchmark in each group: any one-step failure rescued at a tested arm, one-step success regressing at any tested arm, and failure at all tested arms. Selection cycles base families, then conditions, then deterministically hashed case order. Never-rescued cases first match the selected suite/family/perturbation/severity strata where possible; remaining slots use balanced cases, with shortages recorded. All four videos are included in randomized opaque order. The selected sample is qualitative and cannot estimate population rescue rates.

## Separation
Give a fresh reviewer only blind/ and this rubric. Do not give the reviewer source JSONL, private/, the private ID key, this conversation's outcome tables, or the selection coverage counts. The reviewer must not inspect inode provenance or resolve links. Videos use opaque hard-linked filenames; the numeric NPZ omits success flags. Contact sheets include frame/action indices and seconds but no arm, success, selection class, or original condition filename.

A coordinating reviewer holds private/unblinding.json and private/selection.json. A fresh blind reviewer should not be the builder, whose process knows the arm mapping. Keep labels timestamped and finalize before unblinding.

## Rubric
Use the earliest clearly consequential failure, not an inferred internal cause.
- approach: fails to reach a useful interaction pose before meaningful contact.
- grasp: reaches the object but fails to establish or retain the required grasp.
- manipulation: grasps or contacts successfully but required opening, turning, pushing, lifting or transport fails.
- placement: transports toward the target but release, location, orientation or final arrangement fails.
- recovery: an initially disrupted action is followed by repeated unsuccessful retries or failure to re-establish progress.
- unclear: visible evidence does not distinguish these categories, or essential contact/state is occluded.
Record successful completion as completed, not as a failure category. Use a secondary recovery tag when it follows a more specific primary error.

For each opaque clip provide: observed outcome (completed/not completed/unclear), primary stage, optional secondary tag, confidence (high/medium/low), decisive frame/action indices, a literal observation, an interpretation kept separate, and whether full-video viewing was necessary. Task instruction is supplied as context. Do not infer grasp contact solely from gripper command; combine both camera views, actual gripper state, and motion.

Review contact sheets first, then full videos or denser frame crops whenever stage boundaries are ambiguous. Contact sheets sample only 12 frames and cannot alone establish all short contacts or slips. Numeric traces contain executed actions and pre-action EEF quaternion/gripper positions; there is no final post-action pose or force/contact ground truth. Action summaries do not measure intent.

Write blind/labels.jsonl with clip_id plus the fields above; no inferred arm. A second independent reviewer labels at least one clip per selected family and every low-confidence or unclear clip without access to first-review labels. Preserve both labels; adjudicate disagreements after initial labels are locked. Report agreement and exact coverage rather than silently replacing labels.

## Remote viewing without Mac files
Render or crop images on Volt. Return image bytes to the orchestrator only through tool output and an in-memory image emitter; never download or save these images locally. The opaque contact sheet paths can be read remotely by the connector. Do not print private source paths when displaying a blind clip.

## After unblinding
Join finalized labels to private mapping, retain raw file/line/hash references, and report mechanisms across independent base families and conditions. Count how many labels were clear, ambiguous, unseen, and agreed. Mark findings supported by only related states explicitly. The trial population's quantitative statistics always come from all audited records, never this stratified sample.

Historical replay packets require a separate input adapter and historical equivalence gate. Do not mix historical outcome-selected replays with new population packets. Only exactly verified replay videos may be labeled as evidence for the original record; divergent replays remain separate.


Packet scope: describe only the provided clips. Some groups may have unshown counterparts; never infer unshown behavior. The coordinator will separately determine which paired comparisons are supported after labels are finalized.
