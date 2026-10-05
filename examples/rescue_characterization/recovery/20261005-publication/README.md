# Publication-only recovery: 2026-10-05

The fixed-arm scan finished 9,920 episodes. Publication commit ae903b8 was rejected because four raw LIBERO-Plus snapshots exceeded GitHub's file-size limit. No scan evaluation or pipeline.py is rerun.

prepare preserves the original commit through local-only codex/recovery-backup-ae903b8 and copies failed statuses/logs into original-failure. All eight main raw repository snapshots are replaced by line-preserving parts strictly below39,000,000bytes. The index stores original bytes/lines/SHA256 and each part's hash and original line range. Original runs remain untouched on the pod.

Reconstruct exact bytes:

    python examples/rescue_characterization/recovery/20261005-publication/shard_records.py --index examples/rescue_characterization/results/e7537584d348/raw-shards/index.json --output-dir /volt/artifacts/rescue-characterization/reconstructed-main

Omit --output-dir for verification only. Concatenation in index order restores original bytes, including final-newline behavior. Raw source-file/line references remain valid against original runs or reconstruction.

After actual script review, continue pushes the amended unpublished commit, checks remote HEAD, corrects operational publication state with failure preserved, runs independent_audit.py --phase main, publishes the audit, invokes original replay_supervise.py once, publishes audited replay artifacts, then stops at numeric_and_replay_complete_review_pending. Failure-mode review remains pending.

Commands, return codes, errors, PIDs and timestamps are saved under /volt/artifacts/rescue-characterization/recovery-20261005. A failed continuation cannot silently restart: inspect the exact stage and child PID before another explicit recovery. Original pipeline status remains failed as historical evidence; recovery state controls continuation.

The local backup ref contains oversized blobs. Never push it with --all. Push only codex/pi05-rescue-characterization without force.
