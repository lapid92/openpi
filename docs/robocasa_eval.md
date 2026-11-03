# RoboCasa denoising evaluation

`robocasa_eval` consolidates the evaluated pi0.5 and **corrected** SmolVLA protocols for 50 RoboCasa tasks, environment seeds 7/8/9, and denoising counts 1/2/4/10/15/20. It runs locally in the pinned GPU/simulator environments; this package does not submit or collect cloud jobs. The older SmolVLA v1 adapter and results are invalid because of state/action layout errors and are excluded.

The manifest generator carries the 50-task grouping, checkpoint revisions, model split, reset seed, 1.5× horizon, prompt, camera, and replan settings. A new manifest is generated per model/seed/count. Keep its `campaign_id` unique and preserve the exact file for evidence verification. `owner-robocasa-adapter` must point to the pinned LeRobot `lerobot/envs/robocasa.py` at `dd5313e305ff7aaa5b6c7b4769d1b26650ce250b` for SmolVLA. The rollout checks its SHA256 and audits the real first observation and predicted 50-action chunk against that file for every episode. The supplied SHA must be independently verified against the pinned LeRobot checkout.

```sh
export PYTHONPATH="$PWD/scripts:${PYTHONPATH:-}"
python -m robocasa_eval.manifest --model smolvla --seed 7 --steps 2 \
  --campaign-id example-smol-s7-n2 --owner-robocasa-adapter /path/to/pinned/lerobot/envs/robocasa.py \
  --owner-sha256 "$OWNER_SHA256" --checkpoint-file model.safetensors \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --out /path/to/cell/manifest.json
python -m robocasa_eval.worker --manifest /path/to/cell/manifest.json \
  --checkpoint /path/to/checkpoint --shard 0 --out /path/to/cell/shard-0 \
  --server-python /path/to/model/python --sim-python /path/to/simulator/python
```

Run all eight task shards for each cell, with a distinct output directory. The worker checks the requested weight file against the manifest SHA256 and saves a runtime identity receipt. This protects local file identity but does not independently prove the manifest revision label or source revision labels. Verify those against the frozen bundle receipts before claiming campaign parity. The server Python needs the pinned OpenPI/JAX or LeRobot/Torch checkpoint runtime. The simulator Python needs RoboCasa, Gymnasium, the OpenPI client, and the package on `PYTHONPATH`. The original campaign used one H100 per shard and separate pinned environments. The worker's files are local; stage and transfer them using infrastructure outside this package. For the pilot, use the original frozen pilot manifest and validation, since production manifests require 50 tasks × 50 episodes.

The rollout seeds environment construction once per task, then resets sequentially without reseeding. It records XML and initial-state hashes, deterministic per-episode and per-query policy seeds, measured action-network evaluations, query timing, and terminal status. pi0.5 uses the human instruction, padded 224 px images, EE-first 16D state, native RoboCasa action conversion, pretrain split, and 5-action replan. SmolVLA uses a bare task name, native 256 px images, base-first 16D state, signed 12D action slicing, target split, checkpoint camera/resize/normalization processors, and 50-action replan. The server verifies measured NFE. Every new SmolVLA worker run sets `SMOL_VALIDATION_PATH` and requires an exact repeated-input result at all six counts. Historical campaign repeat receipts must be reviewed separately.

`python -m robocasa_eval.report /path/to/cells --output /path/to/report` checks evidence identities, complete cell coverage, NFE, duplicates and scene hashes before writing per-seed/group/overall rates and mean/sample SD across seeds. It compares scene hashes only for matching task and episode keys and reports the observed agreement; equal environment seeds do **not** imply paired scenes.

The pi0.5 campaign had a reproducibility failure against the public submission, and that submission's original NFE is unknown. SmolVLA is target-trained, so `composite_unseen` is an evaluation label rather than proof of training-task holdout. Camera pixel equivalence to training remains unverified. A public SmolVLA full-suite score is unverified and stays null. These model protocols differ in split, prompt, camera preprocessing, and replan; report them descriptively, not as a matched head-to-head comparison.
