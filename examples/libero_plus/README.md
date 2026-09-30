# Frozen π₀.₅ LIBERO-Plus no-training pilot

This is a **small exploratory pilot**, not the official full LIBERO-Plus score. The
benchmark is pinned to `sylvestf/LIBERO-plus` commit
`4976dc30028e805ff8094b55501d532c48fec182`; the asset archive,
classification file, and exact task IDs are recorded in
[`pilot_manifest.json`](pilot_manifest.json). The official benchmark's full
protocol covers many more perturbations. Here, six level-1 `libero_spatial`
perturbations (two robot initial state, two camera viewpoint, two lighting)
are paired across environment seeds 7 and 11, initial-state index 0, and
fixed flow step counts 1, 2, 4, and 10. The six instances use the black-bowl
to plate task family, so category effects cannot be generalized broadly.

The manifest was written before any outcome was inspected. A separate light
condition/seed 3 case is reserved for smoke testing. Each arm uses the same
task BDDL, starting state, environment seed, preprocessing, frozen checkpoint,
action execution (first five actions per chunk), and chunk-indexed Gaussian
noise. Images are rotated 180 degrees and resized to 224 with pad, as in
`examples/libero/main.py`. Each episode gets 10 dummy stabilization actions
and at most 220 policy actions. Success means the simulator returns `done`.

The predictor score is computed from the action-expert features returned by
the **same** cached fixed-step velocity pass. It cannot select a step count.
The primary episode score is sigma at t=1 in the first generated action
chunk. Later scores belong to different closed-loop states in different arms.
The pilot records success, chunk-wise scores and flow times, action and noise
digests, velocity calls, synchronized policy latency, HTTP round-trip
latency, and complete episode wall time. An evaluation or simulation error
is recorded as `status=error` and halts the run, never silently counted as
a task failure.

For analysis, pair by exact task ID and seed. Show all arm success rates by
perturbation category, discordant wins/losses against one step, and pilot-size
uncertainty. Report whether the first-chunk score ranks pairs that gain
closed-loop success with more steps, without treating action agreement or
demonstration error as success. The decision gate is frozen in the manifest:
continue toward a new step predictor only if an extra-step arm yields at least
four paired wins, at most one loss, and wins in at least two categories. No
adaptive threshold or speedup claim is part of this branch.

## Pod commands

All commands run on pod `2vzhlaphss5c` in `/volt/data/openpi_velocity`.
The model process requires `CUDA_VISIBLE_DEVICES=3`; it validates physical
UUID `GPU-4779cdde-a260-f8ec-da6a-6fa390bc7fd7` and a single visible
`cuda:0`. The simulator runs with no CUDA device in its isolated Python 3.8
environment.

```bash
cd /volt/data/openpi_velocity
CUDA_VISIBLE_DEVICES=3 .venv/bin/python examples/libero_plus/policy_server.py \
  --checkpoint /root/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
  --head /volt/data/openpi_velocity_runs/long_3000_20260930/head.npz
# In another pod terminal:
.venv/bin/python examples/libero_plus/launch_simulator.py \
  --smoke --output /volt/data/openpi_evals/libero_plus_pilot/smoke.jsonl
.venv/bin/python examples/libero_plus/launch_simulator.py \
  --output /volt/data/openpi_evals/libero_plus_pilot/pilot.jsonl
```

The server's `/verify` endpoint is only for a separate smoke parity check
against the public fixed sampler. It performs extra reference evaluations;
the pilot client calls only `/infer`, which returns one velocity evaluation
per requested Euler step and derives sigma from the same pass.
