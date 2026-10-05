# Volt-only execution

All scripts run in /volt/code/frozen-flow-study on pod uz2ptakxucbe, branch codex/pi05-two-evaluation-ranking. No local simulation or study output. No training or TVM.

1. Build conditions once in the respective pinned simulator environments. The builder reads only prior declarations and benchmark metadata; generated declaration and separation files are committed.
2. Run .venv/bin/python -m pytest -q examples/flow_probe/test_probe.py examples/flow_probe/test_analysis.py. Independent Reviewer approves concrete code and Test Writer independently audits case separation.
3. Freeze once: .venv/bin/python examples/flow_probe/assemble.py. Commit and push protocol/code before any new scored or smoke inference.
4. Launch one detached smoke supervisor: .venv/bin/python -u examples/flow_probe/supervise.py --manifest examples/flow_probe/protocol.json --output /volt/artifacts/flow-probe/runs --phase smoke
5. Read smoke-status.json, raw parity preparations, both metrics, media audits and smoke-independent-audit.json. Reviewer checks actual artifacts. On failure inspect process UUIDs/commands/locks/records; preserve failed state, no automatic retry or substitution.
6. Only after published independent smoke gate, launch one full supervisor with the same command and --phase full. It refuses existing phase state, pins all source/checkpoint/head/assets, requires all cases, and publishes lossless small raw shards plus raw probe tensor artifacts. It runs independent raw/metric audits before report/publication.
7. Final Reviewer reviews actual raw audits, intervals, warnings, report and Git diff. Publish final independent review evidence and conclusion. Only both primary benchmark gates permit proposing a separate closed-loop test; nothing in this study implements that controller.

Preparation, smoke and full retain exact commands/environment/PIDs and times in phase-status and COMMANDS files. W&B project pi05-two-evaluation-ranking. Full maximum96h; smoke maximum8h; no outcome-based expansion or early favorable stopping.
