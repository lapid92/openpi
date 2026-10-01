#!/bin/bash
set -euo pipefail
phase="${1:?Use smoke or full}"
case "$phase" in smoke|full) ;; *) exit 2 ;; esac
cd /volt/code/frozen-flow-study
export FROZEN_STUDY_PHASE="$phase"
setsid -f bash -c 'trap "" HUP; exec .venv/bin/python -u examples/frozen_flow/supervise.py --manifest examples/frozen_flow/protocol.json --output /volt/artifacts/frozen-flow-study/runs --phase "$FROZEN_STUDY_PHASE"' </dev/null > "/volt/artifacts/frozen-flow-study/${phase}-supervisor.log" 2>&1
sleep 3
