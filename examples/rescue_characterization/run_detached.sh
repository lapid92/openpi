#!/bin/bash
set -euo pipefail
phase="${1:?Use smoke or full}"
case "$phase" in smoke|full) ;; *) exit 2 ;; esac
cd /volt/code/frozen-flow-study
export RESCUE_STUDY_PHASE="$phase"
setsid -f bash -c 'trap "" HUP; exec .venv/bin/python -u examples/rescue_characterization/supervise.py --manifest examples/rescue_characterization/protocol.json --output /volt/artifacts/rescue-characterization/runs --phase "$RESCUE_STUDY_PHASE"' </dev/null > "/volt/artifacts/rescue-characterization/${phase}-supervisor.log" 2>&1
sleep 3
