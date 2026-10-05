#!/usr/bin/env bash
set -euo pipefail
cd /volt/artifacts/rescue-characterization/review-tooling
PYTHON=/volt/code/frozen-flow-study/.venv/bin/python
CONTROL=/volt/artifacts/rescue-characterization/review-control
OUTPUT=/volt/artifacts/rescue-characterization/review-final
"$PYTHON" review_evidence.py join --lock "$CONTROL/baseline-lock.json" --overlap-sources "$CONTROL/overlap-sources.json" --cross-plan "$CONTROL/final-blind-plan.json" --adjudications "$CONTROL/adjudications.json" --output "$OUTPUT"
"$PYTHON" characterize_labels.py --joined "$OUTPUT/label-table.json" --output "$OUTPUT/characterization.json"
"$PYTHON" link_evidence.py --joined "$OUTPUT/label-table.json" --output "$OUTPUT/evidence-links.json"
