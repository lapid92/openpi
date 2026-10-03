# Execution and recovery

All commands run in /volt/code/frozen-flow-study on Volt pod uz2ptakxucbe.

Already launched once:

    bash examples/rescue_characterization/run_detached.sh smoke

The detached continuation is launched once with:

    setsid -f bash -c 'trap "" HUP; exec .venv/bin/python -u examples/rescue_characterization/pipeline.py' </dev/null >/volt/artifacts/rescue-characterization/pipeline.log 2>&1

It waits for published smoke results, independently validates the44episodes and32real-checkpoint parity checks, then launches the full9,920episode scan. After independent raw-record recount and media audit, it runs548historical replay episodes. It publishes code/results/audits to codex/pi05-rescue-characterization. Visual review remains explicitly pending after numeric execution; this is not a completed research conclusion.

State: /volt/artifacts/rescue-characterization/pipeline-status.json
Smoke/full: /volt/artifacts/rescue-characterization/runs/{smoke,full}-status.json
Replay: /volt/artifacts/rescue-characterization/replay/status.json
Independent gates: /volt/artifacts/rescue-characterization/independent-{smoke,final}-audit.json

Do not relaunch either preceding study. Do not duplicate the pipeline or any supervisor. An existing pipeline status file causes a fail-closed refusal; there is no automatic retry or state deletion. On interruption inspect its child supervisors, locks, processes and output ledgers before explicit recovery. SIGTERM/SIGINT are recorded; an existing child may still be running. Preserve failed attempts and commands. A scientific condition is never substituted after outcomes. The independent gate intentionally rejects unplanned retry/preparation history until it has been explicitly audited.

The main manifest e7537584d34855c24c1a38ca11da4f7e479b9bcbfc53d2cc7ec8f7ff46eadf9d is immutable. The separately declared historical replay manifest d9cfdf88561ec89206bd78d7104671e2d72e28cf48b12c947fc59cfd2b18cbdb keeps548episodes outside new population estimates. Replay adds an estimated3–8wall hours /12–32allocatedH100hours and2–15GiBstorage, with a12h hard cap and100GiBreserve. Its1/10cases remain partial: no2/4measurements are imputed.

An exact replay matches all historical recorded initial/stabilized state hashes, chunk action/observation/noise hashes, counts, length and outcome. Old videos were absent, so this cannot establish historical frame-by-frame equality. Any mismatch is reconstruction-not-equivalent and excluded from old-case visual attribution.

W&B project: https://wandb.ai/arm-aair-idit/pi05-rescue-characterization . Supervisors record run URLs, exact child commands and evidence paths. All media and numerical traces stay on the pod; JSONL records include their content hashes. Review selected videos using BLIND_REVIEW.md, then publish label coverage, disagreements, raw links and limitations. Do not infer an online predictor or adaptive speedup.
