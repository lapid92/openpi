# Executed commands

## server-0

```bash
/volt/code/frozen-flow-study/.venv/bin/python /volt/code/frozen-flow-study/examples/rescue_characterization/policy_server.py --manifest /volt/code/frozen-flow-study/examples/rescue_characterization/protocol.json --gpu-uuid GPU-56335fdf-6dc8-1d64-ce39-364d4a532435 --port 8840
```

## server-1

```bash
/volt/code/frozen-flow-study/.venv/bin/python /volt/code/frozen-flow-study/examples/rescue_characterization/policy_server.py --manifest /volt/code/frozen-flow-study/examples/rescue_characterization/protocol.json --gpu-uuid GPU-b786fcd4-f878-69e6-3017-a2ae408151fd --port 8841
```

## server-2

```bash
/volt/code/frozen-flow-study/.venv/bin/python /volt/code/frozen-flow-study/examples/rescue_characterization/policy_server.py --manifest /volt/code/frozen-flow-study/examples/rescue_characterization/protocol.json --gpu-uuid GPU-967ff6df-a7e2-b65a-d051-87819b7725fc --port 8842
```

## server-3

```bash
/volt/code/frozen-flow-study/.venv/bin/python /volt/code/frozen-flow-study/examples/rescue_characterization/policy_server.py --manifest /volt/code/frozen-flow-study/examples/rescue_characterization/protocol.json --gpu-uuid GPU-0cb759da-a716-038e-f0a9-208ec138ccc7 --port 8843
```

## libero-prepare-0

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/rescue_characterization/client.py --manifest /volt/code/frozen-flow-study/examples/rescue_characterization/protocol.json --benchmark libero --server http://127.0.0.1:8840 --output /volt/artifacts/rescue-characterization/runs/libero-prepare-worker-0.jsonl --prepare-only --worker-index 0 --workers 4
```

## libero-prepare-1

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/rescue_characterization/client.py --manifest /volt/code/frozen-flow-study/examples/rescue_characterization/protocol.json --benchmark libero --server http://127.0.0.1:8841 --output /volt/artifacts/rescue-characterization/runs/libero-prepare-worker-1.jsonl --prepare-only --worker-index 1 --workers 4
```

## libero-prepare-2

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/rescue_characterization/client.py --manifest /volt/code/frozen-flow-study/examples/rescue_characterization/protocol.json --benchmark libero --server http://127.0.0.1:8842 --output /volt/artifacts/rescue-characterization/runs/libero-prepare-worker-2.jsonl --prepare-only --worker-index 2 --workers 4
```

## libero-prepare-3

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/rescue_characterization/client.py --manifest /volt/code/frozen-flow-study/examples/rescue_characterization/protocol.json --benchmark libero --server http://127.0.0.1:8843 --output /volt/artifacts/rescue-characterization/runs/libero-prepare-worker-3.jsonl --prepare-only --worker-index 3 --workers 4
```

## libero-smoke-0

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/rescue_characterization/client.py --manifest /volt/code/frozen-flow-study/examples/rescue_characterization/protocol.json --benchmark libero --server http://127.0.0.1:8840 --output /volt/artifacts/rescue-characterization/runs/libero-smoke.jsonl --smoke
```

## libero-smoke-analysis

```bash
/volt/code/frozen-flow-study/.venv/bin/python /volt/code/frozen-flow-study/examples/rescue_characterization/analyze.py --manifest /volt/code/frozen-flow-study/examples/rescue_characterization/protocol.json --benchmark libero --records /volt/artifacts/rescue-characterization/runs/libero-smoke.jsonl --output /volt/artifacts/rescue-characterization/runs/libero-smoke-summary.json --wandb-project pi05-rescue-characterization --wandb-name libero-smoke-e7537584d348 --smoke
```

## libero-smoke-recording-audit

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/rescue_characterization/recording_audit.py --manifest /volt/code/frozen-flow-study/examples/rescue_characterization/protocol.json --records /volt/artifacts/rescue-characterization/runs/libero-smoke.jsonl --output /volt/artifacts/rescue-characterization/runs/libero-smoke-recording-audit.json
```

## libero-smoke-patterns

```bash
/volt/code/frozen-flow-study/.venv/bin/python /volt/code/frozen-flow-study/examples/rescue_characterization/analysis.py --manifest /volt/code/frozen-flow-study/examples/rescue_characterization/protocol.json --benchmark libero --records /volt/artifacts/rescue-characterization/runs/libero-smoke.jsonl --output /volt/artifacts/rescue-characterization/runs/libero-smoke-patterns.json --smoke
```

## libero_plus-prepare-0

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/rescue_characterization/client.py --manifest /volt/code/frozen-flow-study/examples/rescue_characterization/protocol.json --benchmark libero_plus --server http://127.0.0.1:8840 --output /volt/artifacts/rescue-characterization/runs/libero_plus-prepare-worker-0.jsonl --prepare-only --worker-index 0 --workers 4
```

## libero_plus-prepare-1

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/rescue_characterization/client.py --manifest /volt/code/frozen-flow-study/examples/rescue_characterization/protocol.json --benchmark libero_plus --server http://127.0.0.1:8841 --output /volt/artifacts/rescue-characterization/runs/libero_plus-prepare-worker-1.jsonl --prepare-only --worker-index 1 --workers 4
```

## libero_plus-prepare-2

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/rescue_characterization/client.py --manifest /volt/code/frozen-flow-study/examples/rescue_characterization/protocol.json --benchmark libero_plus --server http://127.0.0.1:8842 --output /volt/artifacts/rescue-characterization/runs/libero_plus-prepare-worker-2.jsonl --prepare-only --worker-index 2 --workers 4
```

## libero_plus-prepare-3

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/rescue_characterization/client.py --manifest /volt/code/frozen-flow-study/examples/rescue_characterization/protocol.json --benchmark libero_plus --server http://127.0.0.1:8843 --output /volt/artifacts/rescue-characterization/runs/libero_plus-prepare-worker-3.jsonl --prepare-only --worker-index 3 --workers 4
```

## libero_plus-smoke-0

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/rescue_characterization/client.py --manifest /volt/code/frozen-flow-study/examples/rescue_characterization/protocol.json --benchmark libero_plus --server http://127.0.0.1:8840 --output /volt/artifacts/rescue-characterization/runs/libero_plus-smoke.jsonl --smoke
```

## libero_plus-smoke-analysis

```bash
/volt/code/frozen-flow-study/.venv/bin/python /volt/code/frozen-flow-study/examples/rescue_characterization/analyze.py --manifest /volt/code/frozen-flow-study/examples/rescue_characterization/protocol.json --benchmark libero_plus --records /volt/artifacts/rescue-characterization/runs/libero_plus-smoke.jsonl --output /volt/artifacts/rescue-characterization/runs/libero_plus-smoke-summary.json --wandb-project pi05-rescue-characterization --wandb-name libero_plus-smoke-e7537584d348 --smoke
```

## libero_plus-smoke-recording-audit

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/rescue_characterization/recording_audit.py --manifest /volt/code/frozen-flow-study/examples/rescue_characterization/protocol.json --records /volt/artifacts/rescue-characterization/runs/libero_plus-smoke.jsonl --output /volt/artifacts/rescue-characterization/runs/libero_plus-smoke-recording-audit.json
```

## libero_plus-smoke-patterns

```bash
/volt/code/frozen-flow-study/.venv/bin/python /volt/code/frozen-flow-study/examples/rescue_characterization/analysis.py --manifest /volt/code/frozen-flow-study/examples/rescue_characterization/protocol.json --benchmark libero_plus --records /volt/artifacts/rescue-characterization/runs/libero_plus-smoke.jsonl --output /volt/artifacts/rescue-characterization/runs/libero_plus-smoke-patterns.json --smoke
```

## smoke-report

```bash
/volt/code/frozen-flow-study/.venv/bin/python /volt/code/frozen-flow-study/examples/rescue_characterization/report.py --manifest /volt/code/frozen-flow-study/examples/rescue_characterization/protocol.json --results /volt/artifacts/rescue-characterization/runs --output /volt/artifacts/rescue-characterization/runs/smoke-REPORT.md --smoke
```
