# Executed commands

## server-0

```bash
/volt/code/frozen-flow-study/.venv/bin/python /volt/code/frozen-flow-study/examples/flow_probe/policy_server.py --manifest /volt/code/frozen-flow-study/examples/flow_probe/protocol.json --gpu-uuid GPU-56335fdf-6dc8-1d64-ce39-364d4a532435 --port 8940
```

## server-1

```bash
/volt/code/frozen-flow-study/.venv/bin/python /volt/code/frozen-flow-study/examples/flow_probe/policy_server.py --manifest /volt/code/frozen-flow-study/examples/flow_probe/protocol.json --gpu-uuid GPU-b786fcd4-f878-69e6-3017-a2ae408151fd --port 8941
```

## server-2

```bash
/volt/code/frozen-flow-study/.venv/bin/python /volt/code/frozen-flow-study/examples/flow_probe/policy_server.py --manifest /volt/code/frozen-flow-study/examples/flow_probe/protocol.json --gpu-uuid GPU-967ff6df-a7e2-b65a-d051-87819b7725fc --port 8942
```

## server-3

```bash
/volt/code/frozen-flow-study/.venv/bin/python /volt/code/frozen-flow-study/examples/flow_probe/policy_server.py --manifest /volt/code/frozen-flow-study/examples/flow_probe/protocol.json --gpu-uuid GPU-0cb759da-a716-038e-f0a9-208ec138ccc7 --port 8943
```

## libero-prepare-0

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/flow_probe/client.py --manifest /volt/code/frozen-flow-study/examples/flow_probe/protocol.json --benchmark libero --server http://127.0.0.1:8940 --output /volt/artifacts/flow-probe/runs/libero-full-prepare-worker-0.jsonl --prepare-only --worker-index 0 --workers 4
```

## libero-prepare-1

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/flow_probe/client.py --manifest /volt/code/frozen-flow-study/examples/flow_probe/protocol.json --benchmark libero --server http://127.0.0.1:8941 --output /volt/artifacts/flow-probe/runs/libero-full-prepare-worker-1.jsonl --prepare-only --worker-index 1 --workers 4
```

## libero-prepare-2

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/flow_probe/client.py --manifest /volt/code/frozen-flow-study/examples/flow_probe/protocol.json --benchmark libero --server http://127.0.0.1:8942 --output /volt/artifacts/flow-probe/runs/libero-full-prepare-worker-2.jsonl --prepare-only --worker-index 2 --workers 4
```

## libero-prepare-3

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/flow_probe/client.py --manifest /volt/code/frozen-flow-study/examples/flow_probe/protocol.json --benchmark libero --server http://127.0.0.1:8943 --output /volt/artifacts/flow-probe/runs/libero-full-prepare-worker-3.jsonl --prepare-only --worker-index 3 --workers 4
```

## libero-main-0

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/flow_probe/client.py --manifest /volt/code/frozen-flow-study/examples/flow_probe/protocol.json --benchmark libero --server http://127.0.0.1:8940 --output /volt/artifacts/flow-probe/runs/libero-main-worker-0.jsonl --worker-index 0 --workers 4
```

## libero-main-1

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/flow_probe/client.py --manifest /volt/code/frozen-flow-study/examples/flow_probe/protocol.json --benchmark libero --server http://127.0.0.1:8941 --output /volt/artifacts/flow-probe/runs/libero-main-worker-1.jsonl --worker-index 1 --workers 4
```

## libero-main-2

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/flow_probe/client.py --manifest /volt/code/frozen-flow-study/examples/flow_probe/protocol.json --benchmark libero --server http://127.0.0.1:8942 --output /volt/artifacts/flow-probe/runs/libero-main-worker-2.jsonl --worker-index 2 --workers 4
```

## libero-main-3

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/flow_probe/client.py --manifest /volt/code/frozen-flow-study/examples/flow_probe/protocol.json --benchmark libero --server http://127.0.0.1:8943 --output /volt/artifacts/flow-probe/runs/libero-main-worker-3.jsonl --worker-index 3 --workers 4
```

## libero-metrics

```bash
/volt/code/frozen-flow-study/.venv/bin/python /volt/code/frozen-flow-study/examples/flow_probe/analysis.py --manifest /volt/code/frozen-flow-study/examples/flow_probe/protocol.json --benchmark libero --records /volt/artifacts/flow-probe/runs/libero-main-worker-0.jsonl /volt/artifacts/flow-probe/runs/libero-main-worker-1.jsonl /volt/artifacts/flow-probe/runs/libero-main-worker-2.jsonl /volt/artifacts/flow-probe/runs/libero-main-worker-3.jsonl --output /volt/artifacts/flow-probe/runs/libero-metrics.json
```

## libero-recording-audit

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/rescue_characterization/recording_audit.py --manifest /volt/code/frozen-flow-study/examples/flow_probe/protocol.json --records /volt/artifacts/flow-probe/runs/libero-main-worker-0.jsonl /volt/artifacts/flow-probe/runs/libero-main-worker-1.jsonl /volt/artifacts/flow-probe/runs/libero-main-worker-2.jsonl /volt/artifacts/flow-probe/runs/libero-main-worker-3.jsonl --output /volt/artifacts/flow-probe/runs/libero-recording-audit.json
```

## libero_plus-prepare-0

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/flow_probe/client.py --manifest /volt/code/frozen-flow-study/examples/flow_probe/protocol.json --benchmark libero_plus --server http://127.0.0.1:8940 --output /volt/artifacts/flow-probe/runs/libero_plus-full-prepare-worker-0.jsonl --prepare-only --worker-index 0 --workers 4
```

## libero_plus-prepare-1

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/flow_probe/client.py --manifest /volt/code/frozen-flow-study/examples/flow_probe/protocol.json --benchmark libero_plus --server http://127.0.0.1:8941 --output /volt/artifacts/flow-probe/runs/libero_plus-full-prepare-worker-1.jsonl --prepare-only --worker-index 1 --workers 4
```

## libero_plus-prepare-2

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/flow_probe/client.py --manifest /volt/code/frozen-flow-study/examples/flow_probe/protocol.json --benchmark libero_plus --server http://127.0.0.1:8942 --output /volt/artifacts/flow-probe/runs/libero_plus-full-prepare-worker-2.jsonl --prepare-only --worker-index 2 --workers 4
```

## libero_plus-prepare-3

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/flow_probe/client.py --manifest /volt/code/frozen-flow-study/examples/flow_probe/protocol.json --benchmark libero_plus --server http://127.0.0.1:8943 --output /volt/artifacts/flow-probe/runs/libero_plus-full-prepare-worker-3.jsonl --prepare-only --worker-index 3 --workers 4
```

## libero_plus-main-0

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/flow_probe/client.py --manifest /volt/code/frozen-flow-study/examples/flow_probe/protocol.json --benchmark libero_plus --server http://127.0.0.1:8940 --output /volt/artifacts/flow-probe/runs/libero_plus-main-worker-0.jsonl --worker-index 0 --workers 4
```

## libero_plus-main-1

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/flow_probe/client.py --manifest /volt/code/frozen-flow-study/examples/flow_probe/protocol.json --benchmark libero_plus --server http://127.0.0.1:8941 --output /volt/artifacts/flow-probe/runs/libero_plus-main-worker-1.jsonl --worker-index 1 --workers 4
```

## libero_plus-main-2

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/flow_probe/client.py --manifest /volt/code/frozen-flow-study/examples/flow_probe/protocol.json --benchmark libero_plus --server http://127.0.0.1:8942 --output /volt/artifacts/flow-probe/runs/libero_plus-main-worker-2.jsonl --worker-index 2 --workers 4
```

## libero_plus-main-3

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/flow_probe/client.py --manifest /volt/code/frozen-flow-study/examples/flow_probe/protocol.json --benchmark libero_plus --server http://127.0.0.1:8943 --output /volt/artifacts/flow-probe/runs/libero_plus-main-worker-3.jsonl --worker-index 3 --workers 4
```

## libero_plus-metrics

```bash
/volt/code/frozen-flow-study/.venv/bin/python /volt/code/frozen-flow-study/examples/flow_probe/analysis.py --manifest /volt/code/frozen-flow-study/examples/flow_probe/protocol.json --benchmark libero_plus --records /volt/artifacts/flow-probe/runs/libero_plus-main-worker-0.jsonl /volt/artifacts/flow-probe/runs/libero_plus-main-worker-1.jsonl /volt/artifacts/flow-probe/runs/libero_plus-main-worker-2.jsonl /volt/artifacts/flow-probe/runs/libero_plus-main-worker-3.jsonl --output /volt/artifacts/flow-probe/runs/libero_plus-metrics.json
```

## libero_plus-recording-audit

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/rescue_characterization/recording_audit.py --manifest /volt/code/frozen-flow-study/examples/flow_probe/protocol.json --records /volt/artifacts/flow-probe/runs/libero_plus-main-worker-0.jsonl /volt/artifacts/flow-probe/runs/libero_plus-main-worker-1.jsonl /volt/artifacts/flow-probe/runs/libero_plus-main-worker-2.jsonl /volt/artifacts/flow-probe/runs/libero_plus-main-worker-3.jsonl --output /volt/artifacts/flow-probe/runs/libero_plus-recording-audit.json
```

## independent-audit

```bash
/volt/code/frozen-flow-study/.venv/bin/python /volt/code/frozen-flow-study/examples/flow_probe/recovery/audit_v2.py --manifest /volt/code/frozen-flow-study/examples/flow_probe/protocol.json --phase main --run-dir /volt/artifacts/flow-probe/runs --output /volt/artifacts/flow-probe/runs/full-independent-audit.json
```

## final-report

```bash
/volt/code/frozen-flow-study/.venv/bin/python /volt/code/frozen-flow-study/examples/flow_probe/report.py --manifest /volt/code/frozen-flow-study/examples/flow_probe/protocol.json --results /volt/artifacts/flow-probe/runs --output /volt/artifacts/flow-probe/runs/full-REPORT.md
```
