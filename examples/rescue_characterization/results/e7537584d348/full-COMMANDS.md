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

## libero-main-0

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/rescue_characterization/client.py --manifest /volt/code/frozen-flow-study/examples/rescue_characterization/protocol.json --benchmark libero --server http://127.0.0.1:8840 --output /volt/artifacts/rescue-characterization/runs/libero-main-worker-0.jsonl --worker-index 0 --workers 4
```

## libero-main-1

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/rescue_characterization/client.py --manifest /volt/code/frozen-flow-study/examples/rescue_characterization/protocol.json --benchmark libero --server http://127.0.0.1:8841 --output /volt/artifacts/rescue-characterization/runs/libero-main-worker-1.jsonl --worker-index 1 --workers 4
```

## libero-main-2

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/rescue_characterization/client.py --manifest /volt/code/frozen-flow-study/examples/rescue_characterization/protocol.json --benchmark libero --server http://127.0.0.1:8842 --output /volt/artifacts/rescue-characterization/runs/libero-main-worker-2.jsonl --worker-index 2 --workers 4
```

## libero-main-3

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/rescue_characterization/client.py --manifest /volt/code/frozen-flow-study/examples/rescue_characterization/protocol.json --benchmark libero --server http://127.0.0.1:8843 --output /volt/artifacts/rescue-characterization/runs/libero-main-worker-3.jsonl --worker-index 3 --workers 4
```

## libero-analysis

```bash
/volt/code/frozen-flow-study/.venv/bin/python /volt/code/frozen-flow-study/examples/rescue_characterization/analyze.py --manifest /volt/code/frozen-flow-study/examples/rescue_characterization/protocol.json --benchmark libero --records /volt/artifacts/rescue-characterization/runs/libero-main-worker-0.jsonl /volt/artifacts/rescue-characterization/runs/libero-main-worker-1.jsonl /volt/artifacts/rescue-characterization/runs/libero-main-worker-2.jsonl /volt/artifacts/rescue-characterization/runs/libero-main-worker-3.jsonl --output /volt/artifacts/rescue-characterization/runs/libero-summary.json --wandb-project pi05-rescue-characterization --wandb-name libero-full-e7537584d348
```

## libero-recording-audit

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/rescue_characterization/recording_audit.py --manifest /volt/code/frozen-flow-study/examples/rescue_characterization/protocol.json --records /volt/artifacts/rescue-characterization/runs/libero-main-worker-0.jsonl /volt/artifacts/rescue-characterization/runs/libero-main-worker-1.jsonl /volt/artifacts/rescue-characterization/runs/libero-main-worker-2.jsonl /volt/artifacts/rescue-characterization/runs/libero-main-worker-3.jsonl --output /volt/artifacts/rescue-characterization/runs/libero-recording-audit.json
```

## libero-patterns

```bash
/volt/code/frozen-flow-study/.venv/bin/python /volt/code/frozen-flow-study/examples/rescue_characterization/analysis.py --manifest /volt/code/frozen-flow-study/examples/rescue_characterization/protocol.json --benchmark libero --records /volt/artifacts/rescue-characterization/runs/libero-main-worker-0.jsonl /volt/artifacts/rescue-characterization/runs/libero-main-worker-1.jsonl /volt/artifacts/rescue-characterization/runs/libero-main-worker-2.jsonl /volt/artifacts/rescue-characterization/runs/libero-main-worker-3.jsonl --output /volt/artifacts/rescue-characterization/runs/libero-patterns.json
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

## libero_plus-main-0

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/rescue_characterization/client.py --manifest /volt/code/frozen-flow-study/examples/rescue_characterization/protocol.json --benchmark libero_plus --server http://127.0.0.1:8840 --output /volt/artifacts/rescue-characterization/runs/libero_plus-main-worker-0.jsonl --worker-index 0 --workers 4
```

## libero_plus-main-1

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/rescue_characterization/client.py --manifest /volt/code/frozen-flow-study/examples/rescue_characterization/protocol.json --benchmark libero_plus --server http://127.0.0.1:8841 --output /volt/artifacts/rescue-characterization/runs/libero_plus-main-worker-1.jsonl --worker-index 1 --workers 4
```

## libero_plus-main-2

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/rescue_characterization/client.py --manifest /volt/code/frozen-flow-study/examples/rescue_characterization/protocol.json --benchmark libero_plus --server http://127.0.0.1:8842 --output /volt/artifacts/rescue-characterization/runs/libero_plus-main-worker-2.jsonl --worker-index 2 --workers 4
```

## libero_plus-main-3

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/rescue_characterization/client.py --manifest /volt/code/frozen-flow-study/examples/rescue_characterization/protocol.json --benchmark libero_plus --server http://127.0.0.1:8843 --output /volt/artifacts/rescue-characterization/runs/libero_plus-main-worker-3.jsonl --worker-index 3 --workers 4
```

## libero_plus-analysis

```bash
/volt/code/frozen-flow-study/.venv/bin/python /volt/code/frozen-flow-study/examples/rescue_characterization/analyze.py --manifest /volt/code/frozen-flow-study/examples/rescue_characterization/protocol.json --benchmark libero_plus --records /volt/artifacts/rescue-characterization/runs/libero_plus-main-worker-0.jsonl /volt/artifacts/rescue-characterization/runs/libero_plus-main-worker-1.jsonl /volt/artifacts/rescue-characterization/runs/libero_plus-main-worker-2.jsonl /volt/artifacts/rescue-characterization/runs/libero_plus-main-worker-3.jsonl --output /volt/artifacts/rescue-characterization/runs/libero_plus-summary.json --wandb-project pi05-rescue-characterization --wandb-name libero_plus-full-e7537584d348
```

## libero_plus-recording-audit

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/rescue_characterization/recording_audit.py --manifest /volt/code/frozen-flow-study/examples/rescue_characterization/protocol.json --records /volt/artifacts/rescue-characterization/runs/libero_plus-main-worker-0.jsonl /volt/artifacts/rescue-characterization/runs/libero_plus-main-worker-1.jsonl /volt/artifacts/rescue-characterization/runs/libero_plus-main-worker-2.jsonl /volt/artifacts/rescue-characterization/runs/libero_plus-main-worker-3.jsonl --output /volt/artifacts/rescue-characterization/runs/libero_plus-recording-audit.json
```

## libero_plus-patterns

```bash
/volt/code/frozen-flow-study/.venv/bin/python /volt/code/frozen-flow-study/examples/rescue_characterization/analysis.py --manifest /volt/code/frozen-flow-study/examples/rescue_characterization/protocol.json --benchmark libero_plus --records /volt/artifacts/rescue-characterization/runs/libero_plus-main-worker-0.jsonl /volt/artifacts/rescue-characterization/runs/libero_plus-main-worker-1.jsonl /volt/artifacts/rescue-characterization/runs/libero_plus-main-worker-2.jsonl /volt/artifacts/rescue-characterization/runs/libero_plus-main-worker-3.jsonl --output /volt/artifacts/rescue-characterization/runs/libero_plus-patterns.json
```

## full-report

```bash
/volt/code/frozen-flow-study/.venv/bin/python /volt/code/frozen-flow-study/examples/rescue_characterization/report.py --manifest /volt/code/frozen-flow-study/examples/rescue_characterization/protocol.json --results /volt/artifacts/rescue-characterization/runs --output /volt/artifacts/rescue-characterization/runs/full-REPORT.md
```
