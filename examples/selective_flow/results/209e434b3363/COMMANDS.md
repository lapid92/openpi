# Executed study commands

Historical commands; do not rerun evaluations into these output files.

Launch commands (smoke passed before full):

```bash
cd /volt/code/frozen-flow-study
bash examples/selective_flow/run_detached.sh smoke
bash examples/selective_flow/run_detached.sh full
```

Exact full-supervisor child commands:

## server-0

```bash
/volt/code/frozen-flow-study/.venv/bin/python /volt/code/frozen-flow-study/examples/selective_flow/policy_server.py --manifest /volt/code/frozen-flow-study/examples/selective_flow/protocol.json --gpu-uuid GPU-56335fdf-6dc8-1d64-ce39-364d4a532435 --port 8800
```

## server-1

```bash
/volt/code/frozen-flow-study/.venv/bin/python /volt/code/frozen-flow-study/examples/selective_flow/policy_server.py --manifest /volt/code/frozen-flow-study/examples/selective_flow/protocol.json --gpu-uuid GPU-b786fcd4-f878-69e6-3017-a2ae408151fd --port 8801
```

## server-2

```bash
/volt/code/frozen-flow-study/.venv/bin/python /volt/code/frozen-flow-study/examples/selective_flow/policy_server.py --manifest /volt/code/frozen-flow-study/examples/selective_flow/protocol.json --gpu-uuid GPU-967ff6df-a7e2-b65a-d051-87819b7725fc --port 8802
```

## server-3

```bash
/volt/code/frozen-flow-study/.venv/bin/python /volt/code/frozen-flow-study/examples/selective_flow/policy_server.py --manifest /volt/code/frozen-flow-study/examples/selective_flow/protocol.json --gpu-uuid GPU-0cb759da-a716-038e-f0a9-208ec138ccc7 --port 8803
```

## libero_plus-prepare-0

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/selective_flow/client.py --manifest /volt/code/frozen-flow-study/examples/selective_flow/protocol.json --benchmark libero_plus --server http://127.0.0.1:8800 --output /volt/artifacts/selective-flow-study/runs/libero_plus-prepare-worker-0.jsonl --prepare-only --worker-index 0 --workers 4
```

## libero_plus-prepare-1

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/selective_flow/client.py --manifest /volt/code/frozen-flow-study/examples/selective_flow/protocol.json --benchmark libero_plus --server http://127.0.0.1:8801 --output /volt/artifacts/selective-flow-study/runs/libero_plus-prepare-worker-1.jsonl --prepare-only --worker-index 1 --workers 4
```

## libero_plus-prepare-2

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/selective_flow/client.py --manifest /volt/code/frozen-flow-study/examples/selective_flow/protocol.json --benchmark libero_plus --server http://127.0.0.1:8802 --output /volt/artifacts/selective-flow-study/runs/libero_plus-prepare-worker-2.jsonl --prepare-only --worker-index 2 --workers 4
```

## libero_plus-prepare-3

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/selective_flow/client.py --manifest /volt/code/frozen-flow-study/examples/selective_flow/protocol.json --benchmark libero_plus --server http://127.0.0.1:8803 --output /volt/artifacts/selective-flow-study/runs/libero_plus-prepare-worker-3.jsonl --prepare-only --worker-index 3 --workers 4
```

## libero_plus-main-0

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/selective_flow/client.py --manifest /volt/code/frozen-flow-study/examples/selective_flow/protocol.json --benchmark libero_plus --server http://127.0.0.1:8800 --output /volt/artifacts/selective-flow-study/runs/libero_plus-main-worker-0.jsonl --worker-index 0 --workers 4
```

## libero_plus-main-1

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/selective_flow/client.py --manifest /volt/code/frozen-flow-study/examples/selective_flow/protocol.json --benchmark libero_plus --server http://127.0.0.1:8801 --output /volt/artifacts/selective-flow-study/runs/libero_plus-main-worker-1.jsonl --worker-index 1 --workers 4
```

## libero_plus-main-2

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/selective_flow/client.py --manifest /volt/code/frozen-flow-study/examples/selective_flow/protocol.json --benchmark libero_plus --server http://127.0.0.1:8802 --output /volt/artifacts/selective-flow-study/runs/libero_plus-main-worker-2.jsonl --worker-index 2 --workers 4
```

## libero_plus-main-3

```bash
/volt/envs/libero/bin/python /volt/code/frozen-flow-study/examples/selective_flow/client.py --manifest /volt/code/frozen-flow-study/examples/selective_flow/protocol.json --benchmark libero_plus --server http://127.0.0.1:8803 --output /volt/artifacts/selective-flow-study/runs/libero_plus-main-worker-3.jsonl --worker-index 3 --workers 4
```

## libero_plus-analysis

```bash
/volt/code/frozen-flow-study/.venv/bin/python /volt/code/frozen-flow-study/examples/selective_flow/analyze.py --manifest /volt/code/frozen-flow-study/examples/selective_flow/protocol.json --benchmark libero_plus --records /volt/artifacts/selective-flow-study/runs/libero_plus-main-worker-0.jsonl /volt/artifacts/selective-flow-study/runs/libero_plus-main-worker-1.jsonl /volt/artifacts/selective-flow-study/runs/libero_plus-main-worker-2.jsonl /volt/artifacts/selective-flow-study/runs/libero_plus-main-worker-3.jsonl --output /volt/artifacts/selective-flow-study/runs/libero_plus-summary.json --wandb-project pi05-independent-selective-steps --wandb-name libero_plus-full-209e434b3363
```

## full-report

```bash
/volt/code/frozen-flow-study/.venv/bin/python /volt/code/frozen-flow-study/examples/selective_flow/report.py --manifest /volt/code/frozen-flow-study/examples/selective_flow/protocol.json --results /volt/artifacts/selective-flow-study/runs --output /volt/artifacts/selective-flow-study/runs/full-REPORT.md
```
