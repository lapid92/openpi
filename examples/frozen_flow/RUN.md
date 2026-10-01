# Exact pod launch
Run from /volt/code/frozen-flow-study on uz2ptakxucbe:

~~~bash
bash examples/frozen_flow/run_detached.sh smoke
bash examples/frozen_flow/run_detached.sh full
~~~

The wrapper explicitly ignores HUP before Python starts; plain setsid without this trap exited on terminal disconnect in this Volt environment. Use this wrapper in preference to the abbreviated setsid examples in the generated report. The full command validates complete smoke JSONL and its manifest-bound summary, starts all four UUID-bound servers, prepares each benchmark/GPU, evaluates all enumerated cases, validates records, uploads W&B artifacts, writes the report, commits and pushes results. It does not require a Mac connection once launched.

Inspect /volt/artifacts/frozen-flow-study/runs/{smoke,full}-status.json. Each records exact subprocess argument lists. Exceptions halt and preserve attempted records. Do not delete failures or substitute episodes. The pod's runtime secrets provide W&B and GitHub authentication; no credentials are stored in these files.

The first pre-evaluation launch exited with SIGHUP before any GPU use. The only scientific-protocol-preserving amendment compacted the manifest and changed W&B configuration to reference its hash instead of embedding the full asset inventory. Both pre-run manifest revisions remain in Git history.
