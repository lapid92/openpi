"""Run a task shard with a separate policy server and simulator process."""
from __future__ import annotations

import argparse
import json
import hashlib
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from .protocol import validate_manifest
from .results import save_tables, validate_episode


def wait_for_server(process: subprocess.Popen, port: int, timeout: int = 1200) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError('Policy server exited before becoming ready')
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=1):
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError('Policy startup timeout')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--port', type=int, default=18200)
    parser.add_argument('--server-python', default=sys.executable)
    parser.add_argument('--sim-python', default=sys.executable)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    if not 0 <= args.shard < manifest['shards']:
        raise ValueError('Shard outside manifest')
    owned = manifest['tasks'][args.shard::manifest['shards']]
    weight = (args.checkpoint / manifest.get('checkpoint_file', '')).resolve()
    if not manifest.get('checkpoint_file_sha256') or not weight.is_file() or not weight.is_relative_to(args.checkpoint.resolve()):
        raise ValueError('Manifest must pin an existing weight file inside the checkpoint')
    digest = hashlib.sha256()
    with weight.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != manifest['checkpoint_file_sha256']:
        raise ValueError('Checkpoint weight SHA256 differs from manifest')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'runtime_identity.json').write_text(json.dumps({
        'checkpoint_file': str(weight), 'checkpoint_file_sha256': digest.hexdigest(),
        'checkpoint_revision_claim': manifest['checkpoint_revision'],
        'code_revisions_claim': manifest.get('code_revisions'),
        'verification': 'weight SHA256 checked; source revision labels are not verified by this worker',
    }, indent=2))
    server_env = os.environ.copy()
    package_root = str(Path(__file__).resolve().parent.parent)
    for env in (server_env,):
        env['PYTHONPATH'] = package_root + os.pathsep + env.get('PYTHONPATH', '')
        env.update(PYTHONHASHSEED='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION='0.35')
    if manifest['model'] == 'smolvla':
        server_env['SMOL_VALIDATION_PATH'] = str(args.out/'inference_validation.json')
    server_env.pop('MUJOCO_EGL_DEVICE_ID', None)
    sim_env = server_env.copy()
    sim_env.update(MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', MUJOCO_EGL_DEVICE_ID='0')
    server = subprocess.Popen([
        args.server_python, '-m', 'robocasa_eval.server', '--model', manifest['model'],
        '--checkpoint', str(args.checkpoint), '--steps', str(manifest['denoising_steps']),
        '--port', str(args.port), '--metadata-path', str(args.out/'policy_metadata.json'),
    ], env=server_env, stdout=(args.out/'server.log').open('w'), stderr=subprocess.STDOUT)
    try:
        wait_for_server(server, args.port)
        for task in owned:
            if server.poll() is not None:
                raise RuntimeError('Policy server died during rollout')
            with (args.out/(task['task']+'.log')).open('w') as log:
                subprocess.run([
                    args.sim_python, '-m', 'robocasa_eval.rollout', '--manifest', str(args.manifest),
                    '--task', task['task'], '--out', str(args.out), '--port', str(args.port),
                ], env=sim_env, stdout=log, stderr=subprocess.STDOUT,
                    check=True, timeout=manifest['task_timeout_seconds'])
        rows = [json.loads(path.read_text()) for path in sorted((args.out/'episodes').glob('*.json'))]
        if manifest['model'] == 'smolvla':
            check = json.loads((args.out/'inference_validation.json').read_text())
            required = {1, 2, 4, 10, 15, 20}
            if not check.get('passed') or {r['steps'] for r in check['checks']} != required:
                raise RuntimeError('SmolVLA repeated-input check incomplete')
            if any(not r['repeat_exact'] or r['nfe_first'] != r['steps'] or
                   r['nfe_second'] != r['steps'] or r['shape'] != [50, 12]
                   for r in check['checks']):
                raise RuntimeError('SmolVLA repeated-input check failed')
        seen = {}
        for row in rows:
            key = validate_episode(row, manifest)
            if key in seen and seen[key] != row:
                raise ValueError(f'Conflicting duplicate episode: {key}')
            seen[key] = row
        expected = {(task['task'], i) for task in owned for i in range(manifest['episodes_per_task'])}
        if set(seen) != expected or any(row['status'] == 'infrastructure_failure' for row in seen.values()):
            raise RuntimeError('Shard coverage incomplete')
        save_tables(rows, manifest, args.out/'tables')
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()


if __name__ == '__main__':
    main()
