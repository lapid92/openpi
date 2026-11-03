"""The two evaluated RoboCasa protocols and deterministic query seeds."""
from __future__ import annotations

import hashlib

STEPS = (1, 2, 4, 10, 15, 20)
SEEDS = (7, 8, 9)
GROUPS = ('atomic_seen', 'composite_seen', 'composite_unseen')
RESET_PROTOCOL = 'construction_seed_then_sequential_unseeded_resets'


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:4], 'big')


def validate_manifest(manifest: dict) -> None:
    model = manifest['model']
    if model not in ('pi05', 'smolvla') or manifest['denoising_steps'] not in STEPS:
        raise ValueError('Unsupported model or denoising count')
    if manifest['scene_seed'] not in SEEDS or manifest['policy_seed'] != 0:
        raise ValueError('Expected evaluated environment and policy seeds')
    if manifest['horizon_multiplier'] != 1.5 or manifest['episodes_per_task'] != 50:
        raise ValueError('Expected 50 episodes per task and 1.5x task horizon')
    if manifest['split'] != ('target' if model == 'smolvla' else 'pretrain'):
        raise ValueError('Model split differs from evaluated protocol')
    if manifest['replan_steps'] != (50 if model == 'smolvla' else 5):
        raise ValueError('Model replan interval differs from evaluated protocol')
    if model == 'smolvla' and not all(manifest.get(k) for k in ('owner_robocasa_adapter', 'owner_robocasa_adapter_sha256')):
        raise ValueError('SmolVLA requires a path and SHA256 for the owner adapter audit')
    tasks = manifest['tasks']
    if len(tasks) != 50 or len({task['task'] for task in tasks}) != 50:
        raise ValueError('Expected the 50-task suite without duplicates')
    if any(task['group'] not in GROUPS for task in tasks):
        raise ValueError('Unknown task group')
