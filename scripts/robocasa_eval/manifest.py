"""Create a manifest for one evaluated model, environment seed and step count."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .protocol import SEEDS, STEPS, validate_manifest


def make_manifest(model: str, seed: int, steps: int, campaign_id: str,
                  owner_robocasa_adapter: str | None = None,
                  owner_sha256: str | None = None, checkpoint_file: str | None = None,
                  checkpoint_sha256: str | None = None) -> dict:
    presets = json.loads(Path(__file__).with_name('presets.json').read_text())
    manifest = dict(presets['models'][model], tasks=presets['tasks'])
    manifest.update(campaign_id=campaign_id, scene_seed=seed, denoising_steps=steps)
    if owner_robocasa_adapter:
        manifest['owner_robocasa_adapter'] = owner_robocasa_adapter
        manifest['owner_robocasa_adapter_sha256'] = owner_sha256
    if checkpoint_file and checkpoint_sha256:
        manifest['checkpoint_file'] = checkpoint_file
        manifest['checkpoint_file_sha256'] = checkpoint_sha256
    validate_manifest(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('pi05', 'smolvla'), required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--steps', type=int, choices=STEPS, required=True)
    parser.add_argument('--campaign-id', required=True)
    parser.add_argument('--owner-robocasa-adapter', help='Pinned LeRobot robocasa.py; required for SmolVLA')
    parser.add_argument('--owner-sha256', help='SHA256 of the pinned owner adapter')
    parser.add_argument('--checkpoint-file', required=True, help='Relative path to a checkpoint weight file')
    parser.add_argument('--checkpoint-sha256', required=True, help='SHA256 of that weight file')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    manifest = make_manifest(args.model, args.seed, args.steps, args.campaign_id,
                             args.owner_robocasa_adapter, args.owner_sha256,
                             args.checkpoint_file, args.checkpoint_sha256)
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
