"""Corrected SmolVLA checkpoint processors, cameras, noise and NFE counter."""
from __future__ import annotations
import dataclasses
import hashlib
import json
import pathlib
import time
from .server_common import take_request, finish

class SmolAdapter:
    def __init__(self, checkpoint, steps, device="cuda:0"):
        import torch
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors
        checkpoint = pathlib.Path(checkpoint)
        processor = json.loads((checkpoint / "policy_preprocessor.json").read_text())
        rename = next(s["config"]["rename_map"] for s in processor["steps"]
                      if s["registry_name"] == "rename_observations_processor")
        expected = {
            "observation.images.robot0_agentview_left": "observation.images.camera1",
            "observation.images.robot0_agentview_right": "observation.images.camera2",
            "observation.images.robot0_eye_in_hand": "observation.images.camera3",
        }
        if rename != expected:
            raise ValueError(f"Checkpoint camera mapping differs from audited mapping: {rename}")
        self.device = device
        self.steps = steps
        self.policy = SmolVLAPolicy.from_pretrained(str(checkpoint)).to(device).eval()
        self.policy.config.num_steps = steps
        self.policy.model.config.num_steps = steps
        if self.policy.config.n_obs_steps != 1 or self.policy.config.compile_model:
            raise ValueError("Requires uncompiled n_obs_steps=1 released policy for reliable call counting")
        if self.policy.config.rtc_config is not None and self.policy.config.rtc_config.enabled:
            raise ValueError("RTC is outside the matched checkpoint protocol")
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.calls = 0
        self.warmup_pending = True
        self.hook = self.policy.model.action_out_proj.register_forward_hook(self._count)
        self.metadata = {
            "model": "smolvla", "steps": steps, "checkpoint": str(checkpoint),
            "resolved_config": dataclasses.asdict(self.policy.config),
            "processor_file_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in checkpoint.glob("policy_*processor*") if p.is_file()},
            "noise_rng": "torch.randn(generator=torch.Generator(device).manual_seed(query_seed))",
            "counter": "action_out_proj forward hook; excludes prefix encoding",
            "camera_mapping": expected, "language": "exact caller prompt; saved checkpoint newline/tokenizer processors",
        }

    def _count(self, module, inputs, output):
        self.calls += 1

    def reset(self):
        self.policy.reset()

    def infer(self, obs):
        import numpy as np
        import torch
        started = time.monotonic()
        import os
        validation_path=os.environ.get('SMOL_VALIDATION_PATH')
        if validation_path and not getattr(self,'validation_started',False):
            self.validation_started=True
            original_steps=self.steps
            checks=[]
            for n in (10,4,2,1,15,20):
                self.steps=n;self.policy.config.num_steps=n;self.policy.model.config.num_steps=n
                first=self.infer(dict(obs));second=self.infer(dict(obs))
                equal=np.array_equal(first['actions'],second['actions'])
                checks.append(dict(steps=n,nfe_first=first['action_network_evaluations'],nfe_second=second['action_network_evaluations'],repeat_exact=equal,max_abs_diff=float(np.max(np.abs(first['actions']-second['actions']))),shape=list(first['actions'].shape)))
                if not equal:raise RuntimeError('Pilot repeated-input inference differs')
            self.steps=original_steps;self.policy.config.num_steps=original_steps;self.policy.model.config.num_steps=original_steps
            pathlib.Path(validation_path).write_text(json.dumps(dict(passed=True,checks=checks,camera_shapes={k:list(np.asarray(v).shape) for k,v in obs.items() if 'image' in k},prompt=obs['prompt']),indent=2))
        obs, seed = take_request(obs)
        for name in ("image", "right_image", "wrist_image"):
            if np.asarray(obs["observation/" + name]).shape != (256, 256, 3):
                raise ValueError("SmolVLA requires native 256x256 observations before checkpoint resize")
        # Feed physical camera names to the checkpoint's SAVED rename processor.
        batch = {"observation.state": torch.as_tensor(np.asarray(obs["observation/state"]), dtype=torch.float32),
                 "task": obs["prompt"]}
        for raw, physical in (("image", "robot0_agentview_left"),
                              ("right_image", "robot0_agentview_right"),
                              ("wrist_image", "robot0_eye_in_hand")):
            batch["observation.images." + physical] = torch.from_numpy(
                np.asarray(obs["observation/" + raw]).copy()
            ).permute(2, 0, 1).float().div(255)
        batch = self.pre(batch)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn((1, self.policy.config.chunk_size, self.policy.config.max_action_dim),
                            generator=generator, device=self.device, dtype=torch.float32)
        self.reset()  # Query is stateless; n_obs_steps=1 verified at construction.
        self.calls = 0
        torch.cuda.synchronize(self.device)
        t = time.monotonic()
        with torch.inference_mode():
            actions = self.policy.predict_action_chunk(batch, noise=noise)
        torch.cuda.synchronize(self.device)
        elapsed = time.monotonic() - t
        count = self.calls
        if count != self.steps:
            raise RuntimeError(f"Actual action-network calls {count} != requested {self.steps}")
        actions = self.post(actions).detach().cpu().numpy()[0]
        result = finish(actions, seed, count, elapsed, started, self.metadata)
        result["warmup_query"] = self.warmup_pending
        self.warmup_pending = False
        return result
