"""Pinned released-client environment settings, with durable episode evidence."""
from __future__ import annotations
import argparse
import collections
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
from .results import manifest_digest
from .protocol import validate_manifest, stable_seed
from .smol_interfaces import convert_action as smol_action, convert_state, audit_interface


def atomic(path, value):
    import numpy as np
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,default=lambda x:np.asarray(x).tolist()))
    tmp.replace(path)


def main():
    import numpy as np
    import gymnasium as gym
    import robocasa
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    from robocasa.utils.env_utils import convert_action as pi05_action
    from openpi_client import websocket_client_policy, image_tools
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--task',required=True)
    p.add_argument('--out',required=True);p.add_argument('--port',type=int,default=18200);a=p.parse_args()
    manifest=json.loads(Path(a.manifest).read_text())
    validate_manifest(manifest)
    task=next(t for t in manifest['tasks'] if t['task']==a.task)
    smol=manifest['model']=='smolvla'
    out=Path(a.out);episodes=out/'episodes';scenes=out/'scenes'/a.task
    episodes.mkdir(exist_ok=True);scenes.mkdir(parents=True,exist_ok=True)
    client=websocket_client_policy.WebsocketClientPolicy('127.0.0.1',a.port)
    np.random.seed(manifest['scene_seed'])
    env=None
    # One environment per task; seed its construction once, then reset sequentially
    # exactly as in the released client. Never reseed individual episodes.
    horizon=math.ceil(get_task_horizon(a.task)*manifest['horizon_multiplier'])
    for idx in range(manifest['episodes_per_task']):
        path=episodes/f'{a.task}_{idx:03d}.json'
        if path.exists():raise RuntimeError('Refusing to overwrite an episode')
        record={k:manifest[k] for k in ['campaign_id','protocol_id','checkpoint_revision','denoising_steps']}
        record.update(manifest_sha256=manifest_digest(manifest),task=a.task,group=task['group'],episode_index=idx,started_at=time.time(),
                      model=manifest['model'],split=manifest['split'],prompt=a.task if smol else None,engineering_check=manifest.get('engineering_check',False),success=None,status='running',scene_seed=manifest['scene_seed'],
                      scene_reset_protocol='construction_seed_then_sequential_unseeded_resets',
                      horizon=horizon,replan_steps=manifest['replan_steps'],policy_queries=0,
                      action_network_evaluations=0,control_steps=0)
        try:
            if env is None:env=gym.make('robocasa/'+a.task,split=manifest['split'],seed=manifest['scene_seed'])
            obs,info=env.reset()
            raw=env.unwrapped.env
            xml=raw.sim.model.get_xml().encode();state=raw.sim.get_state().flatten()
            (scenes/f'{idx:03d}.xml.gz').write_bytes(gzip.compress(xml,mtime=0))
            snapshot=dict(initial_state=state,initial_state_sha256=hashlib.sha256(np.asarray(state).tobytes()).hexdigest(),
                          model_xml_sha256=hashlib.sha256(xml).hexdigest(),episode_metadata=raw.get_ep_meta(),
                          scene_rng_state=raw.rng.bit_generator.state,instruction=obs['annotation.human.task_description'],
                          initial_image_sha256={k:hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for k,v in obs.items() if k.startswith('video.')})
            atomic(scenes/f'{idx:03d}.json',snapshot)
            record.update(instruction=snapshot['instruction'],initial_state_sha256=snapshot['initial_state_sha256'],
                          model_xml_sha256=snapshot['model_xml_sha256'],policy_seed=stable_seed('public_smolvla_baseline' if smol else 'public_pi05_baseline',a.task,idx,manifest['policy_seed']))
            plan=collections.deque();timings=[];success=False
            for t in range(horizon):
                if not plan:
                    def image(key):
                        value=np.ascontiguousarray(obs[key])
                        if smol:
                            if value.shape!=(256,256,3) or value.dtype!=np.uint8:raise RuntimeError('Expected native 256 RGB uint8 image')
                            return value
                        return image_tools.convert_to_uint8(image_tools.resize_with_pad(value,224,224))
                    payload={'observation/image':image('video.robot0_agentview_left'),
                             'observation/right_image':image('video.robot0_agentview_right'),
                             'observation/wrist_image':image('video.robot0_eye_in_hand'),
                             'observation/state':convert_state(obs) if smol else np.concatenate([obs['state.'+k] for k in ('end_effector_position_relative','end_effector_rotation_relative','base_position','base_rotation','gripper_qpos')]),
                             'prompt':a.task if smol else snapshot['instruction'],
                             'campaign_policy_seed':stable_seed(record['policy_seed'],record['policy_queries'])}
                    pred=client.infer(payload);nfe=int(pred['action_network_evaluations']);actions=np.asarray(pred['actions'])
                    if nfe!=manifest['denoising_steps']:raise RuntimeError('Measured denoising count mismatch')
                    if actions.ndim!=2 or actions.shape[1]!=12 or len(actions)<manifest['replan_steps'] or not np.isfinite(actions).all():raise RuntimeError('Invalid actions')
                    if smol and record['policy_queries']==0:
                        audit=audit_interface(obs,actions,Path(manifest['owner_robocasa_adapter']),manifest['owner_robocasa_adapter_sha256'])
                        audit_dir=out/'interface';audit_dir.mkdir(exist_ok=True)
                        atomic(audit_dir/f'{a.task}_{idx:03d}.json',audit)
                    record['policy_queries']+=1;record['action_network_evaluations']+=nfe
                    timings.append(dict(seconds=float(pred['inference_seconds']),warmup=bool(pred.get('warmup_query',False))))
                    plan.extend(actions[:manifest['replan_steps']])
                obs,reward,done,truncated,info=env.step(smol_action(plan.popleft()) if smol else pi05_action(plan.popleft()))
                record['control_steps']=t+1
                if info['success']:success=True;break
            record.update(status='success' if success else 'policy_failure',success=success,
                          query_timings=timings,finished_at=time.time())
            atomic(path,record)
            print(json.dumps({k:record[k] for k in ['task','episode_index','status','control_steps']}),flush=True)
        except BaseException as exc:
            record.update(status='infrastructure_failure',error=str(exc),traceback=traceback.format_exc(),finished_at=time.time())
            atomic(path,record)
            # Continuing after reconstructing the environment changes the sequential
            # scene stream. Stop this task and retain missing coverage explicitly.
            raise
    if env is not None:env.close()

if __name__=='__main__':main()
