"""Checkpoint-native interfaces, checked against the pinned owner's evaluator."""
import ast
import hashlib
from pathlib import Path
import numpy as np

FIELDS=('base_position','base_rotation','end_effector_position_relative','end_effector_rotation_relative','gripper_qpos')

def convert_state(obs):
    state=np.concatenate([obs['state.'+k] for k in FIELDS],axis=-1).astype(np.float32)
    if state.shape!=(16,) or not np.isfinite(state).all():raise ValueError('Invalid state')
    return state

def convert_action(action):
    return {'action.base_motion':action[0:4], 'action.control_mode':action[4:5],
            'action.end_effector_position':action[5:8], 'action.end_effector_rotation':action[8:11],
            'action.gripper_close':action[11:12]}

def audit_interface(raw_obs, actions, path, expected_sha256):
    path=Path(path)
    source=path.read_text()
    digest=hashlib.sha256(source.encode()).hexdigest()
    if digest != expected_sha256:raise ValueError('Pinned owner adapter hash mismatch')
    tree=ast.parse(source)
    func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='convert_action')
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),func],type_ignores=[])
    ns={'np':np};exec(compile(ast.fix_missing_locations(module),str(path),'exec'),ns)
    assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='agent_pos' for t in n.targets))
    expected=eval(compile(ast.Expression(assignment.value),str(path),'eval'),{'np':np,'raw_obs':raw_obs})
    actual=convert_state(raw_obs)
    if not np.array_equal(actual,expected):raise ValueError('Owner state mismatch')
    for action in [np.arange(12,dtype=np.float32)]+list(actions):
        a=convert_action(action);b=ns['convert_action'](action)
        if a.keys()!=b.keys() or not all(np.array_equal(a[k],b[k]) for k in a):raise ValueError('Owner action mismatch')
    return dict(passed=True,reference_sha256=digest,state_fields=FIELDS,state_exact=True,action_exact=True,action_chunks_checked=len(actions),state_sha256=hashlib.sha256(actual.tobytes()).hexdigest())
