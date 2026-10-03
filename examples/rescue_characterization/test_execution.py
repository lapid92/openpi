"""Mock-only execution parity and recording tests; no simulator/GPU calls."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import client
import recording
import recording_audit

class FakeEnv:
    instances=[]
    def __init__(self,**kwargs):
        self.calls=[];self.actions=[];self.count=0
        self.__class__.instances.append(self)
    def obs(self):
        return dict(agentview_image=np.zeros((16,16,3),dtype=np.uint8),
                    robot0_eye_in_hand_image=np.zeros((16,16,3),dtype=np.uint8),
                    robot0_eef_pos=np.zeros(3),robot0_eef_quat=np.array([0.,0.,0.,1.]),
                    robot0_gripper_qpos=np.zeros(2))
    def seed(self,seed):self.calls.append("seed")
    def reset(self):self.calls.append("reset")
    def set_init_state(self,state):self.calls.append("set_init");return self.obs()
    def get_sim_state(self):self.calls.append("get_state");return np.zeros(4)
    def step(self,action):
        self.calls.append("step");self.actions.append(list(action));self.count+=1
        return self.obs(),0,False,{}
    def check_success(self):self.calls.append("check_success");return self.count==5
    def close(self):self.calls.append("close")

class MemoryRecorder:
    instances=[]
    def __init__(self,*args):
        self.actions=[];self.success=[];self.observations=[]
        self.__class__.instances.append(self)
    def observe(self,obs):self.observations.append(obs)
    def action(self,obs,action):self.actions.append(list(action))
    def finish(self):return {}
    def close(self):pass

class ExecutionTests(unittest.TestCase):
    def test_recording_preserves_original_environment_call_sequence(self):
        oldspec=importlib.util.spec_from_file_location("original_client",HERE.parent/"frozen_flow"/"client.py")
        old=importlib.util.module_from_spec(oldspec);oldspec.loader.exec_module(old)
        manifest=dict(settings=dict(render_size=16,image_size=16,wait_steps=2,replan_steps=2,max_policy_steps={"suite":8}),
                      noise_policy={"shape":[4,7]},flow_steps=[1,2,4,10],recording={"root":"/unused"})
        condition=dict(bddl_path="/unused",task_index=0,prompt="task",suite="suite",task_name="task",condition_id="test")
        state=np.zeros(4)
        case=dict(seed=3001,init_index=0,initial_state_sha256=client.digest_array(state))
        suite=types.SimpleNamespace(get_task_init_states=lambda _: [state])
        fake_modules={"libero":types.ModuleType("libero"),"libero.libero":types.ModuleType("libero.libero"),
                      "libero.libero.envs":types.SimpleNamespace(OffScreenRenderEnv=FakeEnv)}
        def response(url,payload):
            actions=np.arange(28,dtype=float).reshape(4,7).tolist()
            return dict(actions=actions,velocity_evaluations=payload["flow_steps"],noise_sha256="n",action_sha256="a",policy_ms=1)
        with mock.patch.dict(sys.modules,fake_modules):
            for module in (old,client):
                with mock.patch.object(module,"observation_payload",return_value={"obs":"fixed"}),mock.patch.object(module,"post_json",side_effect=response),mock.patch.object(client,"Recorder",MemoryRecorder):
                    result=module.run_episode(suite,condition,case,2,manifest,"libero","http://fake")
                    self.assertEqual(result["policy_steps"],3)
                    self.assertTrue(result["success"])
        a,b=FakeEnv.instances[-2:]
        self.assertEqual(a.calls,b.calls)
        self.assertEqual(a.actions,b.actions)
        capture=MemoryRecorder.instances[-1]
        self.assertEqual(capture.actions,b.actions[2:])
        self.assertEqual(len(capture.observations),4)
        self.assertEqual(capture.success,[False,False,True])

    def test_rng_hash_does_not_consume_randomness(self):
        np.random.seed(17);before=copy.deepcopy(np.random.get_state())
        recording.rng_hash()
        after=np.random.get_state()
        self.assertEqual(before[0],after[0])
        self.assertTrue(np.array_equal(before[1],after[1]))
        self.assertEqual(before[2:],after[2:])

    def test_recorder_and_audit_detect_action_tampering(self):
        class Writer:
            def __init__(self,path):self.path=Path(path)
            def append_data(self,frame):self.path.write_bytes(b"mock video")
            def close(self):pass
        with tempfile.TemporaryDirectory(dir="/volt/artifacts/rescue-characterization") as temp:
            with mock.patch("imageio.get_writer",side_effect=lambda path,**kw:Writer(path)):
                recorder=recording.Recorder(temp,"libero",{"condition_id":"c"},{"seed":1,"init_index":0},1)
                obs=FakeEnv().obs();recorder.observe(obs)
                actions=np.arange(28,dtype=np.float64).reshape(4,7)
                rng=recording.rng_hash()
                recorder.action(obs,actions[0]);recorder.observe(obs);recorder.success.append(True)
                capture=recorder.finish()
            import hashlib
            row=dict(status="ok",recording=capture,policy_steps=1,success=True,flow_steps=1,
                     total_velocity_evaluations=1,chunks=[dict(chunk_index=0,velocity_evaluations=1,
                     actions=actions.tolist(),noise_sha256="a"*64,observation_sha256="b"*64,action_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),simulator_rng_sha256=rng)])
            manifest=dict(settings={"replan_steps":2,"render_size":16},noise_policy={"shape":[4,7]})
            self.assertEqual(recording_audit.audit_record(row,manifest,inspect_video=False)["steps"],1)
            row["chunks"][0]["actions"][0][0]=100
            with self.assertRaisesRegex(ValueError,"action digest"):
                recording_audit.audit_record(row,manifest,inspect_video=False)

if __name__=="__main__":
    unittest.main()

