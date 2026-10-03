"""Lossless numeric traces and video captured from existing observations only."""
import hashlib
import json
from pathlib import Path
import uuid

def rng_hash():
 import numpy as np
 s=np.random.get_state()
 h=hashlib.sha256(s[0].encode()+s[1].tobytes()+str(s[2:]).encode())
 return h.hexdigest()

class Recorder:
 def __init__(self, root, benchmark, condition, case, arm):
  import imageio
  key=[benchmark,condition["condition_id"],case["seed"],case["init_index"],arm]
  self.key=hashlib.sha256(json.dumps(key,separators=(",",":")).encode()).hexdigest()
  folder=Path(root)/self.key[:2]; folder.mkdir(parents=True,exist_ok=True)
  self.stem=folder/(self.key+"-"+uuid.uuid4().hex[:10])
  self.video=self.stem.with_suffix(".mp4")
  self.trace=self.stem.with_suffix(".npz")
  self.writer=imageio.get_writer(str(self.video),fps=20,codec="libx264",quality=7,macro_block_size=16)
  self.actions=[];self.states=[];self.rng=[];self.success=[];self.frames=0
 def observe(self,obs):
  import numpy as np
  frame=np.concatenate((obs["agentview_image"][::-1,::-1],obs["robot0_eye_in_hand_image"][::-1,::-1]),axis=1)
  self.writer.append_data(np.ascontiguousarray(frame))
  self.frames+=1
 def action(self,obs,action):
  import numpy as np
  self.actions.append(np.asarray(action).copy())
  self.states.append(np.concatenate((obs["robot0_eef_pos"],obs["robot0_eef_quat"],obs["robot0_gripper_qpos"])).copy())
  self.rng.append(rng_hash())
 def finish(self):
  import numpy as np
  self.writer.close()
  np.savez_compressed(self.trace,actions=np.asarray(self.actions),eef_gripper_states=np.asarray(self.states),rng_sha256=np.asarray(self.rng),success=np.asarray(self.success))
  def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
  return dict(video_path=str(self.video),video_sha256=digest(self.video),trace_path=str(self.trace),trace_sha256=digest(self.trace),video_frames=self.frames,executed_actions=len(self.actions))
 def close(self):
  self.writer.close()
