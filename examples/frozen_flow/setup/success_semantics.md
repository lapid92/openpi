# Success semantics source audit

No policy outcomes used for this audit. LIBERO step-returned done is overwritten by task goal satisfaction, not simulator horizon timeout. Plus wrapper modifies camera noise only and returns the same done. Episode runner must separately treat its action budget exhaustion as failure and exceptions as errors.

## LIBERO

/volt/code/frozen-flow-study/third_party/libero/libero/libero/envs/env_wrapper.py:103
```python
103:     def check_success(self):
104:         return self.env._check_success()
105:
106:     @property
107:     def _visualizations(self):
108:         return self.env._visualizations
109:
110:     @property
111:     def robots(self):
112:         return self.env.robots
113:
114:     @property
```

/volt/code/frozen-flow-study/third_party/libero/libero/libero/envs/bddl_base_domain.py:800
```python
800:     def step(self, action):
801:         if self.action_dim == 4 and len(action) > 4:
802:             # Convert OSC_POSITION action
803:             action = np.array(action)
804:             action = np.concatenate((action[:3], action[-1:]), axis=-1)
805:
806:         obs, reward, done, info = super().step(action)
807:         done = self._check_success()
808:
809:         return obs, reward, done, info
810:
811:     def _pre_action(self, action, policy_step=False):
```

/volt/code/frozen-flow-study/third_party/libero/libero/libero/envs/problems/libero_tabletop_manipulation.py:135
```python
135:     def _check_success(self):
136:         """
137:         Check if the goal is achieved. Consider conjunction goals at the moment
138:         """
139:         goal_state = self.parsed_problem["goal_state"]
140:         result = True
141:         for state in goal_state:
142:             result = self._eval_predicate(state) and result
143:         return result
144:
145:     def _eval_predicate(self, state):
146:         if len(state) == 3:
```

## LIBERO-Plus

/volt/benchmarks/libero-plus/libero/libero/envs/env_wrapper.py:350
```python
350:     def check_success(self):
351:         return self.env._check_success()
352:
353:     @property
354:     def _visualizations(self):
355:         return self.env._visualizations
356:
357:     @property
358:     def robots(self):
359:         return self.env.robots
360:
361:     @property
```

/volt/benchmarks/libero-plus/libero/libero/envs/bddl_base_domain.py:801
```python
801:     def step(self, action):
802:         if self.action_dim == 4 and len(action) > 4:
803:             # Convert OSC_POSITION action
804:             action = np.array(action)
805:             action = np.concatenate((action[:3], action[-1:]), axis=-1)
806:
807:         obs, reward, done, info = super().step(action)
808:         done = self._check_success()
809:
810:         return obs, reward, done, info
811:
812:     def _pre_action(self, action, policy_step=False):
```

/volt/benchmarks/libero-plus/libero/libero/envs/problems/libero_tabletop_manipulation.py:253
```python
253:     def _check_success(self):
254:         """
255:         Check if the goal is achieved. Consider conjunction goals at the moment
256:         """
257:         goal_state = self.parsed_problem["goal_state"]
258:         result = True
259:         for state in goal_state:
260:             result = self._eval_predicate(state) and result
261:         return result
262:
263:     def _eval_predicate(self, state):
264:         if len(state) == 3:
```
