# RoboCasa checkpoint discovery (2026-10-01)

Public JAX candidate: https://huggingface.co/changyeon/pi05_robocasa_as50_jax
Revision: 165da7e92fdbd140c4fe3e2a4bad6f0cabcda47e
Contains step `10000` Orbax params and normalization asset `robocasa_lerobot_100demos_pi0/norm_stats.json`. Orbax metadata has non-null commit_timestamp_nsecs=1764066239341937599, verifying completed checkpoint save, not completed intended training. Normalization arrays have 32 state and 32 action components. No README, model card, training configuration or observation/action adapter is provided in the listing. The name is suggestive, not sufficient verification of architecture/interface.

Additional public candidate https://huggingface.co/changyeon/pi05_robocasa_as50_filteredbc_pytorch has step 10000,20000,29999,30000 model.safetensors and metadata.pt, also no model card. Metadata pickle was not executed. Public GitHub username changyeon does not expose matching robotics implementation among listed repositories; identity correspondence is not established.

Public LeRobot candidates exist, including https://huggingface.co/ruiname/pi05-robocasa-10tasks-200k . Its generic card identifies ten task dataset repositories and LeRobot implementation, but does not establish a compatible OpenPI checkpoint/interface. It is a separate implementation requiring its own verification.

Current blocker is therefore NOT absence of public weights. It is absence of a verified checkpoint/model configuration and corresponding RoboCasa observation/action normalization, camera order, control interface and simulator version within the accessible evaluation infrastructure. Do not run the LIBERO adapter or residual head on these candidates. No RoboCasa evaluation performed. Evidence JSON and raw metadata saved alongside this note.
