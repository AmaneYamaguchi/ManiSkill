import gymnasium as gym
import torch
from collections import defaultdict
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
env_id = "PickCube-v1"
num_eval_envs = 64
env_kwargs = dict(obs_mode="state") # modify your env_kwargs here
eval_envs = gym.make(env_id, num_envs=num_eval_envs, reconfiguration_freq=1, **env_kwargs)
# add any other wrappers here
eval_envs = ManiSkillVectorEnv(eval_envs, ignore_terminations=True, record_metrics=True)

# evaluation loop, which will record metrics for complete episodes only
obs, _ = eval_envs.reset(seed=0)
eval_metrics = defaultdict(list)
for _ in range(400):
    action = eval_envs.action_space.sample() # replace with your policy action
    obs, rew, terminated, truncated, info = eval_envs.step(action)
    # note as there are no partial resets, truncated is True for all environments at the same time
    if truncated.any():
        for k, v in info["final_info"]["episode"].items():
            eval_metrics[k].append(v.float())
for k in eval_metrics.keys():
    print(f"{k}_mean: {torch.mean(torch.stack(eval_metrics[k])).item()}")