"""
Standalone evaluation script for Diffusion Policy checkpoints.
Compatible with checkpoints from train.py (state) and train_rgbd.py (rgb/depth/rgb+depth).

Usage (state):
    python evaluate.py --checkpoint runs/<run_name>/checkpoints/best_eval_success_once.pt \
        --env_id PegInsertionSide-v1 --obs_mode state --max_episode_steps 100

Usage (rgbd):
    python evaluate.py --checkpoint runs/<run_name>/checkpoints/best_eval_success_once.pt \
        --env_id PegInsertionSide-v1 --obs_mode rgb+depth --max_episode_steps 100
"""

import os
import random
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from gymnasium.vector.vector_env import VectorEnv
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.evaluate import evaluate
from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.plain_conv import PlainConv
from diffusion_policy.utils import build_state_obs_extractor, convert_obs


@dataclass
class Args:
    checkpoint: str = "runs/checkpoint/best_eval_success_once.pt"
    """path to the checkpoint file (.pt) to evaluate"""
    env_id: str = "PegInsertionSide-v1"
    """the id of the environment"""
    obs_mode: str = "state"
    """observation mode. Use 'state' for train.py checkpoints, 'rgb', 'depth', or 'rgb+depth' for train_rgbd.py checkpoints"""
    control_mode: str = "pd_joint_delta_pos"
    """the control mode to use for the evaluation environments. Must match the control mode used during training."""
    sim_backend: str = "physx_cpu"
    """the simulation backend. can be 'physx_cpu' or 'physx_cuda'"""
    seed: int = 0
    """random seed"""
    num_eval_episodes: int = 100
    """the number of episodes to evaluate the agent on"""
    num_eval_envs: int = 10
    """the number of parallel environments to evaluate the agent on"""
    max_episode_steps: int = 200
    """max steps per episode. Must match the value used during training."""
    capture_video: bool = False
    """whether to capture videos of the evaluation"""
    video_dir: Optional[str] = None
    """directory to save evaluation videos. Defaults to <checkpoint_dir>/eval_videos if capture_video is True."""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""

    # Diffusion Policy architecture -- must match the training run
    obs_horizon: int = 2
    """observation horizon used during training"""
    act_horizon: int = 8
    """action horizon used during training"""
    pred_horizon: int = 16
    """prediction horizon used during training"""
    diffusion_step_embed_dim: int = 64
    unet_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    n_groups: int = 8


# ---------------------------------------------------------------------------
# Agent for state-based observations (matches train.py)
# ---------------------------------------------------------------------------
class AgentState(nn.Module):
    def __init__(self, env: VectorEnv, args: Args):
        super().__init__()
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon
        assert len(env.single_observation_space.shape) == 2  # (obs_horizon, obs_dim)
        assert len(env.single_action_space.shape) == 1  # (act_dim,)
        assert (env.single_action_space.high == 1).all() and (env.single_action_space.low == -1).all()
        self.act_dim = env.single_action_space.shape[0]

        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.act_dim,
            global_cond_dim=np.prod(env.single_observation_space.shape),  # obs_horizon * obs_dim
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        )
        self.num_diffusion_iters = 100
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon',
        )

    def get_action(self, obs_seq):
        # obs_seq: (B, obs_horizon, obs_dim)
        B = obs_seq.shape[0]
        with torch.no_grad():
            obs_cond = obs_seq.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)
            noisy_action_seq = torch.randn(
                (B, self.pred_horizon, self.act_dim), device=obs_seq.device
            )
            for k in self.noise_scheduler.timesteps:
                noise_pred = self.noise_pred_net(
                    sample=noisy_action_seq, timestep=k, global_cond=obs_cond
                )
                noisy_action_seq = self.noise_scheduler.step(
                    model_output=noise_pred, timestep=k, sample=noisy_action_seq
                ).prev_sample
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return noisy_action_seq[:, start:end]  # (B, act_horizon, act_dim)


# ---------------------------------------------------------------------------
# Agent for RGBD observations (matches train_rgbd.py)
# ---------------------------------------------------------------------------
class AgentRGBD(nn.Module):
    def __init__(self, env: VectorEnv, args: Args):
        super().__init__()
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon
        assert len(env.single_observation_space["state"].shape) == 2  # (obs_horizon, obs_dim)
        assert len(env.single_action_space.shape) == 1  # (act_dim,)
        assert (env.single_action_space.high == 1).all() and (env.single_action_space.low == -1).all()
        self.act_dim = env.single_action_space.shape[0]
        obs_state_dim = env.single_observation_space["state"].shape[1]

        self.include_rgb = "rgb" in env.single_observation_space.keys()
        self.include_depth = "depth" in env.single_observation_space.keys()
        total_visual_channels = 0
        if self.include_rgb:
            total_visual_channels += env.single_observation_space["rgb"].shape[-1]
        if self.include_depth:
            total_visual_channels += env.single_observation_space["depth"].shape[-1]

        visual_feature_dim = 256
        self.visual_encoder = PlainConv(
            in_channels=total_visual_channels, out_dim=visual_feature_dim, pool_feature_map=True
        )
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.act_dim,
            global_cond_dim=self.obs_horizon * (visual_feature_dim + obs_state_dim),
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        )
        self.num_diffusion_iters = 100
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

    def encode_obs(self, obs_seq, eval_mode):
        if self.include_rgb:
            rgb = obs_seq["rgb"].float() / 255.0  # (B, obs_horizon, 3*k, H, W)
            img_seq = rgb
        if self.include_depth:
            depth = obs_seq["depth"].float() / 1024.0  # (B, obs_horizon, 1*k, H, W)
            img_seq = depth
        if self.include_rgb and self.include_depth:
            img_seq = torch.cat([rgb, depth], dim=2)  # (B, obs_horizon, C, H, W)
        batch_size = img_seq.shape[0]
        img_seq = img_seq.flatten(end_dim=1)  # (B*obs_horizon, C, H, W)
        visual_feature = self.visual_encoder(img_seq)  # (B*obs_horizon, D)
        visual_feature = visual_feature.reshape(
            batch_size, self.obs_horizon, visual_feature.shape[1]
        )  # (B, obs_horizon, D)
        feature = torch.cat(
            (visual_feature, obs_seq["state"]), dim=-1
        )  # (B, obs_horizon, D+obs_state_dim)
        return feature.flatten(start_dim=1)  # (B, obs_horizon * (D+obs_state_dim))

    def get_action(self, obs_seq):
        # obs_seq: dict with 'state', optionally 'rgb' and/or 'depth'
        B = obs_seq["state"].shape[0]
        with torch.no_grad():
            if self.include_rgb:
                obs_seq["rgb"] = obs_seq["rgb"].permute(0, 1, 4, 2, 3)
            if self.include_depth:
                obs_seq["depth"] = obs_seq["depth"].permute(0, 1, 4, 2, 3)
            obs_cond = self.encode_obs(obs_seq, eval_mode=True)

            noisy_action_seq = torch.randn(
                (B, self.pred_horizon, self.act_dim), device=obs_seq["state"].device
            )
            for k in self.noise_scheduler.timesteps:
                noise_pred = self.noise_pred_net(
                    sample=noisy_action_seq, timestep=k, global_cond=obs_cond
                )
                noisy_action_seq = self.noise_scheduler.step(
                    model_output=noise_pred, timestep=k, sample=noisy_action_seq
                ).prev_sample
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return noisy_action_seq[:, start:end]  # (B, act_horizon, act_dim)


if __name__ == "__main__":
    args = tyro.cli(Args)

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    is_rgbd = args.obs_mode != "state"

    # video output dir
    video_dir = None
    if args.capture_video:
        video_dir = args.video_dir or os.path.join(
            os.path.dirname(args.checkpoint), "eval_videos"
        )

    # build environment
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        human_render_camera_configs=dict(shader_pack="default"),
        max_episode_steps=args.max_episode_steps,
    )
    other_kwargs = dict(obs_horizon=args.obs_horizon)

    wrappers = [FlattenRGBDObservationWrapper] if is_rgbd else []

    envs = make_eval_envs(
        args.env_id,
        args.num_eval_envs,
        args.sim_backend,
        env_kwargs,
        other_kwargs,
        video_dir=video_dir,
        wrappers=wrappers,
    )

    # build agent matching the training architecture
    if is_rgbd:
        agent = AgentRGBD(envs, args).to(device)
    else:
        agent = AgentState(envs, args).to(device)

    # load checkpoint -- prefer ema_agent weights if available
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict_key = "ema_agent" if "ema_agent" in ckpt else "agent"
    agent.load_state_dict(ckpt[state_dict_key])
    print(f"  Loaded weights from '{state_dict_key}' key.")

    # run evaluation
    print(f"\nEvaluating {args.num_eval_episodes} episodes "
          f"({args.num_eval_envs} parallel envs, obs_mode={args.obs_mode}) ...")
    eval_metrics = evaluate(
        args.num_eval_episodes, agent, envs, device, args.sim_backend
    )

    print("\n--- Evaluation Results ---")
    for k, v in eval_metrics.items():
        print(f"  {k}: {np.mean(v):.4f}")

    envs.close()
