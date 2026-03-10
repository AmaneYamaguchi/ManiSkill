"""
Standalone evaluation script for ACT (Action Chunking with Transformers) checkpoints.
Compatible with checkpoints from train.py (state) and train_rgbd.py (rgb / rgb+depth).

Usage (state):
    python evaluate.py --checkpoint runs/<run_name>/checkpoints/best_eval_success_once.pt \
        --env-id PickCube-v1 --obs-mode state --max-episode-steps 100

Usage (rgbd):
    python evaluate.py --checkpoint runs/<run_name>/checkpoints/best_eval_success_once.pt \
        --env-id PickCube-v1 --obs-mode rgb --max-episode-steps 100
"""

import os
import random
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import tyro

from act.evaluate import evaluate
from act.make_env import make_eval_envs
from act.detr.transformer import build_transformer
from act.detr.detr_vae import build_encoder, DETRVAE


@dataclass
class Args:
    checkpoint: str = "runs/checkpoint/best_eval_success_once.pt"
    """path to the checkpoint file (.pt) to evaluate"""
    env_id: str = "PickCube-v1"
    """the id of the environment"""
    obs_mode: str = "state"
    """observation mode. Use 'state' for train.py checkpoints, 'rgb' or 'rgb+depth' for train_rgbd.py checkpoints"""
    control_mode: str = "pd_ee_delta_pos"
    """the control mode. Must match the control mode used during training."""
    sim_backend: str = "physx_cpu"
    """the simulation backend. can be 'physx_cpu' or 'physx_cuda'"""
    seed: int = 0
    """random seed"""
    num_eval_episodes: int = 100
    """the number of episodes to evaluate the agent on"""
    num_eval_envs: int = 10
    """the number of parallel environments to evaluate the agent on"""
    max_episode_steps: int = 100
    """max steps per episode. Must match the value used during training."""
    capture_video: bool = False
    """whether to capture videos of the evaluation"""
    video_dir: Optional[str] = None
    """directory to save evaluation videos. Defaults to <checkpoint_dir>/eval_videos if capture_video is True."""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""

    # ACT architecture -- must match the training run
    kl_weight: float = 10
    num_queries: int = 30
    temporal_agg: bool = True

    # Backbone (RGBD only)
    position_embedding: str = "sine"
    backbone: str = "resnet18"
    lr_backbone: float = 1e-5
    masks: bool = False
    dilation: bool = False
    include_depth: bool = True

    # Transformer
    enc_layers: int = 2
    dec_layers: int = 4
    dim_feedforward: int = 512
    hidden_dim: int = 256
    dropout: float = 0.1
    nheads: int = 4
    pre_norm: bool = False


# ---------------------------------------------------------------------------
# Agent for state-based observations (matches train.py)
# ---------------------------------------------------------------------------
class AgentState(nn.Module):
    def __init__(self, env, args: Args):
        super().__init__()
        assert len(env.single_observation_space.shape) == 1  # (obs_dim,)
        assert len(env.single_action_space.shape) == 1  # (act_dim,)

        self.kl_weight = args.kl_weight
        self.state_dim = env.single_observation_space.shape[0]
        self.act_dim = env.single_action_space.shape[0]

        backbones = None
        transformer = build_transformer(args)
        encoder = build_encoder(args)
        self.model = DETRVAE(
            backbones,
            transformer,
            encoder,
            state_dim=self.state_dim,
            action_dim=self.act_dim,
            num_queries=args.num_queries,
        )

    def get_action(self, obs):
        a_hat, (_, _) = self.model(obs)  # sample from prior
        return a_hat


# ---------------------------------------------------------------------------
# Agent for RGBD observations (matches train_rgbd.py)
# ---------------------------------------------------------------------------
class AgentRGBD(nn.Module):
    def __init__(self, env, args: Args):
        super().__init__()
        assert len(env.single_observation_space["state"].shape) == 1  # (obs_dim,)
        assert len(env.single_observation_space["rgb"].shape) == 4    # (num_cams, C, H, W)
        assert len(env.single_action_space.shape) == 1                # (act_dim,)

        self.state_dim = env.single_observation_space["state"].shape[0]
        self.act_dim = env.single_action_space.shape[0]
        self.kl_weight = args.kl_weight
        self.include_depth = args.include_depth
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        from act.detr.backbone import build_backbone
        backbones = [build_backbone(args)]
        transformer = build_transformer(args)
        encoder = build_encoder(args)
        self.model = DETRVAE(
            backbones,
            transformer,
            encoder,
            state_dim=self.state_dim,
            action_dim=self.act_dim,
            num_queries=args.num_queries,
        )

    def get_action(self, obs):
        obs["rgb"] = obs["rgb"].float() / 255.0
        obs["rgb"] = self.normalize(obs["rgb"])
        if self.include_depth:
            obs["depth"] = obs["depth"].float()
        a_hat, (_, _) = self.model(obs)  # sample from prior
        return a_hat


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
        obs_mode=args.obs_mode if not is_rgbd else ("rgbd" if args.include_depth else "rgb"),
        render_mode="rgb_array",
        max_episode_steps=args.max_episode_steps,
    )
    other_kwargs = None
    wrappers = []
    if is_rgbd:
        from functools import partial as _partial
        from train_rgbd import FlattenRGBDObservationWrapper
        wrappers = [_partial(FlattenRGBDObservationWrapper, depth=args.include_depth)]

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

    # load checkpoint -- prefer ema_agent weights, fall back to agent
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict_key = "ema_agent" if "ema_agent" in ckpt else "agent"
    agent.load_state_dict(ckpt[state_dict_key])
    print(f"  Loaded weights from '{state_dict_key}' key.")

    # load norm_stats from checkpoint (None for delta control modes)
    norm_stats = ckpt.get("norm_stats", None)

    # build eval_kwargs matching act/evaluate.py signature
    eval_kwargs = dict(
        stats=norm_stats,
        num_queries=args.num_queries,
        temporal_agg=args.temporal_agg,
        max_timesteps=args.max_episode_steps,
        device=device,
        sim_backend=args.sim_backend,
    )

    # run evaluation
    print(f"\nEvaluating {args.num_eval_episodes} episodes "
          f"({args.num_eval_envs} parallel envs, obs_mode={args.obs_mode}) ...")
    eval_metrics = evaluate(args.num_eval_episodes, agent, envs, eval_kwargs)

    print("\n--- Evaluation Results ---")
    for k, v in eval_metrics.items():
        print(f"  {k}: {np.mean(v):.4f}")

    envs.close()
