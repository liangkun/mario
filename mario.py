#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 机器学习基础库
import torch
import numpy as np
from torch import nn
from torchvision import transforms as T
from PIL import Image
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer
from torchrl.data import LazyMemmapStorage
import matplotlib.pyplot as plt

# 常用的数据结构和工具
import os
import datetime
import time
import random
from pathlib import Path
from collections import deque
import argparse

# 游戏环境基础库
import gym
from gym.spaces import Box
from gym.wrappers.frame_stack import FrameStack
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

OB_H = 84
OB_W = 84
NUM_SKIP = 4
NUM_STACK = 4
ACTIONS = RIGHT_ONLY

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """每skip帧中返回一帧。
        
           相邻的帧其实差别不大，不用每一帧都“看”，每个skip帧，让模型看到一帧就可以。
        """
        super().__init__(env)
        self._skip = skip
    
    def step(self, action):
        """重复指定的action，将所有的reward累计。"""
        total_reward = 0.0
        for i in range(self._skip):
            next_state, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return next_state, total_reward, done, trunk, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
    
    def permute_orientation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation
    
    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)
        
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
    
    def observation(self, observation):
        transform = T.Compose([
            T.Resize(self.shape, antialias=True),
            T.Normalize(0, 255)
        ])
        observation = transform(observation)
        observation = observation.squeeze(0)
        return observation


class MarioNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim
        if h != OB_H:
            raise ValueError(f"expect input height {OB_H}, got {h}")
        if w != OB_W:
            raise ValueError(f"expect input width {OB_W}, got {w}")
        
        self.online = self.__build_cnn(c, output_dim)

        self.target = self.__build_cnn(c, output_dim)
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad = False
    
    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        if model == "target":
            return self.target(input)
        
        raise ValueError(f"unknown model: {model}")

    def __build_cnn(self, c, output_dim):
        return nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )


class Mario:
    def __init__(self, state_dim, action_dim, options):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = Path(options.save_dir)
    
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        if torch.backends.mps.is_available():
            self.device = "mps"
        
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)

        self.exploration_rate = options.exploration_rate
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = options.exploration_rate_min
        self.curr_step = 0
        self.save_every = 5e5
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000,
                                             device=torch.device("cpu")))
        self.batch_size = options.batch_size
        self.gamma = 0.9
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=options.lr)
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.burnin = options.burnin # 开始训练前最少的动作数
        self.learn_every = 3
        self.sync_every = 1e4

    def act(self, state):
        """在当前状态下选择最优动作（使用epsilon-greedy）。"""
        # 随机探索
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # 选择最优动作
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()
        
        # 降低exploration rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate, self.exploration_rate_min)

        self.curr_step += 1

        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """记录经验到记忆中"""
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
        
        state = first_if_tuple(state).__array__()
        state = torch.tensor(state)
        next_state = first_if_tuple(next_state).__array__()
        next_state = torch.tensor(next_state)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])
        self.memory.add(TensorDict({
            "state": state,
            "next_state": next_state,
            "action": action,
            "reward": reward,
            "done": done
        }, batch_size=[]))

    def recall(self):
        """从记忆中采样经验"""
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def learn(self):
        """学习最优策略"""
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()
        
        if self.curr_step % self.save_every == 0:
            self.save()
        
        if self.curr_step < self.burnin:
            return None, None
        
        if self.curr_step % self.learn_every != 0:
            return None, None
        
        # 强化学习
        state, next_state, action, reward, done = self.recall()
        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)
        loss = self.update_Q_online(td_est, td_tgt)
        return (td_est.mean().item(), loss)

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[np.arange(0, self.batch_size), action]
        return current_Q
    
    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()
    
    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())
    
    def save(self):
        """保存检查点"""
        save_path = self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        torch.save(self.net.state_dict(), save_path)
        print(f"MarioNet saved to {save_path} at step {self.curr_step}.")
    
    def load(self, modelfile):
        self.net.load_state_dict(torch.load(modelfile))


class MetricLogger:
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        self.init_episode()
        self.record_time = time.time()

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_lengths", "ep_avg_losses", "ep_avg_qs", "ep_rewards"]:
            plt.clf()
            plt.plot(getattr(self, f"moving_avg_{metric}"), label=f"moving_avg_{metric}")
            plt.legend()
            plt.savefig(getattr(self, f"{metric}_plot"))

def create_mario_env(options):
    train = options.mode == "train"
    render_mode = "rgb" if train else "human"

    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode=render_mode, apply_api_compatibility=True)
    env = JoypadSpace(env, ACTIONS)

    env = SkipFrame(env, skip=NUM_SKIP)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=NUM_STACK)

    return env


def train(mario, env, options):
    save_dir = Path(options.save_dir)
    logger = MetricLogger(save_dir)

    episodes = options.episodes
    for e in range(episodes):
        state = env.reset()
        steps = 0
        while True:
            action = mario.act(state)
            next_state, reward, done, trunc, info = env.step(action)
            mario.cache(state, next_state, action, reward, done)
            q, loss = mario.learn()
            logger.log_step(reward, loss, q)

            state = next_state

            steps += 1

            if done or info["flag_get"] or steps >= options.max_episode_length:
                break
        
        logger.log_episode()

        if (e % 20 == 0) or (e == episodes - 1):
            logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)


def test(mario, env, options):
    episodes = options.episodes
    for e in range(episodes):
        state = env.reset()
        steps = 0
        episode_reward = 0
        while True:
            action = mario.act(state)
            next_state, reward, done, trunc, info = env.step(action)
            state = next_state
            episode_reward += reward
            steps += 1
            if done or info["flag_get"] or steps > options.max_episode_length:
                break
            time.sleep(0.05)

        print(f"episode: {e}, steps: {steps}, reward: {episode_reward}")


if __name__ == "__main__":
    import sys
    import argparse
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    parser = argparse.ArgumentParser("Robot Mario")
    parser.add_argument("--lr", type=float, default=0.00025)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_episode_length", type=int, default=5000)
    parser.add_argument("--mode", type=str, default="test")
    parser.add_argument("--modelfile", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--burnin", type=int, default=1e4)
    parser.add_argument("--exploration_rate", type=float, default=1.0)
    parser.add_argument("--exploration_rate_min", type=float, default=0.1)
    options = parser.parse_args(sys.argv[1:])

    save_dir = Path(options.save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    env = create_mario_env(options)
    mario = Mario(state_dim=(NUM_STACK, OB_H, OB_W), action_dim=len(ACTIONS), options=options)
    if options.modelfile:
        mario.load(options.modelfile)

    print(f"Using device: {mario.device}")

    if options.mode == "train":
        train(mario, env, options)
    elif options.mode == "test":
        test(mario, env, options)
    else:
        print(f"unknown running mode {options.mode}, specify ether 'test' or 'train'.")
