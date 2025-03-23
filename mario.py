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

# 常用的数据结构和工具
import os
import datetime
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

def create_mario_env():
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode="rgb", apply_api_compatibility=True)
    env = JoypadSpace(env, [["right"], ["right", "A"]])

    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)

    return env



if __name__ == "__main__":
    env = create_mario_env()

    env.reset()
    next_state, reward, done, trunc, info = env.step(action=0)
    print(f"{next_state.shape}\n{reward}\n{done}\n{info}")
    img = Image.fromarray((next_state[1] * 255).numpy())
    img.show()
    