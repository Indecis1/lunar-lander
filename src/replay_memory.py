import random

from collections import deque, namedtuple

import numpy as np


class ReplayMemory:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        random.seed(seed)
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["observation", "action", "reward", "next_observation", "done"])
        self.seed = seed

    def add(self, observation, action, reward, next_observation, done):
        e = self.experience(observation, action, reward, next_observation, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        obs = np.array([e.observation for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None])
        rewards = np.array([e.reward for e in experiences if e is not None])
        next_obs = np.array([e.next_observation for e in experiences if e is not None])
        dones = np.array([e.done for e in experiences if e is not None])

        return obs, actions, rewards, next_obs, dones

    def __len__(self):
        return len(self.memory)
