import numpy as np
import os
import random
import tensorflow as tf

from collections import namedtuple
from replay_memory import ReplayMemory
from utils_model import build_model

AgentCont = namedtuple("AgentConst", ["buffer_size", "batch_size", "update_every", "gamma", "tau", "input_size", "actions_num"])
agent_const = AgentCont(
    100000,
    128,
    5,
    0.995,
    0.01,
    8,
    4
)


class Agent:
    def __init__(self, seed):
        random.seed(seed)
        self.input_size = agent_const.input_size
        self.action_num = agent_const.actions_num
        self.seed = seed
        self.optimizer = tf.keras.optimizers.Adam(lr= 0.001)
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.network_to_train = build_model(self.input_size,  self.action_num , self.loss_fn, self.optimizer)
       

        self.memory = ReplayMemory(self.action_num, agent_const.buffer_size, agent_const.batch_size, seed)

        self.timestep = 0

    def step(self, obs, action, reward, next_obs, done):
        self.memory.add(obs, action, reward, next_obs, done)
        self.timestep += 1
        if len(self.memory) > agent_const.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def play_action(self, obs, epsilon):
        action_logit = self.network_to_train.predict(obs.reshape(1, -1), verbose=0)

        choice = self.greedy_choice(epsilon)
        if choice == "random":
            return np.random.choice(np.arange(self.action_num))
        else:
            return np.argmax(action_logit)

    def greedy_choice(self, epsilon):
        return np.random.choice(["random", "model"], p=[epsilon, 1 - epsilon])

    def learn(self, experiences):
        obs, actions, rewards, next_obs, dones = experiences
        
        q_val = rewards + agent_const.gamma * np.amax(self.network_to_train.predict_on_batch(next_obs), \
                                               axis=1) * (1 - dones)
        target = self.network_to_train.predict_on_batch(obs)
        idx = np.arange(agent_const.batch_size)
        target[[idx], [actions]] = q_val

        self.network_to_train.fit(obs, target, epochs=1, verbose=0)
       

    def save_weights(self, save_dir: str = ""):
        if save_dir != "":
            self.network_to_train.save_weights(os.path.join(save_dir, "network_to_train.weights.h5"))


    @staticmethod
    def load_ai_model_weights(load_path_dir: str):
        model =  build_model(agent_const.input_size,  agent_const.actions_num , tf.keras.losses.MeanSquaredError(), tf.keras.optimizers.Adam(lr= 0.001))
        model.load_weights(os.path.join(load_path_dir, "network_to_train.weights.h5"))
        return model
        