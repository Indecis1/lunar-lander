import numpy as np
import os
import random
import tensorflow as tf

from collections import namedtuple
from replay_memory import ReplayMemory
from utils_model import create_model

AgentCont = namedtuple("AgentConst", ["buffer_size", "batch_size", "update_every", "gamma", "tau", "input_size", "actions_num"])
agent_const = AgentCont(
    2000,
    64,
    5,
    1,
    1e-3,
    (8, ),
    4
)


class Agent:
    def __init__(self, seed):
        random.seed(seed)
        self.input_size = agent_const.input_size
        self.action_num = agent_const.actions_num
        self.seed = seed

        self.network_to_train = create_model(self.input_size, self.action_num)
        self.target_network = create_model(self.input_size, self.action_num)
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_fn = tf.keras.losses.MeanSquaredError()

        self.memory = ReplayMemory(self.action_num, agent_const.buffer_size, agent_const.batch_size, seed)

        self.timestep = 0

    def step(self, obs, action, reward, next_obs, done):
        self.memory.add(obs, action, reward, next_obs, done)
        self.timestep += 1
        if self.timestep % agent_const.update_every == 0:
            if len(self.memory) > agent_const.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)

    def play_action(self, obs, epsilon):
        action_logit = self.network_to_train(obs.reshape(1, -1))

        choice = self.greedy_choice(epsilon)
        if choice == "random":
            return random.choice(np.arange(self.action_num))
        else:
            return np.argmax(action_logit)

    def greedy_choice(self, epsilon):
        return np.random.choice(["random", "model"], p=[epsilon, 1 - epsilon])

    def learn(self, experiences):
        obs, actions, rewards, next_obs, dones = experiences
        alpha = self.optimizer.get_config()["learning_rate"]
        with tf.GradientTape() as tape:
            Q_pred_next = tf.math.reduce_max(self.network_to_train(next_obs, training=True), axis=1)
            Q_pred = tf.math.reduce_max(self.network_to_train(obs, training=True), axis=1)
            updated_Q_pred = Q_pred + alpha * (rewards + agent_const.gamma * Q_pred_next - Q_pred)

            Q_expected = tf.math.reduce_max(self.target_network(obs, training=True), axis=1)
            loss_value = self.loss_fn(Q_expected, updated_Q_pred)
        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, self.network_to_train.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        self.optimizer.apply_gradients(zip(grads, self.network_to_train.trainable_weights))

        self.soft_update(agent_const.tau)

    def soft_update(self, tau: float):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        :param local_model: tf model where weights will be copied from
        :param target_model: tf model where weights will be copied to
        :param tau: interpolation parameter
        """
        local_layer_weights = self.network_to_train.get_weights()
        target_layer_weights = self.target_network.get_weights()
        for i in range(len(target_layer_weights)):
            target_layer_weights[i] = tau * local_layer_weights[i] + (1.0-tau) * target_layer_weights[i]
        self.target_network.set_weights(target_layer_weights)

    def save_weights(self, save_dir: str = ""):
        if save_dir != "":
            self.network_to_train.save_weights(os.path.join(save_dir, "network_to_train.weights.h5"))
            self.target_network.save_weights(os.path.join(save_dir, "target_network.weights.h5"))

    @staticmethod
    def load_ai_model_weights(load_path_dir: str):
        model = create_model(agent_const.input_size, agent_const.actions_num)
        model.load_weights(os.path.join(load_path_dir, "network_to_train.weights.h5"))
        return model
        