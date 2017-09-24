""" dqn.py
"""

import sys
import gym
import random
import numpy as np

from rl.agent import Agent


class DQNAgent(Agent):
    def __init__(self, state_size, action_size, memory, policy, model, 
                 target_model, discount_factor=0.99, batch_size=64, 
                 train_start=1000, load_model=None):

        self.state_size = state_size
        self.action_size = action_size

        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.train_start = train_start

        self.memory = memory
        self.policy = policy

        self.model = model
        self.target_model = target_model

        self.sync_target_model()

        if load_model:
            self.model.load_weights(load_model)

    def sync_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        q_value = self.model.predict(state)
        return self.policy.select_action(
                action=np.argmax(q_value[0]),
                random_action=random.randrange(self.action_size))

    def train_model(self, state, action, reward, next_state, done):
        reward = self.clip_reward(reward)
        self.memory.append(state, action, reward, next_state, done)

        if len(self.memory) < self.train_start:
            return

        minibatch = self.memory.sample(self.batch_size)

        target = self.model.predict(minibatch[0])
        target_val = self.target_model.predict(minibatch[3])

        for i in range(self.batch_size):
            if minibatch[4][i]:
                target[i][minibatch[1][i]] = minibatch[2][i]
            else:
                target[i][minibatch[1][i]] = minibatch[2][i] + self.discount_factor * (
                    np.amax(target_val[i]))

        self.model.fit(minibatch[0], target, batch_size=self.batch_size,
                       epochs=1, verbose=0)

    def clip_reward(self, reward, reward_min=-1., reward_max=1.):
        return np.clip(reward, reward_min, reward_max)