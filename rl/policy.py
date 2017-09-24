""" policy.py
"""

import numpy as np


class EpsilonGreedy:
    def __init__(self, epsilon, epsilon_decay, epsilon_min):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def select_action(self, action, random_action):
        if np.random.rand() < self.epsilon:
            action = random_action

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return action
