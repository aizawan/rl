""" memory.py
"""

import random
import numpy as np
from collections import deque


class ReplayMemory:
    def __init__(self, capacity=2000):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def append(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, n):
        assert len(self.memory) >= n

        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        minibatch = random.sample(self.memory, n)

        for d in minibatch:
            state_batch.append(np.squeeze(d[0]))
            action_batch.append(d[1])
            reward_batch.append(d[2])
            next_state_batch.append(np.squeeze(d[3]))
            done_batch.append(d[4])

        return (np.array(state_batch),
                action_batch,
                reward_batch,
                np.array(next_state_batch),
                done_batch)

    def __len__(self):
        return len(self.memory)
