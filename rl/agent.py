""" agent.py
"""

class Agent:
    def __init__(self, state_size, action_size, **kwargs):
        self.state_size = state_size
        self.action_size = action_size

    def get_action(self, state, **kwargs):
        raise NotImplementedError

    def train_model(self, state, action, reward, next_state, done, **kwargs):
        raise NotImplementedError