""" master.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from rl.utils import make_dirs


class Master:
    def __init__(self,
                 agent,
                 env,
                 episodes,
                 env_name,
                 model_name,
                 snapshot,
                 graph_dir='./output/graph', 
                 model_dir='./output/trained_model'):

        self.agent = agent
        self.env = env
        self.episodes = episodes
        self.env_name = env_name
        self.model_name = model_name
        self.snapshot = snapshot
        self.graph_dir = os.path.join(graph_dir, env_name, model_name)
        self.model_dir = os.path.join(model_dir, env_name, model_name)

        make_dirs(self.graph_dir)
        make_dirs(self.model_dir)

    def play(self, render=False):
        scores, episodes = [], []
        for episode in range(self.episodes):
            continue_game = True
            score = 0
            state = self.env.reset()
            state = self.make_state(state)

            while continue_game:
                if render: self.env.render()

                action = self.agent.get_action(state)
                next_state, reward, done, info = self.env.step(action)
                next_state = self.make_state(next_state)

                self.agent.train_model(state, action, reward, next_state, done)
                score += reward
                state = next_state
                continue_game = not done

                if done:
                    self.agent.sync_target_model()

                    scores.append(score)
                    episodes.append(episode)
                    plt.plot(episodes, scores, 'b')
                    plt.savefig(self.graph_dir + '/graph.png')

                    message = "[{}/{}] ".format(self.env_name, self.model_name) + \
                              "episode: {} ".format(episode) + \
                              "score: {} ".format(score) + \
                              "mean score: {:.3f} ".format(np.mean(scores[-min(10, len(scores)):])) + \
                              "memory length: {} ".format(len(self.agent.memory)) + \
                              "epsilon: {:.3f}".format(self.agent.policy.epsilon)
                    print(message)

            if not episode % self.snapshot:
                self.agent.model.save_weights(self.model_dir + '/model-{}.h5'.format(episode))

    def make_state(self, state):
        if len(state.shape) == 1:
            return np.reshape(state, [1, -1])
        elif len(state.shape) == 3:
            return state[np.newaxis, :, :, :]
