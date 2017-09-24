""" cartpole.py
"""

import gym
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.master import Master
from rl.memory import ReplayMemory
from rl.policy import EpsilonGreedy


def mlp(input_dim, output_dim, loss, optimizer):
    model = Sequential()
    model.add(Dense(24, input_dim=input_dim, activation='relu',
                    kernel_initializer='he_uniform'))
    model.add(Dense(24, activation='relu',
                    kernel_initializer='he_uniform'))
    model.add(Dense(output_dim, activation='linear',
                    kernel_initializer='he_uniform'))
    model.compile(loss=loss, optimizer=optimizer)
    return model


env = gym.make('LunarLander-v2')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

model = mlp(
    input_dim=state_size,
    output_dim=action_size,
    loss='mse',
    optimizer=Adam(lr=0.001))

target_model = mlp(
    input_dim=state_size,
    output_dim=action_size,
    loss='mse',
    optimizer=Adam(lr=0.001))

memory = ReplayMemory(capacity=2000)
policy=EpsilonGreedy(epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01)
agent = DQNAgent(state_size, action_size, memory, policy, model, target_model)

master = Master(
    agent=agent,
    env=env,
    episodes=300,
    env_name='LunarLander-v2',
    model_name='dqn_mlp',
    snapshot=50)

master.play(render=False)
