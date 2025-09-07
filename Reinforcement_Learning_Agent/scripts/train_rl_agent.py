import gym
import numpy as np

# Dummy environment
class SimpleEnv(gym.Env):
    def __init__(self):
        self.state = 0
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(10)
    def step(self, action):
        reward = np.random.rand()
        self.state += 1
        done = self.state>9
        return self.state, reward, done, {}
    def reset(self):
        self.state = 0
        return self.state

env = SimpleEnv()
state = env.reset()
for _ in range(10):
    action = env.action_space.sample()
    state, reward, done, _ = env.step(action)
    if done: state = env.reset()
