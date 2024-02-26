import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

environment_name = 'CartPole-v1'
env = gym.make(environment_name, render_mode = "human")

print(env.reset())

#Example of running random actions for 5 episodes and printing score--------------------------------------
episodes = 5
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0

    while not done:
      env.render()
      action = env.action_space.sample()
      observation, reward, done, truncated, info = env.step(action)
      score += reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()