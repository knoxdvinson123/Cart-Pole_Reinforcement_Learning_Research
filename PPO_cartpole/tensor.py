import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

environment_name = 'CartPole-v1'
env = gym.make(environment_name, render_mode = "human")

PPO_path = os.path.join('Training', 'Logs', 'PPO_3')
model = PPO.load(PPO_path, env=env)

training_log_path = os.path.join(PPO_path, 'PPO_3')
#tensorboard --logdir={training_log_path) #(which is used in command prompt)