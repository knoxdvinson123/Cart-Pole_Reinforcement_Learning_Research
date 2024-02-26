import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


environment_name = 'CartPole-v1'
env = gym.make(environment_name, render_mode = "human")

PPO_path = os.path.join('Training', 'Saved Models', 'PPO_Model_Cartpole')

model = PPO.load(PPO_path, env=env)

log_path = os.path.join('Training', 'Logs')

#this is how to run the evaluation_policy----
print(evaluate_policy(model, env, n_eval_episodes=10, render=True))

