#necessary import statements // dependencies
import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

environment_name = 'CartPole-v1'
env = gym.make(environment_name, render_mode = "human")

# print(env.observation_space)

# training model----------------------------------------------------------

#this is the path to where the files are saved (must create the path/files FIRST) then define the path
log_path = os.path.join('Training', 'Logs')

env = gym.make(environment_name) #creates the environment
env = DummyVecEnv([lambda: env])

#this line defines the agent and the policy being used
model = PPO('MlpPolicy',env, verbose=1, tensorboard_log=log_path) #defines the agent

#train the model- may need more timesteps for more complicated environments Or less if simple
model.learn(total_timesteps=20000)


#how to save a model using model.save()
PPO_path = os.path.join('Training', 'Saved Models', 'PPO_Model_Cartpole')
model.save(PPO_path)

#to reload the model later...
#to delete the model first:
#------------------------------(code below)
##del model
##model = PPO.load(PPO_Path, env=env)
#-------------------------------------
#PPO_Path is the path to the saved model file

#EVALUATION---------------
evaluate_policy(model, env, n_eval_episodes=10, render=True) #render=true (determines if you want to visualize)