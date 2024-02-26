

net_arch = [dict(pi=[128,128,128,128], vf=[128, 128,128,128])] #this defines a new neural network archetecture.
#more specifically a new policy (more informatoin on stable_baselines website

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path, policy_kwargs={'net_arch':net_arch})



#--------------
#ALTERNATE ALGORITHM (like DQN)
from stable_baselines3 import DQN
model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=20000)