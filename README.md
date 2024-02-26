# Cart-Pole_Reinforcement_Learning_Research
An introduction project into the machine learning technique called Reinforcement Learning. 

I am conducting undergradate research in the realm of Reinforcement Learning at Virginia Tech. 

What is Reinforcement Learning (RL)?
Reinforcement Learning allows the "agent", in this case a cart that is balancing a pole, to train over a large number of iterations with a positive and negative reward system. The agent builds on its understanding and creates a nerual network + optimal policy in order to maximize the long term reward.

As an introduction into the use of training algorithms for RL, this environment is hosted by OpenAI Gym (now called gymnasium) and I used the OpenAI API along with a library of RL policy algorithms called stable_baselines_3. As well as using DummyVecEnv to help with the vectorization of the environment.

Algorithm:
I chose to use Proximal Policy Optimization (PPO) for this task.

Environment:
"Cart-Pole-v1"

Goal:
The agent's goal is to keep the pole from falling over by moving the black cart left and right to counter balance.

