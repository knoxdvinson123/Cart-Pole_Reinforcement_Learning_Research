# Cart-Pole_Reinforcement_Learning_Research
Reinforcement Learning Introduction Project - Cart-Pole with PPO

Welcome to the Reinforcement Learning Introduction Project! This repository serves as a comprehensive guide and implementation of the fundamental machine learning technique - Reinforcement Learning (RL). Specifically, it focuses on training an agent, represented by a cart, to balance a pole through the use of the Cart-Pole environment hosted by OpenAI Gym (now called gymnasium).

![image](https://github.com/knoxdvinson123/Cart-Pole_Reinforcement_Learning_Research/assets/154300416/fb942a17-4142-4767-8ed8-f35ec2da6022)
About Reinforcement Learning (RL):
- Reinforcement Learning enables an "agent" to learn and optimize its behavior by interacting with an environment. In this scenario, our agent is the cart, and its objective is to balance a pole. The learning process involves numerous iterations with a positive and negative reward system, allowing the agent to refine its understanding and develop a neural network-based optimal policy to maximize long-term rewards.

Key Features:
   
- Implementation using OpenAI Gym and the stable_baselines_3 library.
- Integration of OpenAI API for enhanced functionality.
- Utilization of DummyVecEnv for efficient vectorization of the RL environment.
   
Algorithm:
- For this project, we have chosen the Proximal Policy Optimization (PPO) algorithm. PPO is a robust and widely used RL algorithm, well-suited for training agents in a variety of environments.

Environment:
- The project focuses on the "Cart-Pole-v1" environment, a classic problem where the agent's goal is to prevent the pole from falling over by skillfully moving the black cart left and right.
