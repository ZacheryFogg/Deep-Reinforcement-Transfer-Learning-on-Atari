Files in repository:

dqn.py - My initial attempt at a dqn from scratch, uses environment functionality from StableBaselines3: https://github.com/DLR-RM/stable-baselines3
After failure with this attempt, I used a model from keras-rl: https://github.com/keras-rl/keras-rl    to verfiy that my model was not faulty
This code currently runs the DQN for a few 1000 timesteps to prove the functionality
   
   
keras-rl/dqn.py - Same architecuture as my own dqn to make sure that my model was not faulty and the lack luster training was a results of how the dqn was not suited for the environment; as opposed to not succeeding due to faulty code on my part 


keras-rl/DoubleDuelingDQN.py - an attempt to see if using double q-learning and dueling architectures could help performance before I took the time to implement it on my own
      
      
ppo.py  - My implementation of PPO. This file went through many iterations as I changed environments. The code in this file will currently just run PPO on the Alien environment for a few 1000 timesteps to prove the functionality
