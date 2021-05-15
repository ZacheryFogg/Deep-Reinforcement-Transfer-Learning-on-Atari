Files in repository:

DQN.ipynb - My initial attempt at a dqn from scratch, uses environment functionality from StableBaselines3: https://github.com/DLR-RM/stable-baselines3
After failure with this attempt, I used a model from keras-rl: https://github.com/keras-rl/keras-rl    to verfiy that my model was not faulty
This code currently runs the DQN for a few 1000 timesteps to prove the functionality
   
   
keras-rl/dqn.py - Same architecuture as my own dqn to make sure that my model was not faulty and the lack luster training was a results of how the dqn was not suited for the environment; as opposed to not succeeding due to faulty code on my part 


keras-rl/DoubleDuelingDQN.py - an attempt to see if using double q-learning and dueling architectures could help performance before I took the time to implement it on my own
      
This was an implementation of PPO on a simple game so that I could learn how it works, but for the overall project, the Stable Baselines version was much faster
toy_ppo_implementation.ipynb  - https://colab.research.google.com/drive/1IAe3Xqa2uhYp3ENXLbg6mLawZEdFrcMA#scrollTo=dBxj7oNm2LqD

Most of the project was spent in this file: 
actual_PPO.ipnb: https://colab.research.google.com/drive/12IE1zIhwgbuj2Zl2YgMCgcxKZ37wq7uH?pli=1#scrollTo=ZbCORSgSvXXg
