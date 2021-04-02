from __future__ import division
import argparse

from PIL import Image
import numpy as np
import gym
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy, GreedyQPolicy
from rl.memory import SequentialMemory
# from rl.memory import PrioritizedMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
# from rl.layers import NoisyNetDense

INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4


# Standard Atari processing


class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert(
            'L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        # saves storage in experience memory
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)


parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='MsPacman-v0')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

# Get the environment and extract the number of actions.
env = gym.make(args.env_name)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n
print("NUMBER OF ACTIONS: " + str(nb_actions))

# Standard DQN model architecture.
input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
model = Sequential()

# (width, height, channels)
model.add(Permute((2, 3, 1), input_shape=input_shape))
model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu'))
model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu'))
model.add(Convolution2D(64, (3, 3), strides=(1, 1), activation='relu'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))  # Noisy Net additonal
model.add(Dense(nb_actions, activation='linear'))
print(model.summary())

# Priortized memory
memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)

num_simulation_steps = 3500000


processor = AtariProcessor()

# policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
#                               nb_steps=1000000)
policy = GreedyQPolicy()

# dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
#                processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
#                train_interval=4, delta_clip=1.)

# updated agent to enable double dqn for off policy learning and dueling network
dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               processor=processor, enable_double_dqn=True, enable_dueling_network=True,
               nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1.)
# target model is updaed to reflect behavior agent every 10000 steps

dqn.compile(Adam(lr=.00025), metrics=['mae'])
folder_path = './output/DoubleDuelingDQNv3/'
# folder_path = './'
prev_weights = './output/DoubleDuelingDQNv3/v2_final_weights.h5'
dqn.load_weights(prev_weights)
final_weights = './output/DoubleDuelingDQNv3/v3_final_weights_v3.1.h5'
if args.mode == 'train':
    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that now you can use the built-in tensorflow.keras callbacks!
    weights_filename = folder_path + \
        f'dd_dqn_{args.env_name}_weights.h5f'
    checkpoint_weights_filename = folder_path + \
        'dd_dqn_' + args.env_name + '_weights_{step}.h5f'
    log_filename = folder_path + \
        f'dd_dqn_{args.env_name}_log.json'
    callbacks = [ModelIntervalCheckpoint(
        checkpoint_weights_filename, interval=250000)]
    callbacks += [FileLogger(log_filename, interval=100)]

    history = dqn.fit(env, callbacks=callbacks,
                      nb_steps=num_simulation_steps, log_interval=10000, verbose=2)

    # After training is done, we save the final weights one more time.
    dqn.save_weights(final_weights, overwrite=True)
    try:
        with open('./output/DoubleDuelingDQNv3/dqn_history', 'wb') as file:
            pickle.dump(history.history, file)
    except:
        pass
        # Finally, evaluate our algorithm for 10 episodes.
    dqn.test(env, nb_episodes=10, visualize=False)
elif args.mode == 'test':
    weights_filename = final_weights
    if args.weights:
        weights_filename = args.weights
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=10, visualize=True)
