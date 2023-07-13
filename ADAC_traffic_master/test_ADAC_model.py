import os
import sys
import gym
import numpy as np
import torch

from ADAC import discrete_BCQ, dac_mdp, utils

sys.path.append("./TrafQ/Environments/gym_gharrafa/")
import gymGharrafa

STATE_DIM = 68
NUM_ACTIONS = 11

# GYM ENVIRONMENT
# 'GUI' : True, 'Play': "action"
ENV_NAME = "gymGharrafa-v2"


# load policy and buffer for recreating model from stored Q values
print('Rebuilding discrete_BCQ and dac_policy objects.')
policy = discrete_BCQ.discrete_BCQ(
    False,
    NUM_ACTIONS,
    STATE_DIM,
    'cpu'
)
policy.load("./models/BCQ_gharaffa-NTFT20_20")
replay_buffer = utils.ReplayBuffer(STATE_DIM, False, {}, 128, 8640, 'cpu')
replay_buffer.load(f"./buffers/stationary-NTFT20_gharaffa-NTFT20_20")

# build DAC MDP
# num_configs = 1 for ADAC
mdp = dac_mdp.dac_builder(NUM_ACTIONS, STATE_DIM, replay_buffer, policy.get_q_model(), 'cpu', 1, 'number', -1,
                          save_to_file=f'./pickled_ADAC/gharaffa_ADAC',
                          build_from_file=f'./pickled_ADAC/gharaffa_ADAC')
dac_policies = mdp.get_policies()
dac_policy = dac_policies[0]  # first generated policy


# simulate using a cyclic policy
env = gym.make(ENV_NAME)
os.system('clear')
env.reset()
obs = env.reset()
steps = 360
for s in range(steps):
    action = (s // 3) % 11  # CYCLIC: switch phases every 3 steps (configure as necessary)
    obs, reward, episode_over, additional = env.step(action)
    print((action, reward))
env.close()
print('\nDONE cyclic policy.')


# simulate using ADAC policy
input('\n[ENTER] to continue to ADAC policy. ')
env = gym.make(ENV_NAME)
os.system('clear')
env.reset()
obs = env.reset()
steps = 360
for s in range(steps):
    action = dac_policy.select_action(obs)
    obs, reward, episode_over, additional = env.step(action)
    print((action, reward))
env.close()
print('\nDONE ADAC policy.')
