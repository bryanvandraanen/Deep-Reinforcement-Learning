import gym
import numpy as np
import torch.multiprocessing as mp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

EPISODES = 5000
MAX_STEPS = 501
PROFICIENCY = 10
TEST_EPISODES = 100

GAMMA = 0.99
LOG_INTERVAL = 10
PLOT_COUNT = 1

TRAIN_RENDER = False
TEST_RENDER = True

VERY_SMALL_NUMBER = 1e-9

# Neural Network Constants
HIDDEN_DIM = 128
DROPOUT = 0.6
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

env = gym.make('CartPole-v1')
env.seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n