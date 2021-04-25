import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys
import gym
from importlib import reload
from configparser import ConfigParser
from datetime import datetime
import tensorflow as tf
import time


# get the relative path
fullpath                = os.path.realpath(__file__)
pref                    = os.path.split(fullpath)[0]

os.environ["RL_PATH"]   = pref

#os.environ["RL_PATH"] = "/Users/ankitgupta/Documents/git/anks/Books/ReinforcementLearning/DeepQLearning"
#pref = os.environ["RL_PATH"]

if f'{pref}/RLLibrary' not in sys.path:
    sys.path.append(f'{pref}/RLLibrary')


import loggingConfig as loggingConfig


logger = loggingConfig.logging
logger.getLogger("PG_Train")


from RLLibrary.Agents.PolicyGradient.A3C.Agents import MasterAgent




if __name__ == "__main__":
    
    save_dir         = f"{pref}/models/Policy/A3C"
    optimizer_args = {"learning_rate": 1e-4}
    model = MasterAgent(game = "CartPole-v0", save_dir=save_dir, **optimizer_args)
    model.train()