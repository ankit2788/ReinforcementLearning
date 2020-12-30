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




os.environ["RL_PATH"] = "/Users/ankitgupta/Documents/git/anks/Books/ReinforcementLearning/DeepQLearning"
pref = os.environ["RL_PATH"]

if f'{pref}/RLLibrary' not in sys.path:
    sys.path.append(f'{pref}/RLLibrary')

# importing personal library
from ActionSelection import ActionExploration
from ConfigReader import Config
from NetworkModels import TabularModel, BaselineModel, DeepNeuralModel
from RLAgents import QLearningAgent, FittedQAgent, DQN





env = gym.make("FrozenLake-v0")
configFile = "/Users/ankitgupta/Documents/git/anks/Books/ReinforcementLearning/DeepQLearning/Configs.ini"

savePath = os.path.join(os.environ["RL_PATH"], "models" )
_time = datetime.now().strftime("%Y%m%d%H%M")


Agent = QLearningAgent(env, configFile=configFile)

Agent.Qmodel.load(modelPath = os.path.join(savePath, "model_QLEARNING_202009090954.csv"))

# play the game with trained model

loss = []
for _thisepisode in tqdm(range(Agent.NbEpisodesTest)):

    # reset the environment
    _currentState = env.reset()
    

    _episodicReward = 0
    _dead = False
    _thisstepsTaken = 0

    while _thisstepsTaken <= Agent.MaxSteps:

        # get the action from agent
        # take the fully greedy action
        action = Agent.getAction(_currentState, _thisepisode, mode = "TEST")

        # perform the action
        _nextState, _reward, _dead, _info = env.step(action)
        _episodicReward += _reward
        _thisstepsTaken += 1


        # if game over, then exit the loop
        if _dead == True:
            break   


    # update Episodic reward & steps taken
    Agent.updateEpisodicInfo(_episodicReward, _thisstepsTaken, mode = "TEST")

Agent.showPerformance(mode = "TEST")



