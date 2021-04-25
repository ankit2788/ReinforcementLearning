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


# importing custom libraries
from ActionSelection import ActionExploration
from ConfigReader import Config
#from RLAgents import QLearningAgent, FittedQAgent, DQN, DoubleDQN
import Agents.PolicyGradient.Reinforce as Reinforce
import Agents.PolicyGradient.ActorCritic as ActorCritic

import utils as RLUtils


import loggingConfig as loggingConfig


logger = loggingConfig.logging
logger.getLogger("PG_Train")



# Run the model
#env         = gym.make("CartPole-v1")
configFile  = os.path.join(pref, "Configs.ini" )
savePath    = os.path.join(os.environ["RL_PATH"], "models" )
_time       = datetime.now().strftime("%Y%m%d%H%M")

env = gym.make("CartPole-v1")
state = env.reset()


# ----- Reinforce
modelParams = {"Name": "Policy", "NetworkShape": [24, 12], "learning_rate": 0.001, \
                "optimizer": "ADAM", "loss": "categorical_crossentropy", }

valueParams = {"Name_value": "Value", "NetworkShape_value": [24, 12], "learning_rate_value": 0.01, \
                "optimizer_value": "ADAM", "loss_value": "mse", }

#ReinforceAgent   = Reinforce.REINFORCE_EAGER(env = env, configFile = configFile, **modelParams)
#ReinforceAgent   = Reinforce.REINFORCE_BASELINE(env = env, configFile = configFile, **modelParams, **valueParams)
"""
Agent = ReinforceAgent
mode = "TRAIN"
for _thisepisode in tqdm(range(Agent.NbEpisodesTrain)):

    # reset the environment
    _currentState = env.reset()
    

    _episodicReward = 0
    _dead = False
    _thisstepsTaken = 0


    while _thisstepsTaken <= Agent.MaxSteps:

        _starttime = time.perf_counter()

        # get the action from agent
        action, actionProb = Agent.getAction(_currentState, mode = mode)

        # perform the action
        _nextState, _reward, _dead, _info = env.step(action)

        # record into memory
        Agent.updateMemory(_currentState, action, _reward, _nextState, _dead, actionProb, value)

        # update States
        _currentState = _nextState
        _thisstepsTaken += 1

        _episodicReward += _reward

        # if game over, then exit the loop
        if _dead == True:

            Agent.train()

            # ---- For logging ------
            # In case of Neural networks, create tensorboard flow
            Agent.updateLoggerInfo(episodeCount = _thisepisode, episodicReward = _episodicReward, \
                                    episodicStepsTaken = _thisstepsTaken, mode = "TRAIN")

            _endtime = time.perf_counter()

            logger.info(f'Episode: {_thisepisode+1} Steps: {_thisstepsTaken} Reward:{_episodicReward} Time taken: {round(_endtime - _starttime,2)} secs ')

            break


"""
# ----------------

# ------Actor Critic
modelParams = {"Name": "ActorCritic", "NetworkShape": [1024, 512], "learning_rate": 0.003, \
                "optimizer": "ADAM", }

ReinforceAgent   = ActorCritic.ActorCritic(env = env, configFile = configFile, **modelParams)
Agent = ReinforceAgent
mode = "TRAIN"
for _thisepisode in tqdm(range(Agent.NbEpisodesTrain)):

    # reset the environment
    _currentState = env.reset()
    

    _episodicReward = 0
    _dead = False
    _thisstepsTaken = 0


    while _thisstepsTaken <= Agent.MaxSteps:

        _starttime = time.perf_counter()

        # get the action from agent
        action, actionProb, value = Agent.getAction(_currentState, mode = mode)            

        # perform the action
        _nextState, _reward, _dead, _info = env.step(action)

        # record into memory
        Agent.updateMemory(_currentState, action, _reward, _nextState, _dead, actionProb, value)

        # learn at every step   
        Agent.train()     


        # update States
        _currentState = _nextState
        _thisstepsTaken += 1

        _episodicReward += _reward

        # if game over, then exit the loop
        if _dead == True:

            # ---- For logging ------
            # In case of Neural networks, create tensorboard flow
            Agent.updateLoggerInfo(episodeCount = _thisepisode, episodicReward = _episodicReward, \
                                    episodicStepsTaken = _thisstepsTaken, mode = "TRAIN")

            _endtime = time.perf_counter()

            logger.info(f'Episode: {_thisepisode+1} Steps: {_thisstepsTaken} Reward:{_episodicReward} Time taken: {round(_endtime - _starttime,2)} secs ')

            break




# save model & save config
#print(f"Model & config save path: {Agent.path}") 
#Agent.PolicyModel.save(name = f"model_{Agent.Name}_{_time}.h5", path = savePath)
#Agent.saveConfig(filename = f"config_{Agent.Name}_{_time}.json", savePath = savePath)




#"""
