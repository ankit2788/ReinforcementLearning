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
import RLAgents as agents
import utils as RLUtils

reload(agents)



# Run the model
env         = gym.make("FrozenLake-v0")
configFile  = os.path.join(pref, "Configs.ini" )
savePath    = os.path.join(os.environ["RL_PATH"], "models" )
_time       = datetime.now().strftime("%Y%m%d%H%M")


#FQAgent         = agents.FittedQAgent(env, configFile)
#QLAgent         = agents.QLearningAgent(env, configFile)
#DQNAgent        = agents.DQN(env = env, configFile = configFile, NetworkShape = [32,32,16])
#DQNAgent        = agents.DQN(env = env, configFile = configFile, NetworkShape = [16])
#DoubleDQNAgent  = agents.DoubleDQN(env = env, configFile = configFile, NetworkShape = [16])
DoubleDQNAgent  = agents.DoubleDQN(env = env, configFile = configFile,  NetworkShape = [16])

#Agent = FQAgent
#Agent = QLAgent
#Agent = DQNAgent
Agent = DoubleDQNAgent

Qvalues = {}

loss = []
mode = "TRAIN"
for _thisepisode in tqdm(range(Agent.NbEpisodesTrain)):

    # reset the environment
    _currentState = env.reset()
    

    _episodicReward = 0
    _dead = False
    _thisstepsTaken = 0

    if Agent.Name.upper() not in ["QLEARNING"]:
        Agent.Tensorboard.step = _thisepisode
        

    while _thisstepsTaken <= Agent.MaxSteps:

        # count the number of times this state explored (Just the state)
        if Agent.Name.upper() == "QLEARNING":
            Agent.countStatesExplored[_currentState] += 1

        # get the action from agent
        action = Agent.getAction(_currentState, mode = mode)

        # perform the action
        _nextState, _reward, _dead, _info = env.step(action)

        # record into memory
        Agent.updateMemory(_currentState, action, _reward, _nextState, _dead)

        # train the agent and update the state
        """
        history = Agent.trainAgent(batch_size = Agent.Qmodel.batchSize)
        if history  is not None:
            loss.append(history.history["loss"])
        """
        Agent.trainAgent(_dead, batch_size = Agent.Qmodel.batchSize)
        
        # update States
        _currentState = _nextState
        _thisstepsTaken += 1

        _episodicReward += _reward

        # if game over, then exit the loop
        if _dead == True:
            break   


    # ---- For logging ------
    # In case of Neural networks, create tensorboard flow
    Agent.updateLoggerInfo(episodeCount = _thisepisode, episodicReward = _episodicReward, \
                            episodicStepsTaken = _thisstepsTaken, mode = "TRAIN")



# save model & save config
print(f"Model & config save path: {Agent.path}") 
Agent.Qmodel.save(name = f"model_{Agent.Name}_{_time}.h5", path = savePath)
Agent.saveConfig(filename = f"config_{Agent.Name}_{_time}.json", savePath = savePath)




# ---------- Policy Gradient -----------

reload(agents)

# Run the model
env         = gym.make("CartPole-v1")
env         = gym.make("FrozenLake-v0")

configFile  = os.path.join(pref, "Configs.ini" )
savePath    = os.path.join(os.environ["RL_PATH"], "models" )
_time       = datetime.now().strftime("%Y%m%d%H%M")

reinforceAgent  = agents.Reinforce(env = env, configFile = configFile,  model = None, NetworkShape = [24,12])

Agent = reinforceAgent




loss = []
mode = "TRAIN"
for _thisepisode in tqdm(range(Agent.NbEpisodesTrain)):

    # reset the environment
    _currentState = env.reset()

    _episodicReward = 0
    _dead = False
    _thisstepsTaken = 0
    Agent.Tensorboard.step = _thisepisode


    while not _dead:

        #print(_thisstepsTaken)
        # get the action from agent
        action, actionProb = Agent.getAction(_currentState, mode = mode)

        # perform the action
        _nextState, _reward, _dead, _info = env.step(action)

        # record into memory
        Agent.updateMemory(_currentState, action, _reward, _nextState, _dead, actionProb)
        _thisstepsTaken += 1
        _episodicReward += _reward
        # update States
        _currentState = _nextState

        # train the agent and update the state
        if _dead:
            Agent.updatePolicy()
        
            # ---- For logging ------
            # In case of Neural networks, create tensorboard flow
            Agent.updateLoggerInfo(episodeCount = _thisepisode, episodicReward = _episodicReward, \
                                    episodicStepsTaken = _thisstepsTaken, mode = "TRAIN")




# save model & save config
print(f"Model & config save path: {Agent.path}") 
Agent.PolicyModel.save(name = f"model_{Agent.Name}_{_time}.h5", path = savePath)
Agent.saveConfig(filename = f"config_{Agent.Name}_{_time}.json", savePath = savePath)





plt.plot(Agent.EpisodicRewards["TRAIN"])
plt.plot(Agent.EpisodicSteps["TRAIN"])

