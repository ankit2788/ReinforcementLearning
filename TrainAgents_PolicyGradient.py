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
import PolicyGradientAgents
import utils as RLUtils


import loggingConfig as loggingConfig


logger = loggingConfig.logging
logger.getLogger("PG_Train")



# Run the model
#env         = gym.make("FrozenLake-v0")
configFile  = os.path.join(pref, "Configs.ini" )
savePath    = os.path.join(os.environ["RL_PATH"], "models" )
_time       = datetime.now().strftime("%Y%m%d%H%M")

env = gym.make("Pong-v0")
state = env.reset()
prev_x = None
score = 0
episode = 0

state_size = 80 * 80
action_size = env.action_space.n

#ReinforceAgent   = PolicyGradientAgents.REINFORCE(env, configFile,  setbaseline = False, NetworkShape = [16])
ReinforceAgent   = PolicyGradientAgents.PGAgent(state_size, action_size)


def preprocess(I):
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float).ravel()

Agent = ReinforceAgent
update_frequency = 10   # update policy after every 10 episodes

while True:
    env.render()

    cur_x = preprocess(state)
    x = cur_x - prev_x if prev_x is not None else np.zeros(state_size)
    prev_x = cur_x

    action, prob = Agent.act(x)
    state, reward, done, info = env.step(action)
    score += reward
    Agent.memorize(x, action, prob, reward)

    if done:
        episode += 1
        Agent.train()
        print('Episode: %d - Score: %f.' % (episode, score))

        # update tensorboard log
        Agent.updateLoggerInfo(episodeCount = episode-1, episodicReward = score)

        score = 0
        state = env.reset()
        prev_x = None

"""
loss = []
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
        Agent.updateMemory(_currentState, action, _reward, _nextState, _dead, actionProb)

        
        #if _thisstepsTaken%40 == 0:
        #    logger.info(f'Episode: {_thisepisode+1} Step: {_thisstepsTaken} ')
        
        
        # update States
        _currentState = _nextState
        _thisstepsTaken += 1

        _episodicReward += _reward

        # if game over, then exit the loop
        if _dead == True:

            #if (_thisepisode+1)%update_frequency == 0:
            Agent.updatePolicy()

            # ---- For logging ------
            # In case of Neural networks, create tensorboard flow
            Agent.updateLoggerInfo(episodeCount = _thisepisode, episodicReward = _episodicReward, \
                                    episodicStepsTaken = _thisstepsTaken, mode = "TRAIN")

            _endtime = time.perf_counter()

            logger.info(f'Episode: {_thisepisode+1} Steps: {_thisstepsTaken} Reward:{_episodicReward} Time taken: {round(_endtime - _starttime,2)} secs ')

            break




# save model & save config
print(f"Model & config save path: {Agent.path}") 
Agent.PolicyModel.save(name = f"model_{Agent.Name}_{_time}.h5", path = savePath)
Agent.saveConfig(filename = f"config_{Agent.Name}_{_time}.json", savePath = savePath)




"""
