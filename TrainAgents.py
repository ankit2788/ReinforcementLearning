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
from RLAgents import QLearningAgent, FittedQAgent, DQN, DoubleDQN
import utils as RLUtils


# Run the model
env         = gym.make("FrozenLake-v0")
configFile  = os.path.join(pref, "Configs.ini" )
savePath    = os.path.join(os.environ["RL_PATH"], "models" )
_time       = datetime.now().strftime("%Y%m%d%H%M")


#FQAgent         = FittedQAgent(env, configFile)
#QLAgent         = QLearningAgent(env, configFile)
#DQNAgent        = DQN(env = env, configFile = configFile, NetworkShape = [32,32,16])
#DQNAgent        = DQN(env = env, configFile = configFile, NetworkShape = [16])
#DoubleDQNAgent  = DoubleDQN(env = env, configFile = configFile, NetworkShape = [16])
DoubleDQNAgent  = DoubleDQN(env = env, configFile = configFile,  NetworkShape = [16])

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




    # after every episode, store the Q values for all state action combination
    if Agent.Name.upper() == "QLEARNING":

        for state in range(Agent.env.observation_space.n):
            try:
                Qvalues[state]
            except KeyError:
                Qvalues[state] = {}

            for action in range(Agent.env.action_space.n):
                try:
                    Qvalues[state][action]

                except KeyError:
                    Qvalues[state][action] = []

                Qvalues[state][action].append(Agent.Qmodel.model[state, action])
        




Agent.plots.showEpisodicReward(mode = "TRAIN")
Agent.plots.showStatesExploration()
Agent.plots.showEpisodicLearning(Qvalues, state=0)
Agent.plots.showEpisodicLearning(Qvalues, state=10)



import random
from utils import getOneHotrepresentation
batch_size = 32

# pick out samples from memory based on batch size
samples = random.sample(Agent.memory, batch_size)


# for all experience in batchsize
curStates       = list(list(zip(*samples)))[0]
actions         = list(list(zip(*samples)))[1]
nextStates      = list(list(zip(*samples)))[3]
rewards         = list(list(zip(*samples)))[2]
done            = list(list(zip(*samples)))[4]


inputStates     = getOneHotrepresentation(curStates, num_classes=Agent.env.observation_space.n)
predict         = Agent.Qmodel.predict(inputStates)

# --------- CHANGE HERE FOR DDQN ---------
# ------ For DQN, target comes from the target model
# using bellman equation 
target          = np.zeros(shape = predict.shape)

# Step 1: get the action that maximizes the Q value for next state using main model
# Step 2: get the Qvalue for above action from TARGET model

nextQvalues_main        = Agent.Qmodel.predict(getOneHotrepresentation(nextStates, num_classes=Agent.env.observation_space.n)) 
nextQvalues_target      = Agent.Targetmodel.predict(getOneHotrepresentation(nextStates, num_classes=Agent.env.observation_space.n)) 



argmax_action           = list(map(np.argmax, nextQvalues_main))

__indexes               = (np.arange(len(nextQvalues_target)), np.array(argmax_action))
selectedQvalues         = nextQvalues_target[__indexes]

#max_nextQvalues = list(map(max, nextQvalues))

# if done (dead), target is just the terminal reward 
for index, action in enumerate(actions):
    target[index][int(action)] = rewards[index] + (1 - done[index]) * Agent.discountfactor * selectedQvalues[index]

# --------- --------- --------- --------- ---------


# mini batch mode learning
self.Qmodel.fit(X_train=inputStates, y_train = np.array(target), \
                batch_size = batch_size, epochs = epochs, verbose = verbose, \
                callbacks= self.callbacks if terminal_state else None)

