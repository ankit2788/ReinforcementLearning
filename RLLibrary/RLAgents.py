import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys
from abc import ABC, abstractclassmethod
import gym
from importlib import reload
from collections import deque
from configparser import ConfigParser
from datetime import datetime
import itertools

from tensorflow.keras.callbacks import History




# Create classes for Different Value based models
# 1. Q Learning - Tabular (very limited)
# 2. Fitted Q learning - Simple neural based
# 3. Deep Q Network
# 4. Double DQN
# 5. Dueling DQN -  TODO


try: 
    pref = os.environ["RL_PATH"]
except KeyError:

    # get the relative path
    fullpath = os.path.realpath(__file__)
    pref     = os.path.split(fullpath)[0]


#os.environ["RL_PATH"] = "/Users/ankitgupta/Documents/git/anks/Books/ReinforcementLearning/DeepQLearning"
#pref = os.environ["RL_PATH"]
"""
if f'{pref}/RLLibrary' not in sys.path:
    sys.path.append(f'{pref}/RLLibrary')
"""


# importing custom libraries

import NetworkModels as networks
from ActionSelection import ActionExploration
from ConfigReader import Config

from Visualizations import QLPerformance, NFQPerformance
from utils import get_val, convertStringtoBoolean, getOneHotrepresentation
from Callbacks import ModifiedTensorBoardCallback, GradCallBack

reload(networks)



# All classes are based on OpenAI environments
class QAgentBase(ABC):

    @abstractclassmethod
    def __init__(self):
        pass


    @abstractclassmethod
    def getAction(self, state):
        pass

    
    @abstractclassmethod
    def updateMemory(self, currentState, currentAction, reward, nextState, dead):
        pass

    @abstractclassmethod
    def trainAgent(self):
        pass

    @abstractclassmethod
    def updateEpisodicInfo(self):
        pass


    @abstractclassmethod
    def saveConfig(self):
        pass



class QLearningAgent(QAgentBase):
    # This is a tabular method
    # Limited state action space

    def __init__(self, env, configFile,  **kwargs):

        self.Name   = "QLEARNING"
        self.env    = env

        # get the config & hyper parameters info
        self.config     = Config(configFile, AgentName=self.Name)
        self.__readConfig()

        # episodic Info
        self.EpisodicRewards    = {"TRAIN": [], "TEST": []}
        self.EpisodicSteps      = {"TRAIN": [], "TEST": []}


        # Action selection method & model used
        self.explore            = ActionExploration(self.config, self.methodExploration)
        
        # initializing the model (In this case: Q values)
        self.Qmodel             = networks.TabularModel(self.config, self.env.observation_space.n, self.env.action_space.n)
        self.Qmodel.init()
        
        # Visualize Peformance
        self.plots              =  QLPerformance(self)

        # initialing exploration count to 0 for all states
        self.countStatesExplored = dict.fromkeys(range(self.env.observation_space.n), 0)




    def __readConfig(self):

        # episodic Info
        self.NbEpisodesTrain        = int(get_val(self.config, tag = "NB_EPISODES_TRAIN", default_value= 2000))
        self.NbEpisodesTest         = int(get_val(self.config, tag = "NB_EPISODES_TEST", default_value= 1000))
        self.MaxSteps               = int(get_val(self.config, tag = "MAX_STEPS_EPISODE", default_value= 100))


        # RL params
        self.discountfactor         = float(get_val(self.config, tag = "DISCOUNT_FACTOR", default_value= 0.9)) 
        self.learning_rate          = float(get_val(self.config, tag = "RL_LEARNING_RATE", default_value= 0.85)) 

        # exploration parameters
        self.methodExploration      = get_val(self.config, tag = "ACTION_SELECTION", default_value= "EPSILON_GREEDY")

        # experience Info
        self.memorySize             = int(get_val(self.config, tag = "MEMORY_SIZE", default_value= 1))
        self.memory                 = deque(maxlen = self.memorySize)

        # save formats
        self.configsaveFormat       = get_val(self.config, tag = "CONFIG_FORMAT", default_value= "json")

        self.path                   = get_val(self.config, tag = "PATH", default_value="models")

        # Need to append the path to relative path
        self.path                   = os.path.join(pref,self.path)



    def getAction(self, state, currentEpisodeNb, mode = "TRAIN"):
        # choose action in given state, and perform this action to obtain the reward

        optimalActionIndex = self._obtainOptimalPolicy(state)

        if mode.upper() == "TRAIN":
            action = self.explore.chooseAction(currentEpisodeNb, range(self.env.action_space.n), \
                                            optimalActionIndex=optimalActionIndex)

        elif mode.upper() == "TEST":
            action = optimalActionIndex
        return action



    def updateMemory(self, currentState, currentAction, reward, nextState, dead):

        if len(self.memory) < self.memorySize:
            self.memory.append([currentState, currentAction, reward, nextState, dead])

        else:
            # delete the last element in the memory list
            del self.memory[-1]
            self.memory.append([currentState, currentAction, reward, nextState, dead])


    def trainAgent(self,batch_size=1) :
        # batch_size isnt used here 

        # if no memory, then nothing to train on
        if len(self.memory) > 0:

            # get the last element in memory
            _curState, _action, _reward, _nextState = self.memory[-1][0], self.memory[-1][1], self.memory[-1][2], self.memory[-1][3]
        
            # update Qvalues (use Bellman equation)
            _targetValue = _reward + self.discountfactor * np.max(self.Qmodel.model[_nextState, :])

            self.Qmodel.model[_curState, _action] = self.Qmodel.model[_curState, _action] + self.learning_rate * (_targetValue - self.Qmodel.model[_curState, _action])

        return None


    def _obtainOptimalPolicy(self, state):
        # identify the optimal Qvalue for exploitation
        _values = self.Qmodel.model[state]
        return np.argmax(_values)


    def updateEpisodicInfo(self, episodeReward, episodeSteps, mode = "TRAIN"):
        self.EpisodicRewards[mode].append(episodeReward)
        self.EpisodicSteps[mode].append(episodeSteps)

         
    
    def saveConfig(self, filename, savePath = ""):

        # save the config in a pickle
        savePath = self.path if savePath else savePath
        self.config.save(filename, savePath, format = self.configsaveFormat.upper())



class FittedQAgent(QAgentBase):

    def __init__(self, env, configFile,  model = None,  **kwargs):

        self.Name   = "NFQ"
        self.env    = env

        # get the config & hyper parameters info
        self.config     = Config(configFile, AgentName=self.Name)
        self.__readConfig()

        # episodic Info
        self.EpisodicRewards    = {"TRAIN": [], "TEST": []}
        self.EpisodicSteps      = {"TRAIN": [], "TEST": []}
        self.trainingiterationCount     = 0 


        # Action selection method & model used
        self.explore            = ActionExploration(self.config, self.methodExploration)

        self.__time             = datetime.now().strftime("%Y%m%d%H%M")

        # tensorboard is used for model performance visualization 
        self.Qmodel     = model
        if model is None:
            self.Qmodel = networks.BaselineModel(self.config, dim_input = self.env.observation_space.n, \
                                dim_output= self.env.action_space.n, time = self.__time, Name = self.Name, **kwargs)

        # initialize the model
        self.Qmodel.init()

        # set the callbacks
        self.setCallbacks()



    def setCallbacks(self):

        self.callbacks      = []
        loggingPath         = f"{pref}/logs/{self.Name}_{self.__time}.log"
        #checkpointPath      = f"{pref}/ModelCheckpoint/{self.Name}_{self.__time}.ckpt"
        
        self.ModelHistory   = History()     # to store learning history
        self.Tensorboard    = ModifiedTensorBoardCallback(model = self.Qmodel.model, log_dir = loggingPath)        

        self.callbacks.append(self.Tensorboard) 
        self.callbacks.append(self.ModelHistory) 

        if  self.log_write_grads:
            gradloggingPath      = f"{pref}/logs/grad_{self.Name}_{self.__time}.log"
            self.gradTensorBoard = GradCallBack(model = self.Qmodel, log_dir = gradloggingPath)

            self.callbacks.append(self.gradTensorBoard)
        


    def __readConfig(self):


        # episodic Info
        self.NbEpisodesTrain        = int(get_val(self.config, tag = "NB_EPISODES_TRAIN", default_value= 2000))
        self.NbEpisodesTest         = int(get_val(self.config, tag = "NB_EPISODES_TEST", default_value= 1000))
        self.MaxSteps               = int(get_val(self.config, tag = "MAX_STEPS_EPISODE", default_value= 100))


        # RL params
        self.discountfactor         = float(get_val(self.config, tag = "DISCOUNT_FACTOR", default_value= 0.9)) 

        # exploration parameters
        self.methodExploration      = get_val(self.config, tag = "ACTION_SELECTION", default_value= "EPSILON_GREEDY")

        # experience Info
        self.memorySize             = int(get_val(self.config, tag = "MEMORY_SIZE", default_value= 1000))
        self.memory                 = deque(maxlen = self.memorySize)

        # TrainingInfo
        self.blnbatchMode           = convertStringtoBoolean(get_val(self.config, tag = "BATCH_MODE", default_value= "TRUE"))

        # Logging Info
        self.AggregateStatsEvery    = int(get_val(self.config, tag = "AGGREGATE_STATS_EVERY", default_value= 100))
        self.log_write_grads        = convertStringtoBoolean(get_val(self.config, tag = "LOG_WRITE_GRADS", default_value="FALSE"))

        # save formats
        self.configsaveFormat       = get_val(self.config, tag = "CONFIG_FORMAT", default_value= "json")

        self.path                   = get_val(self.config, tag = "PATH", default_value="models")

        # Need to append the path to relative path
        self.path                   = os.path.join(pref,self.path)



        
    def getAction(self, state, mode = "TRAIN"):

        # get one hot representation of state
        _state = getOneHotrepresentation(state, num_classes=self.env.observation_space.n)

        _actionsValues = self.Qmodel.predict(_state)[0] 
        _greedyActionIndex = np.argmax(_actionsValues)

        # epsilonGreedyaction        
        if mode.upper() == "TRAIN":
            action = self.explore.chooseAction(self.trainingiterationCount, range(self.env.action_space.n), \
                                            optimalActionIndex=_greedyActionIndex)

        elif mode.upper() == "TEST":
            # select the greedy action
            action = _greedyActionIndex 


        return action


    def trainAgent(self, terminal_state, batch_size, epochs=1, verbose = 0):

        self.trainingiterationCount += 1

        # if memory size is less than batch size, then nothing to train on

        if len(self.memory) < batch_size:
            return None


        # pick out samples from memory based on batch size
        samples = random.sample(self.memory, batch_size)


        # for all experience in batchsize
        curStates       = list(list(zip(*samples)))[0]
        actions         = list(list(zip(*samples)))[1]
        nextStates      = list(list(zip(*samples)))[3]
        rewards         = list(list(zip(*samples)))[2]
        done            = list(list(zip(*samples)))[4]

        inputStates     = getOneHotrepresentation(curStates, num_classes=self.env.observation_space.n)
        target          = self.Qmodel.predict(inputStates)

        # using bellman equation 
        nextQvalues     = self.Qmodel.predict(getOneHotrepresentation(nextStates, num_classes=self.env.observation_space.n)) 
        max_nextQvalues = list(map(max, nextQvalues))

        # if done (dead), target is just the terminal reward 

        for index, action in enumerate(actions):
            if done[index]:
                target[index][int(action)] = rewards[index] 
            else:
                target[index][int(action)] = rewards[index] + self.discountfactor * max_nextQvalues[index]

            if self.blnbatchMode is False:

                self.Qmodel.fit(X_train=inputStates, y_train = np.array(target), \
                            batch_size = batch_size, epochs = epochs, verbose = verbose, \
                            callbacks= self.callbacks if terminal_state else None)


        # mini batch mode learning
        if self.blnbatchMode:
            self.Qmodel.fit(X_train=inputStates, y_train = np.array(target), \
                            batch_size = batch_size, epochs = epochs, verbose = verbose, \
                            callbacks= self.callbacks if terminal_state else None)



        # --- get the gradients if Gradient logging is turned on
        if self.log_write_grads:
            grads = self.Qmodel.get_gradients(X_train, y_train)

            self.gradTensorBoard.step += 1
            self.gradTensorBoard.update_grads(gradStats = grads)


    def updateMemory(self, currentState, currentAction, reward, nextState, dead):
        self.memory.append((currentState, currentAction, reward, nextState, dead))


    def updateEpisodicInfo(self, episodeReward, episodeSteps, mode = "TRAIN"):

        self.EpisodicRewards[mode].append(episodeReward)
        self.EpisodicSteps[mode].append(episodeSteps)

    def saveConfig(self, filename, savePath = ""):

        savePath = self.path if savePath else savePath        
        self.config.save(filename, savePath, format = self.configsaveFormat)



    def updateLoggerInfo(self, episodeCount, episodicReward, episodicStepsTaken, mode = "TRAIN"):
        # This is used to update all the logging related information
        # how the training evolves with episode

        # 1. After every episode, update the episodic reward & steps taken
        self.updateEpisodicInfo(episodicReward, episodicStepsTaken, mode=mode)

        # 2. log model weights & bias Info after every episode 
        self.Tensorboard.update_stats_histogram()

        # 3. Create other logging stats
        if not episodeCount % self.AggregateStatsEvery or episodeCount == 1:

            average_reward  = sum(self.EpisodicRewards[mode][-self.AggregateStatsEvery:])/len(self.EpisodicRewards[mode][-self.AggregateStatsEvery:])
            #min_reward      = min(self.EpisodicRewards[mode][-self.AggregateStatsEvery:])
            max_reward      = max(self.EpisodicRewards[mode][-self.AggregateStatsEvery:])

            average_steps   = sum(self.EpisodicSteps[mode][-self.AggregateStatsEvery:])/len(self.EpisodicSteps[mode][-self.AggregateStatsEvery:])
            max_steps       = max(self.EpisodicSteps[mode][-self.AggregateStatsEvery:])
            #min_steps       = min(self.EpisodicSteps[mode][-self.AggregateStatsEvery:])


            epsilon = self.explore.currentExploration
            self.Tensorboard.update_stats(reward_avg=average_reward, steps_avg=average_steps, \
                                            reward_max=max_reward, steps_max = max_steps, \
                                            epsilon=epsilon)



class DQN(QAgentBase):

    def __init__(self, env, configFile,  model = None, **kwargs):
        
        self.Name   = "DQN"
        self.env    = env

        # get the config & hyper parameters info
        self.config     = Config(configFile, AgentName=self.Name)
        self.__readConfig()

        # episodic Info
        self.EpisodicRewards    = {"TRAIN": [], "TEST": []}
        self.EpisodicSteps      = {"TRAIN": [], "TEST": []}
        self.trainingiterationCount     = 0 

        # Action selection method & model used
        self.explore            = ActionExploration(self.config, self.methodExploration)
        self.__time             = datetime.now().strftime("%Y%m%d%H%M")


        # ---------------- Model setup
        # DQN agent has 2 network
        # 1. Target network --> target is kept fixed for certain iterations after which it is updated 
        # 2. Main network --> regularly trained and updated 


        # tensorboard is used for model performance visualization

        self.Qmodel      = networks.DeepNeuralModel(self.config, dim_input=self.env.observation_space.n, \
                                dim_output = self.env.action_space.n, \
                                time = self.__time, Name = "QModel", **kwargs)
            

        self.Targetmodel = networks.DeepNeuralModel(self.config, dim_input=self.env.observation_space.n, \
                                dim_output = self.env.action_space.n, \
                                time = self.__time, Name = "TargetModel", **kwargs)


        # initialize the model
        self.Qmodel.init()
        self.Targetmodel.init()

        # create Qmodel Callbacks 
        self.setCallbacks()

        # update the target model weights with main network weights
        self.Targetmodel.model.set_weights(self.Qmodel.model.get_weights())
        

    def setCallbacks(self):
        self.callbacks      = []
        loggingPath         = f"{pref}/logs/{self.Name}_{self.__time}.log"
        #checkpointPath      = f"{pref}/ModelCheckpoint/{self.Name}_{self.__time}.ckpt"
        
        self.ModelHistory   = History()     # to store learning history
        self.Tensorboard    = ModifiedTensorBoardCallback(model = self.Qmodel.model, log_dir = loggingPath)        

        self.callbacks.append(self.Tensorboard) 
        self.callbacks.append(self.ModelHistory) 

        if  self.log_write_grads:
            gradloggingPath      = f"{pref}/logs/grad_{self.Name}_{self.__time}.log"
            self.gradTensorBoard = GradCallBack(model = self.Qmodel, log_dir = gradloggingPath)

            self.callbacks.append(self.gradTensorBoard)


    def __readConfig(self):



        # episodic Info
        self.NbEpisodesTrain        = int(get_val(self.config, tag = "NB_EPISODES_TRAIN", default_value= 2000))
        self.NbEpisodesTest         = int(get_val(self.config, tag = "NB_EPISODES_TEST", default_value= 1000))
        self.MaxSteps               = int(get_val(self.config, tag = "MAX_STEPS_EPISODE", default_value= 100))


        # RL params
        self.discountfactor         = float(get_val(self.config, tag = "DISCOUNT_FACTOR", default_value= 0.9)) 
        self.rewardClipping         = convertStringtoBoolean(get_val(self.config, tag = "REWARD_CLIP", default_value= "TRUE"))

        # exploration parameters
        self.methodExploration      = get_val(self.config, tag = "ACTION_SELECTION", default_value= "EPSILON_GREEDY")

        # experience Info
        self.memorySize             = int(get_val(self.config, tag = "MEMORY_SIZE", default_value= 1000))
        self.memory                 = deque(maxlen = self.memorySize)
        self.replayStart            = int(get_val(self.config, tag = "REPLAY_START", default_value= 500))

        # TrainingInfo
        self.blnbatchMode           = convertStringtoBoolean(get_val(self.config, tag = "BATCH_MODE", default_value= "TRUE"))
        self.updateFrequency        = int(get_val(self.config, tag = "TARGET_UPDATE_FREQUENCY", default_value= 1000))

        # Logging Info
        self.AggregateStatsEvery    = int(get_val(self.config, tag = "AGGREGATE_STATS_EVERY", default_value= 100))
        self.log_write_grads        = convertStringtoBoolean(get_val(self.config, tag = "LOG_WRITE_GRADS", default_value="FALSE"))

        # save formats
        self.configsaveFormat       = get_val(self.config, tag = "CONFIG_FORMAT", default_value= "json")
        self.path                   = get_val(self.config, tag = "PATH", default_value="models")

        # Need to append the path to relative path
        self.path                   = os.path.join(pref,self.path)


    def saveConfig(self, filename, savePath = ""):

        savePath = self.path if savePath else savePath        
        self.config.save(filename, savePath, format = self.configsaveFormat)




    def getAction(self, state,  mode = "TRAIN"):

        # get one hot representation of state
        _state = getOneHotrepresentation(state, num_classes=self.env.observation_space.n)

        _actionsValues      = self.Qmodel.predict(_state)[0] 
        _greedyActionIndex  = np.argmax(_actionsValues)

        # epsilonGreedyaction        
        if mode.upper() == "TRAIN":
            action          = self.explore.chooseAction(self.trainingiterationCount, range(self.env.action_space.n), \
                                            optimalActionIndex=_greedyActionIndex)

        elif mode.upper() == "TEST":
            # select the greedy action
            action          = _greedyActionIndex 


        return action

        


    def updateMemory(self, currentState, currentAction, reward, nextState, dead):
        self.memory.append((currentState, currentAction, reward, nextState, dead))


    def trainAgent(self, terminal_state, batch_size, epochs=1, verbose = 0):

        self.trainingiterationCount += 1

        # ------- Replaying the memory (Experience replay) -------

        # dont train until a certain number of iterations (replay start)
        if (len(self.memory) < self.replayStart) or (len(self.memory) < batch_size):
            return None


        # pick out samples from memory based on batch size
        samples = random.sample(self.memory, batch_size)


        # for all experience in batchsize
        curStates       = list(list(zip(*samples)))[0]
        actions         = list(list(zip(*samples)))[1]
        nextStates      = list(list(zip(*samples)))[3]
        rewards         = list(list(zip(*samples)))[2]
        done            = list(list(zip(*samples)))[4]

        # Add pre-processing step if needed

        inputStates     = getOneHotrepresentation(curStates, num_classes=self.env.observation_space.n)
        #predict         = self.Qmodel.predict(inputStates)

        # ------ For DQN, target comes from the target model
        # using bellman equation 

        # Steps:
        # 1. use target model to set the target. 
        # 2. This target needs to be updated based on the bellman equation
        # 2.1. Bellman Equation 1: get the max Q values for next state in the bellman equation

        #target          = np.zeros(shape = predict.shape)
        target          = self.Targetmodel.predict(inputStates)

        nextQvalues     = self.Targetmodel.predict(getOneHotrepresentation(nextStates, num_classes=self.env.observation_space.n)) 
        max_nextQvalues = list(map(max, nextQvalues))

        # if done (dead), target is just the terminal reward 
        for index, action in enumerate(actions):
            if done[index]:
                target[index][int(action)] = rewards[index] 
            else:
                target[index][int(action)] = rewards[index] + self.discountfactor * max_nextQvalues[index]

        # mini batch mode learning
        self.Qmodel.fit(X_train=inputStates, y_train = np.array(target), \
                        batch_size = batch_size, epochs = epochs, verbose = verbose, \
                        callbacks= self.callbacks if terminal_state else None)


        # --- get the gradients if Gradient logging is turned on
        if self.log_write_grads:
            grads = self.Qmodel.get_gradients(X_train, y_train)

            self.gradTensorBoard.step += 1
            self.gradTensorBoard.update_grads(gradStats = grads)


        # ------- Updating the Target network -------

        # update the target network if needed
        if self.trainingiterationCount % self.updateFrequency == 0:
            self.updateTargetNetwork()


    def updateTargetNetwork(self):
        self.Targetmodel.model.set_weights(self.Qmodel.model.get_weights())


    def updateEpisodicInfo(self, episodeReward, episodeSteps, mode = "TRAIN"):

        self.EpisodicRewards[mode].append(episodeReward)
        self.EpisodicSteps[mode].append(episodeSteps)


    def updateLoggerInfo(self, episodeCount, episodicReward, episodicStepsTaken, mode = "TRAIN"):
        # This is used to update all the logging related information
        # how the training evolves with episode

        # 1. After every episode, update the episodic reward & steps taken
        self.updateEpisodicInfo(episodicReward, episodicStepsTaken, mode=mode)

        # 2. log model weights & bias Info after every episode 
        self.Tensorboard.update_stats_histogram()
        #self.Qmodel.tensorboard.update_stats_histogram()
        #self.Targetmodel.tensorboard.update_stats_histogram()

        # 3. Create other logging stats
        if not episodeCount % self.AggregateStatsEvery or episodeCount == 1:

            average_reward  = sum(self.EpisodicRewards[mode][-self.AggregateStatsEvery:])/len(self.EpisodicRewards[mode][-self.AggregateStatsEvery:])
            #min_reward      = min(self.EpisodicRewards[mode][-self.AggregateStatsEvery:])
            max_reward      = max(self.EpisodicRewards[mode][-self.AggregateStatsEvery:])

            average_steps   = sum(self.EpisodicSteps[mode][-self.AggregateStatsEvery:])/len(self.EpisodicSteps[mode][-self.AggregateStatsEvery:])
            max_steps       = max(self.EpisodicSteps[mode][-self.AggregateStatsEvery:])
            #min_steps       = min(self.EpisodicSteps[mode][-self.AggregateStatsEvery:])


            epsilon = self.explore.currentExploration
            self.Tensorboard.update_stats(reward_avg=average_reward, steps_avg=average_steps, \
                                            reward_max=max_reward, steps_max = max_steps, \
                                            epsilon=epsilon)


class DoubleDQN(QAgentBase):

    def __init__(self, env, configFile,  model = None, **kwargs):
        
        self.Name   = "DOUBLEDQN"

        # get the config & hyper parameters info
        self.config     = Config(configFile, AgentName=self.Name)
        self.env    = env

        self.__readConfig()

        # episodic Info
        self.EpisodicRewards    = {"TRAIN": [], "TEST": []}
        self.EpisodicSteps      = {"TRAIN": [], "TEST": []}
        self.trainingiterationCount     = 0 

        # Action selection method & model used
        self.explore            = ActionExploration(self.config, self.methodExploration)
        self.__time             = datetime.now().strftime("%Y%m%d%H%M")


        # ---------------- Model setup
        # DQN agent has 2 network
        # 1. Target network --> target is kept fixed for certain iterations after which it is updated 
        # 2. Main network --> regularly trained and updated 


        # tensorboard is used for model performance visualization

        self.Qmodel      = networks.DeepNeuralModel(self.config, dim_input=self.env.observation_space.n, \
                                dim_output = self.env.action_space.n, \
                                time = self.__time, Name = "QModel",  **kwargs)
            

        self.Targetmodel = networks.DeepNeuralModel(self.config, dim_input=self.env.observation_space.n, \
                                dim_output = self.env.action_space.n, \
                                time = self.__time, Name = "TargetModel",  **kwargs)


        # initialize the model
        self.Qmodel.init()
        self.Targetmodel.init()

        # create Qmodel Callbacks 
        self.setCallbacks()

        # update the target model weights with main network weights
        self.Targetmodel.model.set_weights(self.Qmodel.model.get_weights())
        

    def setCallbacks(self):
        self.callbacks      = []
        loggingPath         = f"{pref}/logs/{self.Name}_{self.__time}.log"
        #checkpointPath      = f"{pref}/ModelCheckpoint/{self.Name}_{self.__time}.ckpt"
        
        self.ModelHistory   = History()     # to store learning history
        self.Tensorboard    = ModifiedTensorBoardCallback(model = self.Qmodel.model, log_dir = loggingPath)        

        self.callbacks.append(self.Tensorboard) 
        self.callbacks.append(self.ModelHistory) 

        if  self.log_write_grads:
            gradloggingPath      = f"{pref}/logs/grad_{self.Name}_{self.__time}.log"
            self.gradTensorBoard = GradCallBack(model = self.Qmodel, log_dir = gradloggingPath)

            self.callbacks.append(self.gradTensorBoard)


    def __readConfig(self):



        # episodic Info
        self.NbEpisodesTrain        = int(get_val(self.config, tag = "NB_EPISODES_TRAIN", default_value= 2000))
        self.NbEpisodesTest         = int(get_val(self.config, tag = "NB_EPISODES_TEST", default_value= 1000))
        self.MaxSteps               = int(get_val(self.config, tag = "MAX_STEPS_EPISODE", default_value= 100))


        # RL params
        self.discountfactor         = float(get_val(self.config, tag = "DISCOUNT_FACTOR", default_value= 0.9)) 
        self.rewardClipping         = convertStringtoBoolean(get_val(self.config, tag = "REWARD_CLIP", default_value= "TRUE"))

        # exploration parameters
        self.methodExploration      = get_val(self.config, tag = "ACTION_SELECTION", default_value= "EPSILON_GREEDY")

        # experience Info
        self.memorySize             = int(get_val(self.config, tag = "MEMORY_SIZE", default_value= 1000))
        self.memory                 = deque(maxlen = self.memorySize)
        self.replayStart            = int(get_val(self.config, tag = "REPLAY_START", default_value= 500))

        # TrainingInfo
        self.blnbatchMode           = convertStringtoBoolean(get_val(self.config, tag = "BATCH_MODE", default_value= "TRUE"))
        self.updateFrequency        = int(get_val(self.config, tag = "TARGET_UPDATE_FREQUENCY", default_value= 1000))

        # Logging Info
        self.AggregateStatsEvery    = int(get_val(self.config, tag = "AGGREGATE_STATS_EVERY", default_value= 100))
        self.log_write_grads        = convertStringtoBoolean(get_val(self.config, tag = "LOG_WRITE_GRADS", default_value="FALSE"))

        # save formats
        self.configsaveFormat       = get_val(self.config, tag = "CONFIG_FORMAT", default_value= "json")

        self.path                   = get_val(self.config, tag = "PATH", default_value="models")

        # Need to append the path to relative path
        self.path                   = os.path.join(pref,self.path)


    def saveConfig(self, filename, savePath = ""):

        savePath = self.path if savePath else savePath        
        self.config.save(filename, savePath, format = self.configsaveFormat)




    def getAction(self, state,  mode = "TRAIN"):

        # get one hot representation of state
        _state = getOneHotrepresentation(state, num_classes=self.env.observation_space.n)

        _actionsValues      = self.Qmodel.predict(_state)[0] 
        _greedyActionIndex  = np.argmax(_actionsValues)

        # epsilonGreedyaction        
        if mode.upper() == "TRAIN":
            action          = self.explore.chooseAction(self.trainingiterationCount, range(self.env.action_space.n), \
                                            optimalActionIndex=_greedyActionIndex)

        elif mode.upper() == "TEST":
            # select the greedy action
            action          = _greedyActionIndex 


        return action



    def updateMemory(self, currentState, currentAction, reward, nextState, dead):
        self.memory.append((currentState, currentAction, reward, nextState, dead))


    def trainAgent(self, terminal_state, batch_size, epochs=1, verbose = 0):

        self.trainingiterationCount += 1

        # ------- Replaying the memory (Experience replay) -------

        # dont train until a certain number of iterations (replay start)
        if (len(self.memory) < self.replayStart) or (len(self.memory) < batch_size):
            return None


        # pick out samples from memory based on batch size
        samples = random.sample(self.memory, batch_size)


        # for all experience in batchsize
        curStates       = list(list(zip(*samples)))[0]
        actions         = list(list(zip(*samples)))[1]
        nextStates      = list(list(zip(*samples)))[3]
        rewards         = list(list(zip(*samples)))[2]
        done            = list(list(zip(*samples)))[4]

        # Add pre-processing step if needed

        inputStates     = getOneHotrepresentation(curStates, num_classes=self.env.observation_space.n)
        #predict         = self.Qmodel.predict(inputStates)

        # --------- CHANGE HERE FOR DDQN ---------
        # using bellman equation 

        # Steps:
        # 1. use target model to set the target. 
        # 2. This target needs to be updated based on the bellman equation
        # 2.1. Bellman Equation 1: get the action that maximizes the Q value for next state using main model
        # 2.2. Bellman Equation 2: get the Qvalue for above action from TARGET model


        target                  = self.Targetmodel.predict(inputStates)
        #target          = np.zeros(shape = predict.shape)


        nextQvalues_main        = self.Qmodel.predict(getOneHotrepresentation(nextStates, num_classes=self.env.observation_space.n)) 
        nextQvalues_target      = self.Targetmodel.predict(getOneHotrepresentation(nextStates, num_classes=self.env.observation_space.n)) 

        argmax_action           = list(map(np.argmax, nextQvalues_main))

        __indexes               = (np.arange(len(nextQvalues_target)), np.array(argmax_action))
        selectedQvalues         = nextQvalues_target[__indexes]

        #max_nextQvalues = list(map(max, nextQvalues))

        # if done (dead), target is just the terminal reward 
        for index, action in enumerate(actions):
            target[index][int(action)] = rewards[index] + (1 - done[index]) * self.discountfactor * selectedQvalues[index]

        # --------- --------- --------- --------- ---------


        # mini batch mode learning
        self.Qmodel.fit(X_train=inputStates, y_train = np.array(target), \
                        batch_size = batch_size, epochs = epochs, verbose = verbose, \
                        callbacks= self.callbacks if terminal_state else None)


        # --- get the gradients if Gradient logging is turned on
        if self.log_write_grads:
            grads = self.Qmodel.get_gradients(X_train, y_train)

            self.gradTensorBoard.step += 1
            self.gradTensorBoard.update_grads(gradStats = grads)


        # ------- Updating the Target network -------

        # update the target network if needed
        if self.trainingiterationCount % self.updateFrequency == 0:
            self.updateTargetNetwork()


    def updateTargetNetwork(self):
        self.Targetmodel.model.set_weights(self.Qmodel.model.get_weights())


    def updateEpisodicInfo(self, episodeReward, episodeSteps, mode = "TRAIN"):

        self.EpisodicRewards[mode].append(episodeReward)
        self.EpisodicSteps[mode].append(episodeSteps)


    def updateLoggerInfo(self, episodeCount, episodicReward, episodicStepsTaken, mode = "TRAIN"):
        # This is used to update all the logging related information
        # how the training evolves with episode

        # 1. After every episode, update the episodic reward & steps taken
        self.updateEpisodicInfo(episodicReward, episodicStepsTaken, mode=mode)

        # 2. log model weights & bias Info after every episode 
        self.Tensorboard.update_stats_histogram()
        #self.Qmodel.tensorboard.update_stats_histogram()
        #self.Targetmodel.tensorboard.update_stats_histogram()

        # 3. Create other logging stats
        if not episodeCount % self.AggregateStatsEvery or episodeCount == 1:

            average_reward  = sum(self.EpisodicRewards[mode][-self.AggregateStatsEvery:])/len(self.EpisodicRewards[mode][-self.AggregateStatsEvery:])
            #min_reward      = min(self.EpisodicRewards[mode][-self.AggregateStatsEvery:])
            max_reward      = max(self.EpisodicRewards[mode][-self.AggregateStatsEvery:])

            average_steps   = sum(self.EpisodicSteps[mode][-self.AggregateStatsEvery:])/len(self.EpisodicSteps[mode][-self.AggregateStatsEvery:])
            max_steps       = max(self.EpisodicSteps[mode][-self.AggregateStatsEvery:])
            #min_steps       = min(self.EpisodicSteps[mode][-self.AggregateStatsEvery:])


            epsilon = self.explore.currentExploration
            self.Tensorboard.update_stats(reward_avg=average_reward, steps_avg=average_steps, \
                                            reward_max=max_reward, steps_max = max_steps, \
                                            epsilon=epsilon)




# ------------------------------------------------------------------------
# ---- Policy Gradient Algos

class Reinforce():
    def __init__(self, env, configFile,  model = None, **kwargs):

        self.Name   = "REINFORCE"
        self.env    = env

        # get the config & hyper parameters info
        self.config     = Config(configFile, AgentName=self.Name)
        self.__readConfig()

        # episodic Info
        self.EpisodicRewards    = {"TRAIN": [], "TEST": []}
        self.EpisodicSteps      = {"TRAIN": [], "TEST": []}
        self.trainingiterationCount     = 0 

        self.__time             = datetime.now().strftime("%Y%m%d%H%M")


        # Create Base Model for Policy (given the state and parameters, what action need to be taken)
        # tensorboard is used for model performance visualization 
        self.PolicyModel        = model
        try:
            self.inputShape = self.env.observation_space.n
        except:
            self.inputShape = self.env.observation_space.shape[0]

        if model is None:
            self.PolicyModel    = networks.DeepNeuralModelClassifier(self.config, dim_input = self.inputShape, \
                                    dim_output= self.env.action_space.n, time = self.__time, Name = self.Name, \
                                    **kwargs)

            self.PolicyModel.init()

        else:
            #self.PolicyModel = None
            setattr(self.PolicyModel, "model", model)
            #self.PolicyModel    = model


        # create Qmodel Callbacks 
        self.setCallbacks()

        

    def setCallbacks(self):
        self.callbacks      = []
        loggingPath         = f"{pref}/logs/{self.Name}_{self.__time}.log"
        #checkpointPath      = f"{pref}/ModelCheckpoint/{self.Name}_{self.__time}.ckpt"
        
        self.ModelHistory   = History()     # to store learning history
        self.Tensorboard    = ModifiedTensorBoardCallback(model = self.PolicyModel.model, log_dir = loggingPath)        

        self.callbacks.append(self.Tensorboard) 
        self.callbacks.append(self.ModelHistory) 

        
        if  self.log_write_grads:
            gradloggingPath      = f"{pref}/logs/grad_{self.Name}_{self.__time}.log"
            self.gradTensorBoard = GradCallBack(model = self.Qmodel, log_dir = gradloggingPath)

            self.callbacks.append(self.gradTensorBoard)




    def __readConfig(self):

        # episodic Info
        self.NbEpisodesTrain        = int(get_val(self.config, tag = "NB_EPISODES_TRAIN", default_value= 2000))
        self.NbEpisodesTest         = int(get_val(self.config, tag = "NB_EPISODES_TEST", default_value= 1000))
        self.MaxSteps               = int(get_val(self.config, tag = "MAX_STEPS_EPISODE", default_value= 100))


        # RL params
        self.discountfactor         = float(get_val(self.config, tag = "DISCOUNT_FACTOR", default_value= 0.9)) 
        self.policy_learning         = float(get_val(self.config, tag = "POLICY_LEARNING", default_value= 0.9)) 
        self.normalizeRewards       = convertStringtoBoolean(get_val(self.config, tag = "NORMALIZE_REWARDS", default_value="TRUE"))

        # memory
        self.memory                 = []

        # TrainingInfo
        self.batchSize              = int(get_val(self.config, tag = "TRAIN_BATCH_SIZE", default_value= 32))
        self.epochs                 = int(get_val(self.config, tag = "EPOCHS", default_value= 32))
        self.blnbatchMode           = convertStringtoBoolean(get_val(self.config, tag = "BATCH_MODE", default_value= "TRUE"))

        # Logging Info
        self.AggregateStatsEvery    = int(get_val(self.config, tag = "AGGREGATE_STATS_EVERY", default_value= 100))
        self.log_write_grads        = convertStringtoBoolean(get_val(self.config, tag = "LOG_WRITE_GRADS", default_value="FALSE"))

        # save formats
        self.configsaveFormat       = get_val(self.config, tag = "CONFIG_FORMAT", default_value= "json")
        self.path                   = get_val(self.config, tag = "PATH", default_value="models")

        # Need to append the path to relative path
        self.path                   = os.path.join(pref,self.path)



    def saveConfig(self, filename, savePath = ""):

        savePath = self.path if savePath else savePath        
        self.config.save(filename, savePath, format = self.configsaveFormat)



    def getAction(self, state, mode = "TRAIN"):
        # based on the state, predict the action to be taken using the network
        
        try:
            self.env.observation_space.n
            _state = getOneHotrepresentation(state, num_classes=self.inputShape)
        except:

            _state = state.reshape([1, state.shape[0]])


        # the model prediction predicts the prob space for all actions
        actionProb         = self.PolicyModel.predict(_state)
        actionProb         = actionProb.flatten()

        # sample the action based on the probability
        action             = np.random.choice(self.env.action_space.n, p = actionProb)


        return action, actionProb


    def discountRewards(self, rewards):

        discounted_rewards=[]
        cumulative_total_return=0
        # iterate the rewards backwards and and calc the total return 
        for reward in rewards[::-1]:      
            cumulative_total_return=(cumulative_total_return* self.discountfactor)+reward
            discounted_rewards.insert(0, cumulative_total_return)


        if self.normalizeRewards:
            discounted_rewards = (discounted_rewards - np.mean(discounted_rewards))/ (np.std(discounted_rewards) + 1e-7)            # to avoid division by 0

        return discounted_rewards

    
    def updateMemory(self, currentState, currentAction, reward, nextState, dead, actionProb):
        self.memory.append((currentState, currentAction, reward, nextState, dead, actionProb))

    def updatePolicy(self):

        

        # for all experience in batchsize
        curStates       = np.vstack(list(list(zip(*self.memory)))[0])
        actions         = np.vstack(list(list(zip(*self.memory)))[1])
        nextStates      = np.vstack(list(list(zip(*self.memory)))[3])
        rewards         = np.vstack(list(list(zip(*self.memory)))[2])
        done            = np.vstack(list(list(zip(*self.memory)))[4])

        actionProb      = np.vstack(list(list(zip(*self.memory)))[5])

        
        # compute the discounted rewards for the entire episode and normalize it
        discountedRewards = self.discountRewards(rewards)


        # ---- Compute the Policy gradient
        gradient = np.subtract(getOneHotrepresentation(actions,self.env.action_space.n ), actionProb)
        gradient *= discountedRewards * self.policy_learning


        try:
            self.env.observation_space.n
            X_train = getOneHotrepresentation(curStates, num_classes=self.inputShape)
        except:
            X_train = curStates    


        y_train = actionProb + np.vstack(gradient)

        # mini batch mode learning
        #self.PolicyModel.fit(X_train=X_train, y_train = y_train, \
        #                batch_size = self.batchSize, epochs = self.epochs, verbose = 0, \
        #                callbacks= self.callbacks)

        history = self.PolicyModel.model.train_on_batch(X_train, y_train)
        

        # --- get the gradients if Gradient logging is turned on
        if self.log_write_grads:
            grads = self.PolicyModel.get_gradients(X_train, y_train)

            self.gradTensorBoard.step += 1
            self.gradTensorBoard.update_grads(gradStats = grads)

        # reset memory
        self.memory = []
        

    def updateEpisodicInfo(self, episodeReward, episodeSteps, mode = "TRAIN"):

        self.EpisodicRewards[mode].append(episodeReward)
        self.EpisodicSteps[mode].append(episodeSteps)


    def updateLoggerInfo(self, episodeCount, episodicReward, episodicStepsTaken, mode = "TRAIN"):
        # This is used to update all the logging related information
        # how the training evolves with episode

        # 1. After every episode, update the episodic reward & steps taken
        self.updateEpisodicInfo(episodicReward, episodicStepsTaken, mode=mode)

        # 2. log model weights & bias Info after every episode 
        self.Tensorboard.update_stats_histogram()
        #self.Qmodel.tensorboard.update_stats_histogram()
        #self.Targetmodel.tensorboard.update_stats_histogram()

        # 3. Create other logging stats
        self.Tensorboard.update_stats(rewards = self.EpisodicRewards[mode][-1])
        if not episodeCount % self.AggregateStatsEvery or episodeCount == 1:

            
            average_reward  = sum(self.EpisodicRewards[mode][-self.AggregateStatsEvery:])/len(self.EpisodicRewards[mode][-self.AggregateStatsEvery:])
            #min_reward      = min(self.EpisodicRewards[mode][-self.AggregateStatsEvery:])
            max_reward      = max(self.EpisodicRewards[mode][-self.AggregateStatsEvery:])

            average_steps   = sum(self.EpisodicSteps[mode][-self.AggregateStatsEvery:])/len(self.EpisodicSteps[mode][-self.AggregateStatsEvery:])
            max_steps       = max(self.EpisodicSteps[mode][-self.AggregateStatsEvery:])
            #min_steps       = min(self.EpisodicSteps[mode][-self.AggregateStatsEvery:])

            self.Tensorboard.update_stats(reward_avg=average_reward, steps_avg=average_steps, \
                                            reward_max=max_reward, steps_max = max_steps)






class ReinforceBaseline():
    def __init__(self, env, configFile,  model = None, **kwargs):

        self.Name   = "REINFORCE_BASELINE"
        self.env    = env

        # get the config & hyper parameters info
        self.config     = Config(configFile, AgentName=self.Name)
        self.__readConfig()

        # episodic Info
        self.EpisodicRewards    = {"TRAIN": [], "TEST": []}
        self.EpisodicSteps      = {"TRAIN": [], "TEST": []}
        self.trainingiterationCount     = 0 

        self.__time             = datetime.now().strftime("%Y%m%d%H%M")


        # Create Base Model for Policy (given the state and parameters, what action need to be taken)
        # tensorboard is used for model performance visualization 
        self.PolicyModel        = model
        try:
            self.inputShape = self.env.observation_space.n
        except:
            self.inputShape = self.env.observation_space.shape[0]

        if model is None:
            self.PolicyModel    = networks.DeepNeuralModelClassifier(self.config, dim_input = self.inputShape, \
                                    dim_output= self.env.action_space.n, time = self.__time, Name = self.Name, \
                                    **kwargs)

            self.PolicyModel.init()

        else:
            #self.PolicyModel = None
            setattr(self.PolicyModel, "model", model)
            #self.PolicyModel    = model


        # Create the baseline Value function model as well
        self.BaselineModel = networks.BaselineModel(self.config, dim_input= self.inputShape, \
                                            dim_output=1, NetworkShape = [], time = self.__time, Name = self.Name)

        self.BaselineModel.init()

        
        # create model Callbacks 
        self.setCallbacks()

        

    def setCallbacks(self):
        self.callbacks      = []
        loggingPath         = f"{pref}/logs/{self.Name}_{self.__time}.log"
        #checkpointPath      = f"{pref}/ModelCheckpoint/{self.Name}_{self.__time}.ckpt"
        
        self.ModelHistory   = History()     # to store learning history
        self.Tensorboard_policy     = ModifiedTensorBoardCallback(model = self.PolicyModel.model, log_dir = loggingPath)        
        self.Tensorboard_value      = ModifiedTensorBoardCallback(model = self.BaselineModel.model, log_dir = loggingPath)        

        self.callbacks.append(self.Tensorboard_policy) 
        self.callbacks.append(self.Tensorboard_value) 
        self.callbacks.append(self.ModelHistory) 

        
        if  self.log_write_grads:
            gradloggingPath      = f"{pref}/logs/grad_{self.Name}_{self.__time}.log"
            self.gradTensorBoard = GradCallBack(model = self.Qmodel, log_dir = gradloggingPath)

            self.callbacks.append(self.gradTensorBoard)




    def __readConfig(self):

        # episodic Info
        self.NbEpisodesTrain        = int(get_val(self.config, tag = "NB_EPISODES_TRAIN", default_value= 2000))
        self.NbEpisodesTest         = int(get_val(self.config, tag = "NB_EPISODES_TEST", default_value= 1000))
        self.MaxSteps               = int(get_val(self.config, tag = "MAX_STEPS_EPISODE", default_value= 100))


        # RL params
        self.discountfactor         = float(get_val(self.config, tag = "DISCOUNT_FACTOR", default_value= 0.9)) 
        self.policy_learning         = float(get_val(self.config, tag = "POLICY_LEARNING", default_value= 0.9)) 
        self.normalizeRewards       = convertStringtoBoolean(get_val(self.config, tag = "NORMALIZE_REWARDS", default_value="TRUE"))

        # memory
        self.memory                 = []

        # TrainingInfo
        self.batchSize              = int(get_val(self.config, tag = "TRAIN_BATCH_SIZE", default_value= 32))
        self.epochs                 = int(get_val(self.config, tag = "EPOCHS", default_value= 32))
        self.blnbatchMode           = convertStringtoBoolean(get_val(self.config, tag = "BATCH_MODE", default_value= "TRUE"))

        # Logging Info
        self.AggregateStatsEvery    = int(get_val(self.config, tag = "AGGREGATE_STATS_EVERY", default_value= 100))
        self.log_write_grads        = convertStringtoBoolean(get_val(self.config, tag = "LOG_WRITE_GRADS", default_value="FALSE"))

        # save formats
        self.configsaveFormat       = get_val(self.config, tag = "CONFIG_FORMAT", default_value= "json")
        self.path                   = get_val(self.config, tag = "PATH", default_value="models")

        # Need to append the path to relative path
        self.path                   = os.path.join(pref,self.path)



    def saveConfig(self, filename, savePath = ""):

        savePath = self.path if savePath else savePath        
        self.config.save(filename, savePath, format = self.configsaveFormat)



    def getAction(self, state, mode = "TRAIN"):
        # based on the state, predict the action to be taken using the network
        
        try:
            self.env.observation_space.n
            _state = getOneHotrepresentation(state, num_classes=self.inputShape)
        except:

            _state = state.reshape([1, state.shape[0]])


        # the model prediction predicts the prob space for all actions
        actionProb         = self.PolicyModel.predict(_state)
        actionProb         = actionProb.flatten()

        # sample the action based on the probability
        action             = np.random.choice(self.env.action_space.n, p = actionProb)


        return action, actionProb


    def discountRewards(self, rewards):

        discounted_rewards=[]
        cumulative_total_return=0
        # iterate the rewards backwards and and calc the total return 
        for reward in rewards[::-1]:      
            cumulative_total_return=(cumulative_total_return* self.discountfactor)+reward
            discounted_rewards.insert(0, cumulative_total_return)


        if self.normalizeRewards:
            discounted_rewards = (discounted_rewards - np.mean(discounted_rewards))/ (np.std(discounted_rewards) + 1e-7)            # to avoid division by 0

        return discounted_rewards

    
    def updateMemory(self, currentState, currentAction, reward, nextState, dead, actionProb):
        self.memory.append((currentState, currentAction, reward, nextState, dead, actionProb))

    def updatePolicy(self):

        

        # for all experience in batchsize
        curStates       = np.vstack(list(list(zip(*self.memory)))[0])
        actions         = np.vstack(list(list(zip(*self.memory)))[1])
        nextStates      = np.vstack(list(list(zip(*self.memory)))[3])
        rewards         = np.vstack(list(list(zip(*self.memory)))[2])
        done            = np.vstack(list(list(zip(*self.memory)))[4])

        actionProb      = np.vstack(list(list(zip(*self.memory)))[5])

        
        # compute the discounted rewards for the entire episode and normalize it
        discountedRewards = self.discountRewards(rewards)


        # ---- Compute the Policy gradient
        gradient = np.subtract(getOneHotrepresentation(actions,self.env.action_space.n ), actionProb)
        gradient *= discountedRewards * self.policy_learning


        try:
            self.env.observation_space.n
            X_train = getOneHotrepresentation(curStates, num_classes=self.inputShape)
        except:
            X_train = curStates    


        y_train = actionProb + np.vstack(gradient)

        # mini batch mode learning
        #self.PolicyModel.fit(X_train=X_train, y_train = y_train, \
        #                batch_size = self.batchSize, epochs = self.epochs, verbose = 0, \
        #                callbacks= self.callbacks)

        history = self.PolicyModel.model.train_on_batch(X_train, y_train)
        

        # --- get the gradients if Gradient logging is turned on
        if self.log_write_grads:
            grads = self.PolicyModel.get_gradients(X_train, y_train)

            self.gradTensorBoard.step += 1
            self.gradTensorBoard.update_grads(gradStats = grads)

        # reset memory
        self.memory = []
        

    def updateEpisodicInfo(self, episodeReward, episodeSteps, mode = "TRAIN"):

        self.EpisodicRewards[mode].append(episodeReward)
        self.EpisodicSteps[mode].append(episodeSteps)


    def updateLoggerInfo(self, episodeCount, episodicReward, episodicStepsTaken, mode = "TRAIN"):
        # This is used to update all the logging related information
        # how the training evolves with episode

        # 1. After every episode, update the episodic reward & steps taken
        self.updateEpisodicInfo(episodicReward, episodicStepsTaken, mode=mode)

        # 2. log model weights & bias Info after every episode 
        self.Tensorboard.update_stats_histogram()
        #self.Qmodel.tensorboard.update_stats_histogram()
        #self.Targetmodel.tensorboard.update_stats_histogram()

        # 3. Create other logging stats
        self.Tensorboard.update_stats(rewards = self.EpisodicRewards[mode][-1])
        if not episodeCount % self.AggregateStatsEvery or episodeCount == 1:

            
            average_reward  = sum(self.EpisodicRewards[mode][-self.AggregateStatsEvery:])/len(self.EpisodicRewards[mode][-self.AggregateStatsEvery:])
            #min_reward      = min(self.EpisodicRewards[mode][-self.AggregateStatsEvery:])
            max_reward      = max(self.EpisodicRewards[mode][-self.AggregateStatsEvery:])

            average_steps   = sum(self.EpisodicSteps[mode][-self.AggregateStatsEvery:])/len(self.EpisodicSteps[mode][-self.AggregateStatsEvery:])
            max_steps       = max(self.EpisodicSteps[mode][-self.AggregateStatsEvery:])
            #min_steps       = min(self.EpisodicSteps[mode][-self.AggregateStatsEvery:])

            self.Tensorboard.update_stats(reward_avg=average_reward, steps_avg=average_steps, \
                                            reward_max=max_reward, steps_max = max_steps)





