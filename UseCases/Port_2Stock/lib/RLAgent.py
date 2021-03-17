import numpy as np
import os
import pandas as pd
from abc import ABC, abstractclassmethod
from datetime import datetime
from collections import deque
from importlib import reload
import random

import tensorflow as tf

from logging.config import dictConfig
import logging

from tensorflow.keras.callbacks import History




# custom libraries
import constants
from ConfigReader import Config
import loggingConfig
import NetworkModels
import utils
import Actions.ActionSelection as actionSelection

reload(actionSelection)
reload(NetworkModels)

from Actions.ActionSelection import ActionExploration
from Callbacks import ModifiedTensorBoardCallback


dictConfig(loggingConfig.DEFAULT_LOGGING)
Logger = logging.getLogger("Agents")




pref = constants.PARENT_PATH




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

    
    
    
class DQN(QAgentBase):

    def __init__(self, env, configFile,  model = None, **kwargs):
        
        
        self.Name   = "DQN"
        self.env    = env

        # get the config & hyper parameters info
        self.config     = Config(configFile, Name=self.Name, Type = "RLAgent")
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

        Logger.info("Setting up Main network and Target Network...")


        # tensorboard is used for model performance visualization

        self.Qmodel      = NetworkModels.DeepNeuralNet(dim_input=self.env.observation_space.n, \
                                dim_output = self.env.action_space.n, \
                                config = self.config, \
                                time = self.__time, Name = "QModel", **kwargs)
            

        self.Targetmodel = NetworkModels.DeepNeuralNet(dim_input=self.env.observation_space.n, \
                                dim_output = self.env.action_space.n, \
                                config = self.config, \
                                time = self.__time, Name = "TargetModel", **kwargs)


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
            #self.gradTensorBoard = GradCallBack(model = self.Qmodel, log_dir = gradloggingPath)

            #self.callbacks.append(self.gradTensorBoard)


    def __readConfig(self):



        # episodic Info
        self.NbEpisodesTrain        = int(utils.get_val(self.config, tag = "NB_EPISODES_TRAIN", default_value= 2000))
        self.NbEpisodesTest         = int(utils.get_val(self.config, tag = "NB_EPISODES_TEST", default_value= 1000))
        self.MaxSteps               = int(utils.get_val(self.config, tag = "MAX_STEPS_EPISODE", default_value= 100))


        # RL params
        self.discountfactor         = float(utils.get_val(self.config, tag = "DISCOUNT_FACTOR", default_value= 0.9)) 
        self.rewardClipping         = utils.convertStringtoBoolean(utils.get_val(self.config, tag = "REWARD_CLIP", default_value= "TRUE"))

        # exploration parameters
        self.methodExploration      = utils.get_val(self.config, tag = "ACTION_SELECTION", default_value= "EPSILON_GREEDY")

        # experience Info
        self.memorySize             = int(utils.get_val(self.config, tag = "MEMORY_SIZE", default_value= 1000))
        self.memory                 = deque(maxlen = self.memorySize)
        self.replayStart            = int(utils.get_val(self.config, tag = "REPLAY_START", default_value= 500))

        # TrainingInfo
        self.blnbatchMode           = utils.convertStringtoBoolean(utils.get_val(self.config, tag = "BATCH_MODE", default_value= "TRUE"))
        self.updateFrequency        = int(utils.get_val(self.config, tag = "TARGET_UPDATE_FREQUENCY", default_value= 1000))

        # Logging Info
        self.AggregateStatsEvery    = int(utils.get_val(self.config, tag = "AGGREGATE_STATS_EVERY", default_value= 100))
        self.log_write_grads        = utils.convertStringtoBoolean(utils.get_val(self.config, tag = "LOG_WRITE_GRADS", default_value="FALSE"))

        # save formats
        self.configsaveFormat       = utils.get_val(self.config, tag = "CONFIG_FORMAT", default_value= "json")
        self.path                   = utils.get_val(self.config, tag = "PATH", default_value="models")

        # Need to append the path to relative path
        self.path                   = os.path.join(pref,self.path)


    def saveConfig(self, filename, savePath = ""):

        savePath = self.path if savePath else savePath        
        self.config.save(filename, savePath, format = self.configsaveFormat)




    def getAction(self, state,  mode = "TRAIN"):

        # get one hot representation of state
        #_state = getOneHotrepresentation(state, num_classes=self.env.observation_space.n)
        
        state = np.array(state).reshape(1, self.env.observation_space.n)
        _actionsValues      = self.Qmodel.predict(state)[0] 
        _greedyActionIndex  = np.argmax(_actionsValues)

        # epsilonGreedyaction        
        if mode.upper() == "TRAIN":
            action          = self.explore.chooseAction(self.trainingiterationCount, range(self.env.action_space.n), \
                                            optimalActionIndex=_greedyActionIndex)

        elif mode.upper() == "TEST":
            # select the greedy action
            _actionsValues      = self.Targetmodel.predict(state)[0] 
            action          = _greedyActionIndex 


        return action

        


    def updateMemory(self, currentState, currentAction, reward, nextState, dead):
        if any(x is None for x in currentState) is False:
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
        inputStates     = np.array(curStates).reshape(len(curStates), self.env.observation_space.n)
        nextStates      = np.array(nextStates).reshape(len(nextStates), self.env.observation_space.n)
        #predict         = self.Qmodel.predict(inputStates)

        # ------ For DQN, target comes from the target model
        # using bellman equation 

        # Steps:
        # 1. use target model to set the target. 
        # 2. This target needs to be updated based on the bellman equation
        # 2.1. Bellman Equation 1: get the max Q values for next state in the bellman equation

        nextQvalues     = self.Targetmodel.predict(nextStates) 
        max_nextQvalues = list(map(max, nextQvalues))

        
        y_train = np.zeros(shape = nextQvalues.shape)
    
    
        # if done (dead), target is just the terminal reward 
        for index, action in enumerate(actions):
            if done[index]:
                y_train[index][int(action)] = rewards[index] 
            else:
                y_train[index][int(action)] = rewards[index] + self.discountfactor * max_nextQvalues[index]

        # mini batch mode learning
        self.Qmodel.fit(X_train=inputStates, y_train = np.array(y_train), \
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


