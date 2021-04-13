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
from Callbacks import ModifiedTensorBoardCallback

import loggingConfig as loggingConfig

reload(networks)



logger = loggingConfig.logging
logger.getLogger("PGAgents")



class REINFORCE():
    """
    Actor only policy gradient algorithm
    An ON Policy method --> Monte Carlo --> requires knowledge of full trajectory
    """

    def __init__(self, env, configFile,  model = None, setbaseline = True, **kwargs):

        self.Name   = "REINFORCE"
        self.env    = env

        self.setbaseline = setbaseline          # if baseline is True --> Reinforce with Baseline algorithm is used

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
            setattr(self.PolicyModel, "model", model)


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
            #self.gradTensorBoard = GradCallBack(model = self.Qmodel, log_dir = gradloggingPath)

            #self.callbacks.append(self.gradTensorBoard)




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

        # norm action probability distribution
        actionProb         /= np.sum(actionProb)

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

        logger.info(f"{self.Name} - Updating Policy ")
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

        self.Tensorboard.step += 1
        
        # 1. After every episode, update the episodic reward & steps taken
        self.updateEpisodicInfo(episodicReward, episodicStepsTaken, mode=mode)

        # 2. log model weights & bias Info after every episode 
        self.Tensorboard.update_stats_histogram(model = self.PolicyModel.model)
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


