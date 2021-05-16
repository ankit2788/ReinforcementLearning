import numpy as np
import random
import os
import sys
from abc import ABC, abstractclassmethod
from configparser import ConfigParser
from datetime import datetime

from tensorflow.keras.callbacks import History
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, Convolution2D, Conv2D
import tensorflow.keras.optimizers as optimizers




# importing custom libraries

import NetworkModels as networks
from ActionSelection import ActionExploration
from ConfigReader import Config

from utils import get_val, convertStringtoBoolean, getOneHotrepresentation
from Callbacks import ModifiedTensorBoardCallback


pref = os.environ["RL_PATH"]



def discountRewards(rewards, discountfactor):
    # discount rewards based on provided discount factor

    discounted_rewards=[]
    cumulative_total_return=0
    # iterate the rewards backwards and and calc the total return 
    for reward in rewards[::-1]:      
        cumulative_total_return=(cumulative_total_return* discountfactor)+reward
        discounted_rewards.insert(0, cumulative_total_return)


    return discounted_rewards




class PGAgent(ABC):

    @abstractclassmethod
    def __init__(self, env, configFile):
        self.Name               = "Test"
        self.env                = env
        self.__time             = datetime.now().strftime("%Y%m%d%H%M")
        self.config             = Config(configFile, AgentName=self.Name)

        # create model
        self.PolicyModel        = None 




        

    @abstractclassmethod
    def readConfig(self, objConfig):

        # episodic Info
        self.NbEpisodesTrain        = int(get_val(objConfig, tag = "NB_EPISODES_TRAIN", default_value= 2000))
        self.NbEpisodesTest         = int(get_val(objConfig, tag = "NB_EPISODES_TEST", default_value= 1000))
        self.MaxSteps               = int(get_val(objConfig, tag = "MAX_STEPS_EPISODE", default_value= 100))


        # RL params
        self.discountfactor         = float(get_val(objConfig, tag = "DISCOUNT_FACTOR", default_value= 0.9)) 
        self.policy_learning_rate   = float(get_val(objConfig, tag = "POLICY_LEARNING", default_value= 0.9)) 


        # TrainingInfo
        self.batchSize              = int(get_val(objConfig, tag = "TRAIN_BATCH_SIZE", default_value= 32))
        self.epochs                 = int(get_val(objConfig, tag = "EPOCHS", default_value= 32))
        self.blnbatchMode           = convertStringtoBoolean(get_val(objConfig, tag = "BATCH_MODE", default_value= "TRUE"))

        # Logging Info
        self.AggregateStatsEvery    = int(get_val(objConfig, tag = "AGGREGATE_STATS_EVERY", default_value= 100))

        # save formats
        self.configsaveFormat       = get_val(objConfig, tag = "CONFIG_FORMAT", default_value= "json")
        self.path                   = get_val(objConfig, tag = "PATH", default_value="models")

        # Need to append the path to relative path
        self.path                   = os.path.join(pref,self.path)

    
    @abstractclassmethod
    def setCallbacks(self,  networkModel, loggingPath):
        # networkModel -> on which Tensorboard should run
        self.callbacks      = []
        #loggingPath         = f"{pref}/logs/{Name}_{timeStamp}.log"
        #checkpointPath      = f"{pref}/ModelCheckpoint/{self.Name}_{self.__time}.ckpt"
        
        self.ModelHistory   = History()     # to store learning history
        self.Tensorboard    = ModifiedTensorBoardCallback(model = networkModel, log_dir = loggingPath)        

        self.callbacks.append(self.Tensorboard) 
        self.callbacks.append(self.ModelHistory) 


    @abstractclassmethod
    def saveConfig(self, filename, savePath = ""):

        savePath = self.path if savePath else savePath        
        self.config.save(filename, savePath, format = self.configsaveFormat)


    @abstractclassmethod
    def getAction(self, state, mode = "TRAIN"):
        pass

    @abstractclassmethod
    def train(self):
        pass

    @abstractclassmethod
    def updateLoggerInfo(self, tensorboard, allEpisodicRewards, allEpisodicSteps, tensorboardNetwork ,\
                    episodeCount, episodicReward, episodicStepsTaken):
        # This is used to update all the logging related information
        # how the training evolves with episode

        tensorboard.step += 1

        # 1. After every episode, update the episodic reward & steps taken
        allEpisodicRewards.append(episodicReward)
        allEpisodicSteps.append(episodicStepsTaken)

        # 2. log model weights & bias Info after every episode 
        tensorboard.update_stats_histogram(model = tensorboardNetwork)

        # 3. Create other logging stats
        tensorboard.update_stats(rewards = episodicReward, steps = episodicStepsTaken)



    

    