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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, Convolution2D, Conv2D
from tensorflow.keras.optimizers import Adam






# Create classes for Different Value based models
# 1. Reinforce (with baseline)
# 2. Actor Critic Method
# 3. A2C/ A3C
# 4. DDPG


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
        self.policy_learning_rate   = float(get_val(self.config, tag = "POLICY_LEARNING", default_value= 0.9)) 
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
        actionProb         = self.PolicyModel.predict(_state).flatten()

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
        discounted_rewards = self.discountRewards(rewards)
        if self.normalizeRewards:
            discounted_rewards = (discounted_rewards - np.mean(discounted_rewards))/ (np.std(discounted_rewards) + 1e-7)            # to avoid division by 0



        # ---- Compute the Policy gradient

        # --- below formulation comes from the derivative of cross entropy loss function wrt to output layer
        # grad(cross entrpy loss) = p_i - y_i --> predicted value  - actual value        
        # https://deepnotes.io/softmax-crossentropy
        # https://cs231n.github.io/neural-networks-2/#losses  --> for more detailed info on losses and its derivation
        # http://karpathy.github.io/2016/05/31/rl/


        # in the below, actualvalue --> 1 for chosen action and 0 for not chosen action --> represented by onehotrepresentation
        #               predictedValue --> predicted probs from network 

        gradient = np.subtract(getOneHotrepresentation(actions,self.env.action_space.n ), actionProb)
        gradient *= discounted_rewards 
        gradient *= self.policy_learning_rate

        # updating actual probabilities (y_train) to take into account the change in policy gradient change
        # \theta = \theta + alpha*rewards * gradient
        y_train = actionProb + np.vstack(gradient)

        # Get X
        try:
            self.env.observation_space.n
            X_train = getOneHotrepresentation(curStates, num_classes=self.inputShape)
        except:
            X_train = curStates    


        logger.info(f"{self.Name} - Updating Policy ")
        history = self.PolicyModel.model.train_on_batch(X_train, y_train)
        

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
        self.Tensorboard.update_stats(rewards = self.EpisodicRewards[mode][-1], steps = episodicStepsTaken)

        """
        if not episodeCount % self.AggregateStatsEvery or episodeCount == 1:

            
            average_reward  = sum(self.EpisodicRewards[mode][-self.AggregateStatsEvery:])/len(self.EpisodicRewards[mode][-self.AggregateStatsEvery:])
            #min_reward      = min(self.EpisodicRewards[mode][-self.AggregateStatsEvery:])
            max_reward      = max(self.EpisodicRewards[mode][-self.AggregateStatsEvery:])

            average_steps   = sum(self.EpisodicSteps[mode][-self.AggregateStatsEvery:])/len(self.EpisodicSteps[mode][-self.AggregateStatsEvery:])
            max_steps       = max(self.EpisodicSteps[mode][-self.AggregateStatsEvery:])
            #min_steps       = min(self.EpisodicSteps[mode][-self.AggregateStatsEvery:])

            self.Tensorboard.update_stats(reward_avg=average_reward, steps_avg=average_steps, \
                                            reward_max=max_reward, steps_max = max_steps)


        """




class REINFORCE_BASELINE():


    """
    An ON Policy method --> Monte Carlo --> requires knowledge of full trajectory
    Baseline reduces the performancevariance
    """

    def __init__(self, env, configFile,  **kwargs):

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

        # Reinforce with Baseline has value function as well which is setup as baseline for performance improvement 
        try:
            self.inputShape = self.env.observation_space.n
        except:
            self.inputShape = self.env.observation_space.shape[0]

        self.PolicyModel    = networks.DeepNeuralModelClassifier(self.config, dim_input = self.inputShape, \
                                dim_output= self.env.action_space.n, time = self.__time, Name = self.Name, \
                                **kwargs) 

        self.PolicyModel.init()

        # set the value network
        self.ValueModel = None
        if self.baseline.upper() == "VALUE":
            self.ValueModel     =  networks.DeepNeuralModel(self.config, dim_input = self.inputShape, \
                                    dim_output= self.env.action_space.n, time = self.__time, Name = self.Name, \
                                    **kwargs)
            self.ValueModel.init()


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

        


    def __readConfig(self):

        # episodic Info
        self.NbEpisodesTrain        = int(get_val(self.config, tag = "NB_EPISODES_TRAIN", default_value= 2000))
        self.NbEpisodesTest         = int(get_val(self.config, tag = "NB_EPISODES_TEST", default_value= 1000))
        self.MaxSteps               = int(get_val(self.config, tag = "MAX_STEPS_EPISODE", default_value= 100))


        # RL params
        self.discountfactor         = float(get_val(self.config, tag = "DISCOUNT_FACTOR", default_value= 0.9)) 
        self.policy_learning_rate   = float(get_val(self.config, tag = "POLICY_LEARNING", default_value= 0.9)) 

        self.baseline               = get_val(self.config, tag = "BASELINE", default_value= "NORMALIZE")
        if self.baseline.upper() not in ["VALUE", "NORMALIZE"]:
            logger.error(f'Baseline has to be either of VALUE or NORMALIZE')
            sys.exit(0)

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
        actionProb         = self.PolicyModel.predict(_state).flatten()

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

        return discounted_rewards

    
    def updateMemory(self, currentState, currentAction, reward, nextState, dead, actionProb):
        self.memory.append((currentState, currentAction, reward, nextState, dead, actionProb))

    def updatePolicy(self):

        
        # ---- Compute the Policy gradient

        # --- below formulation comes from the derivative of cross entropy loss function wrt to output layer
        # grad(cross entrpy loss) = p_i - y_i --> predicted value  - actual value        
        # https://deepnotes.io/softmax-crossentropy
        # https://cs231n.github.io/neural-networks-2/#losses  --> for more detailed info on losses and its derivation
        # http://karpathy.github.io/2016/05/31/rl/


        # in the below, actualvalue --> 1 for chosen action and 0 for not chosen action --> represented by onehotrepresentation
        #               predictedValue --> predicted probs from network 

        """
        G = discountedRewards
        gradient = -(p_i - y_i)           # this is the gradient of cross entropy loss

        # update the target value by gradient
        y_target = p_i +  alpha * G * gradient
        """


        # for all experience in batchsize
        curStates       = np.vstack(list(list(zip(*self.memory)))[0])
        actions         = np.vstack(list(list(zip(*self.memory)))[1])
        nextStates      = np.vstack(list(list(zip(*self.memory)))[3])
        rewards         = np.vstack(list(list(zip(*self.memory)))[2])
        done            = np.vstack(list(list(zip(*self.memory)))[4])

        actionProb      = np.vstack(list(list(zip(*self.memory)))[5])

        

        # Get X
        try:
            self.env.observation_space.n
            X_train = getOneHotrepresentation(curStates, num_classes=self.inputShape)
        except:
            X_train = curStates    


        # compute the discounted rewards for the entire episode and normalize it
        discountedRewards = self.discountRewards(rewards)

        
        if self.baseline.upper() == "VALUE":
            value = self.ValueModel.model.predict(X_train)
            G = discountedRewards - value

        elif self.baseline.upper() == "NORMALIZE":
            G = discountedRewards
            G = (G - np.mean(G))/ (np.std(G) + 1e-7)            # to avoid division by 0


        gradient = np.subtract(getOneHotrepresentation(actions,self.env.action_space.n ), actionProb)
        gradient *= G 
        gradient *= self.policy_learning_rate

        # updating actual probabilities (y_train) to take into account the change in policy gradient change
        # \theta = \theta + alpha*rewards * gradient
        y_train = actionProb + np.vstack(gradient)





        logger.info(f"{self.Name} - Updating Policy ")

        # update policy and also learn the value function
        
        # 1. Update policy
        history = self.PolicyModel.model.train_on_batch(X_train, y_train)

        # 2. Learn target. Use discounted rewards as the target values 
        ## use the observed return Gt as a ‘target’ of the learned value function. 
        # Because Gt is a sample of the true value function for the current policy, this is a reasonable target.

        self.ValueModel.model.train_on_batch(X_train, G)
        
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
        self.Tensorboard.update_stats(rewards = self.EpisodicRewards[mode][-1], steps = episodicStepsTaken)




class REINFORCE_C:
    def __init__(self, env, path=None):
        self.Name = "Reinforce_C"
        self.env=env #import env
        self.state_shape=env.observation_space.shape # the state space
        self.action_shape=env.action_space.n # the action space
        self.gamma=0.99 # decay rate of past observations
        self.alpha=1e-4 # learning rate in the policy gradient
        self.learning_rate=0.01 # learning rate in deep learning

        if not path:
            self.model=self._create_model() #build model
        else:
            self.model=self.load_model(path) #import model

        # record observations
        self.states=[]
        self.gradients=[] 
        self.rewards=[]
        self.probs=[]
        self.discounted_rewards=[]
        self.total_rewards=[]   

        self.EpisodicRewards    = {"TRAIN": [], "TEST": []}
        self.EpisodicSteps      = {"TRAIN": [], "TEST": []}

        self.__time             = datetime.now().strftime("%Y%m%d%H%M")


        loggingPath         = f"{pref}/logs/{self.Name}_{self.__time}.log"
        #checkpointPath      = f"{pref}/ModelCheckpoint/{self.Name}_{self.__time}.ckpt"
        
        self.Tensorboard    = ModifiedTensorBoardCallback(model = self.model, log_dir = loggingPath)        


    def hot_encode_action(self, action):
        '''encoding the actions into a binary list'''

        action_encoded=np.zeros(self.action_shape, np.float32)
        action_encoded[action]=1

        return action_encoded

    def remember(self, state, action, action_prob, reward):
        '''stores observations'''
        encoded_action=self.hot_encode_action(action)
        self.gradients.append(encoded_action-action_prob)
        self.states.append(state)
        self.rewards.append(reward)
        self.probs.append(action_prob)      


    def _create_model(self):
        ''' builds the model using keras'''
        model=Sequential()

        # input shape is of observations
        model.add(Dense(24, input_shape=self.state_shape, activation="relu"))
        # add a relu layer 
        model.add(Dense(12, activation="relu"))

        # output shape is according to the number of action
        # The softmax function outputs a probability distribution over the actions
        model.add(Dense(self.action_shape, activation="softmax")) 
        model.compile(loss="categorical_crossentropy",
                optimizer=Adam(lr=self.learning_rate))
            
        return model          


    def get_action(self, state):
        '''samples the next action based on the policy probabilty distribution 
        of the actions'''

        # transform state
        state=state.reshape([1, state.shape[0]])
        # get action probably
        action_probability_distribution=self.model.predict(state).flatten()
        # norm action probability distribution
        action_probability_distribution/=np.sum(action_probability_distribution)
        
        # sample action
        action=np.random.choice(self.action_shape,1,
                                p=action_probability_distribution)[0]

        return action, action_probability_distribution   


    def get_discounted_rewards(self, rewards): 
        '''Use gamma to calculate the total reward discounting for rewards
        Following - \gamma ^ t * Gt'''
        
        discounted_rewards=[]
        cumulative_total_return=0
        # iterate the rewards backwards and and calc the total return 
        for reward in rewards[::-1]:      
            cumulative_total_return=(cumulative_total_return*self.gamma)+reward
            discounted_rewards.insert(0, cumulative_total_return)

        # normalize discounted rewards
        mean_rewards=np.mean(discounted_rewards)
        std_rewards=np.std(discounted_rewards)
        norm_discounted_rewards=(discounted_rewards-
                            mean_rewards)/(std_rewards+1e-7) # avoiding zero div
        
        return norm_discounted_rewards    



    def update_policy(self):
        '''Updates the policy network using the NN model.
        This function is used after the MC sampling is done - following
        \delta \theta = \alpha * gradient + log pi'''
        
        # get X
        states=np.vstack(self.states)

        # get Y
        gradients=np.vstack(self.gradients)
        rewards=np.vstack(self.rewards)
        discounted_rewards=self.get_discounted_rewards(rewards)
        gradients*=discounted_rewards
        gradients=self.alpha*np.vstack([gradients])+self.probs

        history=self.model.train_on_batch(states, gradients)
        
        self.states, self.probs, self.gradients, self.rewards=[], [], [], []

        return history                 


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
        self.Tensorboard.update_stats_histogram(model = self.model)
        #self.Qmodel.tensorboard.update_stats_histogram()
        #self.Targetmodel.tensorboard.update_stats_histogram()

        # 3. Create other logging stats
        self.Tensorboard.update_stats(rewards = self.EpisodicRewards[mode][-1], steps = episodicStepsTaken)        