import numpy as np
import random
import os
import sys
from importlib import reload
from configparser import ConfigParser
from datetime import datetime

from tensorflow.keras.callbacks import History
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Reshape, Convolution2D, Conv2D, Input
import tensorflow.keras.optimizers as optimizers
import tensorflow as tf


try: 
    pref = os.environ["RL_PATH"]
except KeyError:

    # get the relative path
    fullpath = os.path.realpath(__file__)
    pref     = os.path.split(fullpath)[0]


# importing custom libraries

import NetworkModels as networks
from ConfigReader import Config
from Agents.PolicyGradient import PGAgent, discountRewards

from utils import get_val, convertStringtoBoolean, getOneHotrepresentation
from Callbacks import ModifiedTensorBoardCallback
import loggingConfig as loggingConfig

reload(networks)


logger = loggingConfig.logging
logger.getLogger("PGAgents-ActorCritic")





class ActorCritic(PGAgent):
    """
    Actor Critic  policy gradient algorithm
    Same
    """

    def __init__(self, env, configFile,  **kwargs):

        self.Name   = "ActorCritic"
        self.env    = env

        # get the config & hyper parameters info
        self.config     = Config(configFile, AgentName=self.Name)
        self.readConfig()

        # episodic Info
        self.EpisodicRewards    = {"TRAIN": [], "TEST": []}
        self.EpisodicSteps      = {"TRAIN": [], "TEST": []}
        self.trainingiterationCount     = 0 

        self.__time             = datetime.now().strftime("%Y%m%d%H%M")

        try:
            self.inputShape = self.env.observation_space.n
        except:
            self.inputShape = self.env.observation_space.shape[0]

        # create ActorCritic model  (Both value and policy models are needed)
        self.create_model(**kwargs)


        self.setCallbacks()


    def create_model(self, **kwargs):
        # here we share the networks across the value function and poliy network

        # set the layers
        inputShape          = Input(shape=(self.inputShape,)) 
        neuralNetShape      = kwargs["NetworkShape"] if "NetworkShape" in kwargs.keys() else [64, 32]        

        self.inputLayer     = Dense(units = neuralNetShape[0], activation = "relu")(inputShape)


        # get all hidden layers
        self.hiddenLayer    = self.inputLayer        
        if len(neuralNetShape) > 1:
            for _layer in neuralNetShape[1:]:
                _hiddenLayerinputs = Dense(units = _layer, activation = "relu")(self.hiddenLayer)

            self.hiddenLayer = _hiddenLayerinputs

        self.actorLayer     = Dense(units = self.env.action_space.n, activation = "softmax")(self.hiddenLayer)
        self.criticLayer    = Dense(units = 1, activation = "linear")(self.hiddenLayer)     # since its a value per state, output layer has only 1 dimension

        self.SharedNetwork  = Model(inputs= inputShape, outputs = [self.actorLayer, self.criticLayer], name = "SharedDesign")

        # create compiler and loss function
        optimizer   = kwargs["optimizer"] if "optimizer" in kwargs.keys() else "ADAM" 
        learning_rate = kwargs["learning_rate"] if "learning_rate" in kwargs.keys() else 0.01


        # create optimizer
        if optimizer.upper() == "RMSPROP":            
            optimizer = optimizers.RMSprop(learning_rate = learning_rate)
        elif optimizer.upper() == "ADAM":            
            optimizer = optimizers.Adam(learning_rate = learning_rate)
        setattr(self.SharedNetwork, "optimizer", optimizer)   


        


        

    def setCallbacks(self):
        # ---- sets callback functions to record looging
        loggingPath         = f"{pref}/logs/Policy/{self.Name}_{self.__time}.log"
        super().setCallbacks(networkModel=self.SharedNetwork, loggingPath=loggingPath)



    def readConfig(self):
        # ---- Reads all config frm comfig.ini

        objConfig = self.config
        super().readConfig(objConfig = objConfig)
        self.memory = []


    def saveConfig(self, filename, savePath = ""):
        # ---- saves all config frm comfig.ini
        super().saveConfig(filename=filename, savePath=savePath)


    def getAction(self, state, mode = "TRAIN"):
        # based on the state, predict the action to be taken using the network
        # returns:
        # 1. Action taken
        # 2. Prob distribution of individual action given a state
        # 3. Value of current state given current policy
        
        try:
            self.env.observation_space.n
            _state = getOneHotrepresentation(state, num_classes=self.inputShape)
        except:
            _state = state.reshape([1, state.shape[0]])


        # the model prediction predicts the prob space for all actions
        actionProb, value   = self.SharedNetwork(_state)
        actionProb = actionProb.numpy()[0]
        value = value.numpy()[0]
        
        # norm action probability distribution
        actionProb         /= sum(actionProb)
        

        # sample the action based on the probability
        action             = np.random.choice(self.env.action_space.n, p = actionProb)

        return action, actionProb, value


    
    def updateMemory(self, currentState, currentAction, reward, nextState, dead, actionProb, value):
        self.memory.append((currentState, currentAction, reward, nextState, dead, actionProb, value))

    def train(self):

        with tf.GradientTape() as tape:

            loss_critic = []
            loss_actor  = []

            for index, sample in enumerate(self.memory):



                # get log prob
                state   = tf.Variable([sample[0]], trainable=True, dtype=tf.float32)
                action, reward, nextState  = sample[1], sample[2], sample[3]
                nextState = tf.Variable([nextState], trainable=True, dtype=tf.float32)



                # run shared network to get action prob distro and value associated with that state
                actionProbDistro, _currStateValue = self.SharedNetwork(state, training = True)
                _actionProbDistro, _nextStateValue = self.SharedNetwork(nextState, training = True)

                # compute TD error
                delta = reward + self.discountfactor*_nextStateValue - _currStateValue

                # compute critic loss
                loss_sample_critic = tf.math.square(delta)

                # compute actor loss
                actionProb      = actionProbDistro[0, action]
                loss_sample_actor =  tf.math.log(actionProb) * delta

                loss_actor.append(-loss_sample_actor)


            networkLoss = sum(loss_critic) + sum(loss_actor)

            logger.info(f"{self.Name} - Updating Policy ")

            # performing Backpropagation to update the network
            grads = tape.gradient(networkLoss, self.SharedNetwork.trainable_variables)
            self.SharedNetwork.optimizer.apply_gradients(zip(grads, self.SharedNetwork.trainable_variables))



            # reset memory
            self.memory = []
        


    def updateLoggerInfo(self, episodeCount, episodicReward, episodicStepsTaken, mode = "TRAIN"):
        # This is used to update all the logging related information
        # how the training evolves with episode

        super().updateLoggerInfo(tensorboard = self.Tensorboard, tensorboardNetwork = self.SharedNetwork, \
                                allEpisodicRewards = self.EpisodicRewards[mode], allEpisodicSteps = self.EpisodicSteps[mode], \
                                episodeCount = episodeCount, episodicReward = episodicReward, episodicStepsTaken = episodicStepsTaken)



