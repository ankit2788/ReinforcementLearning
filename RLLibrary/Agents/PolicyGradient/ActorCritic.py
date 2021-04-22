import numpy as np
import random
import os
import sys
from importlib import reload
from configparser import ConfigParser
from datetime import datetime

from tensorflow.keras.callbacks import History
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, Convolution2D, Conv2D
import tensorflow.keras.optimizers as optimizers



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
        self.create_model()


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

        self.actorLayer     = Dense(units = outputDims, activation = "softmax")(self.hiddenLayer)
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
        super().setCallbacks(networkModel=self.SharedNetwork.model, loggingPath=loggingPath)



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
        
        try:
            self.env.observation_space.n
            _state = getOneHotrepresentation(state, num_classes=self.inputShape)
        except:
            _state = state.reshape([1, state.shape[0]])


        # the model prediction predicts the prob space for all actions
        actionProb, ValueFunction   = self.SharedNetwork.model.predict(_state).flatten()

        # norm action probability distribution
        actionProb         /= np.sum(actionProb)

        # sample the action based on the probability
        action             = np.random.choice(self.env.action_space.n, p = actionProb)

        return action, actionProb


    
    def updateMemory(self, currentState, currentAction, reward, nextState, dead, actionProb):
        self.memory.append((currentState, currentAction, reward, nextState, dead, actionProb))

    def train(self):

        

        # for all experience in batchsize
        curStates       = np.vstack(list(list(zip(*self.memory)))[0])
        actions         = np.vstack(list(list(zip(*self.memory)))[1])
        nextStates      = np.vstack(list(list(zip(*self.memory)))[3])
        rewards         = np.vstack(list(list(zip(*self.memory)))[2])
        done            = np.vstack(list(list(zip(*self.memory)))[4])

        actionProb      = np.vstack(list(list(zip(*self.memory)))[5])

        # compute the discounted rewards for the entire episode and normalize it
        discounted_rewards = discountRewards(rewards,  discountfactor=self.discountfactor)
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
        history = self.PolicyNetwork.model.train_on_batch(X_train, y_train)
        

        # reset memory
        self.memory = []
        


    def updateLoggerInfo(self, episodeCount, episodicReward, episodicStepsTaken, mode = "TRAIN"):
        # This is used to update all the logging related information
        # how the training evolves with episode

        super().updateLoggerInfo(tensorboard = self.Tensorboard, tensorboardNetwork = self.PolicyNetwork.model, \
                                allEpisodicRewards = self.EpisodicRewards[mode], allEpisodicSteps = self.EpisodicSteps[mode], \
                                episodeCount = episodeCount, episodicReward = episodicReward, episodicStepsTaken = episodicStepsTaken)



