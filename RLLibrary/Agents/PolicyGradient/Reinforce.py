import numpy as np
import random
import os
import sys
from importlib import reload
from configparser import ConfigParser
from datetime import datetime

from tensorflow.keras.callbacks import History
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Reshape, Convolution2D, Conv2D
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras import backend as K
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
logger.getLogger("PGAgents-Reinforce")





class REINFORCE(PGAgent):
    """
    Actor only policy gradient algorithm
    An ON Policy method --> Monte Carlo --> requires knowledge of full trajectory
    """

    def __init__(self, env, configFile,  **kwargs):

        self.Name   = "REINFORCE"
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

        # create model
        self.PolicyNetwork = networks.DNNClassifier(dim_input = self.inputShape, dim_output = self.env.action_space.n, **kwargs)  
        self.PolicyNetwork.compile(**kwargs)
        self.setCallbacks()

        

    def setCallbacks(self):
        # ---- sets callback functions to record looging
        loggingPath         = f"{pref}/logs/Policy/{self.Name}_{self.__time}.log"
        super().setCallbacks(networkModel=self.PolicyNetwork.model, loggingPath=loggingPath)


    def readConfig(self):
        # ---- Reads all config frm comfig.ini

        objConfig = self.config
        super().readConfig(objConfig = objConfig)

        self.normalizeRewards       = convertStringtoBoolean(get_val(objConfig, tag = "NORMALIZE_REWARDS", default_value="TRUE"))
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
        actionProb         = self.PolicyNetwork.model.predict(_state).flatten()

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
        history = self.PolicyNetwork.model.train_on_batch(X_train, y_train)
        

        # reset memory
        self.memory = []
        


    def updateLoggerInfo(self, episodeCount, episodicReward, episodicStepsTaken, mode = "TRAIN"):
        # This is used to update all the logging related information
        # how the training evolves with episode

        super().updateLoggerInfo(tensorboard = self.Tensorboard, tensorboardNetwork = self.PolicyNetwork.model, \
                                allEpisodicRewards = self.EpisodicRewards[mode], allEpisodicSteps = self.EpisodicSteps[mode], \
                                episodeCount = episodeCount, episodicReward = episodicReward, episodicStepsTaken = episodicStepsTaken)







class REINFORCE_EAGER(PGAgent):
    """
    Actor only policy gradient algorithm
    An ON Policy method --> Monte Carlo --> requires knowledge of full trajectory
    """

    def __init__(self, env, configFile,  **kwargs):

        self.Name   = "REINFORCE"
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


        self.create_model(**kwargs)

        #self.PolicyNetwork.compile(**kwargs)
        self.setCallbacks()

    def create_model(self, **kwargs):

        # create model
        self.PolicyNetwork = networks.DNNClassifier(dim_input = self.inputShape, dim_output = self.env.action_space.n, **kwargs)  

        # create compiler and loss function
        optimizer   = kwargs["optimizer"] if "optimizer" in kwargs.keys() else "ADAM" 
        learning_rate = kwargs["learning_rate"] if "learning_rate" in kwargs.keys() else 0.001


        # create optimizer
        if optimizer.upper() == "RMSPROP":            
            optimizer = optimizers.RMSprop(learning_rate = learning_rate)
        elif optimizer.upper() == "ADAM":            
            optimizer = optimizers.Adam(learning_rate = learning_rate)

        setattr(self.PolicyNetwork, "optimizer", optimizer)      

        

    def setCallbacks(self):
        # ---- sets callback functions to record looging
        loggingPath         = f"{pref}/logs/Policy/{self.Name}_{self.__time}.log"
        super().setCallbacks(networkModel=self.PolicyNetwork.model, loggingPath=loggingPath)


    def readConfig(self):
        # ---- Reads all config frm comfig.ini

        objConfig = self.config
        super().readConfig(objConfig = objConfig)

        self.normalizeRewards       = convertStringtoBoolean(get_val(objConfig, tag = "NORMALIZE_REWARDS", default_value="TRUE"))
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
        actionProb         = self.PolicyNetwork.model.predict(_state).flatten()

        # norm action probability distribution
        actionProb         /= np.sum(actionProb)

        # sample the action based on the probability
        action             = np.random.choice(self.env.action_space.n, p = actionProb)

        return action, actionProb


    
    def updateMemory(self, currentState, currentAction, reward, nextState, dead, actionProb):
        self.memory.append((currentState, currentAction, reward, nextState, dead, actionProb))

    def customLoss():
        log_prob = dist.log_prob(action)
        loss = -log_prob*reward
        return loss 

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
        if self.normalizeRewards:
            discounted_rewards = (discounted_rewards - np.mean(discounted_rewards))/ (np.std(discounted_rewards) + 1e-7)            # to avoid division by 0


        losses = []

        with tf.GradientTape() as tape:
            for index, sample in enumerate(self.memory):

                # get log prob
                state = tf.Variable([sample[0]], trainable=True, dtype=tf.float64)
                prob = self.PolicyNetwork.model(state, training = True)
                action = sample[1]
                actionProb = prob[0, action]
                logProb = tf.math.log(actionProb)

                #actionProb = sample[5]
                #logProb = np.log(actionProb)
                sampleloss = logProb * discounted_rewards[index][0]

                losses.append(-sampleloss)      # this is negative, since we are interested in gradient ascent

            networkLoss = sum(losses)
            
            grads = tape.gradient(networkLoss, self.PolicyNetwork.model.trainable_variables)
            self.PolicyNetwork.optimizer.apply_gradients(zip(grads, self.PolicyNetwork.model.trainable_variables))



        logger.info(f"{self.Name} - Updating Policy ")
        

        # reset memory
        self.memory = []
        


    def updateLoggerInfo(self, episodeCount, episodicReward, episodicStepsTaken, mode = "TRAIN"):
        # This is used to update all the logging related information
        # how the training evolves with episode

        super().updateLoggerInfo(tensorboard = self.Tensorboard, tensorboardNetwork = self.PolicyNetwork.model, \
                                allEpisodicRewards = self.EpisodicRewards[mode], allEpisodicSteps = self.EpisodicSteps[mode], \
                                episodeCount = episodeCount, episodicReward = episodicReward, episodicStepsTaken = episodicStepsTaken)




class REINFORCE_BASELINE(PGAgent):
    """
    Actor only policy gradient algorithm
    An ON Policy method --> Monte Carlo --> requires knowledge of full trajectory
    """

    def __init__(self, env, configFile,  **kwargs):

        self.Name   = "REINFORCE_BASELINE"
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

        # create model  (Both value and policy models are needed)
        self.PolicyNetwork = networks.DNNClassifier(dim_input = self.inputShape, dim_output = self.env.action_space.n, **kwargs)  
        self.PolicyNetwork.compile(**kwargs)

        self.ValueNetwork = None
        if self.baseline.upper() == "VALUE":

            valueParams = {key.split("_value")[0]: kwargs[key] for key in kwargs.keys() if "_VALUE" in key.upper() }
            self.ValueNetwork = networks.DNN(dim_input = self.inputShape, dim_output = self.env.action_space.n, **valueParams)  
            self.ValueNetwork.compile(**valueParams)

        self.setCallbacks()


        

    def setCallbacks(self):
        # ---- sets callback functions to record looging
        loggingPath         = f"{pref}/logs/Policy/{self.Name}_{self.__time}.log"
        super().setCallbacks(networkModel=self.PolicyNetwork.model, loggingPath=loggingPath)



    def readConfig(self):
        # ---- Reads all config frm comfig.ini

        objConfig = self.config
        super().readConfig(objConfig = objConfig)

        self.memory = []

        self.baseline               = get_val(objConfig, tag = "BASELINE", default_value= "NORMALIZE")
        if self.baseline.upper() not in ["VALUE", "NORMALIZE"]:
            logger.error(f'Baseline has to be either of VALUE or NORMALIZE')
            sys.exit(0)
        

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
        actionProb         = self.PolicyNetwork.model.predict(_state).flatten()

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


        # Get X
        try:
            self.env.observation_space.n
            X_train = getOneHotrepresentation(curStates, num_classes=self.inputShape)
        except:
            X_train = curStates    


        # compute the discounted rewards for the entire episode and normalize it
        discounted_rewards = discountRewards(rewards,  discountfactor=self.discountfactor)

        if self.baseline.upper() == "VALUE":
            value = self.ValueNetwork.model.predict(X_train)
            G = discounted_rewards - value
            

        elif self.baseline.upper() == "NORMALIZE":
            G = (discounted_rewards - np.mean(discounted_rewards))/ (np.std(discounted_rewards) + 1e-7)            # to avoid division by 0




        gradient = np.subtract(getOneHotrepresentation(actions,self.env.action_space.n ), actionProb)
        gradient *= G 
        gradient *= self.policy_learning_rate

        # updating actual probabilities (y_train) to take into account the change in policy gradient change
        # \theta = \theta + alpha*rewards * gradient
        y_train = actionProb + np.vstack(gradient)


        logger.info(f"{self.Name} - Updating Policy ")

        # update policy and also learn the value function
        
        # 1. Update policy
        history = self.PolicyNetwork.model.train_on_batch(X_train, y_train)

        # 2. Learn target. Use discounted rewards as the target values 
        ## use the observed return Gt as a ‘target’ of the learned value function. 
        # Because Gt is a sample of the true value function for the current policy, this is a reasonable target.
        if self.baseline.upper() == "VALUE":
            self.ValueNetwork.model.train_on_batch(X_train, G)
        # reset memory
        self.memory = []
        


    def updateLoggerInfo(self, episodeCount, episodicReward, episodicStepsTaken, mode = "TRAIN"):
        # This is used to update all the logging related information
        # how the training evolves with episode

        super().updateLoggerInfo(tensorboard = self.Tensorboard, tensorboardNetwork = self.PolicyNetwork.model, \
                                allEpisodicRewards = self.EpisodicRewards[mode], allEpisodicSteps = self.EpisodicSteps[mode], \
                                episodeCount = episodeCount, episodicReward = episodicReward, episodicStepsTaken = episodicStepsTaken)





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


        loggingPath         = f"{pref}/logs/Policy/{self.Name}_{self.__time}.log"
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