import os, sys
import numpy as np
from importlib import reload
from abc import ABC, abstractclassmethod

import tensorflow as tf

# get the relative path
# fullpath                = os.path.realpath(__file__)
# pref                    = os.path.split(fullpath)[0]


# os.environ["RL_PATH"]   = f'{pref}/../../..'
# pref = f'{pref}/../../..'
# if f'{pref}/RLLibrary' not in sys.path:
#     sys.path.append(f'{pref}')
#     sys.path.append(f'{pref}/RLLibrary')


from RLLibrary.FinUseCases import CustomGym
from RLLibrary.FinUseCases import EnvironmentStorage

from RLLibrary.FinUseCases.PortfolioManagement.ModelManager.A3C import Agent as A3CAgent
from RLLibrary.FinUseCases.PortfolioManagement.ModelManager.A3C import Networks as A3CNet
from RLLibrary.FinUseCases.PortfolioManagement.ModelManager.DQN import Agent as DQNAgent
from RLLibrary.FinUseCases.PortfolioManagement.ModelManager.DQN import Networks as DQNNet

from RLLibrary.FinUseCases.PortfolioManagement import ActionManager

reload(ActionManager)
reload(A3CAgent)

from RLLibrary.utils import constants as constants
DATA_DIR = constants.DATA_DIR
MODEL_DIR = constants.MODEL_DIR

from RLLibrary.utils.loggingConfig import logger
Logger = logger.getLogger("StrategyManager")


# # ------register PortfolioManagement environment onto Custom gym

# CustomGym.register(
#     id = "PortfolioManagement-v0",
#     entry_point = 'FinUseCases.PortfolioManagement.EnvironmentManager:Portfolio',
#     kwargs = {"assets" : ["APA", "BMY"], "initialWeight" : [0.5, 0.5], \
#                     "nhistoricalDays" : 30, \
#                     "startDate" : "2019-01-01", "endDate" : "2019-12-31", \
#                     "actions" : [(-0.1,0.1)], \
#                     "assetDataPath" : os.path.join(DATA_DIR, "PortfolioManagement"), \
#                     "config" : {"initialCash": 1000000, "minCash": 0.02, "transactionFee": 0.0001}, 
#                     "penalizeFactors" : {"Risk": -0.08, "ForbiddenAction": -8}})


# CustomGym.register(
#     id = "PortfolioManagement_CNN-v0",
#     entry_point = 'FinUseCases.PortfolioManagement.EnvironmentManager:Portfolio_MultiStage',
#     kwargs = {"assets" : ["APA", "BMY"], "initialWeight" : [0.5, 0.5], \
#                     "nhistoricalDays" : 30, \
#                     "startDate" : "2019-01-01", "endDate" : "2019-12-31", \
#                     "actions" : [(-0.1,0.1)], \
#                     "assetDataPath" : os.path.join(DATA_DIR, "PortfolioManagement"), \
#                     "config" : {"initialCash": 1000000, "minCash": 0.02, "transactionFee": 0.0001}, 
#                     "penalizeFactors" : {"Risk": -0.08, "ForbiddenAction": -8}})



# ------ Defining Strategies --------

class Strategy(ABC):
    @abstractclassmethod
    def __init__(self, envName, **env_args):
        pass        


    @abstractclassmethod
    def run(self):
        pass

    @abstractclassmethod
    def plotPerformance(self):
        pass



class BuyandHoldStrategy(Strategy):
    # This strategy is just buy and hold

    def __init__(self, envName = "PortfolioManagement-v0", **env_args):

        
        # create the environment
        self.env = CustomGym.make(id = envName, **env_args)

        # update the action space to No actions
        self.actions = [[0] * len(self.env.assets)]             # simply buy and hold
        self.env.action_space = ActionManager.DiscreteActionSpace(self.actions)

        Logger.info(f"{envName} environment created")


        #self.env = CustomGym.make(id = "PortfolioManagement-v0", **env_args)

    def getEnvironmentArgs(self):
        # return the default arguments used in the environment
        return self.env.spec._kwargs



    def run(self):

        Logger.info("Running Equi weighted Strategy")

        currentState = self.env.reset()
        episodeOver = False
        while not episodeOver:
            action = list(self.actions[0])
            newstate, reward, episodeOver, _ = self.env.step(action)
            currentState = newstate

        portHistory = self.env.getPortfolioHistory() 
        return portHistory


    def plotPerformance(self):
        self.env.render()





class RLStrategy_A3C(Strategy):
    # This strategy is just buy and hold

    def __init__(self, envName = "PortfolioManagement-v0", **env_args):

        self.envName = envName
        self.envargs = env_args
        Logger.info("Setting up  A3C agent ")



    def train(self, cores = 1, save_dir = os.path.join(MODEL_DIR, "PortfolioManagement"), MAX_EPISODES = 4000, \
                ActorCriticModel = None, \
                actorHiddenUnits = [32], criticHiddenUnits = [32], optimizer_learning_rate = 1e-4):

        # create the master agent
        Logger.info("Training with A3C agent ")

        self.masterAgent = A3CAgent.MasterAgent(envName = self.envName, cores = cores, save_dir = save_dir, \
                            MAX_EPISODES = MAX_EPISODES, ActorCriticModel = ActorCriticModel, \
                            actorHiddenUnits = actorHiddenUnits, criticHiddenUnits = criticHiddenUnits, \
                            optimizer_learning_rate = optimizer_learning_rate,  \
                            **self.envargs)

        self.masterAgent.train()




    def run(self, modelWeights, actorHiddenUnits = [32], criticHiddenUnits = [32], \
                testPeriod = {"startDate": "2019-01-01", "endDate": "2019-12-31"}):

        # load teh environment

        self.envargs["startDate"] = testPeriod["startDate"]
        self.envargs["endDate"] = testPeriod["endDate"]

        self.env = CustomGym.make(self.envName, **self.envargs)

        model = A3CNet.ActorCritic_FF(state_size=self.env.observation_space.n, action_size=len(self.env.action_space.actions), \
                        actorHiddenUnits=actorHiddenUnits, criticHiddenUnits= criticHiddenUnits)

        
        # compile the model
        model(tf.convert_to_tensor(np.array(self.env.observation_space.currentState)[None, :], dtype = tf.float32))

        # load the weights
        model.load_weights(modelWeights)

        # run the environment
        self.env.reset()
        done = False
        currentState = np.array(self.env.observation_space.currentState)
        forbidden_action_count = 0
        episodic_reward = 0

        while not done:

            # get action probability
            currentState = np.array(currentState)
            probs, _ = model(tf.convert_to_tensor(currentState[None, :], dtype = tf.float32))
            _probs = np.nan_to_num(probs.numpy()[0])

            # choose the action with max prob
            actionIndex = np.argmax(_probs)
            # choose the action based on this prob distribution
            action = self.env.action_space.actions[actionIndex]

            temp = list(_probs)
            if self.env.observation_space.isactionForbidden(actionIndex = actionIndex, allActions = self.env.action_space.actions):
                forbidden_action_count += 1

            next_state, reward, done, _ = self.env.step(action)

            episodic_reward += reward
            currentState = next_state

        print(f'Total episodic reward using A3C-VSM: {episodic_reward}. Forbidden score: {forbidden_action_count}')
        portHistory = self.env.getPortfolioHistory()
        return portHistory


    def plotPerformance(self):
        self.env.render()


class RLStrategy_A3C_CNN(Strategy):
    # This strategy is just buy and hold

    def __init__(self, envName = "PortfolioManagement-v0", **env_args):

        self.envName = envName
        self.envargs = env_args
        Logger.info("Setting up  A3C agent ")



    def train(self, cores = 1, save_dir = os.path.join(MODEL_DIR, "PortfolioManagement"), MAX_EPISODES = 4000, \
                ActorCriticModel = None, \
                actorHiddenUnits = [32], criticHiddenUnits = [32], optimizer_learning_rate = 1e-4):

        # create the master agent
        Logger.info("Training with A3C agent ")

        self.masterAgent = A3CAgent.MasterAgent_Multi(envName = self.envName, cores = cores, save_dir = save_dir, \
                            MAX_EPISODES = MAX_EPISODES, ActorCriticModel = ActorCriticModel, \
                            optimizer_learning_rate = optimizer_learning_rate,  \
                            **self.envargs)

        self.masterAgent.train()





    def run(self, modelWeights, \
                testPeriod = {"startDate": "2019-01-01", "endDate": "2019-12-31"}):

        # load teh environment

        self.envargs["startDate"] = testPeriod["startDate"]
        self.envargs["endDate"] = testPeriod["endDate"]

        self.env = CustomGym.make(self.envName, **self.envargs)

        model = A3CNet.ActorCritic_CNN(nbHistory=self.env.nbHistory, action_size=len(self.env.action_space.actions))

        
        # compile the model
        _randomInitialState = self.env.observation_space.currentState
        model(_randomInitialState)

        # load the weights
        model.load_weights(modelWeights)

        # run the environment
        self.env.reset()
        done = False
        currentState = self.env.observation_space.currentState
        forbidden_action_count = 0
        episodic_reward = 0

        while not done:

            if any(x is None for x in currentState):
                continue

            # get action probability
            probs, _ = model(currentState)
            _probs = np.nan_to_num(probs.numpy()[0])

            # choose the action with max prob
            actionIndex = np.argmax(_probs)
            # choose the action based on this prob distribution
            action = self.env.action_space.actions[actionIndex]

            temp = list(_probs)
            if self.env.observation_space.isactionForbidden(actionIndex = actionIndex, allActions = self.env.action_space.actions):
                forbidden_action_count += 1

            next_state, reward, done, _ = self.env.step(action)

            episodic_reward += reward
            currentState = next_state

        print(f'Total episodic reward using A3C-PVM: {episodic_reward}. Forbidden score: {forbidden_action_count}')
        portHistory = self.env.getPortfolioHistory()
        return portHistory


    def plotPerformance(self):
        self.env.render()


    def plotPerformance(self):
        self.env.render()

class RLStrategy_DQN(Strategy):
    # This strategy is just buy and hold

    def __init__(self, envName = "PortfolioManagement-v0", **env_args):

        self.envName = envName
        self.envargs = env_args



    def train(self, save_dir = os.path.join(MODEL_DIR, "PortfolioManagement"), MAX_EPISODES = 2000, \
                DQNModel = None, hiddenUnits = [32],  batchNormalization = True, \
                dropout_rate = 0.25, optimizer_learning_rate = 1e-4, clipvalue = 100):

        Logger.info("Training with DQN agent ")

        networkArgs = {"Model": DQNModel, "hiddenUnits": hiddenUnits, "batchNormalization": batchNormalization, \
                        "dropout_rate" : dropout_rate, "optimizer_learning_rate": optimizer_learning_rate, "clipvalue": clipvalue}


        self.agent = DQNAgent.DQN(envName = self.envName, save_dir = save_dir, \
                                    networkArgs = networkArgs,  **self.envargs)

        self.agent.train(discount_factor=0.99, MAX_EPISODES = MAX_EPISODES, batch_size = 32)




    def run(self, modelWeights, hiddenUnits = [32], batchNormalization = True, dropoutRate = 0.25, \
                testPeriod = {"startDate": "2019-01-01", "endDate": "2019-12-31"}):

        # load teh environment

        self.envargs["startDate"] = testPeriod["startDate"]
        self.envargs["endDate"] = testPeriod["endDate"]

        self.env = CustomGym.make(self.envName, **self.envargs)

        model = DQNNet.NN_FF(state_size=self.env.observation_space.n, action_size=len(self.env.action_space.actions), \
                            hiddenUnits=hiddenUnits, batchNormalization=batchNormalization, dropout_rate=dropoutRate)

        
        # compile the model
        model(tf.convert_to_tensor(np.array(self.env.observation_space.currentState)[None, :], dtype = tf.float32))

        # load the weights
        model.load_weights(modelWeights)

        # run the environment
        self.env.reset()
        done = False
        currentState = self.env.observation_space.currentState
        forbidden_action_count = 0
        episodic_reward = 0
        episodic_steps = 0


        while not done:

            currentState = np.array(currentState).reshape(1, self.env.observation_space.n)
            actionValues = model(currentState)
            actionIndex = np.argmax(actionValues[0])        # greeedy action


            
            if self.env.observation_space.isactionForbidden(actionIndex = actionIndex, allActions = self.env.action_space.actions):
                forbidden_action_count += 1

                temp = []

                for index in range(len(self.env.action_space.actions)):

                    if not self.env.observation_space.isactionForbidden(actionIndex = actionIndex, allActions = self.env.action_space.actions):

                        temp.append(actionValues[0][index])

            else:
                temp = actionValues

            actionIndex = np.argmax(np.array(temp))                        

            # take step towards action
            action = self.env.action_space.actions[actionIndex]
            next_state, reward, done, _ = self.env.step(action)

            episodic_reward += reward
            episodic_steps += 1
            currentState = next_state

        print(f'Total episodic reward using DQN-VSM: {episodic_reward}. Forbidden score: {forbidden_action_count}')
        portHistory = self.env.getPortfolioHistory()
        return portHistory


    def plotPerformance(self):
        self.env.render()


# a = BuyandHoldStrategy()
# a.run()
 