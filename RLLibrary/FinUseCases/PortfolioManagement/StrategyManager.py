import os, sys
import numpy as np
from importlib import reload
from abc import ABC, abstractclassmethod


# get the relative path
# fullpath                = os.path.realpath(__file__)
# pref                    = os.path.split(fullpath)[0]


# os.environ["RL_PATH"]   = f'{pref}/../../..'
# pref = f'{pref}/../../..'
# if f'{pref}/RLLibrary' not in sys.path:
#     sys.path.append(f'{pref}')
#     sys.path.append(f'{pref}/RLLibrary')


from RLLibrary.FinUseCases import CustomGym
from RLLibrary.FinUseCases.PortfolioManagement.ModelManager.A3C import Agent as A3CAgent
from RLLibrary.FinUseCases.PortfolioManagement import ActionManager

reload(ActionManager)
reload(A3CAgent)

from RLLibrary.utils import constants as constants
DATA_DIR = constants.DATA_DIR
MODEL_DIR = constants.MODEL_DIR

from RLLibrary.utils.loggingConfig import logger
Logger = logger.getLogger("StrategyManager")


# ------register PortfolioManagement environment onto Custom gym

CustomGym.register(
    id = "PortfolioManagement-v0",
    entry_point = 'FinUseCases.PortfolioManagement.EnvironmentManager:Portfolio',
    kwargs = {"assets" : ["APA", "BMY"], "initialWeight" : [0.5, 0.5], \
                    "nhistoricalDays" : 30, \
                    "startDate" : "2019-01-01", "endDate" : "2019-12-31", \
                    "actions" : [(-0.1,0.1)], \
                    "assetDataPath" : os.path.join(DATA_DIR, "PortfolioManagement"), \
                    "config" : {"initialCash": 1000000, "minCash": 0.02, "transactionFee": 0.0001}, 
                    "penalizeFactors" : {"Risk": -0.08, "ForbiddenAction": -8}})



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




    def run(self):

        Logger.info("Learning")

        currentState = self.env.reset()
        episodeOver = False
        while not episodeOver:
            action = list(self.actions[0])
            newstate, reward, episodeOver = self.env.step(action)
            currentState = newstate

        portHistory = self.env.getPortfolioHistory() 
        return portHistory


    def plotPerformance(self):
        self.env.render()



# a = BuyandHoldStrategy()
# a.run()
 