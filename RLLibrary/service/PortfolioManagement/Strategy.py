import os
import numpy as np
from importlib import reload


from RLLibrary.FinUseCases.PortfolioManagement import EnvironmentManager

from RLLibrary.utils import constants as constants
from RLLibrary.utils.loggingConfig import logger

from RLLibrary.FinUseCases import CustomGym

reload(EnvironmentManager)


Logger = logger.getLogger("Strategy")
DATA_DIR = constants.DATA_DIR


# resgiter the environment


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
                    "penalizeFactors" : {"Risk": -0.08, "ForbiddenAction": -8}}


)



class BuyandHoldStrategy():
    # This strategy is just buy and hold

    def __init__(self, assets, initialWeight, \
                    nhistoricalDays = 30, \
                    actions = None, \
                    startDate = "2019-01-01", endDate = "2019-12-31", \
                    assetDataPath = os.path.join(DATA_DIR, "PortfolioManagement"), \
                    config = {"initialCash": 1000000, "minCash": 0.02, "transactionFee": 0.0001}, 
                    penalizeFactors = {"Risk": -0.08, "ForbiddenAction": -8}):

        
        # create the environment
        self.actions = actions if actions is not None else [[0]*len(assets)]

        env_args = {"assets": assets,  "initialWeight" : initialWeight, \
                    "nhistoricalDays" : nhistoricalDays, \
                    "startDate" : startDate, "endDate" : endDate, \
                    "actions" : self.actions, \
                    "assetDataPath" : assetDataPath, \
                    "config" : config, 
                    "penalizeFactors" : penalizeFactors

                    }


        self.env = CustomGym.make(id = "PortfolioManagement-v0", **env_args)




    def run(self):

        Logger.info("Running Equi weighted Strategy")

        currentState = self.env.reset()
        episodeOver = False
        while not episodeOver:
            action = list(self.actions[0])
            #print(action, type(action))
            newstate, reward, episodeOver = self.env.step(action)
            currentState = newstate

        portHistory = self.env.getPortfolioHistory() 
        return portHistory


    def plotPerformance(self):
        self.env.render()





class Strategy_A3C():
    # This strategy is just buy and hold

    def __init__(self, assets, initialWeight, \
                    nhistoricalDays = 30, \
                    actions = None, \
                    startDate = "2019-01-01", endDate = "2019-12-31", \
                    assetDataPath = os.path.join(DATA_DIR, "PortfolioManagement"), \
                    config = {"initialCash": 1000000, "minCash": 0.02, "transactionFee": 0.0001}, 
                    penalizeFactors = {"Risk": -0.08, "ForbiddenAction": -8}):

        
        # create the environment
        if actions is None:
            Logger.error("Please provide discrete action space") 
            raise Exception("Please provide discrete action space")
        self.actions = actions if actions is not None else [[0]*len(assets)]

        self.env = EnvironmentManager.Portfolio(assets, initialWeight, \
                    nhistoricalDays = nhistoricalDays, \
                    startDate = startDate, endDate = endDate, \
                    actions = self.actions, \
                    assetDataPath = assetDataPath, \
                    config = config, \
                    penalizeFactors = penalizeFactors)




    def run(self):

        Logger.info("Learning")

        currentState = self.env.reset()
        episodeOver = False
        while not episodeOver:
            action = list(self.actions[0])
            #print(action, type(action))
            newstate, reward, episodeOver = self.env.step(action)
            currentState = newstate

        portHistory = self.env.getPortfolioHistory() 
        return portHistory


    def plotPerformance(self):
        self.env.render()



        