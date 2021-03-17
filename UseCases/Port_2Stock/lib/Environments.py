import numpy as np
import pandas as pd
import os
from abc import ABC, abstractclassmethod
from logging.config import dictConfig
import logging
from importlib import reload
from configparser import ConfigParser
import collections

# custom
import constants
import loggingConfig
import Actions
import States
import utils
import ConfigReader

reload(Actions)
reload(States)
reload(constants)
reload(utils)

from Actions import DiscreteActionSpace
from States import PortfolioStateSpace
from ConfigReader import Config


dictConfig(loggingConfig.DEFAULT_LOGGING)
Logger = logging.getLogger("Environments")


DATA_DIR = constants.DATA_DIR




class Environment(ABC):
    def __init__(self):
        pass
    
    @abstractclassmethod
    def reset(self):
        pass
    
    @abstractclassmethod
    def step(self, action):
        pass
    
    

class Portfolio(Environment):
    
    def __init__(self, assets, initialWeight, \
                 #configFile = os.path.join(constants.LIB_PATH,"ENVIRON_CONFIG.ini"), \
                    nhistoricalDays = 20, \
                    actions = [(-0.1,0.1)], \
                    config = {"initialCash": constants.MILLION, "minCash": 0.02, "transactionFee": 0.0001}):        
        
        # sets up the portfolio environment
        
        assert np.sum(initialWeight) == 1.0, Logger.error("Asset weight should sum to 100%")        
        
        self.envName = "PORTFOLIO"
        self.assets = assets
        self.initialAssetWeights = initialWeight
        self.nbHistory = nhistoricalDays

        
        # load the data for the assets
        self.__loadData(assets)       

        # load configuration for environment
        
        #self.envConfig     = Config(configFile, Name=self.envName, Type = "Environment")        
        self.__readConfig(config)
                
        # set the action and state space for the environment
        
        
        # initialize state and action space
        self.action_space = DiscreteActionSpace(actions)

        
        # reset the portfolio environment and update the portfolio history
        self.reset()        

        
        # State contains --> 
        # 1. Historical Prices for "nhistoricalDays" days
        # 2. Current Units held for assets
        # 3. Current Portfolio value
        # 4. Current available cash
        # State space size: (n x nbAssets + nbAssets + 1 + 1)

        self.observation_space = PortfolioStateSpace(initialState = self.currentState)
        


                 
        
        
                 
    def __readConfig(self, config):

        # Environment configuration
        """
        self.initialCash        = int(utils.get_val(self.envConfig, tag = "INITIAL_CASH", default_value= constants.MILLION))
        self.minCash            = float(utils.get_val(self.envConfig, tag = "MIN_CASH", default_value= 0.02))
        self.transactionFee    = float(utils.get_val(self.envConfig, tag = "TRANSACTION_COST", default_value= 0.0002))
        """
        
        for key in config:
            setattr(self, key, config[key])
        

        
    def __loadData(self, assets):
        
        data = pd.DataFrame()
        for asset in assets:
            _path = os.path.join(DATA_DIR, f"{asset}.csv")
            _thisasset = pd.read_csv(_path, index_col = "Date")
            
            data = data.merge(_thisasset, how = "outer", left_index = True, right_index = True)
            
            # rename col
            data.rename(columns={'Adj Close': asset}, inplace=True)
            
        
        # fillNAs if any
        data = data.fillna(method = "ffill")
            
        self.HistoricalData = data
        Logger.info("Data Loaded!")
            
            
    def getcurrentState(self):
        
        # state consists of 
        # 1. Historical prices
        # 2. Portfolio holdings
        # 3. Current portfolio valur
        # 4. Current cash available
        # 5. Possible Actions
        
        
        if not hasattr(self, "_priceHistory"):
            __len = (len(self.assets) * self.nbHistory)

            self._priceHistory = [None]*__len
            self._priceHistory = collections.deque(self._priceHistory, maxlen = __len)
                    
        for nbIndex, assetPrice in enumerate(self.currentPrice):
            self._priceHistory.appendleft(assetPrice)
            
        # get possible actions based on current state
        # A_t = f(S_t)
        
        currentPossibleActions = self.__getPossibleActions() 
        
        # Need for normalization of State --> leading to exploding gradients

        state = list(self._priceHistory) + list(self.currentHoldings) + [self.currentPortfolioValue] + [self.currentCash] + currentPossibleActions
        
        self.currentState = state
        return state


    
    def __getPossibleActions(self):
        # No ShortSelling --> cant sell more than available stock
        # No negative Cash
        self.currentPossibleActions = []
        for action in self.action_space.actions:
            
            # based on previous portfolio value, take the action (at yday's close)
            cashValue = self.currentPortfolioValue * np.array(action)
            units_totransact = np.round(np.divide(np.array(cashValue), self.currentPrice),0)

            updatedHoldings = self.currentHoldings + units_totransact
            
            currentTransactionCost = np.round(np.sum(np.abs(cashValue)) * self.transactionFee,2)
            currentCash = self.currentCash -  np.round(currentTransactionCost,2)
            
            negative = sum([1 if item < 0 else 0 for item in updatedHoldings ])
            
            if negative > 0 or currentCash < 0:
                self.currentPossibleActions.append(0)
            else:
                self.currentPossibleActions.append(1)
                
            
        return self.currentPossibleActions

             
        
        
        
    def reset(self):
        """
        Resets the environment
        """
        
        # initialize the nb of Units to be purchased with initialCash amount.
        self.currentIndex = 0
        self.currentDate = self.HistoricalData.index[self.currentIndex]
        self.currentPrice = np.array(list(self.HistoricalData.iloc[self.currentIndex]))
        
        availableCash_forStocks = self.initialCash * (1 - self.minCash)
                         
        self.currentHoldings = np.round(np.divide(availableCash_forStocks * np.array(self.initialAssetWeights), self.currentPrice),0)
        
        assetsValue = np.multiply(self.currentHoldings, self.currentPrice)
        
        
        self.currentTransactionCost = np.round(np.sum(np.abs(assetsValue)) * self.transactionFee,2)         
        self.currentCash = np.round(self.initialCash - np.sum(assetsValue) - self.currentTransactionCost,2)
        self.currentPortfolioValue = np.round(np.matmul(self.currentPrice, self.currentHoldings.T) + self.currentCash,2)
        
        
        # required columns for portfolio History
        columns = ["Date"]
        for asset in self.assets:
            columns.append(f"Asset_{asset}")
        columns.append("Cash")
        columns.append("Cost")
        columns.append("AUM")
        self.__columns = columns
        
        # delete portfolio HIstory
        if hasattr(self, "portfolioHistory"): 
            delattr(self, "portfolioHistory")
        self.updateHistory()
        
        # update current State
        if hasattr(self, "_priceHistory"):
            delattr(self, "_priceHistory")
        self.currentState = self.getcurrentState()
    


        Logger.info("Environment reset")

        
                 
    def updateHistory(self):
                 
        if not hasattr(self, "portfolioHistory"):            
            self.portfolioHistory = pd.DataFrame(columns = self.__columns)
                 
                 
        # portfolio level Info
        port = [self.currentDate]
        for assetHoldings in self.currentHoldings:
            port.append(assetHoldings)
        
        port.append(self.currentCash)
        port.append(self.currentTransactionCost)         
        port.append(self.currentPortfolioValue)
        
        _thisPort = pd.DataFrame([port], columns = self.__columns)
        self.portfolioHistory = self.portfolioHistory.append(_thisPort,ignore_index=True)

        
                 
    
    def getLatestPortfolioInfo(self):
        # update portfolio value with Latest price
        self.currentIndex += 1
        self.currentDate = self.HistoricalData.index[self.currentIndex]
        self.currentPrice = np.array(list(self.HistoricalData.iloc[self.currentIndex]))

        self.currentPortfolioValue = np.matmul(self.currentPrice, self.currentHoldings.T) + self.currentCash

        
    
    
    def step(self, action, penalizefactor = 10):
        """
        # takes the action
        # Returns:
        # 1. Next State
        # 2. Reward after taking this step
        # 3. whether period over (dead)
        """
        
        episodeOver = False
        prevPortValue = self.currentPortfolioValue        
        Logger.debug(f"Action: {action}")
        
        
        # based on previous portfolio value, take the action (at yday's close)
        cashValue = self.currentPortfolioValue * np.array(action)
        units_totransact = np.round(np.divide(np.array(cashValue), self.currentPrice),0)
                 
        self.currentHoldings += units_totransact
                 
        # update cash for transaction cost
        self.currentTransactionCost = np.round(np.sum(np.abs(cashValue)) * self.transactionFee,2)
        self.currentCash -= np.round(self.currentTransactionCost,2)
        self.getLatestPortfolioInfo()  
        
        # get the return by updating the current portfolio value
        reward = (self.currentPortfolioValue - prevPortValue)/prevPortValue
        
        # penalize reward if action taken is forbidden
        _actionIndex = self.action_space.actions.index(action)
        if self.currentPossibleActions[_actionIndex] == 0:
            reward -= penalizefactor
        
        # update state
        newState = self.getcurrentState()
        self.observation_space.updateState(newState = newState)
        
        
        if self.currentIndex == len(self.HistoricalData) - 1:
            episodeOver = True
                 
        # update Portfolio History
        self.updateHistory()
        
        return self.observation_space.currentState, reward, episodeOver
        
        
        
                 
        
        
        
            
        
        
        
        
        
        
        
        
    
        