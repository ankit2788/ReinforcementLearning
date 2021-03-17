import numpy as np
import pandas as pd
import os, sys
from abc import ABC, abstractclassmethod
from logging.config import dictConfig
import logging
from importlib import reload
from configparser import ConfigParser
import collections

# custom
import loggingConfig


LIB_PATH = os.environ["porto"]
LOG_PATH = f'{LIB_PATH}/../logs/'
DATA_PATH = f'{LIB_PATH}/../data/'


logname = os.path.join(LOG_PATH, "Logging.Log")
logging.basicConfig(filename = logname, 
                   filemode = "a", 
                   format = "%(asctime)s %(levelname)-8s %(name)-15s %(message)s",
                   datefmt =  "%Y-%m-%d %H:%M:%S",
                   level = logging.INFO)

                    
Logger = logging.getLogger("Environments")



class DiscreteActionSpace():
    def __init__(self, actions):
        # actions --> list of all actions
        self.actions = actions
        self.n = len(actions)
        
class StateSpace():
    def __init__(self, initialState = None, actionSpaceLength = 7):
        self.currentState = initialState
        self.initialState = initialState
        self.previousState = None
        
        self.actionSpaceLength = actionSpaceLength
        self.n = None
        
        
    def setinitialState(self, initialState):
        self.initialState = initialState
        
    
    def updateState(self, newState, actionSpaceLength = None, normalize = True):
        
        actionSpaceLength = self.actionSpaceLength if actionSpaceLength is None else actionSpaceLength
        
        if self.initialState is None:
            self.initialState = newState
            
        
        self.previousState = self.currentState
        self.currentState = newState
        
        self.n = len(self.currentState)
        if normalize:
            self.normalizedCurrentState = self.normalize(self.currentState, actionSpaceLength)
        else:
            self.normalizedCurrentState = self.currentState

    
    def normalize(self, currentState, actionSpaceLength):

        # normalize by dividing by initial value
        normalizedState = np.divide(np.array(currentState[:-actionSpaceLength]), np.array(self.initialState[:-actionSpaceLength]))
        normalizedState = np.nan_to_num(normalizedState)

        normalizedState = list(normalizedState) + currentState[len(currentState)-actionSpaceLength]
        return normalizedState
    
        


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
    
    def __init__(self, assets, initialWeights, \
                    nbhistoricalDays = 30, \
                    startDate = "2019-01-01", endDate = "2019-12-31", \
                    actions = [(-0.1,0.1)], \
                    normalizeState = True, includeActionsinState = True, 
                    config = {"initialCash": 1000000, "minCash": 0.02, "transactionFee": 0.0000}):        
        
        # sets up the portfolio environment
        
        
        self.envName = self.__class__.__name__
        self.assets = assets
        self.initialWeights = initialWeights
        self.nbhistoricalDays = nbhistoricalDays

        
        # load the data for the assets
        self.__loadData(assets, startDate, endDate)       
        self.blnNormalizeState = normalizeState

        # load configuration for environment
        for key in config:
            setattr(self, key, config[key])
        
        
        # initialize state and action space
        self.action_space = DiscreteActionSpace(actions)
        self.observation_space = StateSpace(actionSpaceLength=self.action_space.n)

        
        # reset the portfolio environment and update the portfolio history
        self.reset(includeActionsinState)        

        
        
    def __loadData(self, assets, startDate = None, endDate = None):
        
        data = pd.DataFrame()
        for asset in assets:
            _path = os.path.join(DATA_PATH, f"{asset}.csv")
            _thisData = pd.read_csv(_path, index_col = "Date")

            if startDate is not None and endDate is not None:
                _thisData = _thisData[(_thisData.index >= startDate) & (_thisData.index <= endDate)]

            # get return
            _thisData["Return"] = _thisData["Adj Close"].pct_change() * 100

            
            data = data.merge(_thisData, how = "outer", left_index = True, right_index = True)
            
            # rename col
            data.rename(columns={'Adj Close': f'Price_{asset}', "Return": f'Ret_{asset}'}, inplace=True)

        
        # fillNAs if any
        data = data.fillna(method = "ffill")
            
        self.HistoricalData = data
        Logger.info("Data Loaded!")


    def reset(self, includeActionsinState = True):
        """
        Resets the environment
        """
        
        self._priceCols = [col for col in self.HistoricalData.columns if "Price" in col]
        self._returnCols = [col for col in self.HistoricalData.columns if "Ret" in col]



        # initialize the nb of Units to be purchased with initialCash amount.
        self.currentIndex = self.nbhistoricalDays
        self.currentDate = self.HistoricalData.index[self.currentIndex]
        self.currentPrice = np.array(list(self.HistoricalData[self._priceCols].iloc[self.currentIndex]))


        
        initialCashforStocks = self.initialCash * (1 - self.minCash)

                         
        self.currentHoldings = np.divide(initialCashforStocks * np.array(self.initialWeights), self.currentPrice)
        self.currentHoldings = np.round(self.currentHoldings, 0)
        self.initialHoldings = self.currentHoldings.copy()


        
        assetsValue = np.multiply(self.currentHoldings, self.currentPrice)
        
        cost = np.sum(np.abs(assetsValue)) * self.transactionFee
        self.currentTransactionCost = np.round(cost,2)         

        self.currentCash = np.round(self.initialCash - np.sum(assetsValue) - cost,2)
        self.initialCash = self.currentCash.copy()


        self.currentPortfolioValue = np.round(np.matmul(self.currentPrice, self.currentHoldings.T) + self.currentCash,2)
        
        
        if hasattr(self, "portHistory"):
            deattr(self, "portHistory")
        self.updatePortfolioHistory()


        if hasattr(self, "returnHistory"):
            delattr(self, "returnHistory")
        self.updateState(includeActionsinState)

    


        Logger.info("Environment reset")        

                 
    def updatePortfolioHistory(self):
                 
        if not hasattr(self, "portfolioHistory"):   
            self.portfolioHistory = []         
                 
                 
        # portfolio level Info
        port = [self.currentDate]
        for assetHoldings in self.currentHoldings:
            port.append(assetHoldings)
        
        port.append(self.currentCash)
        port.append(self.currentTransactionCost)         
        port.append(self.currentPortfolioValue)

        self.portfolioHistory.append(port)
        

    def getPortfolioHistory(self)            :

        # required columns for portfolio History
        columns = ["Date"]
        for asset in self.assets:
            columns.append(f"Asset_{asset}")
        columns.append("Cash")
        columns.append("Cost")
        columns.append("AUM")

        portHistory = pd.DataFrame(self.portfolioHistory, columns = columns)
        portHistory["Date"] = pd.to_datetime(portHistory["Date"]).dt.date

        return portHistory


    def updateState(self, includeActionsinState = True):
        # state consists of 
        # 1. asset returns for last n days
        # 2. asset current holdings
        # 3. Current cash available
        # 4. Possible Actions

        if not hasattr(self, "returnHistory"):
            __len = (len(self.assets) * self.nbhistoricalDays)

            self.returnHistory = [None]*__len
            self.returnHistory = collections.deque(self.returnHistory, maxlen = __len)

            for histIndex in range(self.nbhistoricalDays):
                histReturn = self.HistoricalData[self._returnCols].iloc[histIndex + 1].values

                for ret in np.flipud(histReturn):
                    self.returnHistory.appendleft(ret)


        else:
            histReturn = self.HistoricalData[self._returnCols].iloc[self.currentIndex].values
            for ret in np.flipud(histReturn):
                self.returnHistory.appendleft(ret)
                    
            
        # get possible actions based on current state
        # A_t = f(S_t)
        
        state = list(self.returnHistory) + list(np.divide(self.currentHoldings, self.initialHoldings)) +  [self.currentCash/ self.initialCash] 
        if includeActionsinState:
            self.getFeasibleActions()
            state += list(self._currentPossibleActions)


        self.observation_space.updateState(newState= state, normalize=self.blnNormalizeState)        





    
    def getFeasibleActions(self):
        # No ShortSelling --> cant sell more than available stock
        # No negative Cash
        self._currentPossibleActions = []
        for action in self.action_space.actions:

            updatedHoldings, updatedCash = self._performTrade(action, updateCurrentInfo = False)

            
            # check for negative holdings            
            negative = sum([1 if item < 0 else 0 for item in updatedHoldings ])
            
            if negative > 0 or updatedCash < 0:
                self._currentPossibleActions.append(0)
            else:
                self._currentPossibleActions.append(1)


    def isactionForbidden(self, actionIndex):
        actionPermits = self.observation_space.currentState[(self.observation_space.n - self.action_space.n):]
        if actionPermits[actionIndex] == 1:
            return False
        else:
            return True


    
    def step(self, action, penalizefactor = 10, tradingReward = 0.01, clipReward = True, clipRange = [-5,5]):
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

        # step 1: update cash, holdings and portfolio
        updatedHoldings, updatedCash = self._performTrade(action)

        # step 2: get reward
        reward = self._computeReward(prevPortValue, action, penalizefactor, tradingReward   )
        if clipReward:
            reward = np.clip(reward, clipRange[0], clipRange[1])

        
        if self.currentIndex == len(self.HistoricalData) - 1:
            episodeOver = True


        # Step 3: update State
        self.updateState()
        newState = self.observation_space.currentState

        
        return newState, reward, episodeOver
        
    
        
        
                 
        
        
    def _performTrade(self, action, updateCurrentInfo = True):
        # based on action chosen, perform trading activity

        cashValue = self.currentPortfolioValue * np.array(action)
        units_totransact = np.round(np.divide(np.array(cashValue), self.currentPrice),0)

        updatedHoldings = self.currentHoldings + units_totransact         

        effectiveCashValue = np.multiply(units_totransact, self.currentPrice)
        cost = np.round(np.sum(np.abs(effectiveCashValue)) * self.transactionFee,2)

        updatedCash = self.currentCash - np.sum(effectiveCashValue) - cost

        if updateCurrentInfo:
            self.currentHoldings = updatedHoldings
            self.currentCash = updatedCash

            # update portfolio info
            self.currentIndex += 1
            self.currentDate = self.HistoricalData.index[self.currentIndex]
            self.currentPrice = np.array(list(self.HistoricalData[self._priceCols].iloc[self.currentIndex]))

            self.currentPortfolioValue = np.round(np.matmul(self.currentPrice, self.currentHoldings.T) + self.currentCash, 2)

            self.updatePortfolioHistory()

        return updatedHoldings, updatedCash


    def _computeReward(self, prevPortValue, action, penalizefactor = 10, tradingReward = 0.01):

        # get the return by updating the current portfolio value
        immediateReward_1 = 100 * (self.currentPortfolioValue - prevPortValue)/prevPortValue

        # penalize reward if action taken is forbidden
        immediateReward_2 = 0
        _actionIndex = self.action_space.actions.index(action)
        if self._currentPossibleActions[_actionIndex] == 0:
            immediateReward_2 = -1 * penalizefactor

        reward = immediateReward_1 + immediateReward_2

        return reward



    def plotPortfolio(self):
        portfolio = self.getPortfolioHistory()

        if len(portfolio) <= 10:
            print(f'Portfolio History not long enough')
            sys.exit(0)

        fig, ax = plt.subplots(1,1,figsize = (10,5))
        ax.plot(portfolio["Date"], portfolio["AUM"])
        ax.set_title("Portfolio")

        
        
        
        
        
        
        
    
        