import numpy as np
import pandas as pd
import os
from importlib import reload
import collections
import matplotlib.pyplot as plt

# custom libraries using relative path
from RLLibrary.FinUseCases import Environment
from RLLibrary.FinUseCases.PortfolioManagement import DataManager, TradingManager, RewardManager, StateManager, ActionManager
#from . import Environment

reload(DataManager)
reload(TradingManager)
reload(RewardManager)
reload(StateManager)
reload(ActionManager)

from RLLibrary.utils import constants as constants
from RLLibrary.utils.loggingConfig import logger



Logger = logger.getLogger("Environments")
DATA_DIR = constants.DATA_DIR

class Dummy():
    def __init__(self):
        pass

class Portfolio(Environment):
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, assets, initialWeight, \
                    nhistoricalDays = 30, \
                    startDate = "2019-01-01", endDate = "2019-12-31", \
                    actions = [(-0.1,0.1)], \
                    assetDataPath = os.path.join(DATA_DIR, "PortfolioManagement"), \
                    config = {"initialCash": 1000000, "minCash": 0.02, "transactionFee": 0.0001}, 
                    penalizeFactors = {"Risk": -0.08, "ForbiddenAction": -8}):        


        # sets up the portfolio environment        
        assert np.sum(initialWeight) == 1.0, Logger.error("Asset weight should sum to 100%")        
        
        self.envName = self.__class__.__name__
        self.assets = assets
        self.initialAssetWeights = initialWeight
        self.nbHistory = nhistoricalDays

        self.startDate = startDate
        self.endDate = endDate


        
        # load the data for the assets
        self.DataManager = DataManager.DataMatrices(assets = assets, \
                                                    startDate=startDate, endDate = endDate, \
                                                    filepath=assetDataPath)

        self.DataManager.loadData()
        Logger.info("Data Loaded")

        
        # initialize state and action space
        self.action_space       = ActionManager.DiscreteActionSpace(actions)
        self.observation_space  = StateManager.StateSpace(initialState= None)


        # set up trading and reward managers
        self.TradingManager = TradingManager.Trader(TransactionFee=config["transactionFee"])
        self.RewardManager  = RewardManager.RewardCategory1(riskPenalizeFactor=penalizeFactors["Risk"], \
                                                            actionPenalizeFactor=penalizeFactors["ForbiddenAction"])
        
        # reset the portfolio environment and update the portfolio history
        self.config = Dummy()
        for key in config:
            setattr(self.config, key, config[key])

        self.reset()       


                 
        
        
    def reset(self, ):
        """
        Resets the environment
        """


        __initialCashforStocks = self.config.initialCash * (1 - self.config.minCash)
        
        # initialize the nb of Units to be purchased with initialCash amount.
        self.currentInfo = Dummy()
        self.currentInfo.index  = self.nbHistory         # Index starts at 0  

        self.currentInfo.Date   = self.DataManager.dates[self.currentInfo.index]

        # get Price for this date  (contains Close/ High/ Low and returns for Close)
        _currentData = self.DataManager.getDataForDate(self.currentInfo.Date)
        if _currentData is None:
            Logger.error(f"No Price info found for Date: {self.currentInfo.Date}")
            raise Exception(f"No Price info found for Date: {self.currentInfo.Date}")

        self.currentInfo.Price = Dummy()
        self.currentInfo.Price.Close    = _currentData[0, :, 0]   #  this close is adjusted close
        self.currentInfo.Price.High     = _currentData[0, :, 1]
        self.currentInfo.Price.Low      = _currentData[0, :, 2]

        self.currentInfo.Return         = _currentData[0, :, 3]

        # get current Holdings
        self.currentInfo.Holdings = np.divide(__initialCashforStocks * np.array(self.initialAssetWeights), self.currentInfo.Price.Close)                
        self.currentInfo.Holdings = np.round(self.currentInfo.Holdings,0)
        self.initialHoldings = self.currentInfo.Holdings.copy()

        
        # perform Trade and compute Transaction cost
        updatedCash, transactionCost = self.TradingManager.performTrade(currentCash=self.config.initialCash, \
                                            price = self.currentInfo.Price.Close, \
                                            units = self.currentInfo.Holdings)


        self.currentInfo.Cost = transactionCost
        self.currentInfo.Cash = updatedCash

        self.initialCash = self.currentInfo.Cash.copy()
        
        # get updated portfolio value
        self.currentInfo.Portfolio = np.matmul(self.currentInfo.Price.Close, self.currentInfo.Holdings.T) + self.currentInfo.Cash
        self.currentInfo.Portfolio = np.round(self.currentInfo.Portfolio,2)
        

        # ------------ update Portfolio history  
        self.History = Dummy()   
                
        # update portfolio HIstory
        if hasattr(self.History, "Portfolio"): 
            delattr(self.History, "Portfolio")
        self.updatePortfolioHistory()
        
        # get current State
        if hasattr(self.History, "Returns"):
            delattr(self.History, "Returns")
    
        # ------- update feasible actions as well
        currentState = self.getcurrentState()
        #self.getFeasibleActions()



        Logger.info("Environment reset")    

        
    def updatePortfolioHistory(self):

        if not hasattr(self.History, "Portfolio"): 
            self.History.Portfolio = []

                 
        # portfolio level Info
        _currentPort = [self.currentInfo.Date]
        for assetHoldings in self.currentInfo.Holdings:
            _currentPort.append(assetHoldings)
        
        _currentPort.append(self.currentInfo.Cash)
        _currentPort.append(self.currentInfo.Cost)         
        _currentPort.append(self.currentInfo.Portfolio)
        
        self.History.Portfolio.append(_currentPort)



        
    def getPortfolioHistory(self):
        # get historical portfolio data in a dataframe
        # required columns for portfolio History
        columns = ["Date"]
        for asset in self.assets:
            columns.append(f"Asset_{asset}")
        columns.append("Cash")
        columns.append("Cost")
        columns.append("AUM")

        _portHistory = pd.DataFrame(self.History.Portfolio, columns = columns)
        _portHistory["Date"] = pd.to_datetime(_portHistory["Date"]).dt.date

        return _portHistory


               
    def getcurrentState(self):
        
        # state consists of 
        # 1. Close, High, Low prices for all assets for Historical days --> nxmx3 matrix  (m--> # assets, n --> Historical days)
        # 2. asset returns for past n days  --> nxmx1
        # 2. asset current holdings       --> mx1
        # 4. Current cash available
        # 5. Possible Actions


        # get Asset Price and Returns
        if not hasattr(self.History, "Price"):

            # get historical price & returns data
            _historicalData, _historicalDates = self.DataManager.getDataForRange(startIndex=1, endIndex=self.nbHistory) 

            self.History.Price      = collections.deque(maxlen = self.nbHistory)            
            self.History.Returns    = collections.deque(maxlen = self.nbHistory)

            for index in range(_historicalData.shape[0]):
                self.History.Price.append(_historicalData[index, :, :3])     # Close, High, Low
                self.History.Returns.append(_historicalData[index, :, 3])    # Returns (based on close)
        
        else:
            # only append the current Info
            _currentData = self.DataManager.getDataForDate(self.currentInfo.Date)
            self.History.Price.append(_currentData[0,:,:3])
            self.History.Returns.append(_currentData[0,:,3])



        # get feasible actions based on current portfolio Holdings and Cash
        self.getFeasibleActions()


        state_price     = self.History.Price
        state_return    = self.History.Returns
        state_holding   = self.currentInfo.Holdings/ self.initialHoldings           # normalize holdings
        state_cash      = self.currentInfo.Cash/ self.initialCash                   # normalize Cash
        state_actions   = self.currentInfo.PossibleActions

        currentStatetuple    = (state_price, state_return, state_holding, state_cash, state_actions)

        # update the observation space
        self.observation_space.update(currentState=currentStatetuple)
        currentState    = self.observation_space.currentState 
        
        return currentState
        
        # # update the observation space


    def getFeasibleActions(self)    :
        # No ShortSelling --> cant sell more than available stock
        # No negative Cash
        self.currentInfo.PossibleActions = []

        for action in self.action_space.actions:

            updatedHoldings, updatedCash = self._performTrade(action, updateCurrentInfo = False)

            # check for negative holding
            negative = sum([1 if item < 0 else 0 for item in updatedHoldings ])
            
            if negative > 0 or updatedCash < 0:
                self.currentInfo.PossibleActions.append(0)       # Not feasible action
            else:
                self.currentInfo.PossibleActions.append(1)      # fasible action
                

    
    def step(self, action, penalizefactor = 10, clipReward = True, clipRange = [-5,5]):
        """
        # takes the action
        # Returns:
        # 1. Next State
        # 2. Reward after taking this step
        # 3. whether period over (dead)
        """
        
        episodeOver = False
        prevPortValue = self.currentInfo.Portfolio
        Logger.debug(f"Action: {action}")
        
        
        # step 1: based on action, perform trading activity
        updatedHoldings, updatedCash = self._performTrade(action, updateCurrentInfo = True)

        # step 2: compute reward (a scalar quantity)
        currentReturn   = np.log(self.currentInfo.Portfolio/ prevPortValue)

        actionIndex     = self.action_space.actions.index(action)
        actionFeasible  = True if self.currentInfo.PossibleActions[actionIndex] == 1 else False

        reward = self.RewardManager.computeReward(currentReturn = currentReturn, actionFeasible=actionFeasible)


        if self.currentInfo.index == len(self.DataManager.dates)-1:
            episodeOver = True

        # step 3: update the state
        newState = self.getcurrentState()
        #newState = self.observation_space.currentState

        return newState, reward, episodeOver



    def render(self):
        # renders the current state of environment
        portHistory = self.getPortfolioHistory()
        portHistory["Date"] = pd.to_datetime(portHistory["Date"]).dt.date
        plt.figure(figsize = (10,6))
        plt.plot(portHistory["Date"], portHistory["AUM"])


    def _performTrade(self, action, updateCurrentInfo = True):
        # based on the chosen action, perform Trading Activity

        cashValue       = self.currentInfo.Portfolio * np.array(action)
        units_to_transact = np.round(np.divide(np.array(cashValue), self.currentInfo.Price.Close), 0)
        

        # get revised holdigs
        updatedHoldings = self.currentInfo.Holdings + units_to_transact

        updatedCash, transactionCost = self.TradingManager.performTrade(currentCash=self.currentInfo.Cash, \
                                            price = self.currentInfo.Price.Close, \
                                            units = units_to_transact)



        if updateCurrentInfo:

            # update portfolio info
            self.currentInfo.index += 1

            self.currentInfo.Holdings = updatedHoldings
            self.currentInfo.Cash = updatedCash
            self.currentInfo.Cost = transactionCost

            self.currentInfo.Date = self.DataManager.dates[self.currentInfo.index]

            _currentData = self.DataManager.getDataForDate(self.currentInfo.Date)
            # get Price for this date  (contains Close/ High/ Low and returns for Close)
            if _currentData is None:
                Logger.error(f"No Price info found for Date: {self.currentInfo.Date}")
                raise Exception("No Price info found for Date: {self.currentInfo.Date}")

            self.currentInfo.Price.Close    = _currentData[0, :, 0]   #  this close is adjusted close
            self.currentInfo.Price.High     = _currentData[0, :, 1]
            self.currentInfo.Price.Low      = _currentData[0, :, 2]

            self.currentInfo.Return         = _currentData[0, :, 3]

            self.currentInfo.Portfolio = np.matmul(self.currentInfo.Price.Close, self.currentInfo.Holdings.T) + self.currentInfo.Cash
            self.currentInfo.Portfolio = np.round(self.currentInfo.Portfolio,2)

            # update portfolio Histroy
            self.updatePortfolioHistory()



        return updatedHoldings, updatedCash




        

        
        
                 
        
        
        
            
        
        
        
        
        
        
        
        
    
        