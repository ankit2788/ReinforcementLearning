import numpy as np
import pandas as pd
import os, sys
from importlib import reload
import collections
import matplotlib.pyplot as plt
import datetime
from abc import ABC, abstractclassmethod
from copy import copy

# custom libraries using relative path
from RLLibrary.FinUseCases import Environment
from RLLibrary.FinUseCases.OrderExecution import DataManager, ActionManager, RewardManager, StateManager
#from . import Environment

reload(DataManager)
reload(ActionManager)
reload(RewardManager)
reload(StateManager)

from RLLibrary.utils import constants as constants
from RLLibrary.utils.loggingConfig import logger



Logger = logger.getLogger("Environments")
DATA_DIR = constants.DATA_DIR

class Dummy():
    def __init__(self):
        pass


TIMEZONE = {
    "IST": 5.5,     # +5:0 hours from GMT
    "GMT": 0
}

class OrderExecution(Environment):
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, ticker, \
                    orderConfig = {"initialOrderSize": 10000, "initialTimeHorizon": 100, "orderFactor": 100, \
                                    "TotalIntervals": 50, "startTime": "09:30", "Timezone": "IST"}, \
                    
                    nbHistory = 15, \
                    trainingYear = [], testDate = None, \
                    dataPath = os.path.join(DATA_DIR, "OrderExecution"), \
                    penalizeFactors = {"Impact": -5e-6, "StepReward": 0.1}):        


        """
        Assumption: 
        All orders sent are assumed to be market orders. 
        In this set of environment, we are not dealing with LOB (Limit order book) due to lack of data availability.
        """

        # sets up the portfolio environment        
        
        self.envName = self.__class__.__name__
        self.ticker = ticker

        self.nbHistory = nbHistory      # represents history (in minutes)


        # load the data for the assets
        self.DataManager = DataManager.DataMatrices(ticker =self.ticker, \
                                                    filepath=dataPath)
        if len(trainingYear) == 0:
            if testDate is not None:
                # env set for test mode
                year = [str(testDate[:4])]
                self.DataManager.loadData(years = year)         # load data for specific Date only

                # Also load the Volume statistics 
                self.DataManager.getVolumeStats()

            else:
                Logger.error("Testing period or Training period needs to be provided")
                sys.exit(0)
        else:
            self.DataManager.loadData(years = trainingYear)         

            # Also load the Volume statistics 
            self.DataManager.getVolumeStats()


        self.trainingYears  = trainingYear

        
        # initial values
        self.initialOrderSize   = orderConfig["initialOrderSize"]
        self.orderSizeFactor    = orderConfig["orderFactor"]                # Tradable unit 
        self.initialTimeHorizon = orderConfig["initialTimeHorizon"]         # in minutes
        self.TotalIntervals     = orderConfig["TotalIntervals"]             
        self.TimeGapbetweenIntervals = self.initialTimeHorizon/ self.TotalIntervals

        self.executionStartTime = orderConfig["startTime"]
        self.timezone           = orderConfig["Timezone"]


        # initialize state and action space

        # action space: units to execute (in multiples of orderSizeFactor)
        self.action_space       = ActionManager.DiscreteActionSpace(actions = list(np.arange(int(self.initialOrderSize/ self.orderSizeFactor)+1)))
        self.observation_space  = StateManager.StateSpace(initialState= None)


        # # set up reward managers
        self.RewardManager  = RewardManager.RewardCategory(impactPenalizeFactor=penalizeFactors["Impact"], \
                                                    StepRewardFactor=penalizeFactors["StepReward"])
        
        # reset the portfolio environment and update the portfolio history
        self.reset()       


                 
        
        
    def reset(self, resetDate = None):
        """
        Resets the environment
        """

        if resetDate is None:
            # This is training mode
            # randomly select a date from the available data

            allDates = []
            for year in self.trainingYears:
                allDates += list(self.DataManager.dates[year])

            # randomly choose the date from the list
            resetDate = np.random.choice(allDates)


        # get the data for this date
        year            = str(resetDate)[:4]

        self.envDate    = resetDate
        self.envData    = self.DataManager.data[year][str(resetDate)]       # tuple of times, prices, volumes (per second)
        #Logger.info(f"Env set for {resetDate}")

        # _beginning_epoch    = datetime.datetime(1970, 1, 1)        # epoch time computation starts from here
        # _resetTime          = f"{str(resetDate)} {self.executionStartTime}"
        # _myformat           = "%Y%m%d %H:%M:%S"

        # _myDate             = datetime.datetime.strptime(_resetTime, _myformat)

        # #_startTime          = (_myDate - _beginning_epoch).total_seconds() - TIMEZONE[self.timezone]*60*60      # epoch time



        # initialize the nb of Units to be purchased with initialCash amount.
        self.currentInfo = Dummy()
        self.currentInfo.Time           = self.executionStartTime
        self.currentInfo.OpenInventory  = self.initialOrderSize
        self.currentInfo.TimeLeft       = self.initialTimeHorizon  # minutes
        self.currentInfo.executedInventory   = 0
        self.currentInfo.executedPrice  = np.nan
        self.currentInfo.AvailableInventory  = self.currentInfo.OpenInventory - self.currentInfo.executedInventory

        _tempTimes = list(self.envData[0].reshape(self.envData[0].shape[0],))
        self.currentInfo.Index          = _tempTimes.index(self.currentInfo.Time)

        self.currentInfo.Price = Dummy()
        self.currentInfo.Price.Open = self.envData[1][self.currentInfo.Index][0]
        self.currentInfo.Price.High = self.envData[1][self.currentInfo.Index][1]
        self.currentInfo.Price.Low = self.envData[1][self.currentInfo.Index][2]
        self.currentInfo.Price.Close = self.envData[1][self.currentInfo.Index][3]

        self.currentInfo.Volume     = self.envData[2][self.currentInfo.Index][0]

        self.initialClosePrice = self.envData[1][0][0]


        
        
        # ------------ update Portfolio history  
        self.History = Dummy()   
                
        # update portfolio HIstory
        if hasattr(self.History, "Inventory"): 
            delattr(self.History, "Inventory")
        self.updateInventoryHistory()
        
        # get current State
        currentState = self.getcurrentState()



    def updateInventoryHistory(self):

        if not hasattr(self.History, "Inventory"): 
            self.History.Inventory = []

                 
        # Inventory Info
        _currentInfo = [self.currentInfo.Time]

        _currentInfo.append(self.currentInfo.OpenInventory)
        _currentInfo.append(self.currentInfo.TimeLeft)         
        _currentInfo.append(self.currentInfo.executedInventory)
        _currentInfo.append(self.currentInfo.executedPrice)
        _currentInfo.append(self.currentInfo.AvailableInventory)

        self.History.Inventory.append(_currentInfo)



        
    def getInventoryHistory(self):
        # get historical inventory data in a dataframe
        # required columns for portfolio History
        columns = ["Time"]
        columns.append("OpenInventory")
        columns.append("TimeLeft")
        columns.append("ExecutedInventory")
        columns.append("ExecutedPrice")
        columns.append("AvailableInventory")

        _Inventory = pd.DataFrame(self.History.Inventory, columns = columns)

        return _Inventory


       
               
    def getcurrentState(self):
        
        # state consists of 
        # 1. Close for past n minutes --> nx1 matrix  (n --> Historical minutes) 
        # 2. market Traded volume for past n minutes  --> nx1
        # 2. Open Inventory       --> 1
        # 4. Time Left --> 1



        # get Asset Price and Returns
        _price, _volume = self.envData[1], self.envData[2]
        if not hasattr(self.History, "Price"):

            # get historical price & returns data
            if self.nbHistory >= self.currentInfo.Index + 1:
                _index = self.currentInfo.Index + 1
            else:
                _index = self.nbHistory

            self.History.Price      = collections.deque(maxlen = self.nbHistory)            
            self.History.Volume    = collections.deque(maxlen = self.nbHistory)

            for index in range(_index):
                self.History.Price.append(_price[index][-1])     # only Close price
                self.History.Volume.append(_volume[index][0])    # Volume
        
        else:
            self.History.Price.append(_price[self.currentInfo.Index][-1])  # only Close price
            self.History.Volume.append(_volume[self.currentInfo.Index][0])



        state_price     = self.History.Price
        state_volume    = (self.History.Volume - self.DataManager.VolumeStats.mean)/ self.DataManager.VolumeStats.std       # normalized volume
        state_openInv   = self.currentInfo.AvailableInventory/ self.initialOrderSize     # proportion left
        state_timeleft  = self.currentInfo.TimeLeft/self.initialTimeHorizon           # proportion time left

        # get price tensor normalized with market open
        _initialOpen    = self.envData[1][0][0]     # at market open, what is the OPEN price
        state_price     = np.divide(np.array(state_price), _initialOpen)

        currentState    = list(state_price) + list(state_volume) + [state_openInv] + [state_timeleft]

        # update the observation space
        self.observation_space.update(currentState=currentState)
        currentState    = self.observation_space.currentState 
        
        return currentState
        
    
    def step(self, action):
        """
        # takes the action --> executionSize for the order
        # Returns:
        # 1. Next State
        # 2. Reward after taking this step
        # 3. whether period over (dead)
        """
        
        episodeOver = False
        Logger.debug(f"Action: {action}")
        
        prevPrice = copy(self.currentInfo.Price.Close)

        # step 1: based on action, reduce the order inventory and time left
        inventoryLeft, timeLeft = self._performExecution(action, updateCurrentInfo = True)

        # step 2: compute reward (a scalar quantity)
        
        reward = self.RewardManager.computeReward(deltaPrice = (self.currentInfo.Price.Close - prevPrice)/self.initialClosePrice, \
                                                    executedSize = action * self.orderSizeFactor, currentTradedVolume =self.currentInfo.Volume)

        if self.currentInfo.AvailableInventory <= 0 or self.currentInfo.TimeLeft == 0:
            episodeOver = True

        # step 3: update the state
        newState = self.getcurrentState()
        #newState = self.observation_space.currentState

        return newState, reward, episodeOver, None



    def render(self):
        # renders the current state of environment
        invHistory = self.getInventoryHistory()
        plt.figure(figsize = (10,6))
        plt.plot(invHistory["Time"], invHistory["AvailableInventory"])


    def _performExecution(self, action, updateCurrentInfo = True):
        # based on the chosen action, perform Trading Activity


        self.currentInfo.OpenInventory  = self.currentInfo.AvailableInventory        # current open = previous close

        maxAction       = int(self.currentInfo.OpenInventory/ self.orderSizeFactor)
        action          = min(action, maxAction)

        # print("Action", action)

        inventoryLeft   = self.currentInfo.OpenInventory - action * self.orderSizeFactor
        timeLeft        = self.currentInfo.TimeLeft - 1         # represents 1 minute    (# minimum time)


        if updateCurrentInfo:

            # update portfolio info
            self.currentInfo.Index += 1

            self.currentInfo.Time           = self.envData[0][self.currentInfo.Index][0]

            # update price as well
            self.currentInfo.Price.Open = self.envData[1][self.currentInfo.Index][0]
            self.currentInfo.Price.High = self.envData[1][self.currentInfo.Index][1]
            self.currentInfo.Price.Low = self.envData[1][self.currentInfo.Index][2]
            self.currentInfo.Price.Close = self.envData[1][self.currentInfo.Index][3]

            #self.currentInfo.OpenInventory  = self.currentInfo.AvailableInventory        # current open = previous close
            self.currentInfo.executedInventory = action * self.orderSizeFactor
            self.currentInfo.executedPrice  = self.currentInfo.Price.Close               # market hit (at close)
            self.currentInfo.TimeLeft       = timeLeft
            self.currentInfo.AvailableInventory = inventoryLeft
            
            # update portfolio Histroy
            self.updateInventoryHistory()




        return inventoryLeft, timeLeft

