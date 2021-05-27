import numpy as np




class StateSpace():

    def __init__(self, initialState = None):
        # state only includes previous set of prices & NOT portfolio weights

        self.initialState = initialState
        self.currentState = initialState
        self.previousState = None

        self.n = None


    def setInitialState(self, initialState):
        self.initialState = self.__convertStateToRequiredFormat(currentState = initialState)


    def __convertStateToRequiredFormat(self, currentState):
        return currentState


    def update(self, currentState, normalize = True):
        # current state consists of 
        # 1. OHLC
        # 2. Historical Volume
        # 3. Current open inventory
        # 4. Current time left

        if self.initialState is None:
            self.setInitialState(initialState = currentState)

        self.previousState = self.currentState

        # concatenate returns, current holdings , cash and actionPermission
        self.currentState = self.__convertStateToRequiredFormat(currentState)
        self.currentState = self.normalize(self.currentState)

        self.n = len(self.currentState)


    def normalize(self, currentState):
        # normalize the state by dividing by initial values

        return currentState   

            