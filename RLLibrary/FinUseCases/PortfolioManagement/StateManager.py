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
        # function to convert state tuple to required format

        # concatenate returns, current holdings , cash and actionPermission
        returns     = np.array(currentState[1]).ravel(order = "F")
        holdings    = np.array(currentState[2]).ravel(order = "F") 
        cash        = currentState[3]
        actionPermission = np.array(currentState[4]).ravel(order = "F") 

        updatedState = list(returns) + list(holdings) + [cash] + list(actionPermission)

        return updatedState


    def update(self, currentState, normalize = True):
        # current state consists of 
        # 2. Historical returns
        # 3. Current holdings
        # 4. Current available cash
        # 5. Forbidden Actions

        if self.initialState is None:
            self.setInitialState(initialState = currentState)

        self.previousState = self.currentState

        # concatenate returns, current holdings , cash and actionPermission
        self.currentState = self.__convertStateToRequiredFormat(currentState)
        #self.currentState = self.normalize(self.currentState)

        self.n = len(self.currentState)


    def isactionForbidden(self, actionIndex, allActions):
        actionPermits = self.currentState[self.n - len(allActions):]
        #actionPermits = self.currentState[4]
        if actionPermits[actionIndex] == 1:
            return False
        return True



    def normalize(self, currentState):
        # normalize the state by dividing by initial values

        pass        



            

    

        