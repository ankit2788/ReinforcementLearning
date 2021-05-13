import numpy as np


class RewardCategory1():
    def __init__(self, riskPenalizeFactor, actionPenalizeFactor, clipReward = True, clipRange = [-5,5]):

        self.riskPenalizeFactor = riskPenalizeFactor
        self.actionPenalizeFactor = riskPenalizeFactor

        self.clipReward = clipReward
        self.clipRange = clipRange

        self.runningReturn = []


    def computeReward(self, currentReturn, actionFeasible = True):
        # currentReturn --> log return

        # component 1: log return
        comp1 = 100 * currentReturn
        self.runningReturn.append(comp1)

        # component 2: variance of daily returns
        variance = np.var(self.runningReturn)
        comp2 = variance

        # component 3: penalty for forbidden actions
        comp3 = 0 if actionFeasible else 1

        reward = comp1 + self.riskPenalizeFactor*comp2 + self.actionPenalizeFactor*comp3

        if self.clipReward:
            reward = np.clip(reward, self.clipRange[0], self.clipRange[1])


        return np.round(reward, 2)
