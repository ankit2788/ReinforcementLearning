import numpy as np


class RewardCategory():
    def __init__(self, impactPenalizeFactor, StepRewardFactor):

        self.impactPenalizeFactor = impactPenalizeFactor
        self.stepRewardFactor = StepRewardFactor

        self.runningReturn = []


    def computeReward(self, deltaPrice, executedSize, currentTradedVolume):
        # currentReturn --> log return

        # component 1: price 
        comp1 = deltaPrice*executedSize

        # component 2: order impact penalty
        comp2 = (executedSize/currentTradedVolume)**2
        
        # component 3: Taking another step reward
        comp3 = 1

        reward = comp1 + self.impactPenalizeFactor*comp2 + self.stepRewardFactor*comp3

        return np.round(reward, 2)


