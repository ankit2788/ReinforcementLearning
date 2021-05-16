import numpy as np
import pandas as pd
import os
from abc import ABC, abstractclassmethod
import logging


# custom
from RLLibrary.utils import helper, ConfigReader, constants
from RLLibrary.utils.loggingConfig import logger

Logger = logger.getLogger("Actions")

class Action(ABC):
    
    def __init__(self):
        pass
    
    

class DiscreteActionSpace(Action):
    
    def __init__(self, actions):
        # actions --> list of all actions
        self.actions = actions
        self.n = len(actions)
        Logger.info(f"Action Space set with {self.n} possible actions ")
        


class ContinuousActionSpace(Action):
    def __init__(self):
        Logger.info(f"Continuous action space set")
        





def ActionExploration(method = "EPSILON_GREEDY", **kwargs):

    """
    config: ConfigReader object 
    method: Exploration techniques
    """
    if method.upper() == "EPSILON_GREEDY":

        return EpsilonGreedy(config)

    elif method.upper() == "UCB":
        return None

    else:
        return None


class ActionSelection(ABC):

    @abstractclassmethod
    def __init__(self):
        self.method = None
        pass

    @abstractclassmethod
    def chooseAction(self, actionSpace, optimalAction):
        pass



class EpsilonGreedy(ActionSelection):

    def __init__(self, config, **kwargs):

        """
        config: ConfigReader object 
        """

        self.__readConfig(config)
        self.currentExploration = self.initialExploration
        
        
    def __readConfig(self, config):

        self.initialExploration     = float(config(tag="INITIAL_EXPLORATION"))
        self.finalExploration       = float(config(tag="FINAL_EXPLORATION"))
        self.finalExplorationSize   = int(config(tag="FINAL_EXPLORATION_ITERATION"))

        self.__decayRate = (self.initialExploration - self.finalExploration)/self.finalExplorationSize

        
    def __updateEpsilon(self, currentEpisodeNb):

        if self.currentExploration > self.finalExploration:

            if currentEpisodeNb < self.finalExplorationSize:
                # update current exploration

                self.currentExploration -= self.__decayRate
                 
            

    def chooseAction(self, currentEpisodeNb, actionSpace, optimalActionIndex):

        # returns selected Action Index
        self.__updateEpsilon(currentEpisodeNb)

        if np.random.binomial(1, self.currentExploration) == 1:
            # exploring stage

            return np.random.choice(actionSpace) 

        else:
            # exploiting stage
            return optimalActionIndex

