import numpy as np
import random
from abc import ABC, abstractclassmethod

from logging.config import dictConfig
import logging


import constants
import utils
import loggingConfig
dictConfig(loggingConfig.DEFAULT_LOGGING)
Logger = logging.getLogger("ActionSelection")



def ActionExploration(config, method = "EPSILON_GREEDY"):

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

    #def __init__(self, config, **kwargs):
    def __init__(self, intialExploration, finalExploration, nbIterations, **kwargs):

        """
        config: ConfigReader object 
        """
        """
        self.initialExploration     = float(utils.get_val(config, tag = "INITIAL_EXPLORATION", default_value= 1))
        self.finalExploration       = float(utils.get_val(config, tag = "FINAL_EXPLORATION", default_value= 0.1))
        self.finalExplorationSize   = int(utils.get_val(config, tag = "FINAL_EXPLORATION_ITERATION", default_value= 5000))
        """
        
        self.initialExploration = intialExploration
        self.finalExploration = finalExploration
        self.finalExplorationSize = nbIterations
        
        
        self.__decayRate = (self.initialExploration - self.finalExploration)/self.finalExplorationSize


        self.currentExploration = self.initialExploration
        
        
        
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

