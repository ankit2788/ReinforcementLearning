import numpy as np
import pandas as pd
import os
from abc import ABC, abstractclassmethod
from logging.config import dictConfig
import logging


# custom
import constants
import loggingConfig
dictConfig(loggingConfig.DEFAULT_LOGGING)
Logger = logging.getLogger("States")


class StateSpace(ABC):
    
    def __init__(self):
        pass
    
    
class PortfolioStateSpace(StateSpace):
    def __init__(self, initialState):
        self.currentState = initialState
        self.previousState = None
        self.n = len(initialState)
        
    def updateState(self, newState):
        
        self.previousState = self.currentState
        self.currentState = newState
        