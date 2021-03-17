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
Logger = logging.getLogger("Actions")


class Action(ABC):
    
    def __init__(self):
        pass
    
    

class DiscreteActionSpace(Action):
    
    def __init__(self, actions):
        # actions --> list of all actions
        self.actions = actions
        self.n = len(actions)
        Logger.info(f"Action Space set with {self.n} possible actions ")
        
        


    

