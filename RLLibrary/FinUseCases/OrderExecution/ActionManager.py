from abc import ABC, abstractclassmethod


# custom
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
        

