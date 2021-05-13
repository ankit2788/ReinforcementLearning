import gym, os, sys
from gym.envs.registration import register

from abc import ABC, abstractclassmethod


# get the relative path
fullpath                = os.path.realpath(__file__)
pref                    = os.path.split(fullpath)[0]

os.environ["RL_PATH"]   = f'{pref}/../..'
pref = f'{pref}/../..'
if f'{pref}/RLLibrary' not in sys.path:
    sys.path.append(f'{pref}')
    sys.path.append(f'{pref}/RLLibrary')



class Environment(gym.Env):
    def __init__(self):
        pass
    
    @abstractclassmethod
    def reset(self):
        pass
    
    @abstractclassmethod
    def step(self, action):
        pass



