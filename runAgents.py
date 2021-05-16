import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys
import gym
from importlib import reload
from configparser import ConfigParser
from datetime import datetime
import tensorflow as tf
import time



# get the relative path
fullpath                = os.path.realpath(__file__)
pref                    = os.path.split(fullpath)[0]

os.environ["RL_PATH"]   = pref

#os.environ["RL_PATH"] = "/Users/ankitgupta/Documents/git/anks/Books/ReinforcementLearning/DeepQLearning"
#pref = os.environ["RL_PATH"]

if f'{pref}/RLLibrary' not in sys.path:
    sys.path.append(f'{pref}/RLLibrary')


import loggingConfig as loggingConfig


logger = loggingConfig.logging
logger.getLogger("PG_Train")


from RLLibrary.Agents.PolicyGradient.A3C.Agents import MasterAgent
from RLLibrary.Agents.PolicyGradient.DDPG.Agent import DDPG


def main(agent = "DDPG"):


    modelParams = {"Name": "ActorCritic", "NetworkShape": [1024, 512], "learning_rate": 0.003, \
                    "optimizer": "ADAM", }

    actor_params = {"learning_rate_actor": 0.0001}
    critic_params = {"learning_rate_critic": 0.001}
    save_dir         = f"{pref}/models/Policy/DDPG"

    env = gym.make('Pendulum-v0')
    
    nbEpisodes = 250
    Agent   = DDPG( env, dim_state = env.observation_space.shape, n_actions =  env.action_space.shape[0],  \
                    discount_factor = 0.99, tau = 0.005, \
                    batch_size =64, noise = 0.1, bufferSize = 1000000, save_dir = save_dir, 
                        **actor_params, **critic_params)

    mode = "TRAIN"

    for _thisepisode in tqdm(range(nbEpisodes)):

        _starttime = time.perf_counter()

        # reset the environment
        _currentState = env.reset()

        _episodicReward = 0
        _dead = False
        _thisstepsTaken = 0


        while not _dead:


            # get the action from agent
            action = Agent.getAction(_currentState, mode = mode)            

            # perform the action
            _nextState, _reward, _dead, _info = env.step(action)
            _nextState = np.squeeze(_nextState)
            _reward = np.squeeze(_reward)

            # record into memory
            Agent.updateMemory(_currentState, action, _reward, _nextState, _dead)

            # learn at every step   
            Agent.train()     


            # update States
            _currentState = _nextState
            _thisstepsTaken += 1

            _episodicReward += _reward


            if _thisstepsTaken%20 == 0:
                logger.info(f'Episode: {_thisepisode+1} Steps: {_thisstepsTaken}')

            # if game over, then exit the loop
            if _dead == True:

                # ---- For logging ------
                # In case of Neural networks, create tensorboard flow
                Agent.updateLoggerInfo(episodeCount = _thisepisode, episodicReward = _episodicReward, \
                                        episodicStepsTaken = _thisstepsTaken, mode = "TRAIN")

                _endtime = time.perf_counter()

                logger.info(f'Episode: {_thisepisode+1} Steps: {_thisstepsTaken} Reward:{_episodicReward} Time taken: {round(_endtime - _starttime,2)} secs ')

                break



if __name__ == "__main__":

    agent = sys.argv[1]
    if agent == "A3C":        
        save_dir         = f"{pref}/models/Policy/A3C"
        optimizer_args = {"learning_rate": 1e-4}
        model = MasterAgent(game = "CartPole-v0", save_dir=save_dir, **optimizer_args)
        model.train()

    elif agent == "DDPG":
        main(agent = agent)

