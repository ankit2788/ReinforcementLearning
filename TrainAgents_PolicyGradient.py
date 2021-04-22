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


# importing custom libraries
from ActionSelection import ActionExploration
from ConfigReader import Config
#from RLAgents import QLearningAgent, FittedQAgent, DQN, DoubleDQN
import Agents.PolicyGradient.Reinforce as Reinforce

import utils as RLUtils


import loggingConfig as loggingConfig


logger = loggingConfig.logging
logger.getLogger("PG_Train")



# Run the model
#env         = gym.make("CartPole-v1")
configFile  = os.path.join(pref, "Configs.ini" )
savePath    = os.path.join(os.environ["RL_PATH"], "models" )
_time       = datetime.now().strftime("%Y%m%d%H%M")

env = gym.make("CartPole-v1")
state = env.reset()


#ReinforceAgent   = PolicyGradientAgents.REINFORCE_BASELINE(env, configFile,   NetworkShape = [24,12])
#ReinforceAgent   = PolicyGradientAgents.REINFORCE_C(env = env)
#ReinforceAgent   = PolicyGradientAgents.REINFORCE(env = env, configFile = configFile,   NetworkShape = [24,12])

modelParams = {"Name": "Policy", "NetworkShape": [24, 12], "learning_rate": 0.001, \
                "optimizer": "ADAM", "loss": "categorical_crossentropy", }
valueParams = {"Name_value": "Value", "NetworkShape_value": [24, 12], "learning_rate_value": 0.01, \
                "optimizer_value": "ADAM", "loss_value": "mse", }

ReinforceAgent   = Reinforce.REINFORCE_EAGER(env = env, configFile = configFile, **modelParams)


#ReinforceAgent   = Reinforce.REINFORCE_BASELINE(env = env, configFile = configFile, **modelParams, **valueParams)




Agent = ReinforceAgent
update_frequency = 1   # update policy after every 10 episodes
episodes = 400

"""

total_rewards=np.zeros(episodes)
rollout_n = 1

for episode in range(episodes):
    # each episode is a new game env
    state=env.reset()
    done=False          
    episode_reward=0 #record episode reward
    ep_steps = 0
    
    while not done:
        # play an action and record the game state & reward per episode
        action, prob=Agent.get_action(state)
        next_state, reward, done, _=env.step(action)
        Agent.remember(state, action, prob, reward)
        state=next_state
        episode_reward+=reward
        ep_steps += 1

        #if episode%render_n==0: ## render env to visualize.
            #env.render()
        if done:
            # update policy 
            if episode%rollout_n==0:
                history=Agent.update_policy()

            logger.info(f'Episode: {episode+1} Steps: {ep_steps} Reward:{episode_reward}  ')
            Agent.updateLoggerInfo(episodeCount = episode, episodicReward = episode_reward, \
                                    episodicStepsTaken = ep_steps, mode = "TRAIN")                

        total_rewards[episode]=episode_reward
    


"""

loss = []
mode = "TRAIN"
for _thisepisode in tqdm(range(Agent.NbEpisodesTrain)):

    # reset the environment
    _currentState = env.reset()
    

    _episodicReward = 0
    _dead = False
    _thisstepsTaken = 0


    while _thisstepsTaken <= Agent.MaxSteps:

        _starttime = time.perf_counter()

        # get the action from agent
        if "REINFORCE" in Agent.Name :
            # only actor based methods. Network doesnt return critic values
            action, actionProb = Agent.getAction(_currentState, mode = mode)
        else:
            action, actionProb, value = Agent.getAction(_currentState, mode = mode)            

        # perform the action
        _nextState, _reward, _dead, _info = env.step(action)

        # record into memory
        Agent.updateMemory(_currentState, action, _reward, _nextState, _dead, actionProb)

                
        # update States
        _currentState = _nextState
        _thisstepsTaken += 1

        _episodicReward += _reward

        # if game over, then exit the loop
        if _dead == True:

            #if (_thisepisode+1)%update_frequency == 0:
            Agent.train()

            # ---- For logging ------
            # In case of Neural networks, create tensorboard flow
            Agent.updateLoggerInfo(episodeCount = _thisepisode, episodicReward = _episodicReward, \
                                    episodicStepsTaken = _thisstepsTaken, mode = "TRAIN")

            _endtime = time.perf_counter()

            logger.info(f'Episode: {_thisepisode+1} Steps: {_thisstepsTaken} Reward:{_episodicReward} Time taken: {round(_endtime - _starttime,2)} secs ')

            break




# save model & save config
#print(f"Model & config save path: {Agent.path}") 
#Agent.PolicyModel.save(name = f"model_{Agent.Name}_{_time}.h5", path = savePath)
#Agent.saveConfig(filename = f"config_{Agent.Name}_{_time}.json", savePath = savePath)




#"""
