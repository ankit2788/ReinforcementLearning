# set the environment path
import os, sys
cwd = os.getcwd()
print(cwd)
path = f'{cwd}/../lib'

os.environ["porto"] = path

if os.environ["porto"] not in sys.path:
    sys.path.append(os.environ["porto"])

MODEL_PATH = f'{os.environ["porto"]}/../models/'


from logging.config import dictConfig
import logging
from importlib import reload
import time


import Environments
import Agent

import pandas as pd
import numpy as np
from datetime import datetime


reload(Agent)
reload(Environments)


def main():

    actions = [
    [-0.01,0.01],
    [-0.02,0.02],
    [-0.03,0.03],
    [0.01,-0.01],
    [0.02,-0.02],
    [0.03,-0.03],
    [-0.00,0.00],
    ]

    env = Environments.Portfolio(assets = ["APA", "BMY"], initialWeights = [0.5, 0.5], \
                    nbhistoricalDays = 5, \
                    startDate = "2018-01-01", endDate="2018-12-31", \
                    actions = actions, normalizeState=False, \
                    config = {"initialCash": 1000000, "minCash": 0.02, "transactionFee": 0.0000})


    myagent = Agent.DQNAgent(env)

    episodes = 500
    eps_rewards = []

    discountfactor = 0.99
    eps = Agent.epsilon_exploration(nbframe=0)

    totalFrames = 0

    for ep in range(episodes):
        starttime = time.perf_counter()

        ep_reward = 0
        env.reset()

        current_state = env.observation_space.currentState
        ep_steps = 0
        forbidden_action_count = 0

        done = False

        while not done:

            # get action
            current_state = np.array(current_state).reshape(1, env.observation_space.n)
            actionIndex = myagent.getAction(current_state, eps, mode = "TRAIN")

            if env.isactionForbidden(actionIndex):
                forbidden_action_count += 1

            # take step
            action = env.action_space.actions[actionIndex]
            new_state, reward, done = env.step(action)

            # store into memory
            myagent.updateMemory(current_state, actionIndex, reward, new_state, done)

            # train model
            grads = myagent.train(done, ep+1, discountfactor,  batch_size = 32, epochs=1, verbose = 0 )

            # update state
            ep_steps += 1
            current_state = new_state
            ep_reward += reward



        # update logs
        portfolio = env.getPortfolioHistory()
        epFinalPortValue = portfolio["AUM"].iloc[-1]

        myagent.updateEpisodicInfo(ep+1 , episodeReward = ep_reward, episodeFinalPortValue = epFinalPortValue , epsilon = eps, \
            forbidden_action_count = forbidden_action_count, steps = ep_steps, **grads)

        finishTime = time.perf_counter()
        print(f'Ep: {ep+1}. Time taken: {round(finishTime - starttime), 2} secs. FinalPortValue: {epFinalPortValue}. ForbiddenScore: {round(forbidden_action_count/ ep_steps, 2)}')

        totalFrames += ep*ep_steps
        eps = Agent.epsilon_exploration(nbframe= totalFrames)

        eps_rewards.append(ep_reward)

        if (ep+1)%25 == 0 :
            timestamp = datetime.now().strftime("%Y%m%d%H%M")
            myagent.save(os.path.join(MODEL_PATH, f'{myagent.target_model.name}_EP_{ep+1}_{timestamp}.h5'))





if __name__ == "__main__":
    main()
