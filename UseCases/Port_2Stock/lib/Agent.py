import numpy as np
import os
import pandas as pd
from abc import ABC, abstractclassmethod
from datetime import datetime
from collections import deque
from importlib import reload
import random

import tensorflow as tf

from logging.config import dictConfig
import logging

from tensorflow.keras.callbacks import History
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import History, TensorBoard, ModelCheckpoint
from tensorflow.keras.models import load_model
import tensorflow as tf




# custom libraries
from Callbacks import ModifiedTensorBoardCallback


LIB_PATH = os.environ["porto"]
LOG_PATH = f'{LIB_PATH}/../logs/'
DATA_PATH = f'{LIB_PATH}/../data/'



logname = os.path.join(LOG_PATH, "Logging.Log")
logging.basicConfig(filename = logname, 
                   filemode = "a", 
                   format = "%(asctime)s %(levelname)-8s %(name)-15s %(message)s",
                   datefmt =  "%Y-%m-%d %H:%M:%S",
                   level = logging.INFO)

                    
Logger = logging.getLogger("Agent")



# -------e-greedy for exploration
INITIAL_EXPLORATION = 1
FINAL_EXPLORATION = 1
FINAL_EXPLORATION_FRAME  = 80000

EXPLORATION_RATIO = (INITIAL_EXPLORATION - FINAL_EXPLORATION)/ FINAL_EXPLORATION_FRAME

def epsilon_exploration(nbframe):
    # we start off with more exploration
    if nbframe <= FINAL_EXPLORATION_FRAME:
        return INITIAL_EXPLORATION - EXPLORATION_RATIO*nbframe
    else:
        return FINAL_EXPLORATION


class ExperienceMemory():

    def __init__(self, size, forget_percent=0.05):
        self.maxlen = size
        self.forget_percent = forget_percent
        self.memory = []

    def updateMemory(self, item):
        if len(self.memory) == self.maxlen:

            # forget some percentage with uniform prob
            _itemstokeep = np.round(self.maxlen * (1 - self.forget_percent), 0)
            self.memory = random.sample(self.memory, int(_itemstokeep))
            self.memory.append(item)

        else:
            self.memory.append(item)


    
class DQNAgent():

    def __init__(self, env):
        
        
        self.name   = "DQN"
        self.env    = env



        self.__time             = datetime.now().strftime("%Y%m%d%H%M")


        # ---------------- Model setup
        # DQN agent has 2 network
        # 1. Target network --> target is kept fixed for certain iterations after which it is updated 
        # 2. Main network --> regularly trained and updated 

        Logger.info("Setting up Main network and Target Network...")
        self.train_model = self.createNetwork(name = "Learner")
        self.target_model = self.createNetwork(name = "Target")

        self.target_model.set_weights(self.train_model.get_weights())


        # set call backs
        loggingPath = f'{LOG_PATH}/{self.name}_{self.__time}.log'
        self.Tensorboard = ModifiedTensorBoardCallback(model = self.train_model, log_dir = loggingPath , write_grads = True)
        self.callbacks =  []
        self.callbacks.append(History())
        self.callbacks.append(self.Tensorboard)

        # create memory
        self.memory = ExperienceMemory(size=20000, forget_percent=0.05)

        # some constants
        self.TARGET_MODEL_UPDATE_FREQUENCY = 1
        self.AGGREGATE_STATS_FREQUENCY = 20
        self.COMPUTE_GRADS_FREQUENCY = 10

        self.EpisodicReward = []
        self.EpisodicPortValue = []
        self.trainingCount = 0



    def createNetwork(self, name):

        model = Sequential()

        initializer = tf.keras.initializers.GlorotNormal()

        layer1 = Dense(units = 32, activation= "relu", input_shape = (self.env.observation_space.n, ), kernel_initializer=initializer)
        bn1 = BatchNormalization()
        dropout1 = Dropout(0.25)

        layer2 = Dense(units = 16, activation= "relu", kernel_initializer=initializer)
        bn2 = BatchNormalization()
        dropout2 = Dropout(0.25)

        outputlayer = Dense(units = self.env.action_space.n, activation= "linear", kernel_initializer=initializer)

        model.add(layer1)
        model.add(bn1)
        model.add(dropout1)
        model.add(layer2)
        model.add(bn2)
        model.add(dropout2)
        model.add(outputlayer)

        optimizer = optimizers.Adam()   #trying wth default learning rate of 0.01
        loss = "mse"

        metrics = ["accuracy"]

        model.compile(loss = loss, optimizer=optimizer, metrics=metrics)
        print(model.summary())
        return model



    def getAction(self, state, eps,  mode = "TRAIN"):


        _actionsValues      = self.train_model.predict(state) 
        _greedyActionIndex  = np.argmax(_actionsValues[0])


        # epsilonGreedyaction        
        if mode.upper() == "TRAIN":
            p = np.random.random()
            if p < eps:
                action = np.random.choice(self.env.action_space.n)
            else:
                action          = _greedyActionIndex 


        elif mode.upper() == "TEST":
            # select the greedy action
            action          = _greedyActionIndex 


        return action

        


    def updateMemory(self, currentState, currentAction, reward, nextState, dead):
        if any(x is None for x in currentState) is False:
            self.memory.updateMemory((currentState, currentAction, reward, nextState, dead))


    def train(self, terminal_state, episodeCount, discountfactor = 0.99,  batch_size = 32, epochs=1, verbose = 0):


        # ------- Replaying the memory (Experience replay) -------

        # dont train until a certain number of iterations (replay start)
        if len(self.memory.memory) < batch_size:
            return None

        self.trainingCount += 1


        # pick out samples from memory based on batch size
        samples = random.sample(self.memory.memory, batch_size)


        # for all experience in batchsize
        curStates       = np.array([tup[0][0] for tup in samples])
        actions       = np.array([tup[1] for tup in samples])
        nextStates       = np.array([tup[3] for tup in samples])
        rewards       = np.array([tup[2] for tup in samples])
        done       = np.array([tup[4] for tup in samples])


        # Add pre-processing step if needed
        inputStates     = np.array(curStates).reshape(len(curStates), self.env.observation_space.n)
        nextStates      = np.array(nextStates).reshape(len(nextStates), self.env.observation_space.n)
        #predict         = self.Qmodel.predict(inputStates)

        # ------ For DQN, target comes from the target model
        # using bellman equation 

        # Steps:
        # 1. use target model to set the target. 
        # 2. This target needs to be updated based on the bellman equation
        # 2.1. Bellman Equation 1: get the max Q values for next state in the bellman equation

        curQValues      = self.train_model.predict(nextStates)
        nextQvalues     = self.target_model.predict(nextStates)


        # vectorized computation using bellman equation
        target = rewards + discountfactor * np.amax(nextQvalues, axis=1)
        target[done] = rewards[done]

        target_f = curQValues
        target_f[range(batch_size), actions] = target

        history = self.train_model.fit(np.array(inputStates), np.array(target_f), epochs = 1, \
                    batch_size = batch_size, verbose=verbose, callbacks=self.callbacks if terminal_state else None)


        
        # update Logs
        if self.trainingCount%50 == 0:
            Logger.info(f'Training Count: {self.trainingCount}. Loss: {history.history["loss"][-1]}')

        # update target network
        if terminal_state and episodeCount%self.TARGET_MODEL_UPDATE_FREQUENCY == 0:
            self.target_model.set_weights(self.train_model.get_weights())
            Logger.info("Target Model updated")


        grads = {}
        if terminal_state and episodeCount%self.COMPUTE_GRADS_FREQUENCY == 0:
            with tf.GradientTape(persistent=True) as tape:
                pred_y = self.train_model(inputStates)
                loss = mean_squared_error(pred_y, target_f)

            for var in self.train_model.trainable_variables:
                if "bias" in var.name or "kernel" in var.name:
                    _grad = tape.gradient(loss, var)
                    grads[var.name] = np.mean(_grad)


        return grads        



    def updateEpisodicInfo(self, episodeCount, episodeReward, episodeFinalPortValue, epsilon, forbidden_action_count, steps, **grads):

        self.EpisodicReward.append(episodeReward)
        self.EpisodicPortValue.append(episodeFinalPortValue)

        self.Tensorboard.step += 1
        self.Tensorboard.update_stats_histogram(model = self.train_model)

        average_reward  = sum(self.EpisodicReward[-self.AGGREGATE_STATS_FREQUENCY:])/len(self.EpisodicReward[-self.AGGREGATE_STATS_FREQUENCY:])
        max_reward      = max(self.EpisodicReward[-self.AGGREGATE_STATS_FREQUENCY:])

        average_portValue   = sum(self.EpisodicPortValue[-self.AGGREGATE_STATS_FREQUENCY:])/len(self.EpisodicPortValue[-self.AGGREGATE_STATS_FREQUENCY:])

        self.Tensorboard.update_stats(PortValue = episodeFinalPortValue, AvgPortValue = average_portValue, \
                                         AvgReward=average_reward, MaxReward=max_reward, \
                                        epsilon=epsilon, \
                                        forbidden_score = forbidden_action_count/ steps, **grads)






    def save(self, name):
        self.train_model.save(name)