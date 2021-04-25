import numpy as np
import random
import os
import sys
from datetime import datetime

import tensorflow.keras as keras
from tensorflow.keras.callbacks import History
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Reshape, Convolution2D, Conv2D
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras import backend as K
import tensorflow as tf



"""
Actor critic model to be defined for the environment
"""

class ActorCritic(keras.Model):

    def __init__(self, state_size, action_size):

        super().__init__()

        self.state_size = state_size
        self.action_size =  action_size


        # create 2 separate models for actor and critic

        # --- Actor --> returns prob distribution for all possible states given state
        self.dense1 = Dense(units = 128, activation= "relu")
        self.policy = Dense(units = self.action_size, activation="softmax")
        
        # --- Critic   --> Value function (given a state)
        self.dense2 = Dense(units = 128, activation= "relu")
        self.values = Dense(units = 1, activation="linear")



    def call(self, inputs):

        # forward pass

        hidden_policy = self.dense1(inputs)
        probs = self.policy(hidden_policy)

        hidden_value = self.dense2(inputs)
        values = self.values(hidden_value)

        return probs, values


