import numpy as np
import os

# NN libraries
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
import tensorflow as tf


class ActorNetwork(keras.Model):
    # keras model submoduling
    def __init__(self, action_size, name = "Actor", \
                    chkpt_dir = "/models/ddpg"):

        super().__init__()
        self.model_name = name

        self.n_actions = action_size            # only 1 action to be outputted, but this represents the size of action vector
        self.chkpt_dir = chkpt_dir

        if not os.path.exists(self.chkpt_dir):
            os.makedirs(self.chkpt_dir)

        # create the network
        self.layer1 = Dense(units = 400, activation="relu")
        self.layer2 = Dense(units = 300, activation="relu")

        # this is the deterministic output. 
        # Here, we are range bounding the action to be between -1 adn 1 using tanh activation unit

        # Also, as per the paper, the last layer had weights initialized 
        last_init   = tf.random_uniform_initializer(minval = -0.003, maxval = 0.003)
        self.mu     = Dense(units = self.n_actions, activation="tanh", kernel_initializer=last_init) 


    def call(self, state):

        prob = self.layer1(state)               # 1st layer returns a probabiltiy (since relu unit)
        prob = self.layer2(prob)              # 2nd layer returns a probabiltiy (since relu unit)
        action = self.mu(prob)                  # get the action using last layer 

        return action




class CriticNetwork(keras.Model) :

    def __init__(self, name = "Critic", chkpt_dir = "\models\ddpg"):

        super().__init__()

        self.model_name = name

        self.chkpt_dir = chkpt_dir

        if not os.path.exists(self.chkpt_dir):
            os.makedirs(self.chkpt_dir)

        # create the network
        self.layer1 = Dense(units = 400, activation="relu")
        self.layer2 = Dense(units = 300, activation="relu")

        # this is the deterministic output. 
        # Here, we are range bounding the action to be between -1 adn 1 using tanh activation unit

        # Also, as per the paper, the last layer had weights initialized 
        self.Q      = Dense(units = 1, activation="linear") 


    def call(self, state, action):

        # critic network is a state action value network. 
        # It takes state and action as inputs

        actionValue = self.layer1(tf.concat([state, action], axis = 1))         # input is a combination of state and action
        actionValue = self.layer2(actionValue)

        actionValue = self.Q(actionValue)

        return actionValue










