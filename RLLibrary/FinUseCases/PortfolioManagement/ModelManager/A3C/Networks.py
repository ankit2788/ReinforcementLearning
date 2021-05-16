from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from copy import copy
import tensorflow as tf
import numpy as np


class Dummy():
    def __init__(self):
        pass

class ActorCritic_FF(keras.Model):

    # plain feed forward actor critic model    

    def __init__(self, state_size, action_size, actorHiddenUnits = [32], criticHiddenUnits = [32]):

        super().__init__()

        self.state_size = state_size
        self.action_size =  action_size

        self.actorHiddenUnits   = actorHiddenUnits
        self.criticHiddenUnits  = criticHiddenUnits

        # create 2 separate models for actor and critic
        # self.actor = Dummy()
        # self.critic = Dummy()

        # --- Actor --> returns prob distribution for all possible states given state
        for index, layer in enumerate(actorHiddenUnits):
            layer = Dense(units = layer, activation="relu")
            setattr(self, f"actor_layer{index}", layer)

        # final layer
        layer = Dense(units = self.action_size, activation="softmax")
        setattr(self, f"actor_policy", layer)
        #self.actor.policy = Dense(units = self.action_size, activation="softmax")
        
        # --- Critic   --> Value function (given a state)
        for index, layer in enumerate(criticHiddenUnits):
            layer = Dense(units = layer, activation="relu")
            setattr(self, f"critic_layer{index}", layer)

        # final layer
        layer = Dense(units = 1, activation="linear")
        setattr(self, f"critic_values", layer)
        # self.critic.values = Dense(units = 1, activation="linear")



    def call(self, inputs):

        # simple forward pass
        actor_inputs = copy(inputs)
        critic_inputs = copy(inputs)


        # Actor
        for index, layer in enumerate(self.actorHiddenUnits):
            actor_inputs = getattr(self, f"actor_layer{index}")(actor_inputs)
        
        probs = getattr(self, f"actor_policy")(actor_inputs)

        # Critic
        for index, layer in enumerate(self.criticHiddenUnits):
            critic_inputs = getattr(self, f"critic_layer{index}")(critic_inputs)

        values = getattr(self, f"critic_values")(critic_inputs)

        return probs, values




class ActorCritic_CNN(keras.Model):

    # plain feed forward actor critic model    

    def __init__(self, nbHistory, action_size):

        super().__init__()

        self.action_size =  action_size


        # create 2 separate models for actor and critic
        # self.actor = Dummy()
        # self.critic = Dummy()

        # --- Actor --> returns prob distribution for all possible states given state
        self.actor_layer1 = Conv2D(filters = 2, kernel_size=[1,3], activation = "relu")
        self.actor_layer2 = Conv2D(filters = 20, kernel_size=[1,nbHistory-2], activation = "relu")
        self.actor_layer3 = Conv2D(filters = 1, kernel_size=1, activation = "relu")
        self.actor_layer4 = Flatten()
        self.actor_layer5 = Dense(units = 32, activation="relu")
        self.actor_layer6 = Dense(units = action_size, activation="softmax")

        
        # --- Critic   --> Value function (given a state)
        self.critic_layer1 = Conv2D(filters = 2, kernel_size=[1,3], activation = "relu")
        self.critic_layer2 = Conv2D(filters = 20, kernel_size=[1,nbHistory-2], activation = "relu")
        self.critic_layer3 = Conv2D(filters = 1, kernel_size=1, activation = "relu")
        self.critic_layer4 = Flatten()
        self.critic_layer5 = Dense(units = 32, activation="relu")
        self.critic_layer6 = Dense(units = 1, activation="linear")



    def call(self, inputs):

        # inputs is a tuple

        if len(np.array(inputs).shape) == 1:
            # only a single input
            state_price = inputs[0]
            if (len(state_price.shape)) == 3:
                nbAssets    = state_price.shape[0]
                nbHistory   = state_price.shape[1]
                nbChannels  = state_price.shape[2]
                state_price = np.reshape(state_price, [1, nbAssets, nbHistory, nbChannels])

            state_holdings = inputs[1]
            state_cash = inputs[2]
            state_actions = inputs[3]

            batch_size = 1

        elif len(np.array(inputs).shape) == 2:
            # whole batch is provided
            batch_size = np.array(inputs).shape[0]

            state_price     = []
            state_holdings  = []
            state_cash      = []
            state_actions   = []

            nbAssets = len(inputs[0][1])
            for _index in range(batch_size):
                state_price.append(inputs[_index][0])
                state_holdings.append(inputs[_index][1])
                state_cash.append(inputs[_index][2])
                state_actions.append(inputs[_index][3])

            state_price = np.array(state_price)
            state_holdings = np.array(state_holdings)
            state_cash = np.array(state_cash)
            state_actions = np.array(state_actions)



        curr_holding = np.reshape(state_holdings, [batch_size, nbAssets, 1, 1])
        curr_holding = tf.convert_to_tensor(curr_holding, dtype = tf.float32)

        curr_cash = np.reshape(state_cash, [batch_size, 1, 1, 1])
        curr_cash = tf.convert_to_tensor(curr_cash, dtype = tf.float32)




        # ------ Actor
        hidden1 = self.actor_layer1(state_price)
        hidden2 = self.actor_layer2(hidden1)

        # concatenate the current holdngs as well
        temp_input = tf.concat([hidden2, curr_holding], axis=3)
        hidden3 = self.actor_layer3(temp_input)
        
        # add current cash 
        temp_input = tf.concat([curr_cash, hidden3], axis=1)
        hidden4 = self.actor_layer4(temp_input)

        hidden5 = self.actor_layer5(hidden4)
        probs   = self.actor_layer6(hidden5)            # policy layer

        # ----- Critic
        hidden1 = self.critic_layer1(state_price)
        hidden2 = self.critic_layer2(hidden1)

        # concatenate the current holdngs as well
        temp_input = tf.concat([hidden2, curr_holding], axis=3)
        hidden3 = self.critic_layer3(temp_input)
        
        # add current cash 
        temp_input = tf.concat([curr_cash, hidden3], axis=1)
        hidden4 = self.critic_layer4(temp_input)

        hidden5 = self.critic_layer5(hidden4)
        values   = self.critic_layer6(hidden5)            # policy layer
        
        return probs, values

