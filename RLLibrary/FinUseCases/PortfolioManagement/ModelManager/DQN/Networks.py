from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Dropout
from copy import copy
import tensorflow as tf
import numpy as np


class Dummy():
    def __init__(self):
        pass

class NN_FF(keras.Model):

    # plain feed forward actor critic model    

    def __init__(self, state_size, action_size, hiddenUnits = [32], batchNormalization = True, dropout_rate = 0.25):

        super().__init__()

        self.state_size = state_size
        self.action_size =  action_size

        self.hiddenUnits   = hiddenUnits
        self.batchNormalization = batchNormalization
        self.dropout_rate = dropout_rate

        initializer = keras.initializers.GlorotNormal()

        for index, layer in enumerate(self.hiddenUnits):
            layer = Dense(units = layer, activation="relu", kernel_initializer=initializer)            
            setattr(self, f"layer{index}", layer)

            if self.batchNormalization:
                layer = BatchNormalization()            
                setattr(self, f"layer_BN{index}", layer)

            # add dropout
            layer = Dropout(self.dropout_rate)
            setattr(self, f"layer_dropout{index}", layer)



        # final layer
        layer = Dense(units = self.action_size, activation="linear", kernel_initializer=initializer)
        setattr(self, f"value", layer)



    def call(self, inputs):

        # simple forward pass
        inputs = copy(inputs)


        for index, layer in enumerate(self.hiddenUnits):
            inputs = getattr(self, f"layer{index}")(inputs)

            if self.batchNormalization:
                inputs = getattr(self, f"layer_BN{index}")(inputs)

            # get dropout layer
            inputs = getattr(self, f"layer_dropout{index}")(inputs)

        values = getattr(self, f"value")(inputs)


        return values


