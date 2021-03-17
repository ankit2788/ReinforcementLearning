import numpy as np
import os
import pandas as pd

from datetime import datetime
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import History, TensorBoard, ModelCheckpoint
from tensorflow.keras.models import load_model
import tensorflow as tf

from abc import ABC, abstractclassmethod

from logging.config import dictConfig
import logging




# Custom libraries
#from Callbacks import ModifiedTensorBoardCallback, GradCallBack
from utils import convertStringtoBoolean, get_val
import constants
from ConfigReader import Config
import loggingConfig

dictConfig(loggingConfig.DEFAULT_LOGGING)
Logger = logging.getLogger("NetworkModel")


prefPath = constants.PARENT_PATH



class BaseModel(ABC):

    @abstractclassmethod
    def __init__(self, path):
        self.path = path
        self.model = None
        

    @abstractclassmethod
    def initialize(self):
        pass

    @abstractclassmethod
    def fit(self, X_train, y_train, batch_size, epochs=1, verbose = 1):
        pass


    @abstractclassmethod
    def predict(self, X):
        pass

    @abstractclassmethod
    def summary(self):
        pass

    @abstractclassmethod
    def save(self, name = "", path = ""):
        pass        

    @abstractclassmethod
    def load(self):
        pass

    
    
  
class DeepNeuralNet(BaseModel):

    def __init__(self, dim_input, dim_output, NetworkShape = [], config = None, **kwargs):

        # config: Config object (from ConfigReader module)
        # dim_input: input dimensions
        # dim_output: output dimensions
        # NetworkShape: a list of neurons present in each layer. For baseline model, just use a single layer
        # Example: Shape: [24,24] --> 
        # Inputs --> 24 neurons (layer1)  --> 24 neurons (layer2) --> NbActions (output) 
        

        self.__readConfig(config)
        self.inputDims      = dim_input
        self.outputDims     = dim_output
        self.neuralNetShape = NetworkShape
        try:
            self.__time      = kwargs["time"]
        except KeyError:
            self.__time      = datetime.now().strftime("%Y%m%d%H%M")

        try:
            self.__modelName = kwargs["Name"]
        except KeyError:
            self.__modelName = self.AgentName
            
            
        self.initialize()
        Logger.info(f"{self.__modelName} initialized with {len(self.model.layers)} layers")


    def __readConfig(self, config):
        
        self.batchSize          = int(get_val(config, tag = "TRAIN_BATCH_SIZE", default_value= 32))
        self.epochs             = int(get_val(config, tag = "EPOCHS", default_value=1))
        self.optimizer          = get_val(config, tag = "GRAD_OPTIMIZER", default_value="RMSPROP")
        self.loss               = get_val(config, tag = "GRAD_LOSS", default_value="mse")
        self.dropout_rate       = float(get_val(config, tag = "GRAD_DROPOUT_RATE", default_value=0.0))
        self.learning_rate      = float(get_val(config, tag = "GRAD_LEARNING_RATE", default_value=0.001))
        self.decay_rate         = float(get_val(config, tag = "GRAD_DECAY_RATE", default_value=0.9))
        self.momentum           = float(get_val(config, tag = "GRAD_MOMENTUM", default_value=0.9))
        self.verbosity          = int(get_val(config, tag = "GRAD_VERBOSITY", default_value=0))
        self.modelsaveFormat    = get_val(config, tag = "MODEL_FORMAT", default_value="h5")
        self.AgentName          = get_val(config, tag = "NAME", default_value="DQN")
        self.log_write_grads    = convertStringtoBoolean(get_val(config, tag = "LOG_WRITE_GRADS", default_value="FALSE"))
        self.log_write_hists    = convertStringtoBoolean(get_val(config, tag = "LOG_WRITE_HISTOGRAM", default_value="FALSE"))

        self.path               = get_val(config, tag = "PATH", default_value="models")

        # Need to append the path to relative path
        self.path               = os.path.join(prefPath,self.path)


    def initialize(self):

        # create the model
        model               = Sequential(name = self.__modelName)
        self.history        = History()     # to store learning history

        if not self.neuralNetShape:
            # if shape not provided: then Default model
            layer1 = Dense(units = 64, activation = "relu", input_shape = (self.inputDims,))
            layer2 = Dense(units = 32, activation = "relu")


            outputlayer = Dense(units = self.outputDims, activation = "linear")

            model.add(layer1)
            model.add(layer2)
            model.add(outputlayer)


        else:
            
            layer1 = Dense(units = self.neuralNetShape[0], activation = "relu", input_shape = (self.inputDims,))
            model.add(layer1)

            if len(self.neuralNetShape) > 1:
                for _layer in self.neuralNetShape[1:]:
                    model.add(Dense(units = _layer, activation = "relu"))

            outputlayer = Dense(units = self.outputDims, activation = "linear")
            model.add(outputlayer)


        # create optimizer
        if self.optimizer.upper() == "RMSPROP":            
            optimizer = optimizers.RMSprop(learning_rate = self.learning_rate, momentum = self.momentum, rho = self.decay_rate)
        elif self.optimizer.upper() == "ADAM":            
            optimizer = optimizers.Adam(learning_rate = self.learning_rate)


        # performance metrics
        metrics = [self.loss] if self.loss in ["mae", "mse"] else ["mse"]

        # compile the model
        model.compile(optimizer = optimizer, loss = self.loss, metrics = metrics)
        self.model = model
        #self.model.summary()



    def fit(self, X_train, y_train, batch_size, epochs=1, verbose = 1, callbacks = None):

        batchsize   = batch_size if batch_size else self.batchSize
        epochs      = epochs if epochs else self.epochs
        verbose     = verbose if verbose else self.verbosity   

        #Train the model
        self.model.fit(X_train,y_train,\
                        epochs=epochs,batch_size=batchsize, \
                        verbose=verbose, callbacks = callbacks)


    def predict(self, X):
        return self.model.predict(X)


    def save(self, name = "", path = ""):
        
        savepath = self.path if path == "" else path
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        filename = os.path.join(savepath, name) 
        self.model.save(filepath = filename)


    def summary(self):
        return self.model.summary()

    def load(self, modelPath = ""):

        if modelPath != "":

            # load keras model
            try:
                self.model = load_model(modelPath)
                self.model.summary()
            except ImportError:
                self.model = None

        else:
            self.model = None

    def get_gradients(self, X_train, y_train):
        # compute loss function gradients with respect to model weights and biases

        with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
            tape.watch(self.model.trainable_weights)
            loss = mean_squared_error(y_train, self.model.predict(X_train))

        grads = {}
        for layer in self.model.layers:
            grads[f'{layer.name}'] = tape.gradient(loss, layer.trainable_weights)

        return grads


    

