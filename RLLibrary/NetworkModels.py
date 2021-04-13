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

pref = os.environ["RL_PATH"]

from Callbacks import ModifiedTensorBoardCallback
from utils import convertStringtoBoolean, get_val



class BaseModel(ABC):

    @abstractclassmethod
    def __init__(self, path):
        self.path = path
        self.model = None
        

    @abstractclassmethod
    def init(self):
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


            


class TabularModel():
    def __init__(self, config, nbObservations, NbActions):

        self.nbObservations = nbObservations
        self.nbActions      = NbActions

        self.__readConfig(config)

    def init(self):
        self.model          = np.zeros([self.nbObservations, self.nbActions])

    def __readConfig(self, config):
        self.batchSize          = int(get_val(config, tag = "TRAIN_BATCH_SIZE", default_value= 32))
        self.modelsaveFormat    = get_val(config, tag = "MODEL_FORMAT", default_value="h5")
        self.AgentName          = get_val(config, tag = "NAME", default_value="QLEARNING")
        self.path               = get_val(config, tag = "PATH", default_value="models")

        # Need to append the path to relative path
        self.path               = os.path.join(pref,self.path)



    def save(self, name = "", path = ""):
        savepath = self.path if path == "" else path
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        filename = os.path.join(savepath, name) 
        if self.modelsaveFormat.upper() == "CSV":
            pd.DataFrame(self.model).to_csv(filename)

        else:
            #TODO: create error logger here
            print("Only CSV saving allowed right now")
            


    def load(self, modelPath = ""):

        self.model = None
        if modelPath != "":

            # load csv file
            if os.path.exists(modelPath):                
                self.model = np.array(pd.read_csv(modelPath, index_col = 0))
            else:
                self.model = None

        else:
            self.model = None
    




class BaselineModel(BaseModel):
    def __init__(self, config, dim_input, dim_output, NetworkShape = [], **kwargs):

        # config: Config object (from ConfigReader module)
        # dim_input: input dimensions
        # dim_output: output dimensions
        # NetworkShape: a list of neurons present in each layer. For baseline model, just use a single layer
        
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


    def init(self):
        # ---- single layer model 

        # -------- setting up the model architecture
        model = Sequential(name = self.__modelName)

        layer1 = Dense(units = self.outputDims, activation = "linear", input_shape = (self.inputDims,))
        model.add(layer1)

        # create optimizer
        if self.optimizer.upper() == "RMSPROP":            
            optimizer = optimizers.RMSprop(learning_rate = self.learning_rate, momentum = self.momentum, rho = self.decay_rate)

        # performance metrics
        metrics = [self.loss] if self.loss in ["mae", "mse"] else ["mse"]


        # compile the model
        model.compile(optimizer = optimizer, loss = self.loss, metrics = metrics)
        self.model = model
        self.model.summary()






    def __readConfig(self, config):

        # set default settings

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
        self.AgentName          = get_val(config, tag = "NAME", default_value="NFQ")
        self.log_write_grads    = convertStringtoBoolean(get_val(config, tag = "LOG_WRITE_GRADS", default_value="FALSE"))
        self.log_write_hists    = convertStringtoBoolean(get_val(config, tag = "LOG_WRITE_HISTOGRAM", default_value="FALSE"))

        self.path               = get_val(config, tag = "PATH", default_value="models")

        # Need to append the path to relative path
        self.path               = os.path.join(pref,self.path)



    def fit(self, X_train, y_train, batch_size, epochs=1, verbose = 1, callbacks = None):

        batchsize   = batch_size if batch_size else self.batchSize
        epochs      = epochs if epochs else self.epochs
        verbose     = verbose if verbose else self.verbosity   


        #Train the model
        self.model.fit(X_train,y_train,\
                        epochs=epochs,batch_size=batchsize, \
                        verbose=verbose, callbacks = callbacks)

        """
        # log the Gradients after every fit
        if self.log_write_grads:
            self.gradTensorBoard.step += 1

            grads = self.get_gradients(X_train, y_train)
            self.gradTensorBoard.update_grads(gradStats = grads)
        """

    
    def get_gradients(self, X_train, y_train):
        # compute loss function gradients with respect to model weights and biases

        with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
            tape.watch(self.model.trainable_weights)
            loss = mean_squared_error(y_train, self.model.predict(X_train))

        grads = {}
        for layer in self.model.layers:
            grads[f'{layer.name}'] = tape.gradient(loss, layer.trainable_weights)

        return grads
        



    def predict(self, X):
        return self.model.predict(X)


    def summary(self):
        return self.model.summary()


    def save(self, name = "", path = ""):
        savepath = self.path if path == "" else path
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        filename = os.path.join(savepath, name) 
        self.model.save(filepath = filename)


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
            


    
class DeepNeuralModel(BaseModel):

    def __init__(self, config, dim_input, dim_output, NetworkShape = [], **kwargs):

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
        self.path               = os.path.join(pref,self.path)

    def init(self):

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
        self.model.summary()



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




class DeepNeuralModelClassifier(BaseModel):

    def __init__(self, config, dim_input, dim_output, NetworkShape = [], **kwargs):
        # --- Used as Policy network
        # based on state and network parameters, predict the action to take

        # config: Config object (from ConfigReader module)
        # dim_input: input dimensions
        # dim_output: output dimensions
        # NetworkShape: a list of neurons present in each layer. For baseline model, just use a single layer
        # Example: Shape: [24,24] --> 
        # Inputs --> 24 neurons (layer1)  --> 24 neurons (layer2) --> NbActions (output) 
        

        self.__readConfig(config)

        if type(dim_input) is tuple:
            self.inputDims = dim_input[0]
        else:
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
        self.AgentName          = get_val(config, tag = "NAME", default_value="REINFORCE")
        self.log_write_grads    = convertStringtoBoolean(get_val(config, tag = "LOG_WRITE_GRADS", default_value="FALSE"))
        self.log_write_hists    = convertStringtoBoolean(get_val(config, tag = "LOG_WRITE_HISTOGRAM", default_value="FALSE"))

        self.path               = get_val(config, tag = "PATH", default_value="models")

        # Need to append the path to relative path
        self.path               = os.path.join(pref,self.path)

    def init(self):

        # create the model
        model               = Sequential(name = self.__modelName)
        self.history        = History()     # to store learning history

        if not self.neuralNetShape:
            # if shape not provided: then Default model
            layer1 = Dense(units = 64, activation = "relu", input_shape = (self.inputDims,))
            layer2 = Dense(units = 32, activation = "relu")


            outputlayer = Dense(units = self.outputDims, activation = "softmax")

            model.add(layer1)
            model.add(layer2)
            model.add(outputlayer)


        else:
            
            layer1 = Dense(units = self.neuralNetShape[0], activation = "relu", input_shape = (self.inputDims,))
            model.add(layer1)

            if len(self.neuralNetShape) > 1:
                for _layer in self.neuralNetShape[1:]:
                    model.add(Dense(units = _layer, activation = "relu"))

            outputlayer = Dense(units = self.outputDims, activation = "softmax")
            model.add(outputlayer)


        # create optimizer
        if self.optimizer.upper() == "RMSPROP":            
            optimizer = optimizers.RMSprop(learning_rate = self.learning_rate, momentum = self.momentum, rho = self.decay_rate)
        elif self.optimizer.upper() == "ADAM":            
            optimizer = optimizers.Adam(learning_rate = self.learning_rate)


        # performance metrics
        metrics = None

        # compile the model
        model.compile(optimizer = optimizer, loss = self.loss, metrics = metrics)
        self.model = model
        self.model.summary()



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
