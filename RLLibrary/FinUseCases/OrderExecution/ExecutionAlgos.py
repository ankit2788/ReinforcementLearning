import os, sys
from importlib import reload
import numpy as np
from tensorflow.python.training.tracking.tracking import ResourceTracker

# Load the environments
from RLLibrary.FinUseCases import CustomGym, EnvironmentStorage
import tensorflow as tf


from RLLibrary.utils import constants as constants
from RLLibrary.FinUseCases.OrderExecution import EnvironmentManager
from RLLibrary.FinUseCases.OrderExecution.ModelManager.DQN import Agent as DQNAgent
from RLLibrary.FinUseCases.OrderExecution.ModelManager.A3C import Agent as A3CAgent

from RLLibrary.FinUseCases.OrderExecution.ModelManager.DQN import NetworkManager as DQNNet
from RLLibrary.FinUseCases.OrderExecution.ModelManager.A3C import NetworkManager as A3CNet


reload(EnvironmentManager)
reload(DQNAgent)
reload(A3CAgent)

from RLLibrary.utils import constants as constants
DATA_DIR = constants.DATA_DIR
MODEL_DIR = constants.MODEL_DIR

from RLLibrary.utils.loggingConfig import logger
Logger = logger.getLogger("ExecutionAlgos")


print(CustomGym.registry.all())

class Execution_EquiBalanced():

    def __init__(self, envName = "OrderExecution-v0", **env_args):

        self.envName = envName
        self.envargs = env_args


    def run(self, inidividualOrderSize = 1, orderConfig = {"initialOrderSize": 10000, "initialTimeHorizon": 100, "orderFactor": 100, \
                                "TotalIntervals": 50, "startTime": "09:30", "Timezone": "IST"}, 
                testDate = "20200131"):

        # load teh environment
        self.envargs["orderConfig"] = orderConfig
        self.envargs["testDate"] = testDate
        self.envargs["trainingYear"] = []

        self.env = CustomGym.make(self.envName, **self.envargs)

        # run the environment
        self.env.reset(resetDate = testDate)
        done = False
        episodic_reward = 0
        episodic_steps = 0

        while not done:

            if episodic_steps % self.env.TimeGapbetweenIntervals == 0:
                # take model defined action 
                action = inidividualOrderSize      # lot to execute

            else :
                action = 0
                
            
            if self.env.currentInfo.TimeLeft == 1:
                # if only 1 minute left, execute all remaining orders

                action = self.env.currentInfo.AvailableInventory/ self.env.orderSizeFactor


            next_state, reward, done, _ = self.env.step(action)


            episodic_reward += reward
            episodic_steps += 1

        print(f'Total episodic reward using DQN: {episodic_reward}. Steps taken: {episodic_steps}')
        invHistory = self.env.getInventoryHistory()

        effectiveExecutedPrice = np.sum(invHistory["ExecutedInventory"]*invHistory["ExecutedPrice"])/np.sum(invHistory["ExecutedInventory"])
        return invHistory , np.round(effectiveExecutedPrice, 3),    episodic_reward,   episodic_steps


    def plotPerformance(self):
        self.env.render()



class Execution_DQN():

    def __init__(self, envName = "OrderExecution-v0", **env_args):

        self.envName = envName
        self.envargs = env_args



    def train(self, save_dir = os.path.join(MODEL_DIR, "OrderExecution"), MAX_EPISODES = 500, \
                DQNModel = None, hiddenUnits = [32],  batchNormalization = True, \
                dropout_rate = 0.25, optimizer_learning_rate = 1e-4, clipvalue = 100):

        Logger.info("Training with DQN agent ")

        networkArgs = {"Model": DQNModel, "hiddenUnits": hiddenUnits, "batchNormalization": batchNormalization, \
                        "dropout_rate" : dropout_rate, "optimizer_learning_rate": optimizer_learning_rate, "clipvalue": clipvalue}


        self.agent = DQNAgent.DQN(envName = self.envName, save_dir = save_dir, doubleDQN = False, \
                                    networkArgs = networkArgs,  **self.envargs)

        self.agent.train(discount_factor=0.99, MAX_EPISODES = MAX_EPISODES, batch_size = 32)



    def run(self, modelWeights, hiddenUnits = [32], batchNormalization = True, dropoutRate = 0.25, \
                orderConfig = {"initialOrderSize": 10000, "initialTimeHorizon": 100, "orderFactor": 100, \
                                "TotalIntervals": 50, "startTime": "09:30", "Timezone": "IST"}, 
                testDate = "20200131"):

        # load teh environment
        self.envargs["orderConfig"] = orderConfig
        self.envargs["testDate"] = testDate
        self.envargs["trainingYear"] = []

        self.env = CustomGym.make(self.envName, **self.envargs)

        model = DQNNet.NN_FF(state_size=self.env.observation_space.n, action_size=len(self.env.action_space.actions), \
                            hiddenUnits=hiddenUnits, batchNormalization=batchNormalization, dropout_rate=dropoutRate)

        
        # compile the model
        model(tf.convert_to_tensor(np.array(self.env.observation_space.currentState)[None, :], dtype = tf.float32))

        # load the weights
        model.load_weights(modelWeights)

        # run the environment
        self.env.reset(resetDate = testDate)
        done = False
        currentState = self.env.observation_space.currentState
        episodic_reward = 0
        episodic_steps = 0


        while not done:

            currentState = np.array(currentState).reshape(1, self.env.observation_space.n)
        
            laststep = False
            if episodic_steps % self.env.TimeGapbetweenIntervals == 0:
                # take model defined action 
                actionValues = model(currentState)
                actionIndex = np.argmax(actionValues[0])        # greeedy action
                action = self.env.action_space.actions[actionIndex]

            else :
                actionIndex = 0                     # no order to be executed
                action = 0
                
            
            if self.env.currentInfo.TimeLeft == 1:
                # if only 1 minute left, execute all remaining orders

                action = self.env.currentInfo.AvailableInventory/ self.env.orderSizeFactor
                actionIndex = action            # in this case, action and actionIndex are same


            next_state, reward, done, _ = self.env.step(action)


            episodic_reward += reward
            episodic_steps += 1
            currentState = next_state

        print(f'Total episodic reward using DQN: {episodic_reward}. Steps taken: {episodic_steps}')
        invHistory = self.env.getInventoryHistory()

        effectiveExecutedPrice = np.sum(invHistory["ExecutedInventory"]*invHistory["ExecutedPrice"])/np.sum(invHistory["ExecutedInventory"])
        return invHistory , np.round(effectiveExecutedPrice, 3),    episodic_reward,   episodic_steps


    def plotPerformance(self):
        self.env.render()



class Execution_DDQN():

    def __init__(self, envName = "OrderExecution-v0", **env_args):

        self.envName = envName
        self.envargs = env_args



    def train(self, save_dir = os.path.join(MODEL_DIR, "OrderExecution"), MAX_EPISODES = 500, \
                DQNModel = None, hiddenUnits = [32],  batchNormalization = True, \
                dropout_rate = 0.25, optimizer_learning_rate = 1e-4, clipvalue = 100):

        Logger.info("Training with DQN agent ")

        networkArgs = {"Model": DQNModel, "hiddenUnits": hiddenUnits, "batchNormalization": batchNormalization, \
                        "dropout_rate" : dropout_rate, "optimizer_learning_rate": optimizer_learning_rate, "clipvalue": clipvalue}


        self.agent = DQNAgent.DQN(envName = self.envName, save_dir = save_dir, doubleDQN = True,  \
                                    networkArgs = networkArgs,  **self.envargs)

        self.agent.train(discount_factor=0.99, MAX_EPISODES = MAX_EPISODES, batch_size = 32)


    def run(self, modelWeights, hiddenUnits = [32], batchNormalization = True, dropoutRate = 0.25, \
                orderConfig = {"initialOrderSize": 10000, "initialTimeHorizon": 100, "orderFactor": 100, \
                                "TotalIntervals": 50, "startTime": "09:30", "Timezone": "IST"}, 
                testDate = "20200131"):

        # load teh environment
        self.envargs["orderConfig"] = orderConfig
        self.envargs["testDate"] = testDate
        self.envargs["trainingYear"] = []

        self.env = CustomGym.make(self.envName, **self.envargs)

        model = DQNNet.NN_FF(state_size=self.env.observation_space.n, action_size=len(self.env.action_space.actions), \
                            hiddenUnits=hiddenUnits, batchNormalization=batchNormalization, dropout_rate=dropoutRate)

        
        # compile the model
        model(tf.convert_to_tensor(np.array(self.env.observation_space.currentState)[None, :], dtype = tf.float32))

        # load the weights
        model.load_weights(modelWeights)

        # run the environment
        self.env.reset(resetDate = testDate)
        done = False
        currentState = self.env.observation_space.currentState
        episodic_reward = 0
        episodic_steps = 0


        while not done:

            currentState = np.array(currentState).reshape(1, self.env.observation_space.n)
        
            laststep = False
            if episodic_steps % self.env.TimeGapbetweenIntervals == 0:
                # take model defined action 
                actionValues = model(currentState)
                actionIndex = np.argmax(actionValues[0])        # greeedy action
                action = self.env.action_space.actions[actionIndex]

            else :
                actionIndex = 0                     # no order to be executed
                action = 0
                
            
            if self.env.currentInfo.TimeLeft == 1:
                # if only 1 minute left, execute all remaining orders

                action = self.env.currentInfo.AvailableInventory/ self.env.orderSizeFactor
                actionIndex = action            # in this case, action and actionIndex are same


            next_state, reward, done, _ = self.env.step(action)

            episodic_reward += reward
            episodic_steps += 1
            currentState = next_state

        print(f'Total episodic reward using DDQN: {episodic_reward}. Steps taken: {episodic_steps}')
        invHistory = self.env.getInventoryHistory()
        
        effectiveExecutedPrice = np.sum(invHistory["ExecutedInventory"]*invHistory["ExecutedPrice"])/np.sum(invHistory["ExecutedInventory"])
        return invHistory , np.round(effectiveExecutedPrice, 3),    episodic_reward,   episodic_steps




    def plotPerformance(self):
        self.env.render()





class Execution_A3C():

    def __init__(self, envName = "OrderExecution-v0", **env_args):

        self.envName = envName
        self.envargs = env_args


    def train(self, cores = 1, save_dir = os.path.join(MODEL_DIR, "OrderExecution"), MAX_EPISODES = 2000, \
                ActorCriticModel = None, \
                actorHiddenUnits = [20,20,20], criticHiddenUnits = [20,20,20], optimizer_learning_rate = 1e-4):

        # create the master agent
        Logger.info("Training with A3C agent ")

        self.masterAgent = A3CAgent.MasterAgent(envName = self.envName, cores = cores, save_dir = save_dir, \
                            MAX_EPISODES = MAX_EPISODES, ActorCriticModel = ActorCriticModel, \
                            actorHiddenUnits = actorHiddenUnits, criticHiddenUnits = criticHiddenUnits, \
                            optimizer_learning_rate = optimizer_learning_rate,  \
                            **self.envargs)

        self.masterAgent.train()


    def run(self, modelWeights, actorHiddenUnits = [32], criticHiddenUnits = [32], \
                orderConfig = {"initialOrderSize": 10000, "initialTimeHorizon": 100, "orderFactor": 100, \
                                "TotalIntervals": 50, "startTime": "09:30", "Timezone": "IST"}, 
                testDate = "2020-01-31"):

        # load teh environment
        self.envargs["orderConfig"] = orderConfig
        self.envargs["testDate"] = testDate
        self.envargs["trainingYear"] = []

        self.env = CustomGym.make(self.envName, **self.envargs)

        model = A3CNet.ActorCritic_FF(state_size=self.env.observation_space.n, action_size=len(self.env.action_space.actions), \
                        actorHiddenUnits=actorHiddenUnits, criticHiddenUnits= criticHiddenUnits)

        
        # compile the model
        model(tf.convert_to_tensor(np.array(self.env.observation_space.currentState)[None, :], dtype = tf.float32))

        # load the weights
        model.load_weights(modelWeights)

        # run the environment
        self.env.reset(resetDate = testDate)
        done = False
        currentState = np.array(self.env.observation_space.currentState)
        episodic_reward = 0  
        episodic_steps = 0      



        while not done:

            currentState = np.array(currentState).reshape(1, self.env.observation_space.n)
        
            laststep = False
            if episodic_steps % self.env.TimeGapbetweenIntervals == 0:

                # get action probability
                currentState = np.array(currentState)
                probs, _ = model(tf.convert_to_tensor(currentState[None, :], dtype = tf.float32))
                _probs = np.nan_to_num(probs.numpy()[0])

                # choose the action with max prob
                actionIndex = np.argmax(_probs)
                action = self.env.action_space.actions[actionIndex]

            else :
                actionIndex = 0                     # no order to be executed
                action = 0
                
            
            if self.env.currentInfo.TimeLeft == 1:
                # if only 1 minute left, execute all remaining orders

                action = self.env.currentInfo.AvailableInventory/ self.env.orderSizeFactor
                actionIndex = action            # in this case, action and actionIndex are same


            next_state, reward, done, _ = self.env.step(action)

            episodic_reward += reward
            episodic_steps += 1
            currentState = next_state

        print(f'Total episodic reward using A3C: {episodic_reward}. Steps taken: {episodic_steps}')
        invHistory = self.env.getInventoryHistory()

        effectiveExecutedPrice = np.sum(invHistory["ExecutedInventory"]*invHistory["ExecutedPrice"])/np.sum(invHistory["ExecutedInventory"])
        return invHistory , np.round(effectiveExecutedPrice, 3),    episodic_reward,   episodic_steps




    def plotPerformance(self):
        self.env.render()
