import os, sys
from importlib import reload

# Load the environments
from RLLibrary.FinUseCases import CustomGym, EnvironmentStorage


from RLLibrary.utils import constants as constants
from RLLibrary.FinUseCases.OrderExecution import EnvironmentManager
from RLLibrary.FinUseCases.OrderExecution.ModelManager.DQN import Agent as DQNAgent
from RLLibrary.FinUseCases.OrderExecution.ModelManager.A3C import Agent as A3CAgent

reload(DQNAgent)
reload(A3CAgent)

from RLLibrary.utils import constants as constants
DATA_DIR = constants.DATA_DIR
MODEL_DIR = constants.MODEL_DIR

from RLLibrary.utils.loggingConfig import logger
Logger = logger.getLogger("ExecutionAlgos")


print(CustomGym.registry.all())


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





    # def run(self):

    #     Logger.info("Learning")

    #     currentState = self.env.reset()
    #     episodeOver = False
    #     while not episodeOver:
    #         action = list(self.actions[0])
    #         newstate, reward, episodeOver = self.env.step(action)
    #         currentState = newstate

    #     portHistory = self.env.getPortfolioHistory() 
    #     return portHistory


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




    # def run(self):

    #     Logger.info("Learning")

    #     currentState = self.env.reset()
    #     episodeOver = False
    #     while not episodeOver:
    #         action = list(self.actions[0])
    #         newstate, reward, episodeOver = self.env.step(action)
    #         currentState = newstate

    #     portHistory = self.env.getPortfolioHistory() 
    #     return portHistory


    def plotPerformance(self):
        self.env.render()
