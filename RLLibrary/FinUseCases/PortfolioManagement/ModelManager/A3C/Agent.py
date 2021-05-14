"""
Master Agent --> 
    1. maintains the global network
    2. Holds a shared optimizer across all workers
Local Agent (also called Worker Agents) --> Trained in parallel
"""

import os
import numpy as np
from queue import Queue
import multiprocessing
import sys
from datetime import datetime

import tensorflow as tf
import threading
from importlib import reload
import tqdm



# custom modules

from RLLibrary.utils import constants, Callbacks 
from RLLibrary.FinUseCases import CustomGym

DATA_DIR = constants.DATA_DIR
LOG_DIR = constants.LOG_DIR


from RLLibrary.FinUseCases.PortfolioManagement.ModelManager.A3C import Networks, MemoryManager
from RLLibrary.FinUseCases.PortfolioManagement import EnvironmentManager

reload(Networks)
reload(MemoryManager)
reload(Callbacks)

from RLLibrary.utils.loggingConfig import logger

Logger = logger.getLogger("Agent")




class MasterAgent():

    def __init__(self, envName, save_dir, cores = 1, MAX_EPISODES = 4000, ActorCriticModel = None, \
                        actorHiddenUnits = [32], criticHiddenUnits = [32], \
                        optimizer_learning_rate = 0.0001, 
                        **env_args):

        """
        Creates the master agent for A3C model
        envName --> "PortfolioManagement-v0"
        env_args --> if not provided, then environment uses default settings.
        ActorCriticModel --> keras.Model class else None

        """

        self.Name           = "A3C"
        self.save_dir       = save_dir
        self.MAX_EPISODES   = MAX_EPISODES
        self.envName        = envName
        self.envargs        = env_args
        self.cores          = cores

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # ------------------------------------
        # create the master environment
        self.globalEnvironment  = CustomGym.make(envName, **env_args)
        # assets = ["APA", "BMY"]
        # initialWeight = [0.5, 0.5]
        # actions = [
        #     [-0.03, 0.03], 
        #     [-0.02, 0.02], 
        #     [-0.01, 0.01], 
        #     [0.0, 0.0], 
        #     [0.01, -0.01], 
        #     [0.02, -0.02], 
        #     [0.03, -0.03], 
        # ]

        # self.globalEnvironment = EnvironmentManager.Portfolio(assets, initialWeight, \
        #             nhistoricalDays = 5, \
        #             startDate = "2019-01-01", endDate = "2019-12-31", \
        #             actions = actions, \
        #             assetDataPath = os.path.join(DATA_DIR, "PortfolioManagement"), \
        #             config = {"initialCash": 1000000, "minCash": 0.02, "transactionFee": 0.0000}, \
        #             penalizeFactors = {"Risk": -0.08, "ForbiddenAction": -8})

        # self.globalEnvironment = envName
        # Logger.info("creating custom")
        # self.tempEnv = CustomGym.make("PortfolioManagement-v0")
        # Logger.info("creatde custom gym")

        self.globalEnvironment.reset()

        self.state_size         = self.globalEnvironment.observation_space.n
        self.action_size        = self.globalEnvironment.action_space.n


        networkArgs = {"ActorCriticModel" : ActorCriticModel, \
                        "actorHiddenUnits" : actorHiddenUnits,  "criticHiddenUnits" : criticHiddenUnits, \
                        "optimizer_learning_rate" : optimizer_learning_rate }

        self.initNetwork(**networkArgs)




        # ------------------------------------
        # set the tensorboard callback
        loggingPath = os.path.join(LOG_DIR, "PortfolioManagement")
        if not os.path.exists(loggingPath):
            os.makedirs(loggingPath)

        __time = datetime.now().strftime("%Y%m%d%H%M")
        loggingPath = f'{loggingPath}/{self.Name}_{__time}.log'
        self.Tensorboard = Callbacks.ModifiedTensorBoardCallback(model = self.globalModel, log_dir = loggingPath, write_grads = False)

        self.callbacks = []
        self.callbacks.append(self.Tensorboard)


        # define the progress bar  (only for progress update)
        self.progress_bar = tqdm.tqdm(total = self.MAX_EPISODES + 1)


    def initNetwork(self, **networkArgs):
        # ------------------------------------
        # setup a global network

        self.__ActorCriticModel = networkArgs["ActorCriticModel"] if "ActorCriticModel" in networkArgs.keys() else None
        self.__actorHiddenUnits = networkArgs["actorHiddenUnits"] if "actorHiddenUnits" in networkArgs.keys() else [32]
        self.__criticHiddenUnits = networkArgs["criticHiddenUnits"] if "criticHiddenUnits" in networkArgs.keys() else [32]
        optimizer_learning_rate = networkArgs["optimizer_learning_rate"] if "optimizer_learning_rate" in networkArgs.keys() else 1e-4


        if self.__ActorCriticModel is None:
            globalModel = Networks.ActorCritic_FF(state_size=self.state_size, action_size=self.action_size, \
                                                    actorHiddenUnits = self.__actorHiddenUnits, criticHiddenUnits = self.__criticHiddenUnits) 
                                    
        else:
            globalModel = self.__ActorCriticModel

        self.globalModel = globalModel

        # create a shared optimizer. Note the use of use_locking
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=optimizer_learning_rate, use_locking=True)


        # construct the global Model with some random initial state
        self.globalModel(tf.convert_to_tensor(np.random.random((1, self.state_size)), dtype = tf.float32))
        print(self.globalModel.summary())



    def train(self):
        # This is where multiple local agents are defined
        # Different agents are run & trained on dfferent threads. 
        
        
        reward_queue = Queue()      # a queue to record episodic rewards across all agents
        portfolio_queue = Queue()      # a queue to record final portfolio value across all agents

        # cores = 1
        # cores = multiprocessing.cpu_count()
        Logger.info(f'Created {self.cores} threads')

        # global counter
        global_episode_index = multiprocessing.Value("i", 0)

        # create individual processes
        processes = []
        for _thread in range(self.cores):

            # create each agent object
            worker = WorkerAgent(envName = self.envName,  \
                    global_model = self.globalModel, sharedOptimizer = self.optimizer, \
                    result_queue = reward_queue, portfolio_queue = portfolio_queue, \
                    global_episode_index = global_episode_index, \
                    workerIndex = _thread, save_dir = self.save_dir, MAX_EPISODES = self.MAX_EPISODES, \
                    tensorboard = self.Tensorboard, progress_bar = self.progress_bar, \
                    ActorCriticModel = self.__ActorCriticModel, \
                    actorHiddenUnits = self.__actorHiddenUnits, criticHiddenUnits = self.__criticHiddenUnits, \
                    discount_factor = 0.99, beta_entropy = 0.01, \
                    **self.envargs)

            p = threading.Thread(target = worker.run)
            p.start()
            processes.append(p)


        # wait for all worker agents to complete
        [p.join() for p in processes]







class WorkerAgent(threading.Thread):
    # inheriting multiprocesing properties from Threading class
    best_score = -np.inf



    def __init__(self, envName,  \
                    global_model, sharedOptimizer, \
                    result_queue, portfolio_queue, \
                    global_episode_index, workerIndex, \
                    save_dir, MAX_EPISODES, tensorboard, progress_bar,\
                    ActorCriticModel = None, \
                    actorHiddenUnits = [32], criticHiddenUnits = [32], \
                    discount_factor = 0.99, beta_entropy = 0.01, \
                    **env_args):

        """
        envName --> "PortfolioManagement-v0"
        env_args --> if not provided, then environment uses default settings.
        ActorCriticModel --> keras.Model class else None
        globalModel --> required to copy the parameters frequently
        sharedOptimizer --> same optimizer is used to update parameters
        """

        
        super().__init__()

        self.envName = envName
        # self.envName = "PortfolioManagement-v0"
        self.env = CustomGym.make(self.envName, **env_args)
        self.MAX_EPISODES = MAX_EPISODES

        # assets = ["APA", "BMY"]
        # initialWeight = [0.5, 0.5]
        # actions = [
        #     [-0.03, 0.03], 
        #     [-0.02, 0.02], 
        #     [-0.01, 0.01], 
        #     [0.0, 0.0], 
        #     [0.01, -0.01], 
        #     [0.02, -0.02], 
        #     [0.03, -0.03], 
        # ]

        # self.env = envName
        # # self.env = EnvironmentManager.Portfolio(assets, initialWeight, \
        # #             nhistoricalDays = 5, \
        # #             startDate = "2019-01-01", endDate = "2019-12-31", \
        # #             actions = actions, \
        # #             assetDataPath = os.path.join(DATA_DIR, "PortfolioManagement"), \
        # #             config = {"initialCash": 1000000, "minCash": 0.02, "transactionFee": 0.0000}, \
        # #             penalizeFactors = {"Risk": -0.08, "ForbiddenAction": -8})



        self.state_size = self.env.observation_space.n
        self.action_size = self.env.action_space.n

        self.globalModel = global_model
        self.optimizer = sharedOptimizer

        self.result_queue = result_queue
        self.portfolio_queue = portfolio_queue

        # global episode as worker start index
        self.worker_episode_index = global_episode_index
        self.Name = f'Worker{workerIndex}'
        self.save_dir = save_dir

        self.Tensorboard = tensorboard

        self.discount_factor = discount_factor
        self.beta_entropy = beta_entropy


        # ------------------------------------
        # create the local model
        if ActorCriticModel is None:
            localModel = Networks.ActorCritic_FF(state_size=self.state_size, action_size=self.action_size, \
                                                    actorHiddenUnits = actorHiddenUnits, criticHiddenUnits = criticHiddenUnits) 
                                    
        else:
            localModel = localModel

        self.localModel = localModel
        Logger.info(f"Worker{workerIndex} created")

        # ------------------------------------
        self.globalProgressBar = progress_bar




    def run(self):

        # runs the Agent forever (until all games played)
        # trains the agent while sharing the network parameters 

        memory = MemoryManager.Memory()
        total_steps = 1


        
        while self.worker_episode_index.value <= self.MAX_EPISODES:

            # run the agent for each episode

            self.env.reset()
            current_state = self.env.observation_space.currentState
            memory.reset()

            episodic_reward = 0
            episodic_steps= 0
            self.episodicLoss = 0
            forbidden_action_count = 0


            done = False

            while not done:
                # play the episode until dead

                if any(x is None for x in current_state):
                    continue

                # sample the action based on prob distribution from local model
                current_state = np.array(current_state)
                probs, _ = self.localModel(tf.convert_to_tensor(current_state[None, :], dtype = tf.float32))

                # choose the action based on this probability distribution
                actionIndex = np.random.choice(self.action_size, p = probs.numpy()[0])    # since probs is a tensor. need to convert to array
                action = self.env.action_space.actions[actionIndex]

                if self.env.observation_space.isactionForbidden(actionIndex = actionIndex,allActions = self.env.action_space.actions):
                    forbidden_action_count += 1

                # take a step towards the action
                next_state, reward, done, _ = self.env.step(action)

                episodic_reward += reward

                # store the transition in memory
                memory.update(current_state, action, reward)
                


                if done:
                    # --- this is where the local model is being trained and global model gets udpated

                    # track the variables involved in loss computation using GradientTape (eager execution)
                    with tf.GradientTape() as tape:

                        total_loss = self.compute_loss(done, next_state, memory, discount_factor=self.discount_factor, beta_entropy=self.beta_entropy)

                    self.episodicLoss += total_loss

                    # compute the gradients with respect to local model
                    grads = tape.gradient(total_loss, self.localModel.trainable_weights)

                    # THIS IS IMPORTANT
                    # push the local gradients to global model and udpate global network parameters
                    # then, pull teh global weights to local model

                    self.optimizer.apply_gradients(zip(grads, self.globalModel.trainable_weights))
                    self.localModel.set_weights(self.globalModel.get_weights())

                    memory.reset()

                
                    if episodic_reward > WorkerAgent.best_score:
                        WorkerAgent.best_score = episodic_reward

                        # save the model
                        _time = datetime.now().strftime("%Y%m%d%H%M")
                        path = os.path.join(self.save_dir, f'A3C_EP_{self.worker_episode_index.value}_{_time}.h5')
                        self.globalModel.save_weights(path)


                total_steps += 1
                current_state = next_state
                episodic_steps += 1                 # we are not using it



            # get final portfolio value
            portfolio = self.env.getPortfolioHistory()
            episodeFinalValue = portfolio["AUM"].iloc[-1]


            with self.worker_episode_index.get_lock():
                Logger.info(f'{self.Name} Ep#: {self.worker_episode_index.value} Reward: {np.round(episodic_reward,1)}  FinalPortValye: {np.round(episodeFinalValue,0)}, ForbiddenActionScore: {np.round(forbidden_action_count/ episodic_steps, 2)}  Best Score (GLOBAL): {np.round(WorkerAgent.best_score,2)}  ')

                # update the global episode numner
                self.result_queue.put(episodic_reward)
                self.portfolio_queue.put(episodeFinalValue)

                #update the tensorboard logging
                _grads = {}
                for index, item in enumerate(grads):
                    _grads[f'params_{str(index)}'] = item.numpy()

                self.updateEpisodicInfo(episodeCount = self.worker_episode_index.value, \
                                        episodeReward = episodic_reward, epFinalPortValue = episodeFinalValue, \
                                        forbiddenActionCount = forbidden_action_count, steps = episodic_steps, \
                                        **_grads)

                self.worker_episode_index.value += 1
                self.globalProgressBar.update(1)


        # once the agent has exhausted all runs
        progress_bar.close()
        self.result_queue.put(None)



    def compute_loss(self, done, next_state, memory, discount_factor = 0.99, beta_entropy = 0.01):
        # As per the paper, https://arxiv.org/abs/1602.01783
        # loss has 3 components
        # 1. Critic Loss: simple MSE  (R-V)^2, where R --> discounted reward for state s. V --> value at state S
        # 2. Actor Loss: It is defined in accordance with Policy Gradient  
        #       - sum[grad(log(prob)) * advantage]   --> see paper for more mathematical definition
        # 3. Entropy of policy. Adding this improves exploration  --- HOW???

        
        
        # get the reward value at final step
        if done:
            reward_sum = 0
        else:
            _, values = self.localModel(tf.convert_to_tensor(next_state[None, :], dtype=tf.float32))

        # Here we are assuming 1 step return

        # get discounted returns
        discounted_rewards = []
        for reward in memory.rewards[::-1]:

            reward_sum = reward + discount_factor*reward_sum
            discounted_rewards.append(reward_sum)

        discounted_rewards.reverse()
        discounted_rewards = np.array(discounted_rewards).reshape(-1,1)

        # forward run the netwrok to get probabilities and value functions for all states in memory         
        probs, values = self.localModel(tf.convert_to_tensor(np.vstack(memory.states), dtype = tf.float32))

        advantages = discounted_rewards - values
        advantages = tf.convert_to_tensor(advantages, dtype = tf.float32)


        # critic loss
        loss_critic = tf.math.square(advantages)

        # actor loss
        # get probability of action taken
        actions = [[index, self.env.action_space.actions.index(item)] for index, item in enumerate(memory.actions)]
        prob_action = tf.convert_to_tensor(tf.gather_nd(probs, actions)[:, None])

        loss_actor = tf.math.log(prob_action + 1e-10)
        loss_actor *= - tf.stop_gradient(advantages)    # as per paper, advantages here is asusmed to be independent of parameter

        # entropy loss
        loss_entropy = tf.math.log(prob_action + 1e-10)
        loss_entropy *= - prob_action

        # total loss
        total_loss = tf.reduce_mean(loss_actor + beta_entropy*loss_entropy + loss_critic)
        return total_loss



    def updateEpisodicInfo(self, episodeCount, episodeReward, epFinalPortValue, forbiddenActionCount, steps, **grads):

        AggregateEvery = 20

        self.Tensorboard.step += 1
        self.Tensorboard.update_stats_histogram(model = self.globalModel)

        _reward = list(self.result_queue.queue)
        _portfolio = list(self.portfolio_queue.queue)

        avg_reward = sum(_reward[-AggregateEvery:])/len(_reward[-AggregateEvery:])
        max_reward = max(_reward[-AggregateEvery:])

        avg_port = sum(_portfolio[-AggregateEvery:])/len(_portfolio[-AggregateEvery:])

        self.Tensorboard.update_stats(Portfolio = epFinalPortValue, AvgPortValue = avg_port, \
                                        AvgReward = avg_reward, MaxReward = max_reward, \
                                        ForbiddenActionScore = forbiddenActionCount/steps , \
                                        #**grads \
                                        )









        







        





