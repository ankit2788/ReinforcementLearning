import logging
import os
import numpy as np
from datetime import datetime

import tensorflow as tf
from importlib import reload
import tqdm
import time
import random

from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras import losses
import tensorflow as tf


# custom modules

from RLLibrary.utils import constants, Callbacks 
from RLLibrary.FinUseCases import CustomGym

DATA_DIR = constants.DATA_DIR
LOG_DIR = constants.LOG_DIR


from RLLibrary.FinUseCases.OrderExecution.ModelManager.DQN import NetworkManager, MemoryManager
from RLLibrary.FinUseCases.OrderExecution import EnvironmentManager

reload(NetworkManager)
reload(Callbacks)
reload(MemoryManager)

from RLLibrary.utils.loggingConfig import logger

Logger = logger.getLogger("DQNAgent")



# ------ EPSILON for exploration
INITIAL_EXPLORATION = 1
FINAL_EXPLORATION = 0.1
FINAL_EXPLORATION_FRAME = 6000

EXPLORATION_RATIO = (INITIAL_EXPLORATION - FINAL_EXPLORATION)/ FINAL_EXPLORATION_FRAME

def epsilon_exploration(nbFrames):

    if nbFrames <= FINAL_EXPLORATION_FRAME:
        return INITIAL_EXPLORATION - EXPLORATION_RATIO*nbFrames

    else:
        return FINAL_EXPLORATION







    

class DQN():

    def __init__(self, envName, save_dir,  doubleDQN = False, \
                    networkArgs  = {"Model": None, "hiddenUnits": [32], \
                        "batchNormalization": True, "dropout_rate" : 0.25, "optimizer_learning_rate": 1e-4, "clipvalue": 100},
                        **env_args):

        
        self.doubleDQN = doubleDQN
        self.Name       = "DDQN" if self.doubleDQN else "DQN"
        self.envName    = envName
        self.envargs    = env_args


        target_update_frequency = 50

        self.save_dir = save_dir

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # create the environment
        self.env = CustomGym.make(self.envName, **self.envargs)
        self.env.reset()

        self.train_model    = self.initNetwork(networkArgs)
        self.target_model   = self.initNetwork(networkArgs)

        self.target_model.set_weights(self.train_model.get_weights())

        # set tensorboard callback
        loggingPath = os.path.join(LOG_DIR, "OrderExecution")
        if not os.path.exists(loggingPath):
            os.makedirs(loggingPath)

        __time = datetime.now().strftime("%Y%m%d%H%M")
        loggingPath         = f"{loggingPath}/{self.Name}_{__time}.log"
        self.Tensorboard    = Callbacks.ModifiedTensorBoardCallback(model = self.target_model, log_dir = loggingPath)        

        self.callbacks      = []
        self.callbacks.append(self.Tensorboard) 
        self.callbacks.append(callbacks.History()) 


        # create memory manager
        self.memory = MemoryManager.ExperienceMemory(size = 10000, forget_percent=0.05)
        self.target_update_counter = 0

        self.TARGET_MDOEL_UPDATE_FREQUENCY = target_update_frequency




        # episodic Info
        self.EpisodicRewards    = []
        self.trainingiterationCount     = 0 




    def initNetwork(self, networkArgs):

        if networkArgs["Model"] is None:
            model = NetworkManager.NN_FF(state_size=self.env.observation_space.n, action_size=self.env.action_space.n, \
                hiddenUnits=networkArgs["hiddenUnits"], batchNormalization=networkArgs["batchNormalization"], \
                dropout_rate=networkArgs["dropout_rate"])

        else:
            model = networkArgs["Model"]


        # create optimier
        self.optimizer = optimizers.Adam(learning_rate=networkArgs["optimizer_learning_rate"], clipvalue = networkArgs["clipvalue"])

        # compile the model
        model(tf.convert_to_tensor(np.random.random((1, self.env.observation_space.n)), dtype = tf.float32))

        print(model.summary())
        return model
        



    def train(self, MAX_EPISODES = 1000, discount_factor = 0.99, batch_size = 32):

        eps         = epsilon_exploration(nbFrames=0)
        totalFrames = 0

        episodic_rewards = []

        grads = None
        for ep in tqdm.tqdm(range(MAX_EPISODES)):

            starttime = time.perf_counter()
            ep_reward = 0
            ep_steps = 0
            done = False

            self.env.reset()

            current_state = self.env.observation_space.currentState

            while not done:

                # get action
                current_state = np.array(current_state).reshape(1, self.env.observation_space.n)
                

                trainingRequired = False
                laststep = False
                if ep_steps % self.env.TimeGapbetweenIntervals == 0:
                    # take model defined action 
                    actionIndex = self.getAction(current_state, eps, mode = "TRAIN"    )
                    action = self.env.action_space.actions[actionIndex]
                    trainingRequired = True
                    


                
                if self.env.currentInfo.TimeLeft == 1:
                    # if only 1 minute left, execute all remaining orders

                    action = self.env.currentInfo.AvailableInventory/ self.env.orderSizeFactor
                    actionIndex = action            # in this case, action and actionIndex are same
                    trainingRequired =True
                    #print("Executing remaining order", action)
                    laststep = True


                if trainingRequired:
                    new_state, reward, done, _ = self.env.step(action)
                    # if laststep:
                    #     print("Executing remaining order", action, "index", actionIndex, "remaining", self.env.currentInfo.AvailableInventory, done)


                    # store into experience replay
                    self.updateMemory(current_state, actionIndex, reward, new_state, done)

                    # train teh model
                    grads = self.learn(done, episodeCount = ep, discount_factor = discount_factor, batch_size = batch_size)
                    ep_reward += reward
                    # Logger.info(f'Ep#: {ep} Reward:{reward} Action: {action} Steps: {ep_steps} leftSize: {self.env.currentInfo.AvailableInventory}    penality factor: {self.env.RewardManager.impactPenalizeFactor} {action} {self.env.orderSizeFactor} '   )

                else:
                    action = 0
                    new_state, _ , done, _ = self.env.step(action)


                # update state
                ep_steps += 1
                current_state = new_state


            finishTime = time.perf_counter()

            # update Logs

            # get final portfolio value
            orderHistory = self.env.getInventoryHistory()
            inventory = orderHistory["AvailableInventory"].iloc[-1]


            Logger.info(f'{self.Name} Ep#: {ep} Date: {self.env.envDate} Reward: {np.round(ep_reward,1)} , Steps: {ep_steps} INV: {inventory} Time: {self.env.currentInfo.TimeLeft}')


            #update the tensorboard logging
            _grads = {}
            if grads is not None:
                
                for index, item in enumerate(grads):
                    _grads[f'params_{str(index)}'] = item.numpy()

            self.updateEpisodicInfo(episodeCount = ep, \
                                    episodicReward = ep_reward, \
                                    steps = ep_steps, \
                                    epsilon = eps , \
                                    **_grads)


            # update the epsilo
            totalFrames = totalFrames + ep_steps
            eps = epsilon_exploration(nbFrames=totalFrames)

            episodic_rewards.append(ep_reward)

            if ((ep+1)%100 == 0) or (ep == 0): 

                # save the model
                _time = datetime.now().strftime("%Y%m%d%H%M")
                path = os.path.join(self.save_dir, f'DQN_EP_{ep}_{_time}.h5')
                self.target_model.save_weights(path)


                


    def getAction(self, state, eps, mode = "TRAIN"):

        # get one hot representation of state
        actionsValues       = self.train_model(state)
        actionIndex         = np.argmax(actionsValues)

        # epsilonGreedyaction        
        if mode.upper() == "TRAIN":
            p = np.random.random()
            if p < eps:
                actiontoreturn = np.random.choice(self.env.action_space.n)
            else:
                actiontoreturn = actionIndex

        elif mode.upper() == "TEST":
            # select the greedy action
            actiontoreturn          = actionIndex 


        return actiontoreturn

        


    def updateMemory(self, currentState, currentAction, reward, nextState, dead):
        if any(x is None for x in currentState) is False:
            self.memory.update((currentState, currentAction, nextState, reward,  dead))


    def learn(self, terminal_state, episodeCount, discount_factor = 0.99, batch_size = 32):


        # ------- Replaying the memory (Experience replay) -------

        # dont train until a certain number of iterations (replay start)
        if len(self.memory.memory) < batch_size:
            return None

        self.trainingiterationCount += 1

        # pick out samples from memory based on batch size
        samples = random.sample(self.memory.memory, batch_size)


        # for all experience in batchsize
        curStates       = np.array([tup[0][0] for tup in samples])
        actions         = np.array([int(tup[1]) for tup in samples])
        nextStates      = np.array([tup[2] for tup in samples])
        rewards         = np.array([tup[3] for tup in samples])
        done            = np.array([tup[4] for tup in samples])

        # Add pre-processing step if needed

        inputStates     = np.array(curStates).reshape(len(curStates), self.env.observation_space.n)
        nextStates      = np.array(nextStates).reshape(len(nextStates), self.env.observation_space.n)

        # ------ For DQN, target comes from the target model
        # using bellman equation 

        # Steps:
        # 1. use target model to set the target. 
        # 2. This target needs to be updated based on the bellman equation
        # 2.1. Bellman Equation 1: get the max Q values for next state in the bellman equation




        curr_q_vaues    = self.train_model(inputStates) 
        new_q_values    = self.target_model(nextStates) 


        if not self.doubleDQN:
            # -----DQN algo ------
            # vectorized computation
            target          = rewards + discount_factor * np.amax(new_q_values, axis=1)
            target[done]    = rewards[done]            # end state target is reward itself

            target_f        = curr_q_vaues.numpy().copy()
            target_f[range(batch_size), actions] = target

        else:

            # ----- DDQN Algo -------
            # vectorized computation. 
            # Instead of the max Q value from target network, 
            # we identity the action which generates the max Q value from current network, and get the Q value for that action from target network
            target          = rewards + discount_factor * np.array([new_q_values[index, item] for index, item in enumerate(np.argmax(curr_q_vaues, axis=1))])
            target[done]    = rewards[done]            # end state target is reward itself

            target_f        = curr_q_vaues.numpy().copy()
            target_f[range(batch_size), actions] = target



        # compute teh loss
        with tf.GradientTape() as tape:

            pred_y = self.train_model(inputStates)
            loss = losses.mean_squared_error(pred_y, target_f)
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, self.train_model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.train_model.trainable_weights))


        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter >= self.TARGET_MDOEL_UPDATE_FREQUENCY:

            self.target_model.set_weights(self.train_model.get_weights())
            Logger.info("Target model udpated")
            self.target_update_counter = 0

        return grads




    def updateEpisodicInfo(self, episodeCount, episodicReward, \
                            epsilon, steps, **grads):

        # This is used to update all the logging related information
        # how the training evolves with episode

        # 1. After every episode, update the episodic reward & steps taken
        self.Tensorboard.step += 1
        self.EpisodicRewards.append(episodicReward)

        # 2. log model weights & bias Info after every episode 
        self.Tensorboard.update_stats_histogram(model = self.train_model)

        # 3. Create other logging stats

        AggregateStatsEvery = 20

        average_reward  = sum(self.EpisodicRewards[-AggregateStatsEvery:])/len(self.EpisodicRewards[-AggregateStatsEvery:])
        max_reward      = max(self.EpisodicRewards[-AggregateStatsEvery:])


        self.Tensorboard.update_stats(AvgReward = average_reward, MaxReward = max_reward, \
                                        epsilon = epsilon, \
                                        steps = steps, \
                                        #**grads \
                                        )


