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


from RLLibrary.FinUseCases.PortfolioManagement.ModelManager.DQN import Networks, MemoryManager
from RLLibrary.FinUseCases.PortfolioManagement import EnvironmentManager

reload(Networks)
reload(Callbacks)
reload(MemoryManager)

from RLLibrary.utils.loggingConfig import logger

Logger = logger.getLogger("DQNAgent")



# ------ EPSILON for exploration
INITIAL_EXPLORATION = 1
FINAL_EXPLORATION = 0.1
FINAL_EXPLORATION_FRAME = 200000

EXPLORATION_RATIO = (INITIAL_EXPLORATION - FINAL_EXPLORATION)/ FINAL_EXPLORATION_FRAME

def epsilon_exploration(nbFrames):

    if nbFrames <= FINAL_EXPLORATION_FRAME:
        return INITIAL_EXPLORATION - EXPLORATION_RATIO*nbFrames

    else:
        return FINAL_EXPLORATION







    

class DQN():

    def __init__(self, envName, save_dir,  \
                    networkArgs  = {"Model": None, "hiddenUnits": [32], \
                        "batchNormalization": True, "dropout_rate" : 0.25, "optimizer_learning_rate": 1e-4, "clipvalue": 100},
                        **env_args):

        
        self.Name       = "DQN"
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
        loggingPath = os.path.join(LOG_DIR, "PortfolioManagement")
        if not os.path.exists(loggingPath):
            os.makedirs(loggingPath)

        __time = datetime.now().strftime("%Y%m%d%H%M")
        loggingPath         = f"{loggingPath}/{self.Name}_{__time}.log"
        self.Tensorboard    = Callbacks.ModifiedTensorBoardCallback(model = self.target_model, log_dir = loggingPath)        

        self.callbacks      = []
        self.callbacks.append(self.Tensorboard) 
        self.callbacks.append(callbacks.History()) 


        # create memory manager
        self.memory = MemoryManager.ExperienceMemory(size = 200000, forget_percent=0.05)
        self.target_update_counter = 0

        self.TARGET_MDOEL_UPDATE_FREQUENCY = target_update_frequency




        # episodic Info
        self.EpisodicRewards    = []
        self.EpisodicPortValue      = []
        self.trainingiterationCount     = 0 




    def initNetwork(self, networkArgs):

        if networkArgs["Model"] is None:
            model = Networks.NN_FF(state_size=self.env.observation_space.n, action_size=self.env.action_space.n, \
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
        



    def train(self, MAX_EPISODES = 2000, discount_factor = 0.99, batch_size = 32):

        eps         = epsilon_exploration(nbFrames=0)
        totalFrames = 0

        episodic_rewards = []

        for ep in tqdm.tqdm(range(MAX_EPISODES)):

            starttime = time.perf_counter()
            ep_reward = 0
            ep_steps = 0
            forbidden_action_count = 0
            done = False

            self.env.reset()

            current_state = self.env.observation_space.currentState

            while not done:

                # get action
                current_state = np.array(current_state).reshape(1, self.env.observation_space.n)
                actionIndex = self.getAction(current_state, eps, mode = "TRAIN"    )

                if self.env.observation_space.isactionForbidden(actionIndex = actionIndex, allActions = self.env.action_space.actions):
                    forbidden_action_count += 1

                # take step towards action
                action = self.env.action_space.actions[actionIndex]
                new_state, reward, done, _ = self.env.step(action)

                # store into experience replay
                self.updateMemory(current_state, actionIndex, reward, new_state, done)

                # train teh model
                grads = self.learn(done, episodeCount = ep, discount_factor = discount_factor, batch_size = batch_size)

                # update state
                ep_steps += 1
                current_state = new_state
                ep_reward += reward


            finishTime = time.perf_counter()

            # update Logs

            # get final portfolio value
            portfolio = self.env.getPortfolioHistory()
            episodeFinalValue = portfolio["AUM"].iloc[-1]

            Logger.info(f'{self.Name} Ep#: {ep} Reward: {np.round(ep_reward,1)}  FinalPortValye: {np.round(episodeFinalValue,0)}, ForbiddenActionScore: {np.round(forbidden_action_count/ ep_steps, 2)} ')


            #update the tensorboard logging
            _grads = {}
            for index, item in enumerate(grads):
                _grads[f'params_{str(index)}'] = item.numpy()

            self.updateEpisodicInfo(episodeCount = ep, \
                                    episodicReward = ep_reward, episodeFinalPortValue = episodeFinalValue, \
                                    forbidden_action_count = forbidden_action_count, steps = ep_steps, \
                                    epsilon = eps , \
                                    **_grads)


            # update the epsilo
            nbFrames = totalFrames + ep_steps*ep
            eps = epsilon_exploration(nbFrames=nbFrames)

            episodic_rewards.append(ep_reward)

            if ((ep+1)%50 == 0) or (ep == 0): 

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
        actions         = np.array([tup[1] for tup in samples])
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

        # -----DQN algo ------
        # vectorized computation

        target          = rewards + discount_factor * np.amax(new_q_values, axis=1)
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

            self.target_model.set_weights(self.train_model.get_weights)
            Logger.info("Target model udpated")
            self.target_update_counter = 0

        return grads




    def updateEpisodicInfo(self, episodeCount, episodicReward, episodeFinalPortValue, \
                            epsilon, forbidden_action_count, steps, **grads):

        # This is used to update all the logging related information
        # how the training evolves with episode

        # 1. After every episode, update the episodic reward & steps taken
        self.EpisodicRewards.append(episodicReward)
        self.EpisodicPortValue.append(episodeFinalPortValue)

        # 2. log model weights & bias Info after every episode 
        self.Tensorboard.update_stats_histogram(model = self.train_model)

        # 3. Create other logging stats

        AggregateStatsEvery = 20

        average_reward  = sum(self.EpisodicRewards[-AggregateStatsEvery:])/len(self.EpisodicRewards[-AggregateStatsEvery:])
        max_reward      = max(self.EpisodicRewards[-AggregateStatsEvery:])

        average_portValue   = sum(self.EpisodicPortValue[-AggregateStatsEvery:])/len(self.EpisodicPortValue[-AggregateStatsEvery:])

        self.Tensorboard.update_stats(Portfolio = episodeFinalPortValue, AvgPortValue = average_portValue, \
                                        AvgReward = average_reward, MaxReward = max_reward, \
                                        ForbiddenActionScore = forbidden_action_count/steps , \
                                        epsilon = epsilon, \
                                        #**grads \
                                        )


