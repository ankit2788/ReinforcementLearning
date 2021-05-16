import numpy as np

import os, sys
import matplotlib.pyplot as plt
import gym

from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras.callbacks import History
import tensorflow as tf
from datetime import datetime

# custom libraries

from Agents.PolicyGradient.DDPG import buffer
from Agents.PolicyGradient.DDPG import networks
from Callbacks import ModifiedTensorBoardCallback
import loggingConfig as loggingConfig

logger = loggingConfig.logging
logger.getLogger("DDPG")


pref = os.environ["RL_PATH"]


class DDPG():

    def __init__(self, env, dim_state, n_actions, \
                discount_factor = 0.99, tau = 0.005, batch_size =64, noise = 0.1, 
                bufferSize = 1000000, save_dir = None, 
                **kwargs):


        self.discount_factor = discount_factor
        self.tau = tau          # represents the soft target update rate
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]
        # create the memory
        self.ReplayBuffer = buffer.ReplayBuffer(bufferSize=bufferSize, input_shape=dim_state, n_actions=n_actions)

        # create the noise process
        # TODO
        self.noise = noise

        self.__time             = datetime.now().strftime("%Y%m%d%H%M")


        # create the networks
        self.save_dir = f"{pref}/models/Policy/DDPG" if save_dir is None else save_dir
        self.Actor  = networks.ActorNetwork(action_size=n_actions, chkpt_dir=self.save_dir)
        self.Critic = networks.CriticNetwork(chkpt_dir = self.save_dir)

        self.targetActor    = networks.ActorNetwork(action_size=n_actions, chkpt_dir=self.save_dir, name = "TargetActor")
        self.targetCritic   = networks.CriticNetwork(chkpt_dir = self.save_dir, name = "TargetCritic")

        # compile the networks
        learning_rate_actor = kwargs["learning_rate_actor"] if "learning_rate_actor" in kwargs.keys() else 0.0001
        learning_rate_critic = kwargs["learning_rate_critic"] if "learning_rate_critic" in kwargs.keys() else 0.001

        self.Actor.compile(optimizer = optimizers.Adam(learning_rate=learning_rate_actor))
        self.Critic.compile(optimizer = optimizers.Adam(learning_rate=learning_rate_critic))

        self.targetActor.compile(optimizer = optimizers.Adam(learning_rate=learning_rate_actor))
        self.targetCritic.compile(optimizer = optimizers.Adam(learning_rate=learning_rate_critic))


        # update the target network with main networks
        self.updateNetworks(tau = 1)

        self.setCallbacks()
        self.allEpisodicRewards = []
        self.allEpisodicSteps = []



    def setCallbacks(self):
        # ---- sets callback functions to record looging
        loggingPath         = f"{pref}/logs/Policy/DDPG_{self.__time}.log"

        # networkModel -> on which Tensorboard should run
        self.callbacks      = []
        
        self.ModelHistory   = History()     # to store learning history
        self.Tensorboard    = ModifiedTensorBoardCallback(model = self.Actor, log_dir = loggingPath)        

        self.callbacks.append(self.Tensorboard) 
        self.callbacks.append(self.ModelHistory) 


    def updateNetworks(self, tau = None):
        # updates the target networks

        tau  = self.tau if tau is None else tau

        # tau is required for the soft update of the network weights as described in the paper

        # Actor update
        weights = []
        for index, weight in enumerate(self.Actor.weights):

            _thisweight = tau*weight + (1-tau)*self.targetActor.weights[index]
            weights.append(_thisweight)

        self.targetActor.set_weights(weights)

        # Critic Update
        weights = []
        for index, weight in enumerate(self.Critic.weights):

            _thisweight = tau*weight + (1-tau)*self.targetCritic.weights[index]
            weights.append(_thisweight)

        self.targetCritic.set_weights(weights)



    def updateMemory(self, state, action, reward, nextState, terminal):
        # updaet memory
        self.ReplayBuffer.update(state, action, reward, nextState, terminal)

    def saveModel(self, save_dir = None):

        # save only the target actor and critic models
        self.targetActor.save_weights(self.targetActor.chkpt_dir)
        self.targetCritic.save_weights(self.targetCritic.chkpt_dir)

    def getAction(self, state, mode = "TRAIN"):
        # given the state, get the action. 
        # exploration needed if model is in training model

        state = tf.convert_to_tensor([state], dtype = tf.float32)
        action = self.Actor(state)

        if mode.upper() == "TRAIN":

            # add noise for exploration
            # TODO
            action += tf.random.normal(shape=[self.n_actions],
                    mean=0.0, stddev=self.noise)


        action = tf.clip_by_value(action, self.min_action, self.max_action)                    

        return action



    def train(self):
        # this is the heart of DDPG


        if self.ReplayBuffer.memCounter < self.batch_size:
            return


        # sample a batch from memory
        states, actions, rewards, nextStates, terminals = self.ReplayBuffer.sample(self.batch_size)

        states = tf.convert_to_tensor(states, dtype = tf.float32)
        actions = tf.convert_to_tensor(actions, dtype = tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype = tf.float32)
        nextStates = tf.convert_to_tensor(nextStates, dtype = tf.float32)
        terminals = tf.convert_to_tensor(terminals, dtype = tf.bool)

        # Critic Update
        with tf.GradientTape() as tape:

            actions_next = self.targetActor(nextStates)
            q_next = self.targetCritic(nextStates, actions_next)

            target_q = rewards + self.discount_factor*q_next

            # critic generated q
            current_q = self.Critic(states, actions)

            # loss
            loss_critic = losses.MSE(target_q, current_q)

        grad_critic = tape.gradient(loss_critic, self.Critic.trainable_variables)
        self.Critic.optimizer.apply_gradients(zip(grad_critic, self.Critic.trainable_variables))



        # Actor Update
        with tf.GradientTape() as tape:

            actions_actor = self.Actor(states)
            q_values = self.targetCritic(states, actions_actor)

            # aim is to maximize q_values --> minimize -qValues
            loss_actor = -q_values

        grad_actor = tape.gradient(loss_actor, self.Actor.trainable_variables)
        self.Actor.optimizer.apply_gradients(zip(grad_actor, self.Actor.trainable_variables))


        # now update the network parameters with soft update
        self.updateNetworks(tau = self.tau)


    def updateLoggerInfo(self, episodeCount, episodicReward, episodicStepsTaken, mode = "TRAIN"):
        # This is used to update all the logging related information
        # how the training evolves with episode

        self.Tensorboard.step += 1

        # 1. After every episode, update the episodic reward & steps taken
        self.allEpisodicRewards.append(episodicReward)
        self.allEpisodicSteps.append(episodicStepsTaken)

        # 2. log model weights & bias Info after every episode 
        self.Tensorboard.update_stats_histogram(model = self.Actor)

        # 3. Create other logging stats
        self.Tensorboard.update_stats(rewards = episodicReward, steps = episodicStepsTaken)


















