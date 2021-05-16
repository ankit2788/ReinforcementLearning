"""
Master Agent --> 
    1. maintains the global network
    2. Holds a shared optimizer across all workers
Local Agent (also called Worker Agents) --> Trained in parallel
"""

import os
import gym
import numpy as np
from queue import Queue
import multiprocessing
import sys

import tensorflow as tf
import threading

import matplotlib.pyplot as plt

# custom modules
from Agents.PolicyGradient.A3C import Models


from Callbacks import ModifiedTensorBoardCallback
import loggingConfig as loggingConfig


logger = loggingConfig.logging
logger.getLogger("A3C")


MAX_EPISODES = 2000
T_MAX = 5

class Memory():

    def __init__(self):

        self.reset()


    def reset(self):
        self.states = []
        self.rewards = []
        self.actions = []


    def update(self, state, action, reward):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)


    



class MasterAgent():

    def __init__(self, game, save_dir, **kwargs):

        self.Name = "A3C"
        self.gameName = game
        self.save_dir = save_dir

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        
        # create the master environment
        env = gym.make(self.gameName)
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n


        # create a shared optimizer. Note the use of use_locking
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=kwargs["learning_rate"], use_locking=True)

        # setup a global network
        self.globalModel = Models.ActorCritic(state_size=self.state_size, action_size=self.action_size)

        # construct the global Model with some random initial state
        self.globalModel(tf.convert_to_tensor(np.random.random((1, self.state_size)), dtype = tf.float32))



    def train(self):
        # This is where multiple local agents are defined
        # Different agents are run & trained on dfferent threads. 
        

        
        reward_queue = Queue()      # a queue to record episodic rewards across all agents

        cores = 1
        cores = multiprocessing.cpu_count()
        logger.info(f'Created {cores} threads')

        # global counter
        global_episode_index = multiprocessing.Value("i", 0)

        # create individual processes
        processes = []
        for _thread in range(cores):

            # create each agent object
            worker = WorkerAgent(state_size = self.state_size, action_size = self.action_size, \
                    global_model = self.globalModel, sharedOptimizer = self.optimizer, \
                    result_queue = reward_queue, global_episode_index = global_episode_index, \
                    workerIndex = _thread, gameName = self.gameName , save_dir = self.save_dir)

            p = threading.Thread(target = worker.run)
            p.start()
            processes.append(p)


        # wait for all worker agents to complete
        [p.join() for p in processes]


        # present the final summary 
        # some code to show final summary of all rewards
        moving_average_rewards = []  # record episode reward to plot
        while True:
            reward = reward_queue.get()
            if reward is not None:
                moving_average_rewards.append(reward)
            else:
                break       

        plt.plot(moving_average_rewards)
        plt.ylabel('Moving average ep reward')
        plt.xlabel('Step')
        plt.savefig(os.path.join(self.save_dir,
                                '{} Moving Average.png'.format(self.gameName)))
        plt.show()




class WorkerAgent(threading.Thread):
    # inheriting multiprocesing properties from Threading class
    best_score = 0

    def __init__(self, state_size, action_size, \
                    global_model, sharedOptimizer, \
                    result_queue, global_episode_index, \
                    workerIndex, gameName , save_dir):

        
        super().__init__()

        self.state_size = state_size
        self.action_size = action_size

        self.globalModel = global_model
        self.optimizer = sharedOptimizer

        self.result_queue = result_queue

        # global episode as worker start index
        self.worker_episode_index = global_episode_index


        # create the local model
        self.localModel = Models.ActorCritic(state_size=self.state_size, action_size=self.action_size)
        self.Name = f'Worker{workerIndex}'
        self.gameName = gameName
        self.env = gym.make(self.gameName).unwrapped   # unwrapped to access the behind the scene dynamics of gym environment

        self.save_dir = save_dir




    def run(self):

        # runs the Agent forever (until all games played)
        # trains the agent while sharing the network parameters 

        memory = Memory()
        total_steps = 1


        while self.worker_episode_index.value < MAX_EPISODES:

            # run the agent for each episode

            current_state = self.env.reset()
            memory.reset()

            episodic_reward = 0
            episodic_steps= 0
            self.episodicLoss = 0

            done = False

            while not done:
                # play the episode until dead

                # sample the action based on prob distribution from local model
                probs, _ = self.localModel(tf.convert_to_tensor(current_state[None, :], dtype = tf.float32))
                action = np.random.choice(self.action_size, p = probs.numpy()[0])  # since probs is a tensor. need to convert to array

                # take a step towards the action
                next_state, reward, done, _ = self.env.step(action)

                if done:
                    reward = -1

                episodic_reward += reward

                # store the transition in memory
                memory.update(current_state, action, reward)


                if total_steps%T_MAX == 0 or done:
                    # --- this is where the local model is being trained and global model gets udpated

                    # track the variables involved in loss computation using GradientTape (eager execution)
                    with tf.GradientTape() as tape:

                        total_loss = self.compute_loss(done, next_state, memory)

                    self.episodicLoss += total_loss

                    # compute the gradients with respect to local model
                    grads = tape.gradient(total_loss, self.localModel.trainable_weights)

                    # THIS IS IMPORTANT
                    # push the local gradients to global model and udpate global network parameters
                    # then, pull teh global weights to local model

                    self.optimizer.apply_gradients(zip(grads, self.globalModel.trainable_weights))

                    self.localModel.set_weights(self.globalModel.get_weights())

                    memory.reset()

                
                    if done:
                        if episodic_reward > WorkerAgent.best_score:
                            WorkerAgent.best_score = episodic_reward


                total_steps += 1
                current_state = next_state

                episodic_steps += 1                 # we are not using it


            logger.info(f'{self.Name} Episode: {self.worker_episode_index.value}  Reward: {episodic_reward}.   Best Score (GLOBAL): {WorkerAgent.best_score}  ')

            with self.worker_episode_index.get_lock():
                # update the global episode numner
                self.result_queue.put(episodic_reward)
                self.worker_episode_index.value += 1


        # once the agent has exhausted all runs
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
            reward_sum = values.numpy()[0]

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
        actions= [[index, item] for index, item in enumerate(memory.actions)]
        prob_action = tf.convert_to_tensor(tf.gather_nd(probs, actions)[:, None])

        loss_actor = tf.math.log(prob_action + 1e-10)
        loss_actor *= - tf.stop_gradient(advantages)    # as per paper, advantages here is asusmed to be independent of parameter

        # entropy loss
        loss_entropy = tf.math.log(prob_action + 1e-10)
        loss_entropy *= - prob_action

        # total loss
        total_loss = tf.reduce_mean(loss_actor + beta_entropy*loss_entropy + loss_critic)
        return total_loss







        







        





