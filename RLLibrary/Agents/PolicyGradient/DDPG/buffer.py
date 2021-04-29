import numpy as np



class ReplayBuffer():

    def __init__(self, bufferSize, input_shape, n_actions):

        self.bufferSize = bufferSize
        self.memCounter = 0


        # create storage for state, action, reward and next states
        print((self.bufferSize, *input_shape))
        self.mem_state = np.zeros((self.bufferSize, *input_shape))
        self.mem_action = np.zeros((self.bufferSize, n_actions))
        self.mem_nextstate = np.zeros((self.bufferSize, *input_shape))
        self.mem_reward = np.zeros(self.bufferSize)         # reward is a scalar quantity
        self.mem_terminal = np.zeros(self.bufferSize, dtype=np.bool)


    def update(self, state, action, reward, nextState, dead):

        index = self.memCounter % self.bufferSize

        self.mem_state[index] = state
        self.mem_action[index] = action
        self.mem_reward[index] = reward
        self.mem_nextstate[index] = nextState
        self.mem_terminal[index] = dead

        self.memCounter += 1


    def sample(self, batchSize):

        if self.memCounter >= batchSize:

            # uniformly sample from the memory
            availableSamples = min(self.memCounter, self.bufferSize)

            sampleIndices = np.random.choice(availableSamples, batchSize, replace = False)

            states = self.mem_state[sampleIndices]
            actions = self.mem_action[sampleIndices]
            rewards = self.mem_reward[sampleIndices]
            nextStates = self.mem_nextstate[sampleIndices]
            terminals = self.mem_terminal[sampleIndices]

            return states, actions, rewards, nextStates, terminals

        else:
            return None, None, None, None, None





