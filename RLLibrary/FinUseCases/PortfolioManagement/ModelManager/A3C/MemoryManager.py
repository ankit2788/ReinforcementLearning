

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
