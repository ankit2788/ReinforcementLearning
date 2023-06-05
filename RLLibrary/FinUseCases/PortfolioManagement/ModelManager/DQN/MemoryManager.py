import random
import numpy as np



class ExperienceMemory():

    def __init__(self, size, forget_percent = 0.05):

        self.maxlen = size
        self.forget_percent = forget_percent
        self.memory = []



    def update(self, item):

        if len(self.memory) == self.maxlen:
            # forget some % with uniform prob
            _itemstokeep = np.round(self.maxlen * (1 - self.forget_percent),0)
            self.memory = random.sample(self.memory, int(_itemstokeep))
            self.memory.append(item)

        else:
            self.memory.append(item)
