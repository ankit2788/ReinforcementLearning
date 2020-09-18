from abc import ABC, abstractclassmethod
import matplotlib.pyplot as plt
import numpy as np

class Performance(ABC):
    @abstractclassmethod
    def __init__(self):
        pass

    @abstractclassmethod
    def showEpisodicReward(self):
        pass


class QLPerformance(Performance):

    def __init__(self, RLAgent):

        self.RLAgent = RLAgent

    def showEpisodicReward(self, mode = "TRAIN"):

        print(f'Averge reward obtained: {np.mean(self.RLAgent.EpisodicRewards[mode])}')
        # plot rewards

        fig, ax = plt.subplots(2,1, figsize = (8,10))

        fig.suptitle
        ax[0].plot(self.RLAgent.EpisodicRewards[mode])
        ax[0].set_xlabel("Nb. Episodes")
        ax[0].set_ylabel("Episodic reward")
        ax[0].set_title(f'Episodic reward')

        ax[1].plot(self.RLAgent.EpisodicSteps[mode])
        ax[1].set_xlabel("Nb. Episodes")
        ax[1].set_ylabel("Steps Taken")
        ax[1].set_title(f'Steps taken / episode')

        fig.suptitle(f'{self.RLAgent.Name} - {self.RLAgent.methodExploration}')


    def showStatesExploration(self):
        # shows how many times each State is explored
        # only useful in small envirnment & Q Learning

        lists = sorted(self.RLAgent.countStatesExplored.items()) # sorted by key, return a list of tuples
        x, y = zip(*lists) # unpack a list of pairs into two tuples

        plt.plot(x, y, marker = "o")
        plt.xlabel("State")
        plt.ylabel("Exploration count")

        plt.title("State Exploration")


    def showEpisodicLearning(self, Qvalues, state, action = None):
        # shows how each state action learning evolves
        # shows episodic learning for Q value for state action pair
        # Input:
        #   Qvalues --> state/ action pair for all episodes
        #   state --> state ID
        #   action --> action ID (optional). If not provided, then plot graph for all possible states

        if action is None:
            fig, ax = plt.subplots(nrows=self.RLAgent.env.action_space.n//2, ncols = 2, figsize=(10,6))

            for _actIndex in range(self.RLAgent.env.action_space.n):
                
                ax[_actIndex//2][_actIndex%2].plot(Qvalues[state][_actIndex], label = f'Action: {_actIndex}')
                ax[_actIndex//2][_actIndex%2].set_xlabel("Episodes")
                ax[_actIndex//2][_actIndex%2].set_ylabel("Qvalue")
                ax[_actIndex//2][_actIndex%2].set_title(f'Action: {_actIndex}')

            plt.legend()
            plt.suptitle(f"Q value for State: {state}")


        else:
            plt.plot(Qvalues[state][action], label = f'Action: {action}')
            plt.xlabel("Episodes")
            plt.ylabel("Qvalue")
            plt.title(f'Action: {_actIndex}')
    

class NFQPerformance(Performance):

    def __init__(self, RLAgent):

        self.RLAgent = RLAgent

    def showEpisodicReward(self, mode = "TRAIN"):
        print(f'Averge reward obtained: {np.mean(self.RLAgent.EpisodicRewards[mode])}')
        # plot rewards

        fig, ax = plt.subplots(2,1, figsize = (8,10))

        ax[0].plot(self.RLAgent.EpisodicRewards[mode])
        ax[0].set_xlabel("Nb. Episodes")
        ax[0].set_ylabel("Episodic reward")
        ax[0].set_title(f'Episodic reward')

        ax[1].plot(self.RLAgent.EpisodicSteps[mode])
        ax[1].set_xlabel("Nb. Episodes")
        ax[1].set_ylabel("Steps Taken")
        ax[1].set_title(f'Steps taken / episode')

        fig.suptitle(f'{self.RLAgent.Name} - {self.RLAgent.methodExploration}')




