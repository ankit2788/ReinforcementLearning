import os, sys
from importlib import reload
import multiprocessing


# get the relative path
fullpath                = os.path.realpath(__file__)
pref                    = os.path.split(fullpath)[0]

os.environ["RL_PATH"]   = f'{pref}/../..'
pref = f'{pref}/../..'
if f'{pref}/RLLibrary' not in sys.path:
    sys.path.append(f'{pref}')
    sys.path.append(f'{pref}/RLLibrary')


from RLLibrary.FinUseCases.PortfolioManagement import StrategyManager
from RLLibrary.utils import constants as constants
from RLLibrary.FinUseCases.PortfolioManagement.ModelManager.A3C import Networks

reload(StrategyManager)

DATA_DIR = constants.DATA_DIR
MODEL_DIR = constants.MODEL_DIR


def main(algo, method = "FF"):


    actions = [
        [-0.03, 0.03], 
        [-0.02, 0.02], 
        [-0.01, 0.01], 
        [0.0, 0.0], 
        [0.01, -0.01], 
        [0.02, -0.02], 
        [0.03, -0.03], 
            ]

    path = os.path.join(MODEL_DIR, "PortfolioManagement")    
    path = os.path.join(path, algo)

    if not os.path.exists(path):
        os.makedirs(path)

    path = os.path.join(path, method)

    # ------- Simple FF network
    if algo == "A3C":
        if method == "FF":
            args = {"startDate" : "2018-01-01", "endDate" : "2018-12-31", "actions": actions}
            stgy = StrategyManager.RLStrategy_A3C(envName = "PortfolioManagement-v0", **args)

            cores = multiprocessing.cpu_count()
            stgy.train(cores = cores, save_dir = path, MAX_EPISODES = 4000, \
                        ActorCriticModel = None, \
                        actorHiddenUnits = [32], criticHiddenUnits = [32], optimizer_learning_rate = 1e-4)

        elif method == "CNN": 
            # ------- Simple FF network
            args = {"startDate" : "2018-01-01", "endDate" : "2018-12-31", "actions": actions, "nhistoricalDays": 30}
            stgy = StrategyManager.RLStrategy_A3C_CNN(envName = "PortfolioManagement_CNN-v0", **args)

            ActorCriticModel = Networks.ActorCritic_CNN(nbHistory=args["nhistoricalDays"], action_size=len(actions))
            cores = multiprocessing.cpu_count()
            stgy.train(cores = cores, save_dir = path, MAX_EPISODES = 4000, \
                        ActorCriticModel = ActorCriticModel, \
                        optimizer_learning_rate = 1e-4)

    elif algo == "DQN":
            args = {"startDate" : "2018-01-01", "endDate" : "2018-12-31", "actions": actions, "nhistoricalDays": 30}
            stgy = StrategyManager.RLStrategy_DQN(envName = "PortfolioManagement-v0", **args)

            stgy.train(save_dir=path, MAX_EPISODES=2000, \
                        DQNModel=None, hiddenUnits=[32], batchNormalization=True, dropout_rate=0.25, \
                        optimizer_learning_rate= 1e-4, clipvalue=100 )
        

    # once the strategy is run


if __name__ == "__main__":

    if len(sys.argv) == 3:
        algo    = sys.argv[1]
        method  = sys.argv[2]

    elif len(sys.argv) == 2:
        algo    = sys.argv[1]
        method  = "FF"      # Feed forward network

    elif len(sys.argv) == 1:
        algo    = "A3C"
        method  = "FF"


    main(algo = algo, method=method)




