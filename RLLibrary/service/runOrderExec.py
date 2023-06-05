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


from RLLibrary.FinUseCases.OrderExecution import ExecutionAlgos
from RLLibrary.utils import constants as constants

reload(ExecutionAlgos)

DATA_DIR = constants.DATA_DIR
MODEL_DIR = constants.MODEL_DIR



def main(algo, method = "FF"):


    path = os.path.join(MODEL_DIR, "OrderExecution")    
    path = os.path.join(path, algo)

    if not os.path.exists(path):
        os.makedirs(path)

    path = os.path.join(path, method)



    if algo == "DQN":
        args = {"ticker": "RELIANCE", "trainingYear": ["2018" , "2019"], \
                    "penalizeFactors": {"Impact": -100, "StepReward": 0.1}}

        stgy = ExecutionAlgos.Execution_DQN(envName = "OrderExecution-v0", **args)

        stgy.train(save_dir=path, MAX_EPISODES=1000, \
                    DQNModel=None, hiddenUnits=[20], batchNormalization=False, dropout_rate=0.25, \
                    optimizer_learning_rate= 1e-3, clipvalue=100 )
        


    if algo == "DDQN":
        args = {"ticker": "RELIANCE", "trainingYear": ["2018" , "2019"], \
                    "penalizeFactors": {"Impact": -100, "StepReward": 0.1}}

        stgy = ExecutionAlgos.Execution_DDQN(envName = "OrderExecution-v0", **args)

        stgy.train(save_dir=path, MAX_EPISODES=1000, \
                    DQNModel=None, hiddenUnits=[20], batchNormalization=False, dropout_rate=0.25, \
                    optimizer_learning_rate= 1e-3, clipvalue=100 )



    if algo == "A3C":
        args = {"ticker": "RELIANCE", "trainingYear": ["2018" , "2019"], \
                    "penalizeFactors": {"Impact": -100, "StepReward": 0.1}}

        stgy = ExecutionAlgos.Execution_A3C(envName = "OrderExecution-v0", **args)


        #cores = multiprocessing.cpu_count()
        cores = 3
        stgy.train(cores = cores, save_dir = path, MAX_EPISODES = 2000, \
                    ActorCriticModel = None, \
                    actorHiddenUnits = [20], criticHiddenUnits = [20], optimizer_learning_rate = 1e-3)




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

    #algo = "DQN"
    method = "FF"

    main(algo = algo, method=method)



