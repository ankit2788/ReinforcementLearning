from RLLibrary.FinUseCases import CustomGym
import os


from RLLibrary.utils import constants as constants
DATA_DIR = constants.DATA_DIR
MODEL_DIR = constants.MODEL_DIR

from RLLibrary.utils.loggingConfig import logger
Logger = logger.getLogger("EnvironmentStorage")


# ------register PortfolioManagement environment onto Custom gym

CustomGym.register(
    id = "PortfolioManagement-v0",
    entry_point = 'FinUseCases.PortfolioManagement.EnvironmentManager:Portfolio',
    kwargs = {"assets" : ["APA", "BMY"], "initialWeight" : [0.5, 0.5], \
                    "nhistoricalDays" : 30, \
                    "startDate" : "2019-01-01", "endDate" : "2019-12-31", \
                    "actions" : [(-0.1,0.1)], \
                    "assetDataPath" : os.path.join(DATA_DIR, "PortfolioManagement"), \
                    "config" : {"initialCash": 1000000, "minCash": 0.02, "transactionFee": 0.0001}, 
                    "penalizeFactors" : {"Risk": -0.08, "ForbiddenAction": -8}})


CustomGym.register(
    id = "PortfolioManagement_CNN-v0",
    entry_point = 'FinUseCases.PortfolioManagement.EnvironmentManager:Portfolio_MultiStage',
    kwargs = {"assets" : ["APA", "BMY"], "initialWeight" : [0.5, 0.5], \
                    "nhistoricalDays" : 30, \
                    "startDate" : "2019-01-01", "endDate" : "2019-12-31", \
                    "actions" : [(-0.1,0.1)], \
                    "assetDataPath" : os.path.join(DATA_DIR, "PortfolioManagement"), \
                    "config" : {"initialCash": 1000000, "minCash": 0.02, "transactionFee": 0.0001}, 
                    "penalizeFactors" : {"Risk": -0.08, "ForbiddenAction": -8}})



CustomGym.register(
    id = "OrderExecution-v0",
    entry_point = 'FinUseCases.OrderExecution.EnvironmentManager:OrderExecution',
    kwargs = {"ticker": "RELIANCE", \
                "orderConfig": {"initialOrderSize": 10000, "initialTimeHorizon": 100, "orderFactor": 500, \
                                "TotalIntervals": 50, "startTime": "09:30", "Timezone": "IST"}, \

                "nbHistory" : 15, \
                "trainingYear" : [], "testDate" : None, \
                "dataPath" : os.path.join(DATA_DIR, "OrderExecution"), \
                'penalizeFactors' : {"Impact": -100, "StepReward": 0.1}})
