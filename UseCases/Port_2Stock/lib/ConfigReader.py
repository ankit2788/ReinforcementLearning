import os
from configparser import ConfigParser
import pickle
import json
from logging.config import dictConfig
import logging

import loggingConfig
dictConfig(loggingConfig.DEFAULT_LOGGING)
Logger = logging.getLogger("Configurations")



class Config():

    def __init__(self, file,  Name, Type):

        self.Name      = Name
        self.configType = Type
        
        self.__parser       = ConfigParser()

        if os.path.exists(file):
            self.__parser.read(file)
        else:
            #TODO: set Default path
            pass

    def __call__(self, tag = "INITIAL_EXPLORATION"):

        return self.__parser.get(self.Name, tag)
    

    def save(self, filename, savePath, format = "json"):
        # default format is json
        # other option: pickle

        _config = dict(self.__parser.items(self.Name))

        finalpath = os.path.join(savePath, filename)
        if format.upper() == "PICKLE":
            with open(finalpath, "wb") as file:
                pickle.dump(_config, file)

        elif format.upper() == "JSON":
            with open(finalpath, "w") as file:
                json.dump(_config, file)

        Logger.info(f"{self.Name} {self.configType} config saved")



